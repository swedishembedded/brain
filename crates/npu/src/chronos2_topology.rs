// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build Chronos-2's transformer core as an ONNX graph (fixed sequence length
//! `S`) for whole-graph compilation on OpenVINO (NPU/GPU/CPU).
//!
//! The host keeps the cheap, awkward-in-ONNX pieces (the InstanceNorm+arcsinh
//! scaler, patch features, the patch-embed ResidualBlock, the REG-token splice,
//! and the quantile-head rearrange + denorm) — exactly how the GLM/Qwen NPU
//! exports keep embedding + sampling on host. The graph is the compute-heavy
//! encoder stack + final norm + the trailing quantile head:
//!   inputs:  `emb:[1,S,D]` (assembled token embeddings), `kmask:[1,1,1,S]` (additive)
//!   output:  `qhead:[1,n_out,Q*P]`
//!
//! Chronos-2 specifics vs GLM: attention is **UNSCALED** (no 1/sqrt(d_kv)) and
//! bidirectional (kmask, not causal); RoPE is **half-split / NeoX** (not
//! interleaved); the FFN is **ReLU** (not SwiGLU); RMSNorm weight-only. Brain
//! linear weights `[out,in]` are transposed once to ONNX `[in,out]`; `Int8`
//! stores them per-output-channel and dequantises in-graph.

use chronos2::Chronos2Config;
use onnx::builder::GraphBuilder;
use onnx::graph::Node;

use crate::qwen_topology::Quant;
use crate::topology::WeightSource;

/// Assemble the Chronos-2 graph into `g` (fp32).
pub fn build_chronos2_graph(cfg: &Chronos2Config, w: &dyn WeightSource, s: usize, n_out: usize, g: &mut GraphBuilder) {
    build_chronos2_graph_quant(cfg, w, s, n_out, g, Quant::F32);
}

/// As [`build_chronos2_graph`] with a weight-quantization mode.
pub fn build_chronos2_graph_quant(
    cfg: &Chronos2Config,
    w: &dyn WeightSource,
    s: usize,
    n_out: usize,
    g: &mut GraphBuilder,
    quant: Quant,
) {
    let d = cfg.d_model;
    let heads = cfg.num_heads;
    let hd = cfg.d_kv;
    let inner = cfg.inner_dim();
    let ff = cfg.d_ff;
    let head_out = cfg.head_out_dim();
    let si = s as i64;
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

    tp.g.input_f32("emb", &[1, si, d as i64]);
    tp.g.input_f32("kmask", &[1, 1, 1, si]);
    tp.g.output_f32("qhead", &[1, n_out as i64, head_out as i64]);

    tp.f32("c_eps", &[1], vec![cfg.layer_norm_epsilon]);
    // half-split RoPE cos/sin tables [1,S,1,hd/2]
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0f32; s * half], vec![0f32; s * half]);
    for p in 0..s {
        for j in 0..half {
            let ang = p as f32 * cfg.rope_theta.powf(-(2.0 * j as f32) / hd as f32);
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, si, 1, half as i64], cos);
    tp.f32("rope_sin", &[1, si, 1, half as i64], sin);
    // reshape shapes
    tp.i64("sh_heads", &[4], vec![1, si, heads as i64, hd as i64]);
    tp.i64("sh_flat", &[3], vec![1, si, inner as i64]);
    // half-split slice bounds (axis 3)
    tp.i64("h_lo0", &[1], vec![0]);
    tp.i64("h_hi0", &[1], vec![half as i64]);
    tp.i64("h_lo1", &[1], vec![half as i64]);
    tp.i64("h_hi1", &[1], vec![hd as i64]);
    tp.i64("h_ax", &[1], vec![3]);
    tp.i64("h_st", &[1], vec![1]);
    // slice for the trailing n_out tokens (axis 1)
    tp.i64("t_lo", &[1], vec![(s - n_out) as i64]);
    tp.i64("t_hi", &[1], vec![si]);
    tp.i64("t_ax", &[1], vec![1]);
    tp.i64("t_st", &[1], vec![1]);

    let mut x = "emb".to_string();
    for b in 0..cfg.num_layers {
        let pre = format!("encoder.block.{b}");

        // ---- time self-attention: bidirectional, UNSCALED ----
        let h = tp.rmsnorm(&x, &format!("{pre}.layer.0.layer_norm.weight"), w, d);
        let q = tp.linear(&h, &format!("{pre}.layer.0.self_attention.q.weight"), w, inner, d);
        let k = tp.linear(&h, &format!("{pre}.layer.0.self_attention.k.weight"), w, inner, d);
        let v = tp.linear(&h, &format!("{pre}.layer.0.self_attention.v.weight"), w, inner, d);
        let q4 = tp.reshape(&q, "sh_heads");
        let k4 = tp.reshape(&k, "sh_heads");
        let v4 = tp.reshape(&v, "sh_heads");
        let q4 = tp.rope_neox(&q4);
        let k4 = tp.rope_neox(&k4);
        let qt = tp.transpose(&q4, &[0, 2, 1, 3]); // [1,heads,S,hd]
        let kt = tp.transpose(&k4, &[0, 2, 1, 3]);
        let vt = tp.transpose(&v4, &[0, 2, 1, 3]);
        let ktt = tp.transpose(&kt, &[0, 1, 3, 2]);
        let scores = tp.matmul(&qt, &ktt); // NO scale
        let scores = tp.add(&scores, "kmask"); // additive key mask, broadcast
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &vt); // [1,heads,S,hd]
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]);
        let ctx = tp.reshape(&ctx, "sh_flat"); // [1,S,inner]
        let o = tp.linear(&ctx, &format!("{pre}.layer.0.self_attention.o.weight"), w, d, inner);
        x = tp.add_t(&x, &o);

        // ---- group attention (B=1 degeneration): o(v(rmsnorm(x))) ----
        let h2 = tp.rmsnorm(&x, &format!("{pre}.layer.1.layer_norm.weight"), w, d);
        let vg = tp.linear(&h2, &format!("{pre}.layer.1.self_attention.v.weight"), w, inner, d);
        let og = tp.linear(&vg, &format!("{pre}.layer.1.self_attention.o.weight"), w, d, inner);
        x = tp.add_t(&x, &og);

        // ---- ReLU FFN ----
        let h3 = tp.rmsnorm(&x, &format!("{pre}.layer.2.layer_norm.weight"), w, d);
        let a = tp.linear(&h3, &format!("{pre}.layer.2.mlp.wi.weight"), w, ff, d);
        let ar = tp.unary("Relu", &a);
        let ffo = tp.linear(&ar, &format!("{pre}.layer.2.mlp.wo.weight"), w, d, ff);
        x = tp.add_t(&x, &ffo);
    }

    let xf = tp.rmsnorm(&x, "encoder.final_layer_norm.weight", w, d);
    // trailing n_out tokens
    let tail = {
        let o = tp.tmp("tail");
        tp.g.add(Node::new("Slice", &[&xf, "t_lo", "t_hi", "t_ax", "t_st"], &[&o]));
        o
    };
    // quantile head ResidualBlock (biased) -> qhead
    tp.residual_block(&tail, "output_patch_embedding", w, d, ff, head_out, "qhead");
}

/// ONNX assembly helper (mirrors the GLM topology's `Topo`).
struct Topo<'a> {
    b: crate::topo::TopoBase<'a>,
    quant: Quant,
}

// DSL + shared math emitters live on `TopoBase` (crate::topo); Deref keeps this
// file's call sites unchanged. Only model-specific emission stays here.
impl<'a> std::ops::Deref for Topo<'a> {
    type Target = crate::topo::TopoBase<'a>;
    fn deref(&self) -> &Self::Target {
        &self.b
    }
}
impl<'a> std::ops::DerefMut for Topo<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.b
    }
}

impl<'a> Topo<'a> {
    /// Half-split (NeoX) RoPE on `[1,S,heads,hd]`: pair `(j, j+half)` rotated by
    /// `cos/sin` (broadcast over heads). `first' = first*cos - second*sin`,
    /// `second' = second*cos + first*sin`.
    fn rope_neox(&mut self, x: &str) -> String {
        let first = {
            let o = self.tmp("rf");
            self.g.add(Node::new("Slice", &[x, "h_lo0", "h_hi0", "h_ax", "h_st"], &[&o]));
            o
        };
        let second = {
            let o = self.tmp("rs2");
            self.g.add(Node::new("Slice", &[x, "h_lo1", "h_hi1", "h_ax", "h_st"], &[&o]));
            o
        };
        let fc = self.mul(&first, "rope_cos");
        let ss = self.mul(&second, "rope_sin");
        let new_first = self.sub_t(&fc, &ss);
        let sc = self.mul(&second, "rope_cos");
        let fs = self.mul(&first, "rope_sin");
        let new_second = self.add_t(&sc, &fs);
        self.concat2(&new_first, &new_second, 3)
    }

    fn rmsnorm(&mut self, x: &str, name: &str, w: &dyn WeightSource, dim: usize) -> String {
        let gain = format!("{name}.g");
        self.b.rmsnorm(x, &gain, w.get(name), dim, "c_eps")
    }

    /// Bias-free linear `y = x @ W^T`; brain `[out,in]` transposed to ONNX
    /// `[in,out]`, INT8 per-channel dequant when enabled.
    fn linear(&mut self, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        self.linear_named(x, name, &format!("{name}.wt"), w, out, inp, &o);
        o
    }
    fn linear_named(&mut self, x: &str, name: &str, winit: &str, w: &dyn WeightSource, out: usize, inp: usize, y: &str) {
        // Delegates to the ONE shared weight-quantizing emitter (crate::topo).
        let quant = self.quant;
        crate::topo::linear_quant(&mut self.b, x, name, winit, w, out, inp, quant, y);
    }

    /// A biased `ResidualBlock`: `output(relu(hidden(x))) + residual(x)`. Writes
    /// the final sum to `out_name`. The three linears carry biases (broadcast Add).
    fn residual_block(&mut self, x: &str, prefix: &str, w: &dyn WeightSource, in_dim: usize, h: usize, out_dim: usize, out_name: &str) {
        let hid = self.linear(x, &format!("{prefix}.hidden_layer.weight"), w, h, in_dim);
        let hid = self.bias(&hid, &format!("{prefix}.hidden_layer.bias"), w, h);
        let hr = self.unary("Relu", &hid);
        let o1 = self.linear(&hr, &format!("{prefix}.output_layer.weight"), w, out_dim, h);
        let o1 = self.bias(&o1, &format!("{prefix}.output_layer.bias"), w, out_dim);
        let res = self.linear(x, &format!("{prefix}.residual_layer.weight"), w, out_dim, in_dim);
        let res = self.bias(&res, &format!("{prefix}.residual_layer.bias"), w, out_dim);
        self.node("Add", &[&o1, &res], out_name);
    }
    fn bias(&mut self, x: &str, name: &str, w: &dyn WeightSource, dim: usize) -> String {
        let bn = format!("{name}.b");
        self.f32(&bn, &[dim as i64], w.get(name));
        self.add(x, &bn)
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn builds_a_valid_graph_at_tiny_scale() {
        let cfg = Chronos2Config::tiny();
        let w: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.01; s.iter().product()])).collect();
        let mut g = GraphBuilder::new("chronos2");
        build_chronos2_graph(&cfg, &w, 12, 2, &mut g);
        let bytes = g.finish();
        assert!(bytes.len() > 500, "onnx graph should serialize");
        // sanity: the graph declares the expected I/O tensor names
        let txt = String::from_utf8_lossy(&bytes);
        assert!(txt.contains("emb") && txt.contains("qhead") && txt.contains("kmask"));
    }
}
