// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build FinCast's transformer core as an ONNX graph (fixed sequence length `S`)
//! for whole-graph compilation on OpenVINO (NPU/GPU/CPU).
//!
//! The host keeps the cheap, awkward-in-ONNX pieces (patch features, the
//! `input_ff` patch-embed ResidualBlock, the `freq_emb` add, and the head
//! rearrange/denorm). The graph is the compute-heavy decoder stack + horizon
//! head:
//!   inputs:  `emb:[1,S,D]` (assembled token embeddings),
//!            `amask:[1,1,S,S]` (additive causal + padding mask)
//!   output:  `qhead:[1,S,head_out]`
//!
//! FinCast specifics: attention is **causal**, uses a **fused qkv** projection
//! (biased) with a learned **per-dim query scaling** `scale*softplus(scaling)`
//! folded into a constant, and **no RoPE**. The MLP is a **sparse top-2 MoE**
//! (`num_experts` experts, each `LayerNorm → gate_proj → ReLU → down_proj`)
//! whose deterministic routing is expressed in-graph via `TopK` + a
//! `GreaterOrEqual(prob, 2nd-largest)` mask (no `uniform` stochasticity).
//! The horizon head is a SiLU ResidualBlock.

use fincast::FincastConfig;
use onnx::builder::GraphBuilder;
use onnx::graph::Node;

use crate::qwen_topology::Quant;
use crate::topology::WeightSource;

/// Assemble the FinCast core graph into `g` (fp32).
pub fn build_fincast_graph(cfg: &FincastConfig, w: &dyn WeightSource, s: usize, g: &mut GraphBuilder) {
    build_fincast_graph_quant(cfg, w, s, g, Quant::F32);
}

/// As [`build_fincast_graph`] with a weight-quantization mode.
pub fn build_fincast_graph_quant(cfg: &FincastConfig, w: &dyn WeightSource, s: usize, g: &mut GraphBuilder, quant: Quant) {
    let d = cfg.hidden_size;
    let heads = cfg.num_heads;
    let hd = cfg.head_dim;
    let inner = cfg.inner_dim();
    let ff = cfg.intermediate_size;
    let head_out = cfg.head_out_dim();
    let e = cfg.num_experts;
    let si = s as i64;
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

    tp.g.input_f32("emb", &[1, si, d as i64]);
    tp.g.input_f32("amask", &[1, 1, si, si]);
    tp.g.output_f32("qhead", &[1, si, head_out as i64]);

    tp.f32("c_eps", &[1], vec![cfg.rms_norm_eps]);
    tp.f32("c_ln_eps", &[1], vec![1e-6]);
    // reshape shapes for heads
    tp.i64("sh_heads", &[4], vec![1, si, heads as i64, hd as i64]);
    tp.i64("sh_flat", &[3], vec![1, si, inner as i64]);
    // fused-qkv slice bounds (axis 2)
    tp.i64("q_lo", &[1], vec![0]);
    tp.i64("q_hi", &[1], vec![inner as i64]);
    tp.i64("k_lo", &[1], vec![inner as i64]);
    tp.i64("k_hi", &[1], vec![2 * inner as i64]);
    tp.i64("v_lo", &[1], vec![2 * inner as i64]);
    tp.i64("v_hi", &[1], vec![3 * inner as i64]);
    tp.i64("ax2", &[1], vec![2]);
    tp.i64("st1", &[1], vec![1]);
    // MoE: TopK K + the "2nd largest" slice bounds (axis 2 of [1,S,K])
    tp.i64("topk_k", &[1], vec![cfg.gating_top_n as i64]);
    tp.i64("thr_lo", &[1], vec![(cfg.gating_top_n - 1) as i64]);
    tp.i64("thr_hi", &[1], vec![cfg.gating_top_n as i64]);

    let mut x = "emb".to_string();
    for b in 0..cfg.num_layers {
        let pre = format!("stacked_transformer.layers.{b}");

        // ---- causal attention (fused qkv, per-dim softplus q-scale) ----
        let h = tp.rmsnorm(&x, &format!("{pre}.input_layernorm.weight"), w, d, "c_eps");
        let qkv = tp.linear_bias(&h, &format!("{pre}.self_attn.qkv_proj"), w, cfg.qkv_dim(), d);
        let q = tp.slice(&qkv, "q_lo", "q_hi", "ax2");
        let k = tp.slice(&qkv, "k_lo", "k_hi", "ax2");
        let v = tp.slice(&qkv, "v_lo", "v_hi", "ax2");
        // per-dim q scaling folded into a constant vector [inner]
        let qscale = qscale_const(cfg, w, b);
        let qsname = format!("{pre}.qscale");
        tp.f32(&qsname, &[inner as i64], qscale);
        let q = tp.mul(&q, &qsname);
        let q4 = tp.reshape(&q, "sh_heads");
        let k4 = tp.reshape(&k, "sh_heads");
        let v4 = tp.reshape(&v, "sh_heads");
        let qt = tp.transpose(&q4, &[0, 2, 1, 3]); // [1,heads,S,hd]
        let kt = tp.transpose(&k4, &[0, 2, 1, 3]);
        let vt = tp.transpose(&v4, &[0, 2, 1, 3]);
        let ktt = tp.transpose(&kt, &[0, 1, 3, 2]);
        let scores = tp.matmul(&qt, &ktt); // scale already folded into q
        let scores = tp.add(&scores, "amask");
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &vt);
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]);
        let ctx = tp.reshape(&ctx, "sh_flat");
        let o = tp.linear_bias(&ctx, &format!("{pre}.self_attn.o_proj"), w, d, inner);
        x = tp.add_t(&x, &o);

        // ---- sparse top-2 MoE ----
        let p = tp.rmsnorm(&x, &format!("{pre}.moe.moe_prenorm.gamma"), w, d, "c_eps");
        let glog = tp.linear(&p, &format!("{pre}.moe.moe.gate.to_gates.weight"), w, e, d);
        let gprobs = tp.softmax(&glog, -1); // [1,S,E]
        // TopK values [1,S,K]; the K-th (last) is the routing threshold.
        let (tk_vals, _tk_idx) = tp.topk(&gprobs, "topk_k", 2);
        let thr = tp.slice(&tk_vals, "thr_lo", "thr_hi", "ax2"); // [1,S,1]
        let ge = tp.cmp_ge(&gprobs, &thr); // bool [1,S,E]
        let mask = tp.cast_f32(&ge);
        let gated = tp.mul(&gprobs, &mask);
        let denom = tp.reduce_sum(&gated, -1); // [1,S,1] keepdims
        let wnorm = tp.div(&gated, &denom); // [1,S,E] per-expert combine weight
        // moe_out = p + Σ_e w_e * expert_e(p)
        let mut moe_out = p.clone();
        for ei in 0..e {
            let ep = format!("{pre}.moe.moe.experts.experts.{ei}");
            let ln = tp.layernorm(&p, &format!("{ep}.layer_norm"), w, d, "c_ln_eps");
            let gpj = tp.linear_bias(&ln, &format!("{ep}.gate_proj"), w, d, d);
            let gr = tp.unary("Relu", &gpj);
            let mlp = tp.linear_bias(&gr, &format!("{ep}.down_proj"), w, d, d);
            // slice this expert's weight column [1,S,1] and broadcast-multiply
            let lo = format!("we_lo_{ei}");
            let hi = format!("we_hi_{ei}");
            tp.i64(&lo, &[1], vec![ei as i64]);
            tp.i64(&hi, &[1], vec![(ei + 1) as i64]);
            let we = tp.slice(&wnorm, &lo, &hi, "ax2"); // [1,S,1]
            let contrib = tp.mul(&mlp, &we);
            moe_out = tp.add_t(&moe_out, &contrib);
        }
        x = tp.add_t(&x, &moe_out); // outer residual: block_out = moe_out + x_in
    }

    // horizon head: SiLU ResidualBlock -> qhead [1,S,head_out]
    tp.residual_block_silu(&x, "horizon_ff_layer", w, d, ff, head_out, "qhead");
}

/// Per-dim query scale for layer `b`: `base * softplus(scaling[dd])` tiled over
/// the `num_heads` heads (length `inner`).
fn qscale_const(cfg: &FincastConfig, w: &dyn WeightSource, b: usize) -> Vec<f32> {
    let hd = cfg.head_dim;
    let heads = cfg.num_heads;
    // log2(e): same base change as `fincast::model` — kept identical so the
    // exported ONNX graph matches the in-repo forward bit for bit. (As f32 this
    // is bit-identical to the `1.442695041f32` literal it replaced.)
    let base = std::f32::consts::LOG2_E / (hd as f32).sqrt();
    let scaling = w.get(&format!("stacked_transformer.layers.{b}.self_attn.scaling"));
    let mut out = vec![0f32; heads * hd];
    for h in 0..heads {
        for dd in 0..hd {
            let sp = if scaling[dd] > 20.0 { scaling[dd] } else { (1.0 + scaling[dd].exp()).ln() };
            out[h * hd + dd] = base * sp;
        }
    }
    out
}

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

impl Topo<'_> {
    fn div(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("div");
        self.node("Div", &[a, b], &o);
        o
    }
    fn reduce_sum(&mut self, x: &str, axis: i64) -> String {
        // opset-13 ReduceSum takes `axes` as an INPUT tensor (not an attribute, as
        // ReduceMean still does until opset 18) — pass it as an initializer.
        let ax = format!("sum_axes_{axis}");
        self.i64(&ax, &[1], vec![axis]);
        let o = self.tmp("rsum");
        self.g.add(Node::new("ReduceSum", &[x, &ax], &[&o]).attr_int("keepdims", 1));
        o
    }
    fn topk(&mut self, x: &str, k: &str, largest: i64) -> (String, String) {
        let vals = self.tmp("tkv");
        let idx = self.tmp("tki");
        self.g.add(
            Node::new("TopK", &[x, k], &[&vals, &idx]).attr_int("axis", -1).attr_int("largest", largest).attr_int("sorted", 1),
        );
        (vals, idx)
    }
    fn cmp_ge(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("ge");
        self.node("GreaterOrEqual", &[a, b], &o);
        o
    }
    fn cast_f32(&mut self, x: &str) -> String {
        let o = self.tmp("castf");
        self.g.add(Node::new("Cast", &[x], &[&o]).attr_int("to", 1)); // 1 = FLOAT
        o
    }

    fn rmsnorm(&mut self, x: &str, name: &str, w: &dyn WeightSource, dim: usize, eps: &str) -> String {
        let gain = format!("{name}.g");
        self.b.rmsnorm(x, &gain, w.get(name), dim, eps)
    }

    /// LayerNorm with gain+bias: `(x-mean)/sqrt(var+eps)*g + b`.
    fn layernorm(&mut self, x: &str, prefix: &str, w: &dyn WeightSource, dim: usize, eps: &str) -> String {
        let gain = format!("{prefix}.weight.g");
        let bias = format!("{prefix}.bias.b");
        let gw = w.get(&format!("{prefix}.weight"));
        let bw = w.get(&format!("{prefix}.bias"));
        self.b.layernorm(x, &gain, gw, &bias, bw, dim, eps)
    }

    /// Bias-free linear `y = x @ W^T` (brain `[out,in]` -> ONNX `[in,out]`).
    fn linear(&mut self, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        self.linear_named(x, name, &format!("{name}.wt"), w, out, inp, &o);
        o
    }
    /// Biased linear: `prefix.weight` / `prefix.bias`.
    fn linear_bias(&mut self, x: &str, prefix: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let y = self.linear(x, &format!("{prefix}.weight"), w, out, inp);
        let bn = format!("{prefix}.bias.b");
        self.f32(&bn, &[out as i64], w.get(&format!("{prefix}.bias")));
        self.add(&y, &bn)
    }
    fn linear_named(&mut self, x: &str, name: &str, winit: &str, w: &dyn WeightSource, out: usize, inp: usize, y: &str) {
        crate::topo::linear_quant(&mut self.b, x, name, winit, w, out, inp, self.quant, y);
    }

    /// A biased ResidualBlock with a SiLU hidden nonlinearity:
    /// `output(silu(hidden(x))) + residual(x)`. Writes to `out_name`.
    fn residual_block_silu(&mut self, x: &str, prefix: &str, w: &dyn WeightSource, in_dim: usize, h: usize, out_dim: usize, out_name: &str) {
        let hid = self.linear_bias(x, &format!("{prefix}.hidden_layer.0"), w, h, in_dim);
        let sg = self.unary("Sigmoid", &hid);
        let hr = self.mul(&hid, &sg); // SiLU
        let o1 = self.linear_bias(&hr, &format!("{prefix}.output_layer"), w, out_dim, h);
        let res = self.linear_bias(x, &format!("{prefix}.residual_layer"), w, out_dim, in_dim);
        self.node("Add", &[&o1, &res], out_name);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn builds_a_valid_graph_at_tiny_scale() {
        let cfg = FincastConfig::tiny();
        let w: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.01; s.iter().product()])).collect();
        let mut g = GraphBuilder::new("fincast");
        build_fincast_graph(&cfg, &w, 8, &mut g);
        let bytes = g.finish();
        assert!(bytes.len() > 500, "onnx graph should serialize");
        let txt = String::from_utf8_lossy(&bytes);
        assert!(txt.contains("emb") && txt.contains("qhead") && txt.contains("amask"));
    }
}
