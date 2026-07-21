// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build Kronos's autoregressive-decoder core (`decode_s1`) as an ONNX graph at
//! a fixed context length `T`, for whole-graph compilation on OpenVINO
//! (NPU/GPU/CPU).
//!
//! The host keeps the cheap, awkward-in-ONNX pieces — the hierarchical (s1,s2)
//! embedding gather + `×√d` scale, the `fusion_proj`, and the summed calendar
//! embeddings — exactly how the Chronos-2 / GLM / Qwen exports keep embedding +
//! sampling on host. The graph is the compute-heavy transformer stack + final
//! norm + the `proj_s1` head:
//!   input:   `x:[1,T,D]` (host-assembled fused embedding)
//!   outputs: `ctx:[1,T,D]` (final-norm context, feeds `decode_s2` next),
//!            `s1_logits:[1,T,s1_vocab]`
//!
//! Kronos specifics vs Chronos-2 (see `kronos::config`): attention is **CAUSAL +
//! SCALED** (`1/√head_dim`, lower-triangular additive mask); the q/k/v/out
//! projections are **biased**; RoPE is **half-split / NeoX** (shared with
//! Chronos-2); the FFN is **SwiGLU** (`w2(silu(w1·x)·(w3·x))`, no bias); norm is
//! RMSNorm (eps 1e-5, weight-only). Brain linear weights `[out,in]` are
//! transposed once to ONNX `[in,out]`; `Int8` stores them per-output-channel and
//! dequantises in-graph.

use std::collections::HashMap;

use kronos::KronosConfig;
use onnx::builder::GraphBuilder;
use onnx::graph::Node;

use crate::qwen_topology::Quant;

type W = HashMap<String, Vec<f32>>;

/// Assemble the Kronos decoder `decode_s1` graph into `g` (fp32).
pub fn build_kronos_decoder_graph(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    build_kronos_decoder_graph_quant(cfg, w, t, g, Quant::F32);
}

/// As [`build_kronos_decoder_graph`] with a weight-quantization mode.
pub fn build_kronos_decoder_graph_quant(
    cfg: &KronosConfig,
    w: &W,
    t: usize,
    g: &mut GraphBuilder,
    quant: Quant,
) {
    let d = cfg.d_model;
    let heads = cfg.n_heads;
    let hd = d / heads;
    let ff = cfg.ff_dim;
    let s1v = cfg.s1_vocab();
    let ti = t as i64;
    let mut tp = Topo { g, n: 0, quant };

    tp.g.input_f32("x", &[1, ti, d as i64]);
    tp.g.output_f32("ctx", &[1, ti, d as i64]);
    tp.g.output_f32("s1_logits", &[1, ti, s1v as i64]);

    tp.f32("c_eps", &[1], vec![1e-5]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    // lower-triangular additive causal mask [1,1,T,T]: 0 on/below diag, -1e9 above.
    let mut mask = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask[i * t + j] = -1e9;
            }
        }
    }
    tp.f32("causal_mask", &[1, 1, ti, ti], mask);
    // half-split RoPE cos/sin tables [1,T,1,hd/2] (theta 10000, matching `Ops`).
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0f32; t * half], vec![0f32; t * half]);
    for p in 0..t {
        for j in 0..half {
            let ang = p as f32 * 10000f32.powf(-(2.0 * j as f32) / hd as f32);
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, ti, 1, half as i64], cos);
    tp.f32("rope_sin", &[1, ti, 1, half as i64], sin);
    // reshape shapes
    tp.i64("sh_heads", &[4], vec![1, ti, heads as i64, hd as i64]);
    tp.i64("sh_flat", &[3], vec![1, ti, d as i64]);
    // half-split slice bounds (axis 3)
    tp.i64("h_lo0", &[1], vec![0]);
    tp.i64("h_hi0", &[1], vec![half as i64]);
    tp.i64("h_lo1", &[1], vec![half as i64]);
    tp.i64("h_hi1", &[1], vec![hd as i64]);
    tp.i64("h_ax", &[1], vec![3]);
    tp.i64("h_st", &[1], vec![1]);

    let mut x = "x".to_string();
    for b in 0..cfg.n_layers {
        let pre = format!("transformer.{b}");

        // ---- self-attention: CAUSAL, SCALED, biased q/k/v/out ----
        let h = tp.rmsnorm(&x, &format!("{pre}.norm1.weight"), w, d);
        let q = tp.linear_biased(&h, &format!("{pre}.self_attn.q_proj"), w, d, d);
        let k = tp.linear_biased(&h, &format!("{pre}.self_attn.k_proj"), w, d, d);
        let v = tp.linear_biased(&h, &format!("{pre}.self_attn.v_proj"), w, d, d);
        let q4 = tp.reshape(&q, "sh_heads");
        let k4 = tp.reshape(&k, "sh_heads");
        let v4 = tp.reshape(&v, "sh_heads");
        let q4 = tp.rope_neox(&q4);
        let k4 = tp.rope_neox(&k4);
        let qt = tp.transpose(&q4, &[0, 2, 1, 3]); // [1,heads,T,hd]
        let kt = tp.transpose(&k4, &[0, 2, 1, 3]);
        let vt = tp.transpose(&v4, &[0, 2, 1, 3]);
        let ktt = tp.transpose(&kt, &[0, 1, 3, 2]);
        let scores = tp.matmul(&qt, &ktt);
        let scores = tp.mul(&scores, "c_scale"); // 1/sqrt(hd)
        let scores = tp.add(&scores, "causal_mask"); // lower-triangular, broadcast heads
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &vt); // [1,heads,T,hd]
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]);
        let ctx = tp.reshape(&ctx, "sh_flat"); // [1,T,D]
        let o = tp.linear_biased(&ctx, &format!("{pre}.self_attn.out_proj"), w, d, d);
        x = tp.add_t(&x, &o);

        // ---- SwiGLU FFN (no bias): w2(silu(w1 x) · (w3 x)) ----
        let h2 = tp.rmsnorm(&x, &format!("{pre}.norm2.weight"), w, d);
        let a = tp.linear(&h2, &format!("{pre}.ffn.w1.weight"), w, ff, d);
        let bb = tp.linear(&h2, &format!("{pre}.ffn.w3.weight"), w, ff, d);
        let sa = tp.silu(&a);
        let g_ = tp.mul_t(&sa, &bb);
        let ffo = tp.linear(&g_, &format!("{pre}.ffn.w2.weight"), w, d, ff);
        x = tp.add_t(&x, &ffo);
    }

    // final norm -> `ctx` (graph output + decode_s2 input), then proj_s1 head.
    tp.rmsnorm_to(&x, "norm.weight", w, d, "ctx");
    tp.linear_biased_to("ctx", "head.proj_s1", w, s1v, d, "s1_logits");
}

/// Assemble the Kronos `decode_s2` dependency-layer graph into `g` (fp32).
pub fn build_kronos_dep_graph(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    build_kronos_dep_graph_quant(cfg, w, t, g, Quant::F32);
}

/// As [`build_kronos_dep_graph`] with a weight-quantization mode. Inputs `ctx`
/// (the `decode_s1` final-norm context) and `sib` (host-gathered RAW `emb_s1`
/// rows for the just-sampled s1), both `[1,T,D]`; output `s2_logits:[1,T,s2v]`.
/// Cross-attention is **non-causal, SCALED** with `dep_n_heads` heads (head dim
/// `D/dep_n_heads`, so its own RoPE tables); then `norm(ctx + attn)` → `proj_s2`.
pub fn build_kronos_dep_graph_quant(
    cfg: &KronosConfig,
    w: &W,
    t: usize,
    g: &mut GraphBuilder,
    quant: Quant,
) {
    let d = cfg.d_model;
    let heads = cfg.dep_n_heads;
    let hd = d / heads;
    let s2v = cfg.s2_vocab();
    let ti = t as i64;
    let mut tp = Topo { g, n: 0, quant };

    tp.g.input_f32("ctx", &[1, ti, d as i64]);
    tp.g.input_f32("sib", &[1, ti, d as i64]);
    tp.g.output_f32("s2_logits", &[1, ti, s2v as i64]);

    tp.f32("c_eps", &[1], vec![1e-5]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0f32; t * half], vec![0f32; t * half]);
    for p in 0..t {
        for j in 0..half {
            let ang = p as f32 * 10000f32.powf(-(2.0 * j as f32) / hd as f32);
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, ti, 1, half as i64], cos);
    tp.f32("rope_sin", &[1, ti, 1, half as i64], sin);
    tp.i64("sh_heads", &[4], vec![1, ti, heads as i64, hd as i64]);
    tp.i64("sh_flat", &[3], vec![1, ti, d as i64]);
    tp.i64("h_lo0", &[1], vec![0]);
    tp.i64("h_hi0", &[1], vec![half as i64]);
    tp.i64("h_lo1", &[1], vec![half as i64]);
    tp.i64("h_hi1", &[1], vec![hd as i64]);
    tp.i64("h_ax", &[1], vec![3]);
    tp.i64("h_st", &[1], vec![1]);

    let cx = "dep_layer.cross_attn";
    let q = tp.linear_biased("sib", &format!("{cx}.q_proj"), w, d, d);
    let k = tp.linear_biased("ctx", &format!("{cx}.k_proj"), w, d, d);
    let v = tp.linear_biased("ctx", &format!("{cx}.v_proj"), w, d, d);
    let q4 = tp.reshape(&q, "sh_heads");
    let k4 = tp.reshape(&k, "sh_heads");
    let v4 = tp.reshape(&v, "sh_heads");
    let q4 = tp.rope_neox(&q4);
    let k4 = tp.rope_neox(&k4);
    let qt = tp.transpose(&q4, &[0, 2, 1, 3]);
    let kt = tp.transpose(&k4, &[0, 2, 1, 3]);
    let vt = tp.transpose(&v4, &[0, 2, 1, 3]);
    let ktt = tp.transpose(&kt, &[0, 1, 3, 2]);
    let scores = tp.matmul(&qt, &ktt);
    let scores = tp.mul(&scores, "c_scale"); // non-causal: no mask add
    let probs = tp.softmax(&scores, -1);
    let att = tp.matmul(&probs, &vt);
    let att = tp.transpose(&att, &[0, 2, 1, 3]);
    let att = tp.reshape(&att, "sh_flat");
    let o = tp.linear_biased(&att, &format!("{cx}.out_proj"), w, d, d);
    let sum = tp.add_t("ctx", &o);
    tp.rmsnorm_to(&sum, "dep_layer.norm.weight", w, d, "s2_normed");
    tp.linear_biased_to("s2_normed", "head.proj_s2", w, s2v, d, "s2_logits");
}

/// ONNX assembly helper (mirrors the Chronos-2 topology's `Topo`, extended with
/// biased linears, SiLU, and named-output writers).
struct Topo<'a> {
    g: &'a mut GraphBuilder,
    n: usize,
    quant: Quant,
}

impl<'a> Topo<'a> {
    fn tmp(&mut self, tag: &str) -> String {
        self.n += 1;
        format!("{tag}_{}", self.n)
    }
    fn has(&self, name: &str) -> bool {
        self.g.graph().initializers.iter().any(|t| t.name == name)
    }
    fn f32(&mut self, name: &str, dims: &[i64], data: Vec<f32>) {
        if !self.has(name) {
            self.g.init_f32(name, dims, data);
        }
    }
    fn i64(&mut self, name: &str, dims: &[i64], data: Vec<i64>) {
        if !self.has(name) {
            self.g.init_i64(name, dims, data);
        }
    }
    fn node(&mut self, op: &str, ins: &[&str], out: &str) {
        self.g.add(Node::new(op, ins, &[out]));
    }
    fn unary(&mut self, op: &str, x: &str) -> String {
        let o = self.tmp(&op.to_lowercase());
        self.node(op, &[x], &o);
        o
    }
    fn mul(&mut self, x: &str, c: &str) -> String {
        let o = self.tmp("mul");
        self.node("Mul", &[x, c], &o);
        o
    }
    fn mul_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("mult");
        self.node("Mul", &[a, b], &o);
        o
    }
    fn add(&mut self, x: &str, c: &str) -> String {
        let o = self.tmp("add");
        self.node("Add", &[x, c], &o);
        o
    }
    fn add_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("res");
        self.node("Add", &[a, b], &o);
        o
    }
    fn sub_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("sub");
        self.node("Sub", &[a, b], &o);
        o
    }
    fn matmul(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("mm");
        self.node("MatMul", &[a, b], &o);
        o
    }
    fn reshape(&mut self, x: &str, shape: &str) -> String {
        let o = self.tmp("rs");
        self.node("Reshape", &[x, shape], &o);
        o
    }
    fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        let o = self.tmp("tr");
        self.g.add(Node::new("Transpose", &[x], &[&o]).attr_ints("perm", perm));
        o
    }
    fn softmax(&mut self, x: &str, axis: i64) -> String {
        let o = self.tmp("sm");
        self.g.add(Node::new("Softmax", &[x], &[&o]).attr_int("axis", axis));
        o
    }
    fn concat2(&mut self, a: &str, b: &str, axis: i64) -> String {
        let o = self.tmp("cat");
        self.g.add(Node::new("Concat", &[a, b], &[&o]).attr_int("axis", axis));
        o
    }
    /// SiLU / swish: `x * sigmoid(x)`.
    fn silu(&mut self, x: &str) -> String {
        let s = self.unary("Sigmoid", x);
        self.mul_t(x, &s)
    }

    /// Half-split (NeoX) RoPE on `[1,T,heads,hd]`: pair `(j, j+half)` rotated by
    /// `cos/sin` (broadcast over heads). Shared math with the Chronos-2 export.
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

    fn rmsnorm(&mut self, x: &str, name: &str, w: &W, dim: usize) -> String {
        let o = self.tmp("rmsn");
        self.rmsnorm_to(x, name, w, dim, &o);
        o
    }
    /// RMSNorm writing its final scaled tensor to `out_name`.
    fn rmsnorm_to(&mut self, x: &str, name: &str, w: &W, dim: usize, out_name: &str) {
        let gain = format!("{name}.g");
        self.f32(&gain, &[dim as i64], w[name].clone());
        let sq = self.mul_t(x, x);
        let ms = {
            let o = self.tmp("rms_mean");
            self.g.add(
                Node::new("ReduceMean", &[&sq], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1),
            );
            o
        };
        let mse = self.add(&ms, "c_eps");
        let rms = self.unary("Sqrt", &mse);
        let nrm = {
            let o = self.tmp("rms_div");
            self.node("Div", &[x, &rms], &o);
            o
        };
        self.node("Mul", &[&nrm, &gain], out_name);
    }

    /// Bias-free linear `y = x @ W^T`; brain `[out,in]` transposed to ONNX
    /// `[in,out]`, INT8 per-channel dequant when enabled.
    fn linear(&mut self, x: &str, name: &str, w: &W, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        self.matmul_w(x, name, &format!("{name}.wt"), w, out, inp, &o);
        o
    }
    /// Biased linear `y = x @ W^T + b` (weight `<prefix>.weight`, bias
    /// `<prefix>.bias`), into a fresh tensor.
    fn linear_biased(&mut self, x: &str, prefix: &str, w: &W, out: usize, inp: usize) -> String {
        let o = self.tmp("linb");
        self.linear_biased_to(x, prefix, w, out, inp, &o);
        o
    }
    /// Biased linear writing the bias-add to `out_name`.
    fn linear_biased_to(&mut self, x: &str, prefix: &str, w: &W, out: usize, inp: usize, out_name: &str) {
        let wname = format!("{prefix}.weight");
        let y = self.linear(x, &wname, w, out, inp);
        let bn = format!("{prefix}.bias.b");
        self.f32(&bn, &[out as i64], w[&format!("{prefix}.bias")].clone());
        self.node("Add", &[&y, &bn], out_name);
    }

    fn matmul_w(&mut self, x: &str, name: &str, winit: &str, w: &W, out: usize, inp: usize, y: &str) {
        let qmax = match self.quant {
            Quant::F32 => {
                if !self.has(winit) {
                    let wt = transpose(&w[name], out, inp);
                    self.f32(winit, &[inp as i64, out as i64], wt);
                }
                self.node("MatMul", &[x, winit], y);
                return;
            }
            Quant::Int8 => 127.0f32,
            Quant::Int4 => 7.0f32,
        };
        let wq = format!("{winit}.q");
        if !self.has(&wq) {
            let wt = transpose(&w[name], out, inp);
            let mut scales = vec![0f32; out];
            let mut q = vec![0i8; inp * out];
            for oc in 0..out {
                let mut mx = 0f32;
                for i in 0..inp {
                    mx = mx.max(wt[i * out + oc].abs());
                }
                let sc = if mx > 0.0 { mx / qmax } else { 1.0 };
                scales[oc] = sc;
                for i in 0..inp {
                    q[i * out + oc] = (wt[i * out + oc] / sc).round().clamp(-qmax, qmax) as i8;
                }
            }
            let zp = format!("{winit}.zp");
            if matches!(self.quant, Quant::Int4) {
                self.g.init_i4(&wq, &[inp as i64, out as i64], q);
                self.g.init_i4(&zp, &[out as i64], vec![0i8; out]);
            } else {
                self.g.init_i8(&wq, &[inp as i64, out as i64], q);
                self.g.init_i8(&zp, &[out as i64], vec![0i8; out]);
            }
            self.f32(&format!("{winit}.s"), &[out as i64], scales);
            self.g.add(Node::new("DequantizeLinear", &[&wq, &format!("{winit}.s"), &zp], &[winit]).attr_int("axis", 1));
        }
        self.node("MatMul", &[x, winit], y);
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_valid_graph_at_tiny_scale() {
        let cfg = KronosConfig::tiny();
        let w: W = cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.01; s.iter().product()])).collect();
        let mut g = GraphBuilder::new("kronos_decoder");
        build_kronos_decoder_graph(&cfg, &w, 8, &mut g);
        let bytes = g.finish();
        assert!(bytes.len() > 500, "onnx graph should serialize");
        let txt = String::from_utf8_lossy(&bytes);
        assert!(txt.contains("x") && txt.contains("s1_logits") && txt.contains("ctx"));
    }
}
