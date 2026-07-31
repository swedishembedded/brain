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
pub fn build_kronos_decoder_graph_quant(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder, quant: Quant) {
    s1_stack(cfg, w, t, g, quant, false);
}

/// s1 PREFILL: the full-window decoder, additionally emitting per-layer RoPE'd
/// `k_{l}`/`v_{l}` `[1,heads,T,hd]` — these seed the single-token decode's KV cache
/// (`build_kronos_s1_decode_graph`). Same math as the plain decoder graph.
pub fn build_kronos_s1_prefill_graph_quant(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder, quant: Quant) {
    s1_stack(cfg, w, t, g, quant, true);
}
pub fn build_kronos_s1_prefill_graph(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    s1_stack(cfg, w, t, g, Quant::F32, true);
}

fn s1_stack(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder, quant: Quant, emit_kv: bool) {
    let d = cfg.d_model;
    let heads = cfg.n_heads;
    let hd = d / heads;
    let ff = cfg.ff_dim;
    let s1v = cfg.s1_vocab();
    let ti = t as i64;
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

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
        // When seeding the decode cache, the RoPE'd K/V transposes write directly to
        // the graph outputs `k_{b}`/`v_{b}` (a declared output can still be consumed
        // downstream — writing via an Identity lets OpenVINO fold+drop the output).
        let (kt, vt) = if emit_kv {
            let (kn, vn) = (format!("k_{b}"), format!("v_{b}"));
            tp.g.output_f32(&kn, &[1, heads as i64, t as i64, hd as i64]);
            tp.g.output_f32(&vn, &[1, heads as i64, t as i64, hd as i64]);
            tp.g.add(Node::new("Transpose", &[&k4], &[&kn]).attr_ints("perm", &[0, 2, 1, 3]));
            tp.g.add(Node::new("Transpose", &[&v4], &[&vn]).attr_ints("perm", &[0, 2, 1, 3]));
            (kn, vn)
        } else {
            (tp.transpose(&k4, &[0, 2, 1, 3]), tp.transpose(&v4, &[0, 2, 1, 3]))
        };
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

/// dep PREFILL: project the whole context `ctx:[1,T,D]` to the cross-attention
/// K/V once, `dep_k`/`dep_v:[1,dep_heads,T,dep_hd]` (K RoPE'd at each position),
/// seeding the dep decode cache — mirrors host `ensure_dep_kv`.
pub fn build_kronos_dep_prefill_graph(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    build_kronos_dep_prefill_graph_quant(cfg, w, t, g, Quant::F32);
}
pub fn build_kronos_dep_prefill_graph_quant(cfg: &KronosConfig, w: &W, t: usize, g: &mut GraphBuilder, quant: Quant) {
    let d = cfg.d_model;
    let heads = cfg.dep_n_heads;
    let hd = d / heads;
    let half = hd / 2;
    let ti = t as i64;
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

    tp.g.input_f32("ctx", &[1, ti, d as i64]);
    tp.g.output_f32("dep_k", &[1, heads as i64, ti, hd as i64]);
    tp.g.output_f32("dep_v", &[1, heads as i64, ti, hd as i64]);

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
    tp.i64("h_lo0", &[1], vec![0]);
    tp.i64("h_hi0", &[1], vec![half as i64]);
    tp.i64("h_lo1", &[1], vec![half as i64]);
    tp.i64("h_hi1", &[1], vec![hd as i64]);
    tp.i64("h_ax", &[1], vec![3]);
    tp.i64("h_st", &[1], vec![1]);

    let cx = "dep_layer.cross_attn";
    let k = tp.linear_biased("ctx", &format!("{cx}.k_proj"), w, d, d);
    let v = tp.linear_biased("ctx", &format!("{cx}.v_proj"), w, d, d);
    let k4 = tp.reshape(&k, "sh_heads");
    let v4 = tp.reshape(&v, "sh_heads");
    let k4 = tp.rope_neox(&k4);
    tp.g.add(Node::new("Transpose", &[&k4], &["dep_k"]).attr_ints("perm", &[0, 2, 1, 3]));
    tp.g.add(Node::new("Transpose", &[&v4], &["dep_v"]).attr_ints("perm", &[0, 2, 1, 3]));
}

/// dep DECODE (single token): `sib:[1,1,D]` (RAW emb_s1 of the sampled s1) +
/// `ctx_last:[1,1,D]` (this position's s1 context) + `rope_cos`/`rope_sin:
/// [1,1,1,dep_hd/2]` + `dep_mask:[1,1,1,cap]` + `past_dep_k`/`past_dep_v:
/// [1,dep_heads,cap,dep_hd]` → `new_dep_k`/`new_dep_v:[1,dep_heads,1,dep_hd]`,
/// `s2_logits:[1,1,s2v]`. The dep cross-attn is non-causal but during the rollout
/// only positions `0..=pos` exist, so `dep_mask` (0 on filled slots) + the self
/// score attends exactly the same set the full-window dep graph would.
pub fn build_kronos_dep_decode_graph(cfg: &KronosConfig, w: &W, cap: usize, g: &mut GraphBuilder) {
    build_kronos_dep_decode_graph_quant(cfg, w, cap, g, Quant::F32);
}
pub fn build_kronos_dep_decode_graph_quant(cfg: &KronosConfig, w: &W, cap: usize, g: &mut GraphBuilder, quant: Quant) {
    let d = cfg.d_model;
    let heads = cfg.dep_n_heads;
    let hd = d / heads;
    let half = hd / 2;
    let s2v = cfg.s2_vocab();
    let ci = cap as i64;
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

    tp.g.input_f32("sib", &[1, 1, d as i64]);
    tp.g.input_f32("ctx_last", &[1, 1, d as i64]);
    tp.g.input_f32("rope_cos", &[1, 1, 1, half as i64]);
    tp.g.input_f32("rope_sin", &[1, 1, 1, half as i64]);
    tp.g.input_f32("dep_mask", &[1, 1, 1, ci]);
    tp.g.input_f32("past_dep_k", &[1, heads as i64, ci, hd as i64]);
    tp.g.input_f32("past_dep_v", &[1, heads as i64, ci, hd as i64]);
    tp.g.output_f32("new_dep_k", &[1, heads as i64, 1, hd as i64]);
    tp.g.output_f32("new_dep_v", &[1, heads as i64, 1, hd as i64]);
    tp.g.output_f32("s2_logits", &[1, 1, s2v as i64]);

    tp.f32("c_eps", &[1], vec![1e-5]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    tp.i64("sh_heads1", &[4], vec![1, 1, heads as i64, hd as i64]);
    tp.i64("sh_flat1", &[3], vec![1, 1, d as i64]);
    tp.i64("h_lo0", &[1], vec![0]);
    tp.i64("h_hi0", &[1], vec![half as i64]);
    tp.i64("h_lo1", &[1], vec![half as i64]);
    tp.i64("h_hi1", &[1], vec![hd as i64]);
    tp.i64("h_ax", &[1], vec![3]);
    tp.i64("h_st", &[1], vec![1]);
    tp.i64("sl_ax", &[1], vec![3]);
    tp.i64("sl_0", &[1], vec![0]);
    tp.i64("sl_cap", &[1], vec![ci]);
    tp.i64("sl_cap1", &[1], vec![ci + 1]);

    let cx = "dep_layer.cross_attn";
    let q = tp.linear_biased("sib", &format!("{cx}.q_proj"), w, d, d);
    let k = tp.linear_biased("ctx_last", &format!("{cx}.k_proj"), w, d, d);
    let v = tp.linear_biased("ctx_last", &format!("{cx}.v_proj"), w, d, d);
    let q4 = tp.reshape(&q, "sh_heads1");
    let k4 = tp.reshape(&k, "sh_heads1");
    let v4 = tp.reshape(&v, "sh_heads1");
    let q4 = tp.rope_neox(&q4);
    let k4 = tp.rope_neox(&k4);
    let qt = tp.transpose(&q4, &[0, 2, 1, 3]); // [1,heads,1,hd]
    tp.g.add(Node::new("Transpose", &[&k4], &["new_dep_k"]).attr_ints("perm", &[0, 2, 1, 3]));
    tp.g.add(Node::new("Transpose", &[&v4], &["new_dep_v"]).attr_ints("perm", &[0, 2, 1, 3]));
    let pkt = tp.transpose("past_dep_k", &[0, 1, 3, 2]);
    let sp = tp.matmul(&qt, &pkt);
    let sp = tp.mul(&sp, "c_scale");
    let sp = tp.add(&sp, "dep_mask");
    let nkt = tp.transpose("new_dep_k", &[0, 1, 3, 2]);
    let ss = tp.matmul(&qt, &nkt);
    let ss = tp.mul(&ss, "c_scale");
    let scores = tp.concat2(&sp, &ss, 3);
    let probs = tp.softmax(&scores, -1);
    let pp = tp.slice(&probs, "sl_0", "sl_cap", "sl_ax");
    let ps = tp.slice(&probs, "sl_cap", "sl_cap1", "sl_ax");
    let cp = tp.matmul(&pp, "past_dep_v");
    let cs = tp.matmul(&ps, "new_dep_v");
    let ctxa = tp.add_t(&cp, &cs);
    let ctxa = tp.transpose(&ctxa, &[0, 2, 1, 3]);
    let ctxa = tp.reshape(&ctxa, "sh_flat1");
    let o = tp.linear_biased(&ctxa, &format!("{cx}.out_proj"), w, d, d);
    let sum = tp.add_t("ctx_last", &o);
    tp.rmsnorm_to(&sum, "dep_layer.norm.weight", w, d, "s2_normed");
    tp.linear_biased_to("s2_normed", "head.proj_s2", w, s2v, d, "s2_logits");
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
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

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

// ============================ cached rollout graphs ============================
//
// The full-window graphs above recompute all T positions every decode step
// (O(T²)/step). These give the NPU the same KV-cache + shared-prefill the host
// `forecast_cached` has: a PREFILL graph seeds a fixed-`cap` KV cache over the
// context, then a single-token DECODE graph appends one token, attending the
// cached K/V (O(cap)/step). Mirrors qwen's build_talker_{prefill,decode}_graph.
// RoPE is shift-invariant (`q_i·k_j` depends only on `i−j`), so keys cached at
// absolute positions stay valid — the same property the qwen decode graph uses.

/// s1 DECODE (single token): `x:[1,1,D]` + per-layer `past_k_{l}`/`past_v_{l}:
/// [1,heads,cap,hd]` + `rope_cos`/`rope_sin:[1,1,1,hd/2]` (this token's absolute
/// position) + `past_mask:[1,1,1,cap]` (additive, 0 on filled slots) → per-layer
/// `new_k_{l}`/`new_v_{l}:[1,heads,1,hd]`, `ctx:[1,1,D]`, `s1_logits:[1,1,s1v]`.
pub fn build_kronos_s1_decode_graph(cfg: &KronosConfig, w: &W, cap: usize, g: &mut GraphBuilder) {
    build_kronos_s1_decode_graph_quant(cfg, w, cap, g, Quant::F32);
}
pub fn build_kronos_s1_decode_graph_quant(cfg: &KronosConfig, w: &W, cap: usize, g: &mut GraphBuilder, quant: Quant) {
    let d = cfg.d_model;
    let heads = cfg.n_heads;
    let hd = d / heads;
    let half = hd / 2;
    let ff = cfg.ff_dim;
    let s1v = cfg.s1_vocab();
    let ci = cap as i64;
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

    tp.g.input_f32("x", &[1, 1, d as i64]);
    tp.g.input_f32("rope_cos", &[1, 1, 1, half as i64]);
    tp.g.input_f32("rope_sin", &[1, 1, 1, half as i64]);
    tp.g.input_f32("past_mask", &[1, 1, 1, ci]);
    for b in 0..cfg.n_layers {
        tp.g.input_f32(&format!("past_k_{b}"), &[1, heads as i64, ci, hd as i64]);
        tp.g.input_f32(&format!("past_v_{b}"), &[1, heads as i64, ci, hd as i64]);
        tp.g.output_f32(&format!("new_k_{b}"), &[1, heads as i64, 1, hd as i64]);
        tp.g.output_f32(&format!("new_v_{b}"), &[1, heads as i64, 1, hd as i64]);
    }
    tp.g.output_f32("ctx", &[1, 1, d as i64]);
    tp.g.output_f32("s1_logits", &[1, 1, s1v as i64]);

    tp.f32("c_eps", &[1], vec![1e-5]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    tp.i64("sh_heads1", &[4], vec![1, 1, heads as i64, hd as i64]);
    tp.i64("sh_flat1", &[3], vec![1, 1, d as i64]);
    tp.i64("h_lo0", &[1], vec![0]);
    tp.i64("h_hi0", &[1], vec![half as i64]);
    tp.i64("h_lo1", &[1], vec![half as i64]);
    tp.i64("h_hi1", &[1], vec![hd as i64]);
    tp.i64("h_ax", &[1], vec![3]);
    tp.i64("h_st", &[1], vec![1]);
    // probs split (axis 3): past [0,cap), self [cap,cap+1).
    tp.i64("sl_ax", &[1], vec![3]);
    tp.i64("sl_0", &[1], vec![0]);
    tp.i64("sl_cap", &[1], vec![ci]);
    tp.i64("sl_cap1", &[1], vec![ci + 1]);

    let mut x = "x".to_string();
    for b in 0..cfg.n_layers {
        let pre = format!("transformer.{b}");
        let h = tp.rmsnorm(&x, &format!("{pre}.norm1.weight"), w, d);
        let q = tp.linear_biased(&h, &format!("{pre}.self_attn.q_proj"), w, d, d);
        let k = tp.linear_biased(&h, &format!("{pre}.self_attn.k_proj"), w, d, d);
        let v = tp.linear_biased(&h, &format!("{pre}.self_attn.v_proj"), w, d, d);
        let q4 = tp.reshape(&q, "sh_heads1");
        let k4 = tp.reshape(&k, "sh_heads1");
        let v4 = tp.reshape(&v, "sh_heads1");
        let q4 = tp.rope_neox(&q4);
        let k4 = tp.rope_neox(&k4);
        let qt = tp.transpose(&q4, &[0, 2, 1, 3]); // [1,heads,1,hd]
        // this token's k/v → [1,heads,1,hd]: graph outputs, host appends to the cache.
        let new_k = format!("new_k_{b}");
        let new_v = format!("new_v_{b}");
        tp.g.add(Node::new("Transpose", &[&k4], &[&new_k]).attr_ints("perm", &[0, 2, 1, 3]));
        tp.g.add(Node::new("Transpose", &[&v4], &[&new_v]).attr_ints("perm", &[0, 2, 1, 3]));
        // scores: [ q·pastᵀ·scale + mask | q·newᵀ·scale ] → softmax over cap+1.
        let pkt = tp.transpose(&format!("past_k_{b}"), &[0, 1, 3, 2]); // [1,heads,hd,cap]
        let sp = tp.matmul(&qt, &pkt);
        let sp = tp.mul(&sp, "c_scale");
        let sp = tp.add(&sp, "past_mask");
        let nkt = tp.transpose(&new_k, &[0, 1, 3, 2]); // [1,heads,hd,1]
        let ss = tp.matmul(&qt, &nkt);
        let ss = tp.mul(&ss, "c_scale");
        let scores = tp.concat2(&sp, &ss, 3); // [1,heads,1,cap+1]
        let probs = tp.softmax(&scores, -1);
        let pp = tp.slice(&probs, "sl_0", "sl_cap", "sl_ax"); // [1,heads,1,cap]
        let ps = tp.slice(&probs, "sl_cap", "sl_cap1", "sl_ax"); // [1,heads,1,1]
        let cp = tp.matmul(&pp, &format!("past_v_{b}")); // [1,heads,1,hd]
        let cs = tp.matmul(&ps, &new_v);
        let ctxh = tp.add_t(&cp, &cs);
        let ctxh = tp.transpose(&ctxh, &[0, 2, 1, 3]); // [1,1,heads,hd]
        let ctxh = tp.reshape(&ctxh, "sh_flat1"); // [1,1,D]
        let o = tp.linear_biased(&ctxh, &format!("{pre}.self_attn.out_proj"), w, d, d);
        x = tp.add_t(&x, &o);
        // SwiGLU FFN (no bias)
        let h2 = tp.rmsnorm(&x, &format!("{pre}.norm2.weight"), w, d);
        let a = tp.linear(&h2, &format!("{pre}.ffn.w1.weight"), w, ff, d);
        let bb = tp.linear(&h2, &format!("{pre}.ffn.w3.weight"), w, ff, d);
        let sa = tp.silu(&a);
        let g_ = tp.mul_t(&sa, &bb);
        let ffo = tp.linear(&g_, &format!("{pre}.ffn.w2.weight"), w, d, ff);
        x = tp.add_t(&x, &ffo);
    }
    tp.rmsnorm_to(&x, "norm.weight", w, d, "ctx");
    tp.linear_biased_to("ctx", "head.proj_s1", w, s1v, d, "s1_logits");
}

/// ONNX assembly helper (mirrors the Chronos-2 topology's `Topo`, extended with
/// biased linears, SiLU, and named-output writers).
struct Topo<'a> {
    b: crate::topo::TopoBase<'a>,
    quant: Quant,
}

// DSL + shared math emitters live on `TopoBase` (crate::topo); Deref keeps this
// file's call sites unchanged. Only kronos-specific emission stays here.
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
        let gain = format!("{name}.g");
        self.b.rmsnorm(x, &gain, w[name].clone(), dim, "c_eps")
    }
    /// RMSNorm writing its final scaled tensor to `out_name`.
    fn rmsnorm_to(&mut self, x: &str, name: &str, w: &W, dim: usize, out_name: &str) {
        let gain = format!("{name}.g");
        self.b.rmsnorm_to(x, &gain, w[name].clone(), dim, "c_eps", out_name);
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
