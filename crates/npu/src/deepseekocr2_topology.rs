// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build DeepSeek-OCR-2's new vision-tower resampler (`crates/deepseekocr2`'s
//! `encoder::Resampler`: a 24-layer Qwen2-shaped GQA tower run as a
//! learned-query resampler under a prefix-LM mask, then a linear projector)
//! as a fixed-shape ONNX graph, for a best-effort OpenVINO/NPU compile
//! attempt. Pure Rust - no NPU needed to produce the file.
//!
//! **Scope boundary, stated plainly: this graph does NOT include SAM.** Its
//! input is `sam_tokens:[1,n_query,d]` - the per-view token grid SAM has
//! already produced - matching the scope `crates/deepseekocr2`'s own
//! `encoder.rs` has held since M3 ("SAM's already-produced per-view token
//! grid, taken as a plain host slice"). SAM's windowed attention with
//! decomposed relative-position bias has no ONNX export precedent anywhere
//! in this crate today, so exporting it is real, separate, unstarted work,
//! not folded into this file by assumption.
//!
//! **The prefix-LM mask exports exactly as the plan predicted: a static
//! additive buffer, not data-dependent control flow.** `allow(i,j) =
//! (i<P && j<P) || (j<=i)` (image rows bidirectional among themselves, query
//! rows causal over everything before them) depends only on `P = n_query`,
//! which is fixed at export time (one graph per view size, the same
//! "static shapes; one graph per bucket" convention `lfm_topology.rs`
//! already uses) - so it bakes into one `[1,1,2P,2P]` initializer exactly
//! like `nemotron_topology.rs::attn_mask_host`'s own precomputed padding
//! mask, added to the scaled scores before `Softmax`.
//!
//! GQA (14 heads / 2 KV heads) is expressed by literally repeating each KV
//! head `group` times via `Reshape`+`Expand`+`Reshape` before the attention
//! matmuls (`qwen_topology.rs`'s own GQA emitters use the identical shape).

use onnx::builder::GraphBuilder;

use deepseekocr2::config::Qwen2EncoderConfig;

use crate::topo::TopoBase;
use crate::topology::WeightSource;

/// Assemble the resampler+projector graph into `g`. `w` is the real (or tiny)
/// checkpoint's flat tensors (`vision.encoder.blocks.{l}.*`/
/// `vision.encoder.norm.weight`/`vision.query_{local,global}.weight`/
/// `vision.projector.fc.*` naming - `crates/gguf/src/deepseekocr2_vision.rs`'s
/// classifier output). `n_query` and `local` select which learned query bank
/// this graph's view uses (144/local or 256/global); `decoder_hidden` is the
/// projector's output width.
pub fn build_resampler_graph(cfg: &Qwen2EncoderConfig, w: &dyn WeightSource, n_query: usize, local: bool, decoder_hidden: usize, g: &mut GraphBuilder) {
    let mut tp = TopoBase::new(g);
    let d = cfg.d_model as usize;
    let nh = cfg.n_heads as usize;
    let nkv = cfg.n_kv_heads as usize;
    let hd = cfg.head_dim() as usize;
    let half = hd / 2;
    let group = (cfg.n_heads / cfg.n_kv_heads) as usize;
    let p_ = n_query; // one view's token count == its query-bank size (M0 fact)
    let seq = 2 * p_;
    let si = seq as i64;
    let theta = cfg.rope_theta;
    let eps = cfg.rms_eps;

    tp.g.input_f32("sam_tokens", &[1, p_ as i64, d as i64]);
    tp.g.output_f32("projected", &[1, p_ as i64, decoder_hidden as i64]);

    // ---- concat with the matching learned query bank ----
    let qb_name = if local { "vision.query_local.weight" } else { "vision.query_global.weight" };
    tp.f32("query_bank", &[1, p_ as i64, d as i64], w.get(qb_name));
    let mut x = tp.concat2("sam_tokens", "query_bank", 1); // [1,2P,d]

    // ---- shared constants ----
    tp.f32("c_eps", &[1], vec![eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    let (mut cos, mut sin) = (vec![0f32; seq * hd], vec![0f32; seq * hd]);
    for pos in 0..seq {
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = pos as f32 * theta.powf(-2.0 * m / hd as f32);
            cos[pos * hd + j] = ang.cos();
            sin[pos * hd + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, si, 1, hd as i64], cos);
    tp.f32("rope_sin", &[1, si, 1, hd as i64], sin);

    // Prefix-LM mask: allow(i,j) = (i<P && j<P) || (j<=i). Bakes the same
    // formula `attn_prefix_mask.wgsl` computes at inference time, verified
    // by M3's mutation check on the tiny golden.
    let mut mask = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let allow = (i < p_ && j < p_) || (j <= i);
            if !allow {
                mask[i * seq + j] = -1.0e9;
            }
        }
    }
    tp.f32("prefix_mask", &[1, 1, si, si], mask);

    tp.i64("rh_ax", &[1], vec![3]);
    tp.i64("rh_lo0", &[1], vec![0]);
    tp.i64("rh_hi0", &[1], vec![half as i64]);
    tp.i64("rh_lo1", &[1], vec![half as i64]);
    tp.i64("rh_hi1", &[1], vec![hd as i64]);
    tp.i64("sh_q", &[4], vec![1, si, nh as i64, hd as i64]);
    tp.i64("sh_kv", &[4], vec![1, si, nkv as i64, hd as i64]);
    tp.i64("sh_kv5", &[5], vec![1, nkv as i64, 1, si, hd as i64]);
    tp.i64("sh_exp", &[5], vec![1, nkv as i64, group as i64, si, hd as i64]);
    tp.i64("sh_nh", &[4], vec![1, nh as i64, si, hd as i64]);
    tp.i64("sh_ctx", &[3], vec![1, si, (nh * hd) as i64]);

    for l in 0..cfg.n_layers as usize {
        let p = |leaf: &str| format!("vision.encoder.blocks.{l}.{leaf}");

        let xn1 = rmsnorm(&mut tp, &x, &p("norm1.weight"), w, d);
        let q = linear_bias(&mut tp, &xn1, &p("attn.q.weight"), Some(&p("attn.q.bias")), w, d, d);
        let k = linear_bias(&mut tp, &xn1, &p("attn.k.weight"), Some(&p("attn.k.bias")), w, nkv * hd, d);
        let v = linear_bias(&mut tp, &xn1, &p("attn.v.weight"), Some(&p("attn.v.bias")), w, nkv * hd, d);
        let q = tp.reshape(&q, "sh_q");
        let k = tp.reshape(&k, "sh_kv");
        let v = tp.reshape(&v, "sh_kv");
        let q = rope_half_split(&mut tp, &q);
        let k = rope_half_split(&mut tp, &k);
        let q = tp.transpose(&q, &[0, 2, 1, 3]); // [1,nh,2P,hd]
        let k = expand_kv(&mut tp, &k); // [1,nkv,2P,hd] -> [1,nh,2P,hd]
        let v = expand_kv(&mut tp, &v);
        let kt = tp.transpose(&k, &[0, 1, 3, 2]); // [1,nh,hd,2P]
        let scores = tp.matmul(&q, &kt);
        let scores = tp.mul(&scores, "c_scale");
        let scores = tp.add_t(&scores, "prefix_mask");
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &v); // [1,nh,2P,hd]
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]);
        let ctx = tp.reshape(&ctx, "sh_ctx");
        let attn_out = linear(&mut tp, &ctx, &p("attn.out.weight"), w, d, d);
        x = tp.add_t(&x, &attn_out);

        let xn2 = rmsnorm(&mut tp, &x, &p("norm2.weight"), w, d);
        let ff = cfg.ffn_hidden as usize;
        let gate = linear(&mut tp, &xn2, &p("mlp.gate.weight"), w, ff, d);
        let up = linear(&mut tp, &xn2, &p("mlp.up.weight"), w, ff, d);
        let act = tp.silu(&gate);
        let h = tp.mul_t(&act, &up);
        let mlp_out = linear(&mut tp, &h, &p("mlp.down.weight"), w, d, ff);
        x = tp.add_t(&x, &mlp_out);
    }

    let xf = rmsnorm(&mut tp, &x, "vision.encoder.norm.weight", w, d);

    // Slice the query half only: rows [P, 2P).
    tp.i64("q_lo", &[1], vec![p_ as i64]);
    tp.i64("q_hi", &[1], vec![seq as i64]);
    tp.i64("q_ax", &[1], vec![1]);
    let query_half = tp.slice(&xf, "q_lo", "q_hi", "q_ax"); // [1,P,d]

    linear_bias_to(&mut tp, &query_half, "vision.projector.fc.weight", Some("vision.projector.fc.bias"), w, decoder_hidden, d, "projected");
}

fn rmsnorm(tp: &mut TopoBase, x: &str, name: &str, w: &dyn WeightSource, dim: usize) -> String {
    let gain = format!("{name}.g");
    tp.rmsnorm(x, &gain, w.get(name), dim, "c_eps")
}

fn linear(tp: &mut TopoBase, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
    linear_bias(tp, x, name, None, w, out, inp)
}

fn linear_bias(tp: &mut TopoBase, x: &str, name: &str, bias: Option<&str>, w: &dyn WeightSource, out: usize, inp: usize) -> String {
    let y = tp.tmp("lin");
    linear_bias_to(tp, x, name, bias, w, out, inp, &y);
    y
}

/// `y = x . Wᵀ [+ bias]` (brain stores `[out,in]`; ONNX `MatMul` wants
/// `[in,out]`, so the weight is transposed once into a fresh initializer).
fn linear_bias_to(tp: &mut TopoBase, x: &str, name: &str, bias: Option<&str>, w: &dyn WeightSource, out: usize, inp: usize, y: &str) {
    let winit = format!("{name}.wt");
    if !tp.has(&winit) {
        let raw = w.get(name);
        let mut t = vec![0f32; raw.len()];
        for r in 0..out {
            for c in 0..inp {
                t[c * out + r] = raw[r * inp + c];
            }
        }
        tp.f32(&winit, &[inp as i64, out as i64], t);
    }
    match bias {
        None => tp.node("MatMul", &[x, &winit], y),
        Some(bname) => {
            let pre = tp.tmp("lin_pre");
            tp.node("MatMul", &[x, &winit], &pre);
            if !tp.has(bname) {
                let bd = w.get(bname);
                tp.f32(bname, &[out as i64], bd);
            }
            tp.node("Add", &[&pre, bname], y);
        }
    }
}

/// RoPE, half-split (NeoX) convention, over `[1,S,heads,hd]`.
fn rope_half_split(tp: &mut TopoBase, x: &str) -> String {
    let x_hi = tp.slice(x, "rh_lo1", "rh_hi1", "rh_ax");
    let x_lo = tp.slice(x, "rh_lo0", "rh_hi0", "rh_ax");
    let neg_hi = tp.unary("Neg", &x_hi);
    let rot = tp.concat2(&neg_hi, &x_lo, 3);
    let a = tp.mul(x, "rope_cos");
    let b = tp.mul(&rot, "rope_sin");
    tp.add_t(&a, &b)
}

/// GQA head-repeat: `[1,nkv,S,hd] -> [1,nkv,1,S,hd] -Expand-> [1,nkv,group,S,hd]
/// -> [1,nh,S,hd]`.
fn expand_kv(tp: &mut TopoBase, x: &str) -> String {
    let r5 = tp.reshape(x, "sh_kv5");
    let e = tp.tmp("exp");
    tp.node("Expand", &[&r5, "sh_exp"], &e);
    tp.reshape(&e, "sh_nh")
}

