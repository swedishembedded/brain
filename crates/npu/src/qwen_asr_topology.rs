// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-ASR **audio encoder** as an OpenVINO-compilable ONNX graph. Ports the
//! transformer HEAD (the compute-dominant part): the 24 windowed ViT blocks +
//! `ln_post` + the multi-modal projector, over already-packed post-CNN tokens
//! `[n_audio, d_model]` → `audio_embeds [n_audio, output_dim]`. Op-for-op with
//! `qwen_asr::encoder::AudioEncoder::encode_packed`.
//!
//! The conv stem + valid-position packing (data-dependent gather) stay on host —
//! exactly the `build_nemotron_head` vs `build_nemotron_encoder` split. The block
//! window attention (block-diagonal `cu_seqlens` spans) is a build-time additive
//! `[1,n,n]` mask, like the nemotron band mask.
//!
//! Standard pre-LN ViT block (no RoPE / QK-norm / LayerScale), fused QKV, erf-GELU.
//! Reuses the generic ONNX helpers from [`crate::nemotron_topology`].

use onnx::{GraphBuilder, Node};

use crate::nemotron_topology::{add_t, layernorm_onnx, linear_nb, reshape, transpose};
use crate::topology::WeightSource;

/// Config subset the audio-encoder head needs (mirrors `AudioEncoderConfig`).
#[derive(Clone, Copy, Debug)]
pub struct QwenAsrTopo {
    pub d_model: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub ffn_dim: u32,
    pub n_layers: u32,
    pub output_dim: u32,
    pub eps: f32,
}

impl Default for QwenAsrTopo {
    fn default() -> Self {
        QwenAsrTopo { d_model: 1024, n_heads: 16, head_dim: 64, ffn_dim: 4096, n_layers: 24, output_dim: 2048, eps: 1e-5 }
    }
}

const NEG_MASK: f32 = -1.0e9;

/// Block-diagonal additive attention mask `[1, n, n]`: 0 where queries/keys share
/// a `cu_seqlens` span, else a large negative — constant for a fixed span layout.
fn window_mask_host(spans: &[(u32, u32)], n: u32) -> Vec<f32> {
    let n = n as usize;
    let mut m = vec![NEG_MASK; n * n];
    for &(row0, len) in spans {
        let (a, b) = (row0 as usize, (row0 + len) as usize);
        for i in a..b.min(n) {
            for j in a..b.min(n) {
                m[i * n + j] = 0.0;
            }
        }
    }
    m
}

/// `x @ W^T + b`, returning the output name (weight `[out, in]`, bias `[out]`).
fn linear_bias(g: &mut GraphBuilder, w: &dyn WeightSource, x: &str, wname: &str, bname: &str, out: u32, inn: u32, tag: &str) -> String {
    let mm = linear_nb(g, w, x, wname, out, inn, tag);
    let bn = format!("{tag}.b");
    g.init_f32(&bn, &[out as i64], w.get(bname));
    let o = format!("{tag}.bias");
    g.add(Node::new("Add", &[&mm, &bn], &[&o]).name(&format!("{tag}.bias")));
    o
}

/// erf-GELU: `0.5·x·(1 + erf(x/√2))` (torch `F.gelu`, exact — not the tanh approx).
fn gelu_erf(g: &mut GraphBuilder, x: &str, tag: &str) -> String {
    let inv = format!("{tag}.inv");
    g.init_f32(&inv, &[1], vec![std::f32::consts::FRAC_1_SQRT_2]);
    let xs = format!("{tag}.xs");
    g.add(Node::new("Mul", &[x, &inv], &[&xs]).name(&format!("{tag}.xs")));
    let er = format!("{tag}.erf");
    g.add(Node::new("Erf", &[&xs], &[&er]).name(&format!("{tag}.erf")));
    let one = format!("{tag}.one");
    g.init_f32(&one, &[1], vec![1.0]);
    let e1 = format!("{tag}.e1");
    g.add(Node::new("Add", &[&er, &one], &[&e1]).name(&format!("{tag}.e1")));
    let half = format!("{tag}.half");
    g.init_f32(&half, &[1], vec![0.5]);
    let hx = format!("{tag}.hx");
    g.add(Node::new("Mul", &[x, &half], &[&hx]).name(&format!("{tag}.hx")));
    let o = format!("{tag}.gelu");
    g.add(Node::new("Mul", &[&hx, &e1], &[&o]).name(&format!("{tag}.gelu")));
    o
}

/// Slice columns `[lo, hi)` of a `[n, *]` tensor (axis 1).
fn slice_cols(g: &mut GraphBuilder, x: &str, lo: u32, hi: u32, tag: &str) -> String {
    let (s, e, a) = (format!("{tag}.s"), format!("{tag}.e"), format!("{tag}.a"));
    g.init_i64(&s, &[1], vec![lo as i64]);
    g.init_i64(&e, &[1], vec![hi as i64]);
    g.init_i64(&a, &[1], vec![1]);
    let o = format!("{tag}.slice");
    g.add(Node::new("Slice", &[x, &s, &e, &a], &[&o]).name(&format!("{tag}.slice")));
    o
}

/// One windowed ViT block (pre-LN, fused QKV, block-diagonal window mask, erf-GELU MLP).
fn vit_block_onnx(g: &mut GraphBuilder, topo: &QwenAsrTopo, w: &dyn WeightSource, mask: &str, x: &str, blk: u32, n: u32) -> String {
    let (c, heads, hd, ffn) = (topo.d_model, topo.n_heads, topo.head_dim, topo.ffn_dim);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let tag = format!("blk{blk}");
    let p = format!("blocks.{blk}");

    // ---- attention ----
    let ln1 = layernorm_onnx(g, w, x, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"), c, topo.eps, &format!("{tag}.n1"));
    let qkv = linear_bias(g, w, &ln1, &format!("{p}.qkv.weight"), &format!("{p}.qkv.bias"), 3 * c, c, &format!("{tag}.qkv"));
    let to_heads = |g: &mut GraphBuilder, name: &str, sub: &str| -> String {
        let r = reshape(g, name, &[n as i64, heads as i64, hd as i64], &format!("{tag}.{sub}.r"));
        transpose(g, &r, &[1, 0, 2], &format!("{tag}.{sub}.t")) // [heads, n, hd]
    };
    let q = slice_cols(g, &qkv, 0, c, &format!("{tag}.q"));
    let k = slice_cols(g, &qkv, c, 2 * c, &format!("{tag}.k"));
    let v = slice_cols(g, &qkv, 2 * c, 3 * c, &format!("{tag}.v"));
    let qh = to_heads(g, &q, "qh");
    let kh = to_heads(g, &k, "kh");
    let vh = to_heads(g, &v, "vh");
    let kt = transpose(g, &kh, &[0, 2, 1], &format!("{tag}.kt")); // [heads, hd, n]
    let sc = format!("{tag}.sc");
    g.add(Node::new("MatMul", &[&qh, &kt], &[&sc]).name(&format!("{tag}.sc"))); // [heads, n, n]
    let scn = format!("{tag}.sck");
    g.init_f32(&scn, &[1], vec![scale]);
    let scaled = format!("{tag}.scaled");
    g.add(Node::new("Mul", &[&sc, &scn], &[&scaled]).name(&format!("{tag}.scaled")));
    let masked = add_t(g, &scaled, mask, &format!("{tag}.msk"));
    let probs = format!("{tag}.probs");
    g.add(Node::new("Softmax", &[&masked], &[&probs]).name(&format!("{tag}.softmax")).attr_int("axis", -1));
    let ctxh = format!("{tag}.ctxh");
    g.add(Node::new("MatMul", &[&probs, &vh], &[&ctxh]).name(&format!("{tag}.ctx"))); // [heads, n, hd]
    let ctx_tp = transpose(g, &ctxh, &[1, 0, 2], &format!("{tag}.ctxtp")); // [n, heads, hd]
    let ctx = reshape(g, &ctx_tp, &[n as i64, c as i64], &format!("{tag}.ctxflat"));
    let attn = linear_bias(g, w, &ctx, &format!("{p}.proj.weight"), &format!("{p}.proj.bias"), c, c, &format!("{tag}.proj"));
    let x1 = add_t(g, x, &attn, &format!("{tag}.res1"));

    // ---- MLP (erf-GELU) ----
    let ln2 = layernorm_onnx(g, w, &x1, &format!("{p}.norm2.weight"), &format!("{p}.norm2.bias"), c, topo.eps, &format!("{tag}.n2"));
    let h1 = linear_bias(g, w, &ln2, &format!("{p}.fc1.weight"), &format!("{p}.fc1.bias"), ffn, c, &format!("{tag}.fc1"));
    let act = gelu_erf(g, &h1, &format!("{tag}.act"));
    let h2 = linear_bias(g, w, &act, &format!("{p}.fc2.weight"), &format!("{p}.fc2.bias"), c, ffn, &format!("{tag}.fc2"));
    add_t(g, &x1, &h2, &format!("{tag}.res2"))
}

/// The audio-encoder head: packed tokens `[n_audio, d_model]` (named `input_name`)
/// → `ln_post` → multi-modal projector → `audio_embeds [n_audio, output_dim]`
/// (named `out_name`). `spans` are the `cu_seqlens` attention windows.
#[allow(clippy::too_many_arguments)]
pub fn build_qwen_asr_head(g: &mut GraphBuilder, topo: &QwenAsrTopo, w: &dyn WeightSource, n_audio: u32, spans: &[(u32, u32)], input_name: &str, out_name: &str) {
    let (c, out) = (topo.d_model, topo.output_dim);
    let maskv = window_mask_host(spans, n_audio);
    let maskn = "enc.wmask";
    g.init_f32(maskn, &[1, n_audio as i64, n_audio as i64], maskv);
    let mut cur = input_name.to_string();
    for b in 0..topo.n_layers {
        cur = vit_block_onnx(g, topo, w, maskn, &cur, b, n_audio);
    }
    // ln_post → encoder_out
    let enc = layernorm_onnx(g, w, &cur, "ln_post.weight", "ln_post.bias", c, topo.eps, "enc.lnpost");
    // multi-modal projector: Linear(d→d)+bias → erf-GELU → Linear(d→out)+bias
    let f1 = linear_bias(g, w, &enc, "multi_modal_projector.linear_1.weight", "multi_modal_projector.linear_1.bias", c, c, "proj.l1");
    let act = gelu_erf(g, &f1, "proj.act");
    // final projector linear writes the graph output directly.
    let mm = linear_nb(g, w, &act, "multi_modal_projector.linear_2.weight", out, c, "proj.l2");
    let bn = "proj.l2.b";
    g.init_f32(bn, &[out as i64], w.get("multi_modal_projector.linear_2.bias"));
    g.add(Node::new("Add", &[&mm, bn], &[out_name]).name("proj.out"));
}
