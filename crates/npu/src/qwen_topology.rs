// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build a Qwen3 decoder as an ONNX graph (fixed sequence length `T`) using
//! brain's pure-Rust ONNX serializer, for whole-graph compilation on OpenVINO
//! (NPU / GPU / CPU). The graph takes `input_ids:[1,T]` (int64) and produces
//! `logits:[1,T,vocab]` (f32) — a cache-free prefill that mirrors brain's own
//! recompute-per-step inference. Standard ONNX ops only (Gather/MatMul/Mul/Add/
//! Sigmoid/Softmax/ReduceMean/Sqrt/Div/Reshape/Transpose/Expand/Slice/Neg/
//! Concat), so OpenVINO's ONNX frontend maps it directly.
//!
//! Conventions: brain stores linear weights `[out,in]`; ONNX `MatMul(x,B)` needs
//! `B=[in,out]`, so every linear weight is transposed once at export. The tied
//! lm_head reuses `tok.weight`. RoPE uses Qwen's half-split (NeoX) convention via
//! precomputed cos/sin tables; QK-norm is RMSNorm over `head_dim`.

use std::collections::HashMap;

use onnx::builder::GraphBuilder;
use onnx::graph::Node;
use qwen::config::QwenConfig;

type W = HashMap<String, Vec<f32>>;

/// Assembles the decoder graph into `g`. `w` is the brain checkpoint's tensors
/// (role ""), `t` the fixed sequence length. Input `input_ids:[1,T]` (int64),
/// output `logits:[1,T,vocab]` (f32).
pub fn build_qwen_graph(cfg: &QwenConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    let mut tp = Topo { g, n: 0 };
    let d = cfg.d_model as usize;
    let vocab = cfg.vocab as usize;
    let ti = t as i64;

    tp.g.input_i64("input_ids", &[1, ti]);
    tp.g.output_f32("logits", &[1, ti, vocab as i64]);

    // Token embedding: Gather(tok.weight[vocab,d], ids) -> [1,T,d].
    tp.f32("tok.weight", &[vocab as i64, d as i64], w["tok.weight"].clone());
    let x = tp.gather("tok.weight", "input_ids", 0, "emb");

    let xf = build_stack(&mut tp, cfg, w, t, &x);
    // lm_head. Tied models (`tie_embeddings`) reuse `tok.weight`; untied models
    // (e.g. the Qwen3-TTS Talker, `tie_embeddings = false`) have a separate
    // `lm_head.weight`. Both are `[vocab,d]`, transposed to `[d,vocab]` for MatMul.
    let head = if cfg.tie_embeddings { "tok.weight" } else { "lm_head.weight" };
    tp.linear_named(&xf, head, "lm_head.w", w, vocab, d, "logits");
}

/// Assemble the Talker decoder as an **input-embedding**-driven graph: input
/// `inputs_embeds:[1,T,d]` (f32), output `hidden:[1,T,d]` (f32, the post-final-
/// norm hidden states). This is the shape the autoregressive Talker loop needs —
/// it feeds the text/codec/speaker-conditioned embedding stream straight into the
/// residual stream (no token-id Gather) and reads back the per-position hidden
/// state, exactly mirroring [`tts::gen::TalkerGen::forward`]. The codebook-0 head
/// (`codec_head_logits`) and the MTP residual fill stay on the host, so this
/// graph stops at the final RMSNorm and emits the hidden state, not logits.
pub fn build_talker_hidden_graph(cfg: &QwenConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    let mut tp = Topo { g, n: 0 };
    let d = cfg.d_model as usize;
    let ti = t as i64;

    tp.g.input_f32("inputs_embeds", &[1, ti, d as i64]);
    tp.g.output_f32("hidden", &[1, ti, d as i64]);

    let xf = build_stack(&mut tp, cfg, w, t, "inputs_embeds");
    // Surface the final-norm hidden state as the graph output `hidden`.
    tp.node("Identity", &[&xf], "hidden");
}

/// Build the shared decoder body (constants + `n_layers` blocks + final RMSNorm)
/// onto `tp`, reading the residual stream from `x_in` (`[1,T,d]`) and returning
/// the name of the final-norm hidden states (`[1,T,d]`). Used by both the
/// token-id graph ([`build_qwen_graph`]) and the input-embedding Talker graph
/// ([`build_talker_hidden_graph`]).
fn build_stack(tp: &mut Topo, cfg: &QwenConfig, w: &W, t: usize, x_in: &str) -> String {
    let d = cfg.d_model as usize;
    let nh = cfg.n_heads as usize;
    let nkv = cfg.n_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let half = hd / 2;
    let group = nh / nkv;
    let hq = nh * hd;
    let hkv = nkv * hd;
    let ff = cfg.d_ff as usize;
    let eps = cfg.rms_eps;
    let ti = t as i64;

    // Shared constants.
    tp.f32("c_eps", &[1], vec![eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    tp.f32("c_two", &[1], vec![2.0]);
    // RoPE cos/sin tables [1,T,1,hd] (half-split / NeoX: emb = cat(freqs,freqs)).
    let (mut cos, mut sin) = (vec![0f32; t * hd], vec![0f32; t * hd]);
    for p in 0..t {
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = p as f32 * cfg.rope_theta.powf(-2.0 * m / hd as f32);
            cos[p * hd + j] = ang.cos();
            sin[p * hd + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, ti, 1, hd as i64], cos);
    tp.f32("rope_sin", &[1, ti, 1, hd as i64], sin);
    // Causal mask [1,1,T,T]: 0 for j<=i, -1e9 otherwise.
    let mut mask = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask[i * t + j] = -1.0e9;
            }
        }
    }
    tp.f32("causal_mask", &[1, 1, ti, ti], mask);
    // rotate_half slice bounds (along last axis).
    tp.i64("rh_ax", &[1], vec![3]);
    tp.i64("rh_lo0", &[1], vec![0]);
    tp.i64("rh_hi0", &[1], vec![half as i64]);
    tp.i64("rh_lo1", &[1], vec![half as i64]);
    tp.i64("rh_hi1", &[1], vec![hd as i64]);
    // Reshape/expand target shapes.
    tp.i64("sh_q", &[4], vec![1, ti, nh as i64, hd as i64]);
    tp.i64("sh_kv", &[4], vec![1, ti, nkv as i64, hd as i64]);
    tp.i64("sh_q5", &[5], vec![1, nkv as i64, 1, ti, hd as i64]);
    tp.i64("sh_exp", &[5], vec![1, nkv as i64, group as i64, ti, hd as i64]);
    tp.i64("sh_nh", &[4], vec![1, nh as i64, ti, hd as i64]);
    tp.i64("sh_ctx", &[3], vec![1, ti, hq as i64]);

    // Residual stream starts at the caller-provided input (token embeddings or
    // the input-embedding stream), already `[1,T,d]`.
    let mut x = x_in.to_string();

    for l in 0..cfg.n_layers as usize {
        let p = |s: &str| format!("blocks.{l}.{s}");
        // --- attention ---
        let h1 = tp.rmsnorm(&x, &p("ln1.weight"), w, d);
        let q = tp.linear(&h1, &p("attn.wq.weight"), w, hq, d);
        let k = tp.linear(&h1, &p("attn.wk.weight"), w, hkv, d);
        let v = tp.linear(&h1, &p("attn.wv.weight"), w, hkv, d);
        let q = tp.reshape(&q, "sh_q");
        let k = tp.reshape(&k, "sh_kv");
        let v = tp.reshape(&v, "sh_kv");
        // QK-norm (RMSNorm over head_dim) then RoPE.
        let q = tp.rmsnorm(&q, &p("attn.q_norm.weight"), w, hd);
        let k = tp.rmsnorm(&k, &p("attn.k_norm.weight"), w, hd);
        let q = tp.rope(&q);
        let k = tp.rope(&k);
        // To [1,heads,T,hd].
        let q = tp.transpose(&q, &[0, 2, 1, 3]);
        let k = tp.transpose(&k, &[0, 2, 1, 3]);
        let v = tp.transpose(&v, &[0, 2, 1, 3]);
        // GQA: expand kv heads from nkv to nh by repeating each `group` times.
        let k = tp.expand_kv(&k);
        let v = tp.expand_kv(&v);
        // scores = q @ k^T * scale + mask ; softmax ; @ v.
        let kt = tp.transpose(&k, &[0, 1, 3, 2]);
        let scores = tp.matmul(&q, &kt);
        let scores = tp.mul(&scores, "c_scale");
        let scores = tp.add(&scores, "causal_mask");
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &v); // [1,nh,T,hd]
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]); // [1,T,nh,hd]
        let ctx = tp.reshape(&ctx, "sh_ctx"); // [1,T,Hq]
        let attn = tp.linear(&ctx, &p("attn.wo.weight"), w, d, hq);
        x = tp.add_t(&x, &attn);
        // --- SwiGLU MLP ---
        let h2 = tp.rmsnorm(&x, &p("ln2.weight"), w, d);
        let gate = tp.linear(&h2, &p("mlp.gate.weight"), w, ff, d);
        let up = tp.linear(&h2, &p("mlp.up.weight"), w, ff, d);
        let sig = tp.unary("Sigmoid", &gate);
        let silu = tp.mul_t(&gate, &sig);
        let hmul = tp.mul_t(&silu, &up);
        let down = tp.linear(&hmul, &p("mlp.down.weight"), w, d, ff);
        x = tp.add_t(&x, &down);
    }

    tp.rmsnorm(&x, "norm.weight", w, d)
}

/// ONNX graph assembly helper: unique temp names + node/initializer emission.
struct Topo<'a> {
    g: &'a mut GraphBuilder,
    n: usize,
}

impl<'a> Topo<'a> {
    fn tmp(&mut self, tag: &str) -> String {
        self.n += 1;
        format!("{tag}_{}", self.n)
    }
    fn f32(&mut self, name: &str, dims: &[i64], data: Vec<f32>) {
        self.g.init_f32(name, dims, data);
    }
    fn i64(&mut self, name: &str, dims: &[i64], data: Vec<i64>) {
        self.g.init_i64(name, dims, data);
    }
    fn node(&mut self, op: &str, ins: &[&str], out: &str) {
        self.g.add(Node::new(op, ins, &[out]));
    }

    fn gather(&mut self, data: &str, idx: &str, axis: i64, tag: &str) -> String {
        let o = self.tmp(tag);
        self.g.add(Node::new("Gather", &[data, idx], &[&o]).attr_int("axis", axis));
        o
    }
    fn unary(&mut self, op: &str, x: &str) -> String {
        let o = self.tmp(&op.to_lowercase());
        self.node(op, &[x], &o);
        o
    }
    /// Binary op with an initializer/const second operand (by name).
    fn mul(&mut self, x: &str, c: &str) -> String {
        let o = self.tmp("mul");
        self.node("Mul", &[x, c], &o);
        o
    }
    fn add(&mut self, x: &str, c: &str) -> String {
        let o = self.tmp("add");
        self.node("Add", &[x, c], &o);
        o
    }
    fn mul_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("mul");
        self.node("Mul", &[a, b], &o);
        o
    }
    fn add_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("res");
        self.node("Add", &[a, b], &o);
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
    fn expand_kv(&mut self, x: &str) -> String {
        // [1,nkv,T,hd] -> [1,nkv,1,T,hd] -> Expand [1,nkv,group,T,hd] -> [1,nh,T,hd]
        let r5 = self.reshape(x, "sh_q5");
        let e = self.tmp("exp");
        self.node("Expand", &[&r5, "sh_exp"], &e);
        self.reshape(&e, "sh_nh")
    }

    /// RMSNorm over the last `dim` axis with gain `name` (from `w`).
    fn rmsnorm(&mut self, x: &str, name: &str, w: &W, dim: usize) -> String {
        let gain = format!("{name}.g");
        if !self.has(&gain) {
            self.f32(&gain, &[dim as i64], w[name].clone());
        }
        let sq = self.mul_t(x, x);
        let ms = {
            let o = self.tmp("rms_mean");
            self.g.add(Node::new("ReduceMean", &[&sq], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1));
            o
        };
        let mse = self.add(&ms, "c_eps");
        let rms = self.unary("Sqrt", &mse);
        let nrm = {
            let o = self.tmp("rms_div");
            self.node("Div", &[x, &rms], &o);
            o
        };
        self.mul(&nrm, &gain)
    }

    /// RoPE (half-split) on [1,T,heads,hd]: x*cos + rotate_half(x)*sin.
    fn rope(&mut self, x: &str) -> String {
        // rotate_half = concat(-x[..,half:], x[..,:half])
        let x2 = {
            let o = self.tmp("rh_x2");
            self.g.add(Node::new("Slice", &[x, "rh_lo1", "rh_hi1", "rh_ax"], &[&o]));
            o
        };
        let x1 = {
            let o = self.tmp("rh_x1");
            self.g.add(Node::new("Slice", &[x, "rh_lo0", "rh_hi0", "rh_ax"], &[&o]));
            o
        };
        let nx2 = self.unary("Neg", &x2);
        let rot = {
            let o = self.tmp("rh");
            self.g.add(Node::new("Concat", &[&nx2, &x1], &[&o]).attr_int("axis", 3));
            o
        };
        let a = self.mul(x, "rope_cos");
        let b = self.mul(&rot, "rope_sin");
        self.add_t(&a, &b)
    }

    /// Linear `y = x · Wᵀ` with brain weight `name` ([out,in]); transposed to
    /// [in,out] for ONNX MatMul. Output is a fresh temp.
    fn linear(&mut self, x: &str, name: &str, w: &W, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        self.linear_named(x, name, &format!("{name}.wt"), w, out, inp, &o);
        o
    }
    /// As [`linear`] but writes to an explicit output name; `winit` names the
    /// transposed weight initializer.
    fn linear_named(&mut self, x: &str, name: &str, winit: &str, w: &W, out: usize, inp: usize, y: &str) {
        if !self.has(winit) {
            let wt = transpose(&w[name], out, inp); // [out,in] -> [in,out]
            self.f32(winit, &[inp as i64, out as i64], wt);
        }
        self.node("MatMul", &[x, winit], y);
    }

    fn has(&self, name: &str) -> bool {
        self.g.graph().initializers.iter().any(|t| t.name == name)
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
