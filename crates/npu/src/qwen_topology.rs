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
    let mut tp = Topo { g, n: 0, quant: Quant::F32 };
    let d = cfg.d_model as usize;
    let vocab = cfg.vocab as usize;
    let ti = t as i64;

    tp.g.input_i64("input_ids", &[1, ti]);
    tp.g.output_f32("logits", &[1, ti, vocab as i64]);

    // Token embedding: Gather(tok.weight[vocab,d], ids) -> [1,T,d].
    tp.f32("tok.weight", &[vocab as i64, d as i64], w["tok.weight"].clone());
    let x = tp.gather("tok.weight", "input_ids", 0, "emb");

    let xf = build_stack(&mut tp, cfg, w, t, &x, false);
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
pub fn build_talker_hidden_graph(cfg: &QwenConfig, w: &W, t: usize, quant: bool, g: &mut GraphBuilder) {
    let mut tp = Topo { g, n: 0, quant: Quant::from_bool(quant) };
    let d = cfg.d_model as usize;
    let ti = t as i64;

    tp.g.input_f32("inputs_embeds", &[1, ti, d as i64]);
    tp.g.output_f32("hidden", &[1, ti, d as i64]);

    let xf = build_stack(&mut tp, cfg, w, t, "inputs_embeds", false);
    // Surface the final-norm hidden state as the graph output `hidden`.
    tp.node("Identity", &[&xf], "hidden");
}

/// Build the **prefill** Talker graph: like [`build_talker_hidden_graph`] (full
/// context, `inputs_embeds:[1,T,d] -> hidden:[1,T,d]`) but additionally emits the
/// per-layer post-QK-norm/post-RoPE K/V (`k_{l}`/`v_{l}:[1,nkv,T,hd]`). Running
/// this once seeds the resident decode cache for the whole prompt prefix in a
/// single inference, instead of streaming the prefix token-by-token through the
/// decode-step graph.
pub fn build_talker_prefill_graph(cfg: &QwenConfig, w: &W, t: usize, quant: Quant, g: &mut GraphBuilder) {
    let mut tp = Topo { g, n: 0, quant };
    let d = cfg.d_model as usize;
    let ti = t as i64;
    tp.g.input_f32("inputs_embeds", &[1, ti, d as i64]);
    tp.g.output_f32("hidden", &[1, ti, d as i64]);
    let xf = build_stack(&mut tp, cfg, w, t, "inputs_embeds", true);
    tp.node("Identity", &[&xf], "hidden");
}

/// Build the **KV-cache decode-step** Talker graph: process ONE new token given
/// the per-layer past key/value cache, so each generated frame is O(1)
/// projections + O(t) attention instead of re-running the whole context. This is
/// the resident-cache counterpart of [`build_talker_hidden_graph`] (which is the
/// cache-free prefill).
///
/// Inputs: `x:[1,1,d]` (new token embedding); `rope_cos`/`rope_sin:[1,1,1,hd]`
/// (the rotary tables for THIS absolute position, supplied by the host so no
/// position arithmetic lives in the graph); `past_mask:[1,1,1,cap]` (additive 0
/// for already-filled cache slots, -inf otherwise); and per layer
/// `past_k_{l}`/`past_v_{l}:[1,nkv,cap,hd]` (post-QK-norm, post-RoPE, transposed).
/// Outputs: `hidden:[1,1,d]` and per layer `new_k_{l}`/`new_v_{l}:[1,nkv,1,hd]`
/// (this token's k/v, which the host writes into the cache at the current slot).
/// Attention is over the (masked) past cache concatenated with the new token's
/// own key/value, so the static `cap` shapes never change.
pub fn build_talker_decode_graph(cfg: &QwenConfig, w: &W, cap: usize, quant: Quant, g: &mut GraphBuilder) {
    let mut tp = Topo { g, n: 0, quant };
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
    let capi = cap as i64;
    let nl = cfg.n_layers as usize;

    // ---- I/O ----
    tp.g.input_f32("x", &[1, 1, d as i64]);
    tp.g.input_f32("rope_cos", &[1, 1, 1, hd as i64]);
    tp.g.input_f32("rope_sin", &[1, 1, 1, hd as i64]);
    tp.g.input_f32("past_mask", &[1, 1, 1, capi]);
    for l in 0..nl {
        tp.g.input_f32(&format!("past_k_{l}"), &[1, nkv as i64, capi, hd as i64]);
        tp.g.input_f32(&format!("past_v_{l}"), &[1, nkv as i64, capi, hd as i64]);
        tp.g.output_f32(&format!("new_k_{l}"), &[1, nkv as i64, 1, hd as i64]);
        tp.g.output_f32(&format!("new_v_{l}"), &[1, nkv as i64, 1, hd as i64]);
    }
    tp.g.output_f32("hidden", &[1, 1, d as i64]);

    // ---- constants ----
    tp.f32("c_eps", &[1], vec![eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    tp.i64("rh_ax", &[1], vec![3]);
    tp.i64("rh_lo0", &[1], vec![0]);
    tp.i64("rh_hi0", &[1], vec![half as i64]);
    tp.i64("rh_lo1", &[1], vec![half as i64]);
    tp.i64("rh_hi1", &[1], vec![hd as i64]);
    tp.i64("dsh_q1", &[4], vec![1, 1, nh as i64, hd as i64]);
    tp.i64("dsh_kv1", &[4], vec![1, 1, nkv as i64, hd as i64]);
    tp.i64("dsh_ctx1", &[3], vec![1, 1, hq as i64]);
    // GQA head-expand shapes: cache (cap) and single-token.
    tp.i64("dsh_kv5c", &[5], vec![1, nkv as i64, 1, capi, hd as i64]);
    tp.i64("dsh_expc", &[5], vec![1, nkv as i64, group as i64, capi, hd as i64]);
    tp.i64("dsh_nhc", &[4], vec![1, nh as i64, capi, hd as i64]);
    tp.i64("dsh_kv51", &[5], vec![1, nkv as i64, 1, 1, hd as i64]);
    tp.i64("dsh_exp1", &[5], vec![1, nkv as i64, group as i64, 1, hd as i64]);
    tp.i64("dsh_nh1", &[4], vec![1, nh as i64, 1, hd as i64]);
    // probs split (axis 3): past [0,cap), self [cap,cap+1).
    tp.i64("sl_ax", &[1], vec![3]);
    tp.i64("sl_0", &[1], vec![0]);
    tp.i64("sl_cap", &[1], vec![capi]);
    tp.i64("sl_cap1", &[1], vec![capi + 1]);

    let mut x = "x".to_string();
    for l in 0..nl {
        let p = |s: &str| format!("blocks.{l}.{s}");
        // --- attention ---
        let h1 = tp.rmsnorm(&x, &p("ln1.weight"), w, d);
        let q = tp.linear(&h1, &p("attn.wq.weight"), w, hq, d);
        let k = tp.linear(&h1, &p("attn.wk.weight"), w, hkv, d);
        let v = tp.linear(&h1, &p("attn.wv.weight"), w, hkv, d);
        let q = tp.reshape(&q, "dsh_q1");
        let k = tp.reshape(&k, "dsh_kv1");
        let v = tp.reshape(&v, "dsh_kv1");
        let q = tp.rmsnorm(&q, &p("attn.q_norm.weight"), w, hd);
        let k = tp.rmsnorm(&k, &p("attn.k_norm.weight"), w, hd);
        let q = tp.rope(&q);
        let k = tp.rope(&k);
        let q = tp.transpose(&q, &[0, 2, 1, 3]); // [1,nh,1,hd]
        // This token's k/v, transposed to [1,nkv,1,hd] — graph outputs (host caches).
        let new_k = format!("new_k_{l}");
        let new_v = format!("new_v_{l}");
        tp.g.add(Node::new("Transpose", &[&k], &[&new_k]).attr_ints("perm", &[0, 2, 1, 3]));
        tp.g.add(Node::new("Transpose", &[&v], &[&new_v]).attr_ints("perm", &[0, 2, 1, 3]));
        // Expand kv heads (nkv->nh) for both the cache and the new token.
        let pk_e = tp.expand_to(&format!("past_k_{l}"), "dsh_kv5c", "dsh_expc", "dsh_nhc");
        let pv_e = tp.expand_to(&format!("past_v_{l}"), "dsh_kv5c", "dsh_expc", "dsh_nhc");
        let nk_e = tp.expand_to(&new_k, "dsh_kv51", "dsh_exp1", "dsh_nh1");
        let nv_e = tp.expand_to(&new_v, "dsh_kv51", "dsh_exp1", "dsh_nh1");
        // scores: [q·pastᵀ*scale + mask | q·newᵀ*scale] -> softmax over cap+1.
        let pkt = tp.transpose(&pk_e, &[0, 1, 3, 2]); // [1,nh,hd,cap]
        let sp = tp.matmul(&q, &pkt);
        let sp = tp.mul(&sp, "c_scale");
        let sp = tp.add(&sp, "past_mask");
        let nkt = tp.transpose(&nk_e, &[0, 1, 3, 2]); // [1,nh,hd,1]
        let ss = tp.matmul(&q, &nkt);
        let ss = tp.mul(&ss, "c_scale");
        let scores = tp.concat2(&sp, &ss, 3); // [1,nh,1,cap+1]
        let probs = tp.softmax(&scores, -1);
        let pp = tp.slice(&probs, "sl_0", "sl_cap", "sl_ax"); // [1,nh,1,cap]
        let ps = tp.slice(&probs, "sl_cap", "sl_cap1", "sl_ax"); // [1,nh,1,1]
        let cp = tp.matmul(&pp, &pv_e); // [1,nh,1,hd]
        let cs = tp.matmul(&ps, &nv_e); // [1,nh,1,hd]
        let ctx = tp.add_t(&cp, &cs);
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]); // [1,1,nh,hd]
        let ctx = tp.reshape(&ctx, "dsh_ctx1"); // [1,1,hq]
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
    let xf = tp.rmsnorm(&x, "norm.weight", w, d);
    tp.node("Identity", &[&xf], "hidden");
}

/// Build the **fused single-infer MTP** graph: the whole per-frame residual
/// code-prediction (16 autoregressive substeps) collapsed into ONE inference,
/// instead of 15 tiny per-substep NPU infers (dispatch-bound). Inputs
/// `talker_hidden:[1,1,emb]` + `cb0_embed:[1,1,emb]`; outputs `codes:[1,1,nres]`
/// (f32, host rounds to the residual codebook ids) + `res_sum:[1,1,emb]`
/// (Σ residual codec-embeddings, the Talker feedback). The 16 decoder positions
/// are unrolled with the per-layer K/V grown in-graph (`Concat`) and constant
/// per-position RoPE; each position `k≥1` does lm_head → ArgMax → Gather the next
/// input embedding (the autoregressive feedback), all on-device.
///
/// `emb` = embedding_dim (Talker width, feedback), `d` = MTP decoder width,
/// `vocab` = residual codebook size, `n_groups` = num_code_groups (16 → 15 residuals).
/// Weights: `small_to_mtp_projection.{weight,bias}` (emb→d; Identity if emb==d),
/// `blocks.{l}.*`, `norm.weight`, `codec_embedding.{i}.weight[vocab,emb]`,
/// `lm_head.{i}.weight[vocab,d]` for i in 0..nres.
pub fn build_mtp_fused_graph(cfg: &QwenConfig, emb: usize, vocab: usize, n_groups: usize, w: &W, g: &mut GraphBuilder) {
    let mut tp = Topo { g, n: 0, quant: Quant::F32 };
    let d = cfg.d_model as usize;
    let nh = cfg.n_heads as usize;
    let nkv = cfg.n_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let half = hd / 2;
    let hq = nh * hd;
    let nl = cfg.n_layers as usize;
    let nres = n_groups - 1;
    let has_proj = emb != d;

    tp.g.input_f32("talker_hidden", &[1, 1, emb as i64]);
    tp.g.input_f32("cb0_embed", &[1, 1, emb as i64]);
    tp.g.output_f32("codes", &[1, 1, nres as i64]);
    tp.g.output_f32("res_sum", &[1, 1, emb as i64]);

    // shared constants
    tp.f32("c_eps", &[1], vec![cfg.rms_eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    tp.i64("rh_ax", &[1], vec![3]);
    tp.i64("rh_lo0", &[1], vec![0]);
    tp.i64("rh_hi0", &[1], vec![half as i64]);
    tp.i64("rh_lo1", &[1], vec![half as i64]);
    tp.i64("rh_hi1", &[1], vec![hd as i64]);
    tp.i64("mf_q1", &[4], vec![1, 1, nh as i64, hd as i64]);
    tp.i64("mf_kv1", &[4], vec![1, 1, nkv as i64, hd as i64]);
    tp.i64("mf_ctx1", &[3], vec![1, 1, hq as i64]);
    tp.i64("mf_sq2", &[1], vec![2]); // squeeze axis for the argmax index

    // Per-layer growing K/V accumulators (post-QK-norm/post-RoPE, [1,nkv,S,hd]).
    let mut kacc: Vec<Option<String>> = vec![None; nl];
    let mut vacc: Vec<Option<String>> = vec![None; nl];

    // pos 0: project the Talker hidden and run one decode step (seeds the cache; no head).
    let x0 = mtp_project(&mut tp, "talker_hidden", w, d, emb, has_proj);
    let _ = mtp_decode_step(&mut tp, &mut kacc, &mut vacc, cfg, w, &x0, 0);

    tp.f32("mf_zero", &[1, 1, emb as i64], vec![0.0; emb]);
    let mut res_sum = "mf_zero".to_string();
    let mut input_raw = "cb0_embed".to_string();
    let mut code_cols: Vec<String> = Vec::with_capacity(nres);
    for k in 1..=nres {
        let pin = mtp_project(&mut tp, &input_raw, w, d, emb, has_proj);
        let hidden = mtp_decode_step(&mut tp, &mut kacc, &mut vacc, cfg, w, &pin, k);
        let logits = tp.linear(&hidden, &format!("lm_head.{}.weight", k - 1), w, vocab, d); // [1,1,vocab]
        // greedy: first max index (matches CpuMtp argmax tie-to-lowest).
        let idx = tp.tmp("mf_argmax");
        tp.g.add(Node::new("ArgMax", &[&logits], &[&idx]).attr_int("axis", -1).attr_int("keepdims", 1)); // [1,1,1] i64
        // codes column: cast to f32 for the graph output.
        let idxf = tp.tmp("mf_idxf");
        tp.g.add(Node::new("Cast", &[&idx], &[&idxf]).attr_int("to", 1)); // to FLOAT
        code_cols.push(idxf);
        // r = codec_embedding[k-1][idx]  ([1,1,emb])
        let tbl = format!("codec_embedding.{}.weight", k - 1);
        let tf = format!("{tbl}.f");
        if !tp.has(&tf) {
            tp.f32(&tf, &[vocab as i64, emb as i64], w[&tbl].clone());
        }
        let idx2 = tp.tmp("mf_idx2");
        tp.g.add(Node::new("Squeeze", &[&idx, "mf_sq2"], &[&idx2])); // [1,1]
        let r = tp.tmp("mf_r");
        tp.g.add(Node::new("Gather", &[&tf, &idx2], &[&r]).attr_int("axis", 0)); // [1,1,emb]
        res_sum = tp.add_t(&res_sum, &r);
        input_raw = r;
    }
    // codes [1,1,nres] (f32) via concat along the last axis; res_sum passthrough.
    let refs: Vec<&str> = code_cols.iter().map(|s| s.as_str()).collect();
    let codes_cat = tp.tmp("mf_codes");
    tp.g.add(Node::new("Concat", &refs, &[&codes_cat]).attr_int("axis", 2));
    tp.node("Identity", &[&codes_cat], "codes");
    tp.node("Identity", &[&res_sum], "res_sum");
}

/// Project an `[1,1,emb]` embedding to the MTP decoder width `[1,1,d]` via
/// `small_to_mtp_projection` (Identity when `emb == d`, the 0.6B case).
fn mtp_project(tp: &mut Topo, x: &str, w: &W, d: usize, emb: usize, has_proj: bool) -> String {
    if !has_proj {
        return x.to_string();
    }
    let y = tp.linear(x, "small_to_mtp_projection.weight", w, d, emb); // [1,1,d]
    let bn = "small_to_mtp_projection.bias.b";
    if !tp.has(bn) {
        tp.f32(bn, &[d as i64], w["small_to_mtp_projection.bias"].clone());
    }
    tp.add(&y, bn)
}

/// One MTP decoder step at absolute position `pos`, appending this token's K/V to
/// the per-layer accumulators (grown via `Concat`) and returning the final-norm
/// hidden `[1,1,d]`. Attention is over positions `0..=pos` (causal by construction,
/// so no mask); RoPE uses constants baked for `pos`.
fn mtp_decode_step(
    tp: &mut Topo,
    kacc: &mut [Option<String>],
    vacc: &mut [Option<String>],
    cfg: &QwenConfig,
    w: &W,
    x_in: &str,
    pos: usize,
) -> String {
    let d = cfg.d_model as usize;
    let nh = cfg.n_heads as usize;
    let nkv = cfg.n_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let group = nh / nkv;
    let hq = nh * hd;
    let hkv = nkv * hd;
    let ff = cfg.d_ff as usize;
    let s = pos + 1; // sequence length after appending this token
    let mut x = x_in.to_string();
    for l in 0..cfg.n_layers as usize {
        let p = |t: &str| format!("blocks.{l}.{t}");
        let h1 = tp.rmsnorm(&x, &p("ln1.weight"), w, d);
        let q = tp.linear(&h1, &p("attn.wq.weight"), w, hq, d);
        let k = tp.linear(&h1, &p("attn.wk.weight"), w, hkv, d);
        let v = tp.linear(&h1, &p("attn.wv.weight"), w, hkv, d);
        let q = tp.reshape(&q, "mf_q1");
        let k = tp.reshape(&k, "mf_kv1");
        let v = tp.reshape(&v, "mf_kv1");
        let q = tp.rmsnorm(&q, &p("attn.q_norm.weight"), w, hd);
        let k = tp.rmsnorm(&k, &p("attn.k_norm.weight"), w, hd);
        let q = tp.rope_at(&q, pos, hd, cfg.rope_theta);
        let k = tp.rope_at(&k, pos, hd, cfg.rope_theta);
        let q = tp.transpose(&q, &[0, 2, 1, 3]); // [1,nh,1,hd]
        let knew = tp.transpose(&k, &[0, 2, 1, 3]); // [1,nkv,1,hd]
        let vnew = tp.transpose(&v, &[0, 2, 1, 3]); // [1,nkv,1,hd]
        // Append to the accumulated K/V along the sequence axis (2).
        let kall = match &kacc[l] {
            None => knew.clone(),
            Some(prev) => tp.concat2(prev, &knew, 2),
        };
        let vall = match &vacc[l] {
            None => vnew.clone(),
            Some(prev) => tp.concat2(prev, &vnew, 2),
        };
        kacc[l] = Some(kall.clone());
        vacc[l] = Some(vall.clone());
        // GQA head-expand [1,nkv,S,hd] -> [1,nh,S,hd] (shapes depend on S).
        let sh5 = format!("mf_kv5_{s}");
        let she = format!("mf_exp_{s}");
        let shn = format!("mf_nh_{s}");
        if !tp.has(&sh5) {
            tp.i64(&sh5, &[5], vec![1, nkv as i64, 1, s as i64, hd as i64]);
            tp.i64(&she, &[5], vec![1, nkv as i64, group as i64, s as i64, hd as i64]);
            tp.i64(&shn, &[4], vec![1, nh as i64, s as i64, hd as i64]);
        }
        let kexp = tp.expand_to(&kall, &sh5, &she, &shn); // [1,nh,S,hd]
        let vexp = tp.expand_to(&vall, &sh5, &she, &shn);
        let kt = tp.transpose(&kexp, &[0, 1, 3, 2]); // [1,nh,hd,S]
        let scores = tp.matmul(&q, &kt); // [1,nh,1,S]
        let scores = tp.mul(&scores, "c_scale");
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &vexp); // [1,nh,1,hd]
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]); // [1,1,nh,hd]
        let ctx = tp.reshape(&ctx, "mf_ctx1"); // [1,1,hq]
        let attn = tp.linear(&ctx, &p("attn.wo.weight"), w, d, hq);
        x = tp.add_t(&x, &attn);
        let h2 = tp.rmsnorm(&x, &p("ln2.weight"), w, d);
        let gate = tp.linear(&h2, &p("mlp.gate.weight"), w, ff, d);
        let up = tp.linear(&h2, &p("mlp.up.weight"), w, ff, d);
        let sig = tp.unary("Sigmoid", &gate);
        let silu = tp.mul_t(&gate, &sig);
        let hmul = tp.mul_t(&silu, &up);
        let down = tp.linear(&hmul, &p("mlp.down.weight"), w, d, ff);
        x = tp.add_t(&x, &down);
        let _ = hkv;
    }
    tp.rmsnorm(&x, "norm.weight", w, d)
}

/// Build the shared decoder body (constants + `n_layers` blocks + final RMSNorm)
/// onto `tp`, reading the residual stream from `x_in` (`[1,T,d]`) and returning
/// the name of the final-norm hidden states (`[1,T,d]`). Used by both the
/// token-id graph ([`build_qwen_graph`]) and the input-embedding Talker graph
/// ([`build_talker_hidden_graph`]).
fn build_stack(tp: &mut Topo, cfg: &QwenConfig, w: &W, t: usize, x_in: &str, emit_kv: bool) -> String {
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
        let k = tp.transpose(&k, &[0, 2, 1, 3]); // [1,nkv,T,hd]
        let v = tp.transpose(&v, &[0, 2, 1, 3]); // [1,nkv,T,hd]
        // Prefill mode: surface the per-layer post-QK-norm/post-RoPE K/V (before
        // the GQA head-expand) as graph outputs so the host can seed the decode
        // KV cache in one inference (vs streaming the prefix token-by-token).
        if emit_kv {
            let nkv = cfg.n_kv_heads as i64;
            let hd = cfg.head_dim as i64;
            tp.g.output_f32(&format!("k_{l}"), &[1, nkv, t as i64, hd]);
            tp.g.output_f32(&format!("v_{l}"), &[1, nkv, t as i64, hd]);
            tp.node("Identity", &[&k], &format!("k_{l}"));
            tp.node("Identity", &[&v], &format!("v_{l}"));
        }
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

/// Weight quantization for the linear layers. `Int8`/`Int4` store weights as
/// per-output-channel symmetric integers dequantised in-graph (`DequantizeLinear`
/// -> MatMul): ~4x / ~8x smaller than fp32, so the 1.7B Talker fits the NPU and —
/// being weight-bandwidth bound — decodes faster. Norms/RoPE/mask stay fp32.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    F32,
    Int8,
    Int4,
}

impl Quant {
    pub fn from_bool(int8: bool) -> Quant {
        if int8 {
            Quant::Int8
        } else {
            Quant::F32
        }
    }
}

/// ONNX graph assembly helper: unique temp names + node/initializer emission.
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
    /// GQA head-expand with caller-named reshape/expand target shapes (decode graph
    /// needs both a `cap`-length and a single-token variant).
    fn expand_to(&mut self, x: &str, sh5: &str, shexp: &str, shnh: &str) -> String {
        let r5 = self.reshape(x, sh5);
        let e = self.tmp("exp");
        self.node("Expand", &[&r5, shexp], &e);
        self.reshape(&e, shnh)
    }
    /// `Slice(x, lo, hi, axis)` by initializer names.
    fn slice(&mut self, x: &str, lo: &str, hi: &str, ax: &str) -> String {
        let o = self.tmp("sl");
        self.g.add(Node::new("Slice", &[x, lo, hi, ax], &[&o]));
        o
    }
    /// `Concat([a,b], axis)`.
    fn concat2(&mut self, a: &str, b: &str, axis: i64) -> String {
        let o = self.tmp("cat");
        self.g.add(Node::new("Concat", &[a, b], &[&o]).attr_int("axis", axis));
        o
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

    /// RoPE (half-split) at a FIXED absolute position `pos` on `[1,1,heads,hd]` —
    /// the fused-MTP unroll processes one token at a known position, so this token's
    /// cos/sin are baked as constants (broadcast `[1,1,1,hd]` over the head axis).
    /// Reuses the shared `rh_*` rotate-half slice bounds.
    fn rope_at(&mut self, x: &str, pos: usize, hd: usize, theta: f32) -> String {
        let half = hd / 2;
        let cosn = format!("mf_cos_{pos}");
        let sinn = format!("mf_sin_{pos}");
        if !self.has(&cosn) {
            let (mut cos, mut sin) = (vec![0f32; hd], vec![0f32; hd]);
            for j in 0..hd {
                let m = (j % half) as f32;
                let ang = pos as f32 * theta.powf(-2.0 * m / hd as f32);
                cos[j] = ang.cos();
                sin[j] = ang.sin();
            }
            self.f32(&cosn, &[1, 1, 1, hd as i64], cos);
            self.f32(&sinn, &[1, 1, 1, hd as i64], sin);
        }
        let x2 = {
            let o = self.tmp("rha_x2");
            self.g.add(Node::new("Slice", &[x, "rh_lo1", "rh_hi1", "rh_ax"], &[&o]));
            o
        };
        let x1 = {
            let o = self.tmp("rha_x1");
            self.g.add(Node::new("Slice", &[x, "rh_lo0", "rh_hi0", "rh_ax"], &[&o]));
            o
        };
        let nx2 = self.unary("Neg", &x2);
        let rot = {
            let o = self.tmp("rha");
            self.g.add(Node::new("Concat", &[&nx2, &x1], &[&o]).attr_int("axis", 3));
            o
        };
        let a = self.mul(x, &cosn);
        let b = self.mul(&rot, &sinn);
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
    /// transposed weight initializer. In `quant` mode the weight is stored as
    /// per-output-channel symmetric INT8 and dequantised in-graph.
    fn linear_named(&mut self, x: &str, name: &str, winit: &str, w: &W, out: usize, inp: usize, y: &str) {
        let (q4, qmax) = match self.quant {
            Quant::F32 => {
                if !self.has(winit) {
                    let wt = transpose(&w[name], out, inp); // [out,in] -> [in,out]
                    self.f32(winit, &[inp as i64, out as i64], wt);
                }
                self.node("MatMul", &[x, winit], y);
                return;
            }
            Quant::Int8 => (false, 127.0f32),
            Quant::Int4 => (true, 7.0f32),
        };
        // Per-output-channel (axis=1) symmetric integer weights: scale[o]=max|col|/qmax,
        // dequantised in-graph (`DequantizeLinear`) then MatMul. INT4 packs 2/byte.
        let wq = format!("{winit}.q");
        if !self.has(&wq) {
            let wt = transpose(&w[name], out, inp); // [out,in] -> [in,out]
            let mut scales = vec![0f32; out];
            let mut q = vec![0i8; inp * out];
            for o in 0..out {
                let mut mx = 0f32;
                for i in 0..inp {
                    mx = mx.max(wt[i * out + o].abs());
                }
                let s = if mx > 0.0 { mx / qmax } else { 1.0 };
                scales[o] = s;
                for i in 0..inp {
                    let v = (wt[i * out + o] / s).round().clamp(-qmax, qmax);
                    q[i * out + o] = v as i8;
                }
            }
            let zp = format!("{winit}.zp");
            if q4 {
                self.g.init_i4(&wq, &[inp as i64, out as i64], q);
                self.g.init_i4(&zp, &[out as i64], vec![0i8; out]);
            } else {
                self.g.init_i8(&wq, &[inp as i64, out as i64], q);
                self.g.init_i8(&zp, &[out as i64], vec![0i8; out]);
            }
            self.f32(&format!("{winit}.s"), &[out as i64], scales);
            self.g.add(
                Node::new("DequantizeLinear", &[&wq, &format!("{winit}.s"), &zp], &[winit]).attr_int("axis", 1),
            );
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
