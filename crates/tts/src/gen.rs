// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS **Talker generation** engine: an inference-only forward of the
//! 28-layer Talker decoder that accepts an *arbitrary input-embedding* sequence
//! (`[T, d_model]`) — the text/codec/speaker-conditioned prefix the autoregressive
//! voice synthesis needs — rather than token ids.
//!
//! The shared [`qwen::Qwen`] backbone (used for parity/training in [`crate::talker`])
//! only embeds *token ids* through its `tok.weight` table and exposes neither an
//! input-embedding entry point nor the per-position hidden state the MTP needs. So
//! generation builds its own forward graph from the shared `model::block`
//! step-builders (exactly as [`crate::mtp`] does), feeding `inputs_embeds` straight
//! into the residual stream and reading back the final-norm hidden states. The
//! decoder weights, the codec embedding/head tables and the text-projection
//! front-end are all loaded once from the same brain checkpoint that
//! [`crate::talker::TalkerModel`] uses — no second copy of the 0.6 B decoder.
//!
//! There is no KV-cache (none exists in the shared engine), so each step re-runs
//! the forward over the whole growing context — correct but `O(T²)`; keep the
//! generated length modest on the CPU backend.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use paramstore::ParamStore;

use crate::config::TalkerConfig;
use crate::talker::TextProjection;

// ---- kernel pipeline table (forward subset; indices match block::KernelIds) ----
const MATMUL: usize = 0;
const RMSNORM: usize = 1;
const RMS_INV: usize = 2;
const ROPE: usize = 3;
const GQA_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const GQA_APPLY: usize = 6;
const SILU_MUL: usize = 7;
const ADD2: usize = 8;

const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rope_base", kernels::ROPE_BASE),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
];

/// Per-layer / shared GPU scratch (reused across all layers since the forward is
/// strictly sequential and only the final hidden state is read back).
struct Scratch {
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    scores: DeviceBuffer,
    xn_final: DeviceBuffer,
}

/// Inference-only Talker, specialised for autoregressive generation from input
/// embeddings.
pub struct TalkerGen {
    pub cfg: TalkerConfig,
    gpu: Gpu,
    ps: ParamStore,
    max_t: u32,
    // residual stream snapshots between layers (res[0] = input embeds).
    res: Vec<DeviceBuffer>,
    sc: Scratch,
    // CPU tables.
    pub text: TextProjection,
    codec_embedding: Vec<f32>, // talker codec table [vocab, d] (= tok.weight)
    codec_head: Vec<f32>,      // [vocab, d]
}

fn only_fwd_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM,
        rmsnorm_dw: RMSNORM,
        rope: ROPE,
        rope_bwd: ROPE,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: GQA_SCORES,
        gqa_dv: GQA_APPLY,
        gqa_dq: GQA_APPLY,
        gqa_dk: GQA_APPLY,
        silu_mul: SILU_MUL,
        silu_da: SILU_MUL,
        silu_db: SILU_MUL,
    }
}

impl TalkerGen {
    fn decoder_param_list(cfg: &TalkerConfig) -> Vec<(String, usize)> {
        let d = cfg.d_model as usize;
        let ff = cfg.d_ff as usize;
        let hq = cfg.q_dim() as usize;
        let hkv = cfg.kv_dim() as usize;
        let hd = cfg.head_dim as usize;
        let mut out = Vec::new();
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("attn.wq.weight"), hq * d));
            out.push((p("attn.wk.weight"), hkv * d));
            out.push((p("attn.wv.weight"), hkv * d));
            out.push((p("attn.q_norm.weight"), hd));
            out.push((p("attn.k_norm.weight"), hd));
            out.push((p("attn.wo.weight"), d * hq));
            out.push((p("ln2.weight"), d));
            out.push((p("mlp.gate.weight"), ff * d));
            out.push((p("mlp.up.weight"), ff * d));
            out.push((p("mlp.down.weight"), d * ff));
        }
        out.push(("norm.weight".to_string(), d));
        out
    }

    /// Load an inference Talker for generation from the brain checkpoint written
    /// by [`crate::import::import_talker`]. `max_t` is the largest context length
    /// (prefix + generated frames) the buffers are sized for.
    pub fn load(path: &str, max_t: u32) -> TalkerGen {
        let c = checkpoint::load(path);
        let qcfg = qwen::QwenConfig::from_json(&c.header["config"]);
        let mut cfg = TalkerConfig::from_qwen(&qcfg);
        let take = |name: &str| {
            c.find(name, "")
                .cloned()
                .unwrap_or_else(|| panic!("import: missing tensor {name}"))
        };

        let gpu = Gpu::new(PIPELINES);
        let mut decoder = std::collections::HashMap::new();
        for (n, _) in Self::decoder_param_list(&cfg) {
            decoder.insert(n.clone(), take(&n));
        }
        let roles = Self::decoder_param_list(&cfg)
            .into_iter()
            .map(|(n, c)| (n, c, paramstore::Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &decoder);

        // CPU tables.
        let codec_embedding = take("tok.weight");
        let codec_head = take("lm_head.weight");
        let fc1_w = take("text_projection.fc1.weight");
        let fc1_b = take("text_projection.fc1.bias");
        let fc2_w = take("text_projection.fc2.weight");
        let fc2_b = take("text_projection.fc2.bias");
        let text_embedding = c.find("text_embedding.weight", "").cloned();
        let inter = fc1_b.len();
        let in_dim = fc1_w.len() / inter;
        let out = fc2_b.len();
        let text_vocab = text_embedding.as_ref().map(|e| e.len() / in_dim).unwrap_or(0);
        cfg.text_hidden_size = in_dim as u32;
        if text_vocab > 0 {
            cfg.text_vocab_size = text_vocab as u32;
        }
        let text = TextProjection {
            text_embedding,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            in_dim,
            inter,
            out,
            text_vocab,
        };

        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let n = max_t as u64;
        let bht2 = (cfg.n_heads * max_t * max_t) as u64;
        let st = |x: u64| gpu.storage(x);
        let mut res = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(n * d));
        }
        let sc = Scratch {
            xn1: st(n * d),
            q_pre: st(n * hq),
            q: st(n * hq),
            k_pre: st(n * hkv),
            k: st(n * hkv),
            v: st(n * hkv),
            probs: st(bht2),
            ctx: st(n * hq),
            xmid: st(n * d),
            xn2: st(n * d),
            gate_pre: st(n * ff),
            up: st(n * ff),
            h: st(n * ff),
            proj: st(n * d),
            mlp_out: st(n * d),
            scores: st(bht2),
            xn_final: st(n * d),
        };

        TalkerGen {
            cfg,
            gpu,
            ps,
            max_t,
            res,
            sc,
            text,
            codec_embedding,
            codec_head,
        }
    }

    /// d_model.
    pub fn d(&self) -> usize {
        self.cfg.d_model as usize
    }

    /// Talker codebook-0 embedding row for `id` (`[d_model]`).
    pub fn codec_embed(&self, id: u32) -> &[f32] {
        let d = self.d();
        let s = id as usize * d;
        &self.codec_embedding[s..s + d]
    }

    fn forward_steps(&self, n: u32) -> Vec<Step> {
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let ids = only_fwd_ids();
        let ga = Gqa {
            b: 1,
            t: n,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
        };
        let theta = c.rope_theta;
        let g = &self.gpu;
        let sc = &self.sc;
        let w = |name: &str| self.ps.w(name);
        let mut s: Vec<Step> = Vec::new();

        for l in 0..c.n_layers as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            s.push(block::rmsnorm_fwd(g, &ids, &self.res[l], w(&p("ln1.weight")), &sc.xn1, d, n));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wq.weight")), &sc.q_pre], &[n, d, hq], n * hq));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wk.weight")), &sc.k_pre], &[n, d, hkv], n * hkv));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wv.weight")), &sc.v], &[n, d, hkv], n * hkv));
            s.push(block::rmsnorm_fwd(g, &ids, &sc.q_pre, w(&p("attn.q_norm.weight")), &sc.q, hd, n * nh));
            s.push(block::rmsnorm_fwd(g, &ids, &sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, hd, n * nkv));
            s.push(block::rope_fwd(g, &ids, &sc.q, n, nh, hd, hq, n, theta));
            s.push(block::rope_fwd(g, &ids, &sc.k, n, nkv, hd, hkv, n, theta));
            s.extend(block::gqa_fwd(g, &ids, &ga, &sc.q, &sc.k, &sc.v, &sc.scores, &sc.probs, &sc.ctx));
            s.push(g.step(MATMUL, &[&sc.ctx, w(&p("attn.wo.weight")), &sc.proj], &[n, hq, d], n * d));
            s.push(g.step(ADD2, &[&self.res[l], &sc.proj, &sc.xmid], &[n * d], n * d));
            s.push(block::rmsnorm_fwd(g, &ids, &sc.xmid, w(&p("ln2.weight")), &sc.xn2, d, n));
            s.push(g.step(MATMUL, &[&sc.xn2, w(&p("mlp.gate.weight")), &sc.gate_pre], &[n, d, ff], n * ff));
            s.push(g.step(MATMUL, &[&sc.xn2, w(&p("mlp.up.weight")), &sc.up], &[n, d, ff], n * ff));
            s.push(block::swiglu_fwd(g, &ids, &sc.gate_pre, &sc.up, &sc.h, n * ff));
            s.push(g.step(MATMUL, &[&sc.h, w(&p("mlp.down.weight")), &sc.mlp_out], &[n, ff, d], n * d));
            s.push(g.step(ADD2, &[&sc.xmid, &sc.mlp_out, &self.res[l + 1]], &[n * d], n * d));
        }
        let last = c.n_layers as usize;
        s.push(block::rmsnorm_fwd(g, &ids, &self.res[last], w("norm.weight"), &sc.xn_final, d, n));
        s
    }

    /// Run the decoder over `inputs_embeds` (`[n, d_model]`, row-major) and return
    /// the final-norm hidden states (`[n, d_model]`).
    pub fn forward(&self, inputs_embeds: &[f32]) -> Vec<f32> {
        let d = self.d();
        let n = inputs_embeds.len() / d;
        assert_eq!(inputs_embeds.len(), n * d);
        assert!(n as u32 <= self.max_t, "context {n} exceeds max_t {}", self.max_t);
        self.gpu.write(&self.res[0], bytemuck::cast_slice(inputs_embeds));
        let steps = self.forward_steps(n as u32);
        self.gpu.submit(&[], &steps);
        self.gpu.read(&self.sc.xn_final, n * d)
    }

    /// Codebook-0 logits (`[vocab]`) for a single final-norm hidden row.
    pub fn codec_head_logits(&self, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.d();
        let v = self.cfg.vocab as usize;
        assert_eq!(hidden_row.len(), d);
        let mut out = vec![0.0f32; v];
        for (o, dst) in out.iter_mut().enumerate() {
            let wrow = &self.codec_head[o * d..(o + 1) * d];
            let mut acc = 0.0f32;
            for k in 0..d {
                acc += wrow[k] * hidden_row[k];
            }
            *dst = acc;
        }
        out
    }
}
