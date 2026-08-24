// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS **Talker generation** engine: an inference-only forward of the
//! 28-layer Talker decoder that accepts an *arbitrary input-embedding* sequence
//! (`[T, d_model]`) — the text/codec/speaker-conditioned prefix the autoregressive
//! voice synthesis needs — rather than token ids.
//!
//! The shared [`qwen3::Qwen`] backbone (used for parity/training in [`crate::talker`])
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
// Incremental KV-cache decode kernels (single new token vs the growing cache).
const ATTN_DECODE_SCORES: usize = 9;
const DECODE_SOFTMAX: usize = 10;
const ATTN_DECODE_APPLY: usize = 11;
const KV_APPEND: usize = 12;
const ROPE_AT: usize = 13;

pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rope_base", kernels::ROPE_BASE),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("kv_append", kernels::KV_APPEND),
    ("rope_at", kernels::ROPE_AT),
];

/// Which position-dependent uniform a cached decode step needs refreshed each token.
#[derive(Clone, Copy)]
enum PosUniform {
    RopeQ,
    RopeK,
    Append,
    Scores,
    Softmax,
    Apply,
}

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

/// A built incremental-decode tape: the recorded step list plus the
/// `(buffer, uniform)` pairs whose position fields are rewritten each step.
type DecodeTape = (Vec<Step>, Vec<(DeviceBuffer, PosUniform)>);

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
    // Persistent per-layer KV cache for incremental decode ([max_t, kv_dim] each).
    kcache: Vec<DeviceBuffer>,
    vcache: Vec<DeviceBuffer>,
    // Next absolute position the incremental `step` will decode (cache fill level).
    dec_pos: std::cell::Cell<u32>,
    // Cached decode tape (built once, reused every token) + the position-dependent
    // uniform buffers to refresh per step — eliminates per-token tape rebuilding.
    dec_cache: std::cell::RefCell<Option<DecodeTape>>,
    // CPU tables.
    pub text: TextProjection,
    codec_embedding: Vec<f32>, // talker codec table [vocab, d] (= tok.weight)
    codec_head: Vec<f32>,      // [vocab, d]
}

fn only_fwd_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        // No backward graph is built here, so every backward slot is
        // UNREGISTERED rather than a stand-in index for another live kernel.
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dw: block::UNREGISTERED,
        rope: ROPE,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_mul: SILU_MUL,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
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
        Self::load_on(Gpu::new(PIPELINES), path, max_t)
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`) so a
    /// process holds ONE device however many components it loads.
    pub fn load_on(gpu: Gpu, path: &str, max_t: u32) -> TalkerGen {
        let c = checkpoint::load(path);
        let qcfg = qwen3::QwenConfig::from_json(&c.header["config"]);
        let mut cfg = TalkerConfig::from_qwen(&qcfg);
        let take = |name: &str| {
            c.find(name, "")
                .cloned()
                .unwrap_or_else(|| panic!("import: missing tensor {name}"))
        };

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
        let mut kcache = Vec::new();
        let mut vcache = Vec::new();
        for _ in 0..cfg.n_layers {
            kcache.push(st(n * hkv));
            vcache.push(st(n * hkv));
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
            kcache,
            vcache,
            dec_pos: std::cell::Cell::new(0),
            dec_cache: std::cell::RefCell::new(None),
            text,
            codec_embedding,
            codec_head,
        }
    }

    /// Test-only: build a decoder-only Talker from an in-memory weight map (the
    /// `decoder_param_list` leaves), for KV-parity tests without a checkpoint.
    #[cfg(test)]
    pub(crate) fn from_decoder_map(cfg: TalkerConfig, map: &std::collections::HashMap<String, Vec<f32>>, max_t: u32) -> TalkerGen {
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        let roles = Self::decoder_param_list(&cfg).into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, map);
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
        let mut kcache = Vec::new();
        let mut vcache = Vec::new();
        for _ in 0..cfg.n_layers {
            kcache.push(st(n * hkv));
            vcache.push(st(n * hkv));
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
        let text = TextProjection {
            text_embedding: None,
            fc1_w: Vec::new(),
            fc1_b: Vec::new(),
            fc2_w: Vec::new(),
            fc2_b: Vec::new(),
            in_dim: 0,
            inter: 0,
            out: 0,
            text_vocab: 0,
        };
        TalkerGen { cfg, gpu, ps, max_t, res, sc, kcache, vcache, dec_pos: std::cell::Cell::new(0), dec_cache: std::cell::RefCell::new(None), text, codec_embedding: Vec::new(), codec_head: Vec::new() }
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

    /// Reset the incremental KV cache to an empty sequence (next `step` is pos 0).
    pub fn reset_cache(&self) {
        self.dec_pos.set(0);
    }

    /// The absolute position the next [`Self::step`] will decode.
    pub fn cache_pos(&self) -> u32 {
        self.dec_pos.get()
    }

    /// **Incremental KV-cache decode** of a single new token embedding (`[d_model]`)
    /// at the current cache position, returning the final-norm hidden state
    /// (`[d_model]`). This is the `O(T)`-per-token twin of [`Self::forward`]'s
    /// `O(T²)` recompute: the same Qwen3 block math, but K/V for the new token are
    /// projected, QK-normed, RoPE'd at the absolute position, appended to the
    /// persistent per-layer cache, and attended by a single query over positions
    /// `0..=pos`. Expressed entirely in the WGSL op set, so it runs on whatever
    /// backend `Gpu` selected (GPU or the wgsl-cpu JIT) — one engine, any device.
    pub fn step(&self, embed: &[f32]) -> Vec<f32> {
        let pos = self.dec_pos.get();
        let hidden = self.decode_cached(embed, pos);
        self.dec_pos.set(pos + 1);
        hidden
    }

    /// Position-dependent uniform contents for a cached decode step at `pos`.
    fn pu_params(&self, k: PosUniform, pos: u32) -> Vec<u32> {
        let c = &self.cfg;
        let (hd, hq, hkv) = (c.head_dim, c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let group = nh / nkv;
        let cap = self.max_t;
        let t = pos + 1;
        let scale = (1.0f32 / (hd as f32).sqrt()).to_bits();
        let theta = c.rope_theta.to_bits();
        match k {
            PosUniform::RopeQ => vec![1, nh, hd, hq, 0, pos, theta],
            PosUniform::RopeK => vec![1, nkv, hd, hkv, 0, pos, theta],
            PosUniform::Append => vec![hkv, pos],
            PosUniform::Scores => vec![nh, group, hd, t, cap, hkv, scale],
            PosUniform::Softmax => vec![nh, t, cap],
            PosUniform::Apply => vec![nh, group, hd, t, cap, hkv],
        }
    }

    /// Build the decode tape ONCE: constant-shape steps bake their uniforms; the
    /// seven position-dependent steps per layer bind reusable uniform buffers that
    /// [`Self::decode_cached`] refreshes each token. Reused across all tokens, this
    /// removes the ~34ms/token host tape-rebuild the profiler flagged.
    fn build_dec_cache(&self) -> (Vec<Step>, Vec<(DeviceBuffer, PosUniform)>) {
        let c = &self.cfg;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let half = hd / 2;
        let cap = self.max_t;
        let g = &self.gpu;
        let sc = &self.sc;
        let ids = only_fwd_ids();
        let w = |name: &str| self.ps.w(name);
        let mut s: Vec<Step> = Vec::new();
        let mut pus: Vec<(DeviceBuffer, PosUniform)> = Vec::new();
        // A pos-dependent step: allocate its reusable uniform, record via step_buf,
        // return both so the caller pushes into `s` and `pus` (no captured borrows).
        let posstep = |kind: usize, nfields: usize, bufs: &[&DeviceBuffer], threads: u32| -> (Step, DeviceBuffer) {
            let ub = g.uniform_dynamic(nfields);
            let st = g.step_buf(kind, &ub, bufs, threads);
            (st, ub)
        };
        let add_pos = |s: &mut Vec<Step>, pus: &mut Vec<(DeviceBuffer, PosUniform)>, pair: (Step, DeviceBuffer), pu: PosUniform| {
            s.push(pair.0);
            pus.push((pair.1, pu));
        };
        for l in 0..c.n_layers as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            s.push(block::rmsnorm_fwd(g, &ids, &self.res[l], w(&p("ln1.weight")), &sc.xn1, d, 1));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wq.weight")), &sc.q_pre], &[1, d, hq], hq));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wk.weight")), &sc.k_pre], &[1, d, hkv], hkv));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wv.weight")), &sc.v], &[1, d, hkv], hkv));
            s.push(block::rmsnorm_fwd(g, &ids, &sc.q_pre, w(&p("attn.q_norm.weight")), &sc.q, hd, nh));
            s.push(block::rmsnorm_fwd(g, &ids, &sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, hd, nkv));
            add_pos(&mut s, &mut pus, posstep(ROPE_AT, 7, &[&sc.q], nh * half), PosUniform::RopeQ);
            add_pos(&mut s, &mut pus, posstep(ROPE_AT, 7, &[&sc.k], nkv * half), PosUniform::RopeK);
            add_pos(&mut s, &mut pus, posstep(KV_APPEND, 2, &[&sc.k, &self.kcache[l]], hkv), PosUniform::Append);
            add_pos(&mut s, &mut pus, posstep(KV_APPEND, 2, &[&sc.v, &self.vcache[l]], hkv), PosUniform::Append);
            add_pos(&mut s, &mut pus, posstep(ATTN_DECODE_SCORES, 7, &[&sc.q, &self.kcache[l], &sc.scores], nh * cap), PosUniform::Scores);
            add_pos(&mut s, &mut pus, posstep(DECODE_SOFTMAX, 3, &[&sc.scores, &sc.probs], nh), PosUniform::Softmax);
            add_pos(&mut s, &mut pus, posstep(ATTN_DECODE_APPLY, 6, &[&sc.probs, &self.vcache[l], &sc.ctx], nh * hd), PosUniform::Apply);
            s.push(g.step(MATMUL, &[&sc.ctx, w(&p("attn.wo.weight")), &sc.proj], &[1, hq, d], d));
            s.push(g.step(ADD2, &[&self.res[l], &sc.proj, &sc.xmid], &[d], d));
            s.push(block::rmsnorm_fwd(g, &ids, &sc.xmid, w(&p("ln2.weight")), &sc.xn2, d, 1));
            s.push(g.step(MATMUL, &[&sc.xn2, w(&p("mlp.gate.weight")), &sc.gate_pre], &[1, d, ff], ff));
            s.push(g.step(MATMUL, &[&sc.xn2, w(&p("mlp.up.weight")), &sc.up], &[1, d, ff], ff));
            s.push(block::swiglu_fwd(g, &ids, &sc.gate_pre, &sc.up, &sc.h, ff));
            s.push(g.step(MATMUL, &[&sc.h, w(&p("mlp.down.weight")), &sc.mlp_out], &[1, ff, d], d));
            s.push(g.step(ADD2, &[&sc.xmid, &sc.mlp_out, &self.res[l + 1]], &[d], d));
        }
        let last = c.n_layers as usize;
        s.push(block::rmsnorm_fwd(g, &ids, &self.res[last], w("norm.weight"), &sc.xn_final, d, 1));
        (s, pus)
    }

    /// Incremental decode using the cached tape: refresh the position-dependent
    /// uniforms, submit the prebuilt steps, read the hidden state.
    fn decode_cached(&self, embed: &[f32], pos: u32) -> Vec<f32> {
        assert_eq!(embed.len(), self.cfg.d_model as usize, "step embed must be [d_model]");
        assert!(pos < self.max_t, "decode pos {pos} exceeds max_t {}", self.max_t);
        if self.dec_cache.borrow().is_none() {
            *self.dec_cache.borrow_mut() = Some(self.build_dec_cache());
        }
        let g = &self.gpu;
        g.write(&self.res[0], bytemuck::cast_slice(embed));
        let cache = self.dec_cache.borrow();
        let (steps, pus) = cache.as_ref().unwrap();
        for (ub, k) in pus {
            g.write(ub, &self.pu_params(*k, pos));
        }
        g.submit(&[], steps);
        g.read(&self.sc.xn_final, self.cfg.d_model as usize)
    }

    /// Codebook-0 logits (`[vocab]`) for a single final-norm hidden row. Shared
    /// host head used by the GPU/CPU recompute loop and the NPU loop alike.
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

impl crate::prompt::TalkerHost for TalkerGen {
    fn d(&self) -> usize {
        self.cfg.d_model as usize
    }
    fn text(&self) -> &TextProjection {
        &self.text
    }
    fn codec_embed(&self, id: u32) -> &[f32] {
        TalkerGen::codec_embed(self, id)
    }
}

#[cfg(test)]
mod kv_tests {
    use super::*;
    use crate::config::TalkerConfig;
    use data::rng::Rng;
    use std::collections::HashMap;

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The incremental KV-cache `step` must reproduce the `O(T²)` `forward`
    /// recompute for every prefix (the cache is algebraically exact) — same engine,
    /// same weights, so any difference is only attention reduction order.
    #[test]
    fn kv_step_matches_full_recompute() {
        let cfg = TalkerConfig::tiny(); // d16 L2 GQA 4/2 hd8 ff32
        let d = cfg.d_model as usize;
        let max_t = 8u32;
        let seq = 6usize;
        let mut rng = Rng::new(1234);

        // Random decoder weights; norm/qk-norm/ln weights initialised to 1.
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, count) in TalkerGen::decoder_param_list(&cfg) {
            let v = if name.contains("ln") || name.ends_with("norm.weight") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.08).collect()
            };
            map.insert(name, v);
        }
        let tg = TalkerGen::from_decoder_map(cfg.clone(), &map, max_t);

        let embeds: Vec<f32> = (0..seq * d).map(|_| rng.next_gaussian() as f32).collect();

        // Incremental: feed one token at a time through the KV cache.
        tg.reset_cache();
        let inc: Vec<Vec<f32>> = (0..seq).map(|i| tg.step(&embeds[i * d..(i + 1) * d])).collect();

        // Reference: full recompute of each prefix; compare the last row.
        for i in 0..seq {
            let full = tg.forward(&embeds[..(i + 1) * d]);
            let last = &full[i * d..(i + 1) * d];
            let err = maxabs(&inc[i], last);
            assert!(err < 2e-3, "prefix {i}: KV step vs full recompute maxabs={err}");
        }
    }

    /// The shared-engine KV `step` must also match the INDEPENDENT hand-rolled
    /// `CpuTalker` KV oracle on the same weights — the direct evidence that the two
    /// tts engines (GPU `TalkerGen` + CPU `CpuTalker`) can collapse into this one.
    #[test]
    fn kv_step_matches_cpu_talker() {
        use crate::gen_kv::CpuTalker;
        let cfg = TalkerConfig::tiny();
        let d = cfg.d_model as usize;
        let map = random_decoder(&cfg, 20240727);
        let tg = TalkerGen::from_decoder_map(cfg.clone(), &map, 8);
        let mut cpu = CpuTalker::from_decoder_map(cfg.clone(), &map);
        let mut rng = Rng::new(5);
        tg.reset_cache();
        cpu.reset();
        let mut worst = 0f32;
        for _ in 0..6 {
            let e: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32).collect();
            worst = worst.max(maxabs(&tg.step(&e), &cpu.step(&e)));
        }
        println!("kv_step vs CpuTalker: worst maxabs = {worst}");
        assert!(worst < 1e-2, "engine KV step vs CpuTalker oracle maxabs={worst}");
    }

    fn medium_cfg() -> TalkerConfig {
        let mut c = TalkerConfig::tiny();
        c.n_layers = 12;
        c.d_model = 512;
        c.head_dim = 128;
        c.n_heads = 8;
        c.n_kv_heads = 4;
        c.d_ff = 2048;
        c
    }

    fn random_decoder(cfg: &TalkerConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(seed);
        let mut map = HashMap::new();
        for (name, count) in TalkerGen::decoder_param_list(cfg) {
            let v = if name.contains("ln") || name.ends_with("norm.weight") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.02).collect()
            };
            map.insert(name, v);
        }
        map
    }

    /// Throughput: incremental KV `step` (O(T)/token) vs `forward` full-recompute
    /// (O(T²)) at growing context. Run with:
    ///   cargo test -p brain-tts --lib bench_kv_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_kv_decode() {
        let cfg = medium_cfg();
        let d = cfg.d_model as usize;
        let max_t = 1200u32;
        let tg = TalkerGen::from_decoder_map(cfg.clone(), &random_decoder(&cfg, 7), max_t);
        let mut rng = Rng::new(99);
        let embed: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32).collect();
        println!("Talker medium: d={} L={} heads={}/{} hd={} ff={}", cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff);
        for &ctx in &[128usize, 512, 1024] {
            // Prime the KV cache to `ctx` positions, then time one incremental step.
            tg.reset_cache();
            for _ in 0..ctx {
                tg.step(&embed);
            }
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                tg.step(&embed);
            }
            let kv_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            // O(T²) baseline: recompute the whole ctx-length prefix per token.
            let prefix: Vec<f32> = (0..ctx * d).map(|i| embed[i % d]).collect();
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                tg.forward(&prefix);
            }
            let full_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!("ctx={ctx:>5}: KV step {kv_ms:>7.2} ms/tok ({:>6.1} tok/s)  |  full-recompute {full_ms:>8.2} ms/tok ({:>5.1} tok/s)  |  speedup {:.1}x", 1e3 / kv_ms, 1e3 / full_ms, full_ms / kv_ms);
        }
    }
}
