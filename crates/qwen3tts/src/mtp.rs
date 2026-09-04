// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS **MTP code-predictor**: a small (5-layer) Qwen3 decoder that fills
//! the residual codebooks 1..15 of one acoustic frame, conditioned on the
//! Talker hidden state and codebook-0.
//!
//! Per `modeling_qwen3_tts.py` (`Qwen3TTSTalkerCodePredictorModel` +
//! `forward_finetune` / `forward_sub_talker_finetune`), one frame is processed as
//! a length-`num_code_groups` sequence of *input embeddings* under full (causal)
//! attention:
//!   pos 0 : the Talker hidden state (`small_to_mtp_projection` is `Identity`
//!           here, since the MTP and Talker share `hidden_size = 1024`),
//!   pos 1 : `talker.codec_embedding(codebook0)`  (the Talker's table),
//!   pos i (2..15) : `code_predictor.codec_embedding[i-2](codebook_{i-1})`.
//! The per-position output head `lm_head[i-1]` reads `hidden[:, i]` to predict
//! codebook `i` (positions 1..15 → 15 residual codebooks).
//!
//! The decoder block is the same Qwen3 block as the Talker (RMSNorm, GQA with
//! per-head QK-norm, half-split RoPE base 1e6, SwiGLU), so it is built from the
//! shared `model::block` step-builders. The decoder runs on the GPU engine; the
//! (tiny) input-embedding gather and the per-position output heads run on the
//! CPU. This is an inference forward (no backward) — the Talker decoder carries
//! the gradient-checked block coverage.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use paramstore::ParamStore;

use crate::config::MtpConfig;

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
// Coalesced RMSNorm - the throughput twin of `RMSNORM`, selected by
// `block::rms_variant` inside `block::rmsnorm_fwd`.
const RMSNORM_ROWS: usize = 9;
// Incremental KV-cache decode kernels (one new position vs the growing
// per-frame cache) - the same five the Talker's own decode tape uses.
const ATTN_DECODE_SCORES: usize = 10;
const DECODE_SOFTMAX: usize = 11;
const ATTN_DECODE_APPLY: usize = 12;
const KV_APPEND: usize = 13;
const ROPE_AT: usize = 14;
// The fp32 GEMM tier `block::gemm_variant` selects between: the
// workgroup-per-output-column decode GEMV (`matmul_gemv`, which
// `gpu_core::upgrade` transparently substitutes `matmul_gemv_reg` for on a
// capable device) and the 128x128 register-tiled GEMM. Both are bit-identical
// to the naive `MATMUL` they replace; only the thread mapping differs.
const MATMUL_GEMV: usize = 15;
const MATMUL_REG3: usize = 16;

// Appended, never reordered: `qwen3omnimoe` builds its own `Gpu` from this
// exact table (`caps.rs`'s `new_like(qwen3tts::mtp::PIPELINES)`), so every
// index above is part of this module's contract with that crate.
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
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("kv_append", kernels::KV_APPEND),
    ("rope_at", kernels::ROPE_AT),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("matmul_reg3", kernels::MATMUL_REG3),
];

/// One-row scratch for the incremental (KV-cached) decode tape. The
/// full-recompute tape keeps a per-layer `Layer` because it holds all
/// `num_code_groups` rows of every layer live in one submit; a decode step is
/// strictly sequential over one row, so one shared set suffices.
struct DecScratch {
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn_final: DeviceBuffer,
}

struct Layer {
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
}

pub struct MtpModel {
    pub cfg: MtpConfig,
    gpu: Gpu,
    ps: ParamStore,
    t: u32,
    // GPU forward scratch
    res: Vec<DeviceBuffer>,
    layers: Vec<Layer>,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    scores: DeviceBuffer,
    xn_final: DeviceBuffer,
    fwd_steps: Vec<Step>,
    // Incremental-decode state: a per-layer key/value cache holding this
    // frame's `num_code_groups` positions, one-row scratch, and one PREBUILT
    // tape per position. The MTP's sequence length is fixed at
    // `num_code_groups`, so every position's uniforms are compile-time
    // constants of the model - unlike the Talker, whose position runs to
    // `max_t` and therefore rewrites dynamic uniform buffers per step.
    kcache: Vec<DeviceBuffer>,
    vcache: Vec<DeviceBuffer>,
    dec: DecScratch,
    dec_tapes: std::cell::RefCell<Option<Vec<Vec<Step>>>>,
    // CPU input-embedding tables (residual codebooks) and output heads.
    codec_embedding: Vec<Vec<f32>>, // [n_residual][vocab*embedding_dim]
    lm_head: Vec<Vec<f32>>,         // [n_residual][vocab*d_model]
    // `Some((weight[d_model*embedding_dim], bias[d_model]))` when the Talker
    // hidden width (`embedding_dim`) differs from this MTP's own internal
    // width (`d_model`) -- the 1.7B family (embedding_dim 2048, d_model
    // 1024); `None` when they're equal (the 0.6B family), where the HF
    // checkpoint carries no such tensor at all and the projection really is
    // Identity. See `MtpConfig::embedding_dim`'s doc comment.
    small_to_mtp_projection: Option<(Vec<f32>, Vec<f32>)>,
}

/// Temperature/top-k/top-p sampling over one residual codebook's logit row.
/// No EOS masking or repetition penalty -- the residual codebooks have no EOS
/// token and the reference's `subtalker_*` config carries no repetition
/// penalty for them either, only the three sampling knobs this mirrors.
fn sample_residual(row: &[f32], opts: &crate::pipeline::ResidualOpts, rng: &mut data::rng::Rng) -> usize {
    if opts.temperature <= 0.0 {
        let mut best = 0usize;
        for j in 1..row.len() {
            if row[j] > row[best] {
                best = j;
            }
        }
        return best;
    }
    let mut scaled: Vec<f32> = row.iter().map(|&l| l / opts.temperature).collect();
    if opts.top_k > 0 && opts.top_k < scaled.len() {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        let threshold = scaled[idx[opts.top_k - 1]];
        for x in scaled.iter_mut() {
            if *x < threshold {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    if opts.top_p > 0.0 && opts.top_p < 1.0 {
        let max0 = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut ranked: Vec<(usize, f32)> =
            scaled.iter().enumerate().filter(|&(_, &x)| x.is_finite()).map(|(i, &x)| (i, (x - max0).exp())).collect();
        let z: f32 = ranked.iter().map(|&(_, p)| p).sum();
        if z > 0.0 {
            ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cum = 0.0f32;
            let mut cutoff = ranked.len();
            for (rank, &(_, p)) in ranked.iter().enumerate() {
                cum += p / z;
                if cum >= opts.top_p {
                    cutoff = rank + 1;
                    break;
                }
            }
            let keep: std::collections::HashSet<usize> = ranked[..cutoff].iter().map(|&(i, _)| i).collect();
            for (i, x) in scaled.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in scaled.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i;
        }
    }
    scaled.len() - 1
}

impl MtpModel {
    /// The fp32 GEMM tier for this device - the same rule `qwen3::serve`,
    /// `flux1`/`flux2` and `model::rowemit` use. Both fast kernels cooperate
    /// across a workgroup, so a device without `workgroup_reductions`
    /// (`backend-cpu`) keeps the naive reference, which that backend routes to
    /// its AVX2 GEMM anyway. Every variant is bit-identical to that reference;
    /// only the thread mapping differs.
    fn gemm_tier(&self) -> block::GemmVariants {
        if self.gpu.caps().workgroup_reductions {
            block::GemmVariants::Fast { gemv: Some(MATMUL_GEMV), tiled: MATMUL_REG3 }
        } else {
            block::GemmVariants::Reference(MATMUL)
        }
    }

    /// One `out[m,n] = x[m,k] @ w[n,k]^T` dispatch through [`Self::gemm_tier`].
    fn mm(&self, tier: block::GemmVariants, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        let (kind, threads) = block::gemm_variant(tier, m, n);
        self.gpu.step(kind, &[x, w, out], &[m, k, n], threads)
    }

    fn only_fwd_ids() -> KernelIds {
        // Forward needs rmsnorm/rms_inv, rope, gqa scores/apply/softmax, silu_mul.
        // No backward graph is built, so every backward slot is UNREGISTERED -
        // out of range of PIPELINES, so reaching one is a panic rather than a
        // silent dispatch of whichever kernel the stand-in index named.
        KernelIds {
            rmsnorm: RMSNORM,
            rms_inv: RMS_INV,
            rmsnorm_dx: block::UNREGISTERED,
            rmsnorm_dx_rows: block::UNREGISTERED,
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
            rmsnorm_rows: RMSNORM_ROWS,
        }
    }

    /// Decoder block parameter list (blocks + final norm); the codec embeddings
    /// and heads live on the CPU.
    pub(crate) fn decoder_param_list(cfg: &MtpConfig) -> Vec<(String, usize)> {
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

    /// Build on an existing device handle (see `gpu_core::Gpu::share`) so a
    /// process holds ONE device however many components it loads. `pub`
    /// (not just `pub(crate)`) so a caller with weights already in hand --
    /// e.g. a real-weight parity test reading straight from an HF mmap,
    /// bypassing `ParamStore`/file I/O entirely, the same pattern
    /// `crates/omni`'s other real-weight tests use -- doesn't need a round
    /// trip through a brain checkpoint file first.
    pub fn build_on(
        gpu: Gpu,
        cfg: MtpConfig,
        decoder: std::collections::HashMap<String, Vec<f32>>,
        codec_embedding: Vec<Vec<f32>>,
        lm_head: Vec<Vec<f32>>,
    ) -> MtpModel {
        Self::build_on_with_projection(gpu, cfg, decoder, codec_embedding, lm_head, None)
    }

    /// Same as [`Self::build_on`], with an explicit `small_to_mtp_projection`
    /// (see the field doc on [`MtpModel`]).
    pub fn build_on_with_projection(
        gpu: Gpu,
        cfg: MtpConfig,
        decoder: std::collections::HashMap<String, Vec<f32>>,
        codec_embedding: Vec<Vec<f32>>,
        lm_head: Vec<Vec<f32>>,
        small_to_mtp_projection: Option<(Vec<f32>, Vec<f32>)>,
    ) -> MtpModel {
        assert!(
            small_to_mtp_projection.is_some() || cfg.embedding_dim == cfg.d_model,
            "embedding_dim ({}) != d_model ({}) requires a small_to_mtp_projection",
            cfg.embedding_dim,
            cfg.d_model
        );
        let t = cfg.num_code_groups;
        let roles = Self::decoder_param_list(&cfg)
            .into_iter()
            .map(|(n, c)| (n, c, paramstore::Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &decoder);

        let n = t as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let bht2 = (cfg.n_heads * t * t) as u64;
        let st = |x: u64| gpu.storage(x);

        let mut res = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(n * d));
        }
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
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
            });
        }
        let nht = (cfg.n_heads * t) as u64;
        let dec = DecScratch {
            xn1: st(d),
            q_pre: st(hq),
            q: st(hq),
            k_pre: st(hkv),
            k: st(hkv),
            v: st(hkv),
            scores: st(nht),
            probs: st(nht),
            ctx: st(hq),
            xmid: st(d),
            xn2: st(d),
            gate_pre: st(ff),
            up: st(ff),
            h: st(ff),
            proj: st(d),
            mlp_out: st(d),
            xn_final: st(d),
        };
        let mut kcache = Vec::new();
        let mut vcache = Vec::new();
        for _ in 0..cfg.n_layers {
            kcache.push(st(n * hkv));
            vcache.push(st(n * hkv));
        }
        let mut m = MtpModel {
            cfg,
            t,
            ps,
            res,
            layers,
            proj: st(n * d),
            mlp_out: st(n * d),
            scores: st(bht2),
            xn_final: st(n * d),
            fwd_steps: Vec::new(),
            kcache,
            vcache,
            dec,
            dec_tapes: std::cell::RefCell::new(None),
            codec_embedding,
            lm_head,
            small_to_mtp_projection,
            gpu,
        };
        m.fwd_steps = m.forward_steps();
        m
    }

    /// Project an `embedding_dim`-wide row (a Talker hidden state, a
    /// codebook-0 embedding, or a raw residual-codebook embedding) down to
    /// this MTP's own `d_model` width, via `small_to_mtp_projection` when the
    /// two widths differ, or a straight copy when they're equal (the 0.6B
    /// family, where the reference itself has no such tensor).
    fn project_to_hidden(&self, x: &[f32]) -> Vec<f32> {
        let e = self.cfg.embedding_dim as usize;
        assert_eq!(x.len(), e, "expected an embedding_dim-wide row");
        match &self.small_to_mtp_projection {
            Some((w, b)) => {
                let d = self.cfg.d_model as usize;
                let mut out = model::hostmath::matvec(w, x, d, e);
                for (o, bi) in out.iter_mut().zip(b) {
                    *o += bi;
                }
                out
            }
            None => x.to_vec(),
        }
    }

    fn forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let ids = Self::only_fwd_ids();
        let tier = self.gemm_tier();
        let ga = Gqa {
            b: 1,
            t: n,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
        };
        let theta = c.rope_theta;
        let g = &self.gpu;
        let w = |name: &str| self.ps.w(name);
        let mut s: Vec<Step> = Vec::new();

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &self.res[l],
                w(&p("ln1.weight")),
                &lb.xn1,
                d,
                n,
            ));
            s.push(self.mm(tier, &lb.xn1, w(&p("attn.wq.weight")), &lb.q_pre, n, d, hq));
            s.push(self.mm(tier, &lb.xn1, w(&p("attn.wk.weight")), &lb.k_pre, n, d, hkv));
            s.push(self.mm(tier, &lb.xn1, w(&p("attn.wv.weight")), &lb.v, n, d, hkv));
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &lb.q_pre,
                w(&p("attn.q_norm.weight")),
                &lb.q,
                hd,
                n * nh,
            ));
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &lb.k_pre,
                w(&p("attn.k_norm.weight")),
                &lb.k,
                hd,
                n * nkv,
            ));
            s.push(block::rope_fwd(g, &ids, &lb.q, n, nh, hd, hq, n, theta));
            s.push(block::rope_fwd(g, &ids, &lb.k, n, nkv, hd, hkv, n, theta));
            s.extend(block::gqa_fwd(
                g,
                &ids,
                &ga,
                &lb.q,
                &lb.k,
                &lb.v,
                &self.scores,
                &lb.probs,
                &lb.ctx,
            ));
            s.push(self.mm(tier, &lb.ctx, w(&p("attn.wo.weight")), &self.proj, n, hq, d));
            s.push(g.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &lb.xmid,
                w(&p("ln2.weight")),
                &lb.xn2,
                d,
                n,
            ));
            s.push(self.mm(tier, &lb.xn2, w(&p("mlp.gate.weight")), &lb.gate_pre, n, d, ff));
            s.push(self.mm(tier, &lb.xn2, w(&p("mlp.up.weight")), &lb.up, n, d, ff));
            s.push(block::swiglu_fwd(
                g,
                &ids,
                &lb.gate_pre,
                &lb.up,
                &lb.h,
                n * ff,
            ));
            s.push(self.mm(tier, &lb.h, w(&p("mlp.down.weight")), &self.mlp_out, n, ff, d));
            s.push(g.step(
                ADD2,
                &[&lb.xmid, &self.mlp_out, &self.res[l + 1]],
                &[n * d],
                n * d,
            ));
        }
        let last = c.n_layers as usize;
        s.push(block::rmsnorm_fwd(
            g,
            &ids,
            &self.res[last],
            w("norm.weight"),
            &self.xn_final,
            d,
            n,
        ));
        s
    }

    /// One prebuilt incremental-decode tape per position `0..num_code_groups`.
    ///
    /// The same 5-layer block [`Self::forward_steps`] records, but for ONE new
    /// row against a per-layer key/value cache: `O(1)` projections and
    /// `O(pos)` attention instead of the whole `num_code_groups`-long sequence
    /// re-projected from scratch. Every uniform is a constant of `(layer,
    /// pos)`, so all `num_code_groups` tapes are recorded once and replayed -
    /// no per-step tape rebuild and no per-step uniform rewrite.
    ///
    /// The cache needs no explicit reset between frames: position `pos`'s tape
    /// always WRITES cache row `pos` before attending, and `attn_decode_*`
    /// only ever read rows `0..=pos`, so the previous frame's rows above `pos`
    /// are unreachable rather than stale.
    fn build_dec_tapes(&self) -> Vec<Vec<Step>> {
        let c = &self.cfg;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let half = hd / 2;
        let cap = self.t;
        let theta = c.rope_theta.to_bits();
        let g = &self.gpu;
        let sc = &self.dec;
        let ids = Self::only_fwd_ids();
        let tier = self.gemm_tier();
        let gd = block::GqaDecodeIds {
            kv_append: KV_APPEND,
            attn_decode_scores: ATTN_DECODE_SCORES,
            decode_softmax: DECODE_SOFTMAX,
            attn_decode_apply: ATTN_DECODE_APPLY,
        };
        let w = |name: &str| self.ps.w(name);
        (0..cap)
            .map(|pos| {
                let mut s: Vec<Step> = Vec::new();
                for l in 0..c.n_layers as usize {
                    let p = |name: &str| format!("blocks.{l}.{name}");
                    s.push(block::rmsnorm_fwd(g, &ids, &self.res[l], w(&p("ln1.weight")), &sc.xn1, d, 1));
                    s.push(self.mm(tier, &sc.xn1, w(&p("attn.wq.weight")), &sc.q_pre, 1, d, hq));
                    s.push(self.mm(tier, &sc.xn1, w(&p("attn.wk.weight")), &sc.k_pre, 1, d, hkv));
                    s.push(self.mm(tier, &sc.xn1, w(&p("attn.wv.weight")), &sc.v, 1, d, hkv));
                    s.push(block::rmsnorm_fwd(g, &ids, &sc.q_pre, w(&p("attn.q_norm.weight")), &sc.q, hd, nh));
                    s.push(block::rmsnorm_fwd(g, &ids, &sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, hd, nkv));
                    s.push(g.step(ROPE_AT, &[&sc.q], &[1, nh, hd, hq, 0, pos, theta], nh * half));
                    s.push(g.step(ROPE_AT, &[&sc.k], &[1, nkv, hd, hkv, 0, pos, theta], nkv * half));
                    s.extend(block::gqa_decode_step(
                        g,
                        &gd,
                        nh,
                        nkv,
                        hd,
                        pos,
                        cap,
                        &sc.q,
                        &sc.k,
                        &sc.v,
                        &self.kcache[l],
                        &self.vcache[l],
                        &sc.scores,
                        &sc.probs,
                        &sc.ctx,
                    ));
                    s.push(self.mm(tier, &sc.ctx, w(&p("attn.wo.weight")), &sc.proj, 1, hq, d));
                    s.push(g.step(ADD2, &[&self.res[l], &sc.proj, &sc.xmid], &[d], d));
                    s.push(block::rmsnorm_fwd(g, &ids, &sc.xmid, w(&p("ln2.weight")), &sc.xn2, d, 1));
                    s.push(self.mm(tier, &sc.xn2, w(&p("mlp.gate.weight")), &sc.gate_pre, 1, d, ff));
                    s.push(self.mm(tier, &sc.xn2, w(&p("mlp.up.weight")), &sc.up, 1, d, ff));
                    s.push(block::swiglu_fwd(g, &ids, &sc.gate_pre, &sc.up, &sc.h, ff));
                    s.push(self.mm(tier, &sc.h, w(&p("mlp.down.weight")), &sc.mlp_out, 1, ff, d));
                    s.push(g.step(ADD2, &[&sc.xmid, &sc.mlp_out, &self.res[l + 1]], &[d], d));
                }
                s.push(block::rmsnorm_fwd(g, &ids, &self.res[c.n_layers as usize], w("norm.weight"), &sc.xn_final, d, 1));
                s
            })
            .collect()
    }

    /// Record (never read back) one incremental decode step: put `embed`'s
    /// key/value into the cache at `pos`. Position 0 carries the Talker hidden
    /// state, which no output head reads, so seeding the cache with it must
    /// not cost a host round trip - `Gpu::read` is the only blocking call on
    /// this path, and skipping it here keeps a frame at exactly one device
    /// round trip per PREDICTED codebook.
    fn dec_submit(&self, embed: &[f32], pos: u32) {
        let d = self.cfg.d_model as usize;
        assert_eq!(embed.len(), d, "dec_step embed must be [d_model]");
        assert!(pos < self.t, "dec_step pos {pos} exceeds num_code_groups {}", self.t);
        if self.dec_tapes.borrow().is_none() {
            *self.dec_tapes.borrow_mut() = Some(self.build_dec_tapes());
        }
        let g = &self.gpu;
        // `res[0]` is `[num_code_groups, d_model]`; a decode step uses row 0
        // only, the same way the Talker's own cached step writes its `res[0]`.
        // `Gpu::write` submits everything recorded before it first, so the
        // previous position's tape can never read this row.
        g.write(&self.res[0], bytemuck::cast_slice(embed));
        let tapes = self.dec_tapes.borrow();
        g.submit(&[], &tapes.as_ref().unwrap()[pos as usize]);
    }

    /// [`Self::dec_submit`] plus the readback: this position's final-norm
    /// hidden state (`[d_model]`), which its output head then reads.
    fn dec_step(&self, embed: &[f32], pos: u32) -> Vec<f32> {
        self.dec_submit(embed, pos);
        self.gpu.read(&self.dec.xn_final, self.cfg.d_model as usize)
    }

    /// Run the decoder over an assembled `[num_code_groups, d_model]` input
    /// embedding sequence and return the final-norm hidden states,
    /// `[num_code_groups, d_model]`.
    fn hidden(&self, inputs_embeds: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        let t = self.t as usize;
        assert_eq!(
            inputs_embeds.len(),
            t * d,
            "inputs_embeds must be [num_code_groups, d_model]"
        );
        self.gpu
            .write(&self.res[0], bytemuck::cast_slice(inputs_embeds));
        self.gpu.submit(&[], &self.fwd_steps);
        self.gpu.read(&self.xn_final, t * d)
    }

    /// `lm_head[idx]` applied to one final-norm hidden row -> `[vocab]`.
    ///
    /// `hostmath::matvec` is the AVX2+FMA, rayon-over-rows `matmul_abt`; the
    /// scalar `for o { for k { } }` loop this replaced was the single largest
    /// host term in a real synth run (`[2048, 1024]` per head, 15 heads per
    /// residual step, 15 residual steps per audio frame).
    fn head_row(&self, idx: usize, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        let v = self.cfg.vocab as usize;
        model::hostmath::matvec(&self.lm_head[idx], hidden_row, v, d)
    }

    /// Run the decoder over an assembled `[num_code_groups, d_model]` input
    /// embedding sequence and return the residual-codebook logits, shape
    /// `[(num_code_groups - 1) * vocab]` (row `i` = logits for codebook `i+1`,
    /// produced by `lm_head[i]` from decoder position `i+1`).
    ///
    /// Every head is evaluated here, which is what a caller wanting the whole
    /// logit block (parity dumps, tests) asks for. The autoregressive
    /// generation loop needs exactly ONE of those rows per step and must use
    /// [`Self::logits_at`] instead.
    pub fn logits(&self, inputs_embeds: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        let v = self.cfg.vocab as usize;
        let t = self.t as usize;
        let hidden = self.hidden(inputs_embeds);
        let mut out = vec![0.0f32; (t - 1) * v];
        for i in 1..t {
            let row = self.head_row(i - 1, &hidden[i * d..(i + 1) * d]);
            out[(i - 1) * v..i * v].copy_from_slice(&row);
        }
        out
    }

    /// The single logit row [`Self::logits`] would place at `(k - 1) * vocab`:
    /// decoder position `k`'s hidden state through `lm_head[k - 1]`.
    /// Identical arithmetic, `num_code_groups - 1` times less of it - the
    /// generation loop discards every other row.
    fn logits_at(&self, inputs_embeds: &[f32], k: usize) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        let hidden = self.hidden(inputs_embeds);
        self.head_row(k - 1, &hidden[k * d..(k + 1) * d])
    }

    /// Assemble the input-embedding sequence for one frame. `talker_hidden` is the
    /// Talker hidden state (`[d_model]`); `cb0_embed` is the Talker codec-0
    /// embedding (`[d_model]`, supplied by the Talker since the MTP does not own
    /// that table); `residual_codes` are codebooks `1..=(num_code_groups-2)`
    /// (length `num_code_groups - 2`), embedded by the MTP's own tables. Returns
    /// `[num_code_groups, d_model]`.
    pub fn assemble(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
        residual_codes: &[u32],
    ) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        let e = self.cfg.embedding_dim as usize;
        let t = self.t as usize;
        assert_eq!(talker_hidden.len(), e);
        assert_eq!(cb0_embed.len(), e);
        assert_eq!(residual_codes.len(), t.saturating_sub(2));
        let mut out = vec![0.0f32; t * d];
        out[0..d].copy_from_slice(&self.project_to_hidden(talker_hidden));
        out[d..2 * d].copy_from_slice(&self.project_to_hidden(cb0_embed));
        for (i, &code) in residual_codes.iter().enumerate() {
            // position 2+i embeds codebook (i+1) via codec_embedding[i].
            let tbl = &self.codec_embedding[i];
            let src = code as usize * e;
            let row = self.project_to_hidden(&tbl[src..src + e]);
            out[(2 + i) * d..(3 + i) * d].copy_from_slice(&row);
        }
        out
    }

    /// Per-frame residual codebook generation (greedy). Given the Talker final
    /// hidden state at this frame (`talker_hidden`, `[d_model]`) and the Talker
    /// codebook-0 embedding (`cb0_embed`, `[d_model]`, from the Talker's own
    /// table), autoregressively predict residual codebooks `1..=15` and return
    /// `(codes, residual_embed_sum)`:
    ///   * `codes` — the 15 residual codebook ids (codebooks 1..15),
    ///   * `residual_embed_sum` — `Σ_{i=1}^{15} codec_embedding[i-1][code_i]`
    ///     (`[d_model]`), the residual part of the frame's feedback embedding.
    ///
    /// Mirrors `code_predictor.generate` in `modeling_qwen3_tts.py`: position 0 is
    /// the Talker hidden, position 1 the cb0 embed, position `i+1` (`i>=1`) the
    /// embedding of codebook `i`; `lm_head[i-1]` reads hidden position `i+1` to
    /// predict codebook `i+1`. Because attention is causal, predicting codebook
    /// `k` only needs positions `0..=k` filled, so we grow the sequence in place
    /// (future positions stay zero and never influence the read position).
    pub fn generate_residuals(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let mut rng = data::rng::Rng::new(0);
        self.generate_residuals_with(talker_hidden, cb0_embed, None, &mut rng)
    }

    /// Same as [`Self::generate_residuals`], with optional independent sampling
    /// on the residual codebooks (`residual = None` is the greedy argmax;
    /// `rng` is only consulted when `residual.is_some()`).
    /// See `crate::pipeline::ResidualOpts` / `GenOpts::residual` for why this
    /// exists: the reference's own `subtalker_*` config keys sample these
    /// codebooks too, and they carry most of the acoustic detail.
    ///
    /// **KV-cached**: one incremental decoder step per position, not one full
    /// re-forward of the growing `[num_code_groups, d_model]` sequence per
    /// residual codebook. Algebraically the same thing - attention is causal,
    /// so position `k`'s hidden state only ever depended on positions `0..=k`,
    /// which the cache holds exactly - but `num_code_groups` times less
    /// decoder arithmetic per audio frame. Gated against the recompute it
    /// replaces by `kv_cached_residuals_match_the_full_recompute`; the
    /// recompute itself is kept as [`Self::generate_residuals_recompute`].
    pub fn generate_residuals_with(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
        residual: Option<&crate::pipeline::ResidualOpts>,
        rng: &mut data::rng::Rng,
    ) -> (Vec<u32>, Vec<f32>) {
        let e = self.cfg.embedding_dim as usize;
        let v = self.cfg.vocab as usize;
        let nres = self.t as usize - 1; // 15
        assert_eq!(talker_hidden.len(), e);
        assert_eq!(cb0_embed.len(), e);

        let mut codes = vec![0u32; nres];
        // `res_sum` feeds back into the TALKER's own embedding stream
        // (`pipeline::generate_codes`'s `feed`), which is `embedding_dim`
        // wide -- NOT this MTP's internal `d_model`. Accumulate the RAW
        // (unprojected) codec_embedding rows, matching `codec_embed`'s own
        // contract above.
        let mut res_sum = vec![0.0f32; e];

        // pos 0: the Talker hidden state. No head reads it; it is decoded only
        // to put its key/value into the cache.
        let _ = self.dec_step(&self.project_to_hidden(talker_hidden), 0);
        // pos k (1..=nres): input is codebook (k-1)'s embedding (pos 1 = cb0);
        // `lm_head[k-1]` reads pos k to predict codebook k.
        let mut input_raw = cb0_embed.to_vec();
        for k in 1..=nres {
            let hidden = self.dec_step(&self.project_to_hidden(&input_raw), k as u32);
            let row = self.head_row(k - 1, &hidden);
            let best = match residual {
                Some(ro) => sample_residual(&row, ro, rng),
                None => {
                    let mut best = 0usize;
                    for j in 1..v {
                        if row[j] > row[best] {
                            best = j;
                        }
                    }
                    best
                }
            };
            codes[k - 1] = best as u32;
            // codec_embedding[k-1] embeds codebook k.
            let r = &self.codec_embedding[k - 1][best * e..(best + 1) * e];
            for j in 0..e {
                res_sum[j] += r[j];
            }
            if k < nres {
                input_raw = r.to_vec();
            }
        }
        (codes, res_sum)
    }

    /// The `O(num_code_groups^2)` full-recompute residual generation
    /// [`Self::generate_residuals_with`] replaced: one whole re-forward of the
    /// growing input-embedding sequence per residual codebook. Kept as the
    /// reference the cached path is gated against, and as the shape
    /// `MtpModel::logits` still serves for parity dumps.
    pub fn generate_residuals_recompute(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
        residual: Option<&crate::pipeline::ResidualOpts>,
        rng: &mut data::rng::Rng,
    ) -> (Vec<u32>, Vec<f32>) {
        let d = self.cfg.d_model as usize;
        let e = self.cfg.embedding_dim as usize;
        let v = self.cfg.vocab as usize;
        let t = self.t as usize; // num_code_groups (16)
        let nres = t - 1; // 15
        assert_eq!(talker_hidden.len(), e);
        assert_eq!(cb0_embed.len(), e);

        let mut emb = vec![0.0f32; t * d];
        emb[0..d].copy_from_slice(&self.project_to_hidden(talker_hidden));
        emb[d..2 * d].copy_from_slice(&self.project_to_hidden(cb0_embed));

        let mut codes = vec![0u32; nres];
        let mut res_sum = vec![0.0f32; e];
        // k = codebook index being predicted (1..=15); head index = k-1; read pos = k.
        for k in 1..=nres {
            // Only row `k-1` of the `[(t-1), vocab]` logit block is read here,
            // so only that row is computed.
            let row = &self.logits_at(&emb, k)[..];
            let best = match residual {
                Some(ro) => sample_residual(row, ro, rng),
                None => {
                    let mut best = 0usize;
                    for j in 1..v {
                        if row[j] > row[best] {
                            best = j;
                        }
                    }
                    best
                }
            };
            codes[k - 1] = best as u32;
            // codec_embedding[k-1] embeds codebook k.
            let tbl = &self.codec_embedding[k - 1];
            let r = &tbl[best * e..(best + 1) * e];
            for j in 0..e {
                res_sum[j] += r[j];
            }
            if k < nres {
                // position k+1 carries the embedding of codebook k for the next step.
                let projected = self.project_to_hidden(r);
                emb[(k + 1) * d..(k + 2) * d].copy_from_slice(&projected);
            }
        }
        (codes, res_sum)
    }

    /// Residual codebook embedding row: `codec_embedding[residual_idx][code]`
    /// (`[embedding_dim]` -- the Talker's own hidden width, NOT `d_model`;
    /// they coincide on the 0.6B family but not the 1.7B). `residual_idx` is
    /// `0..=14` (codebook `residual_idx + 1`). Used to build the
    /// reference-audio codec embedding in ICL voice-clone prompts, which are
    /// added directly into the Talker's own embedding stream -- so this
    /// deliberately returns the UNPROJECTED row, not `project_to_hidden`'s
    /// `d_model`-wide output (that projection is this MTP's own internal
    /// concern, not part of the Talker-side contract this method serves).
    pub fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32] {
        let e = self.cfg.embedding_dim as usize;
        let s = code as usize * e;
        &self.codec_embedding[residual_idx][s..s + e]
    }

    /// Load an inference-only MTP from a brain checkpoint written by
    /// [`crate::import::import_mtp`].
    pub fn load_inference(path: &str) -> MtpModel {
        Self::load_inference_on(Gpu::new(PIPELINES), path)
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`).
    pub fn load_inference_on(gpu: Gpu, path: &str) -> MtpModel {
        let c = checkpoint::load(path);
        let cfg = MtpConfig::from_brain_json(&c.header["config"]);
        let take = |name: &str| {
            c.find(name, "")
                .cloned()
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        let mut decoder = std::collections::HashMap::new();
        for (n, _) in Self::decoder_param_list(&cfg) {
            let d = take(&n);
            decoder.insert(n, d);
        }
        let nres = cfg.n_residual() as usize;
        let codec_embedding = (0..nres)
            .map(|i| take(&format!("codec_embedding.{i}.weight")))
            .collect();
        let lm_head = (0..nres)
            .map(|i| take(&format!("lm_head.{i}.weight")))
            .collect();
        // Present only when `embedding_dim != d_model` (the 1.7B family);
        // `import::import_mtp` writes it exactly then, per the HF checkpoint.
        let projection = c
            .find("small_to_mtp_projection.weight", "")
            .cloned()
            .map(|w| (w, take("small_to_mtp_projection.bias")));
        MtpModel::build_on_with_projection(gpu, cfg, decoder, codec_embedding, lm_head, projection)
    }

    /// Build a randomly-initialised MTP for tests.
    pub fn new_synthetic(cfg: MtpConfig, seed: u64) -> MtpModel {
        Self::new_synthetic_on(Gpu::new(PIPELINES), cfg, seed)
    }

    pub(crate) fn new_synthetic_on(gpu: Gpu, cfg: MtpConfig, seed: u64) -> MtpModel {
        use data::rng::Rng;
        let mut rng = Rng::new(seed);
        let mut normal = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
        };
        let proj_std = 0.02f32 / ((2.0 * cfg.n_layers as f32).sqrt());
        let mut decoder = std::collections::HashMap::new();
        for (n, numel) in Self::decoder_param_list(&cfg) {
            let v = if n.ends_with("norm.weight")
                || n.ends_with("ln1.weight")
                || n.ends_with("ln2.weight")
            {
                vec![1.0; numel]
            } else if n.ends_with("attn.wo.weight") || n.ends_with("mlp.down.weight") {
                normal(numel, proj_std)
            } else {
                normal(numel, 0.02)
            };
            decoder.insert(n, v);
        }
        let nres = cfg.n_residual() as usize;
        let d = cfg.d_model as usize;
        let e = cfg.embedding_dim as usize;
        let v = cfg.vocab as usize;
        // Embedding tables are `embedding_dim`-wide (the Talker's own hidden
        // width); only `lm_head` (reading the internal `d_model`-wide hidden
        // state) stays at `d_model`.
        let codec_embedding = (0..nres).map(|_| normal(v * e, 0.02)).collect();
        let lm_head = (0..nres).map(|_| normal(v * d, 0.02)).collect();
        let projection = if e != d { Some((normal(d * e, 0.02), normal(d, 0.0))) } else { None };
        MtpModel::build_on_with_projection(gpu, cfg, decoder, codec_embedding, lm_head, projection)
    }
}

impl crate::prompt::MtpHost for MtpModel {
    fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32] {
        MtpModel::codec_embed(self, residual_idx, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    #[test]
    fn forward_shape_and_finite() {
        if gpu_disabled() {
            return;
        }
        let cfg = MtpConfig::tiny();
        let t = cfg.num_code_groups as usize;
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let m = MtpModel::new_synthetic_on(gpu_core::testgpu::dev(PIPELINES), cfg, 5);
        let embeds: Vec<f32> = (0..t * d).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let logits = m.logits(&embeds);
        assert_eq!(logits.len(), (t - 1) * v);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn assemble_layout() {
        if gpu_disabled() {
            return;
        }
        let cfg = MtpConfig::tiny(); // num_code_groups = 4 -> residual_codes len 2
        let d = cfg.d_model as usize;
        let m = MtpModel::new_synthetic_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1);
        let th = vec![0.5f32; d];
        let cb0 = vec![-0.5f32; d];
        let embeds = m.assemble(&th, &cb0, &[1, 2]);
        assert_eq!(embeds.len(), 4 * d);
        assert_eq!(&embeds[0..d], &th[..]);
        assert_eq!(&embeds[d..2 * d], &cb0[..]);
        let logits = m.logits(&embeds);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    /// Regression for the 1.7B-family bug found running `brain qwen3tts
    /// design` against a real `Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint:
    /// it panicked in `assert_eq!(cb0_embed.len(), d)` because that build
    /// assumed `small_to_mtp_projection` was always Identity (true only when
    /// `embedding_dim == d_model`, the 0.6B case). `tiny_projected` sets
    /// `embedding_dim=24 != d_model=16`, matching the real 1.7B checkpoint's
    /// `hidden_size=2048 != code_predictor.hidden_size=1024` shape mismatch.
    #[test]
    fn assemble_projects_embedding_dim_rows_down_to_d_model() {
        if gpu_disabled() {
            return;
        }
        let cfg = MtpConfig::tiny_projected();
        let (d, e) = (cfg.d_model as usize, cfg.embedding_dim as usize);
        assert_ne!(d, e, "test config must actually exercise the projection");
        let m = MtpModel::new_synthetic_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1);
        let th = vec![0.5f32; e];
        let cb0 = vec![-0.5f32; e];
        // Would panic before the fix (assert_eq! on mismatched widths); now
        // produces a properly `d_model`-wide sequence.
        let embeds = m.assemble(&th, &cb0, &[1, 2]);
        assert_eq!(embeds.len(), 4 * d);
        assert!(embeds.iter().all(|x| x.is_finite()));
        // The projection must actually run (not silently truncate/passthrough):
        // pos 0's d_model-wide row is NOT simply th's first d entries.
        assert_ne!(&embeds[0..d], &th[0..d]);

        let (codes, res_sum) = m.generate_residuals_with(&th, &cb0, None, &mut data::rng::Rng::new(1));
        assert_eq!(codes.len(), 3); // num_code_groups(4) - 1
        assert_eq!(res_sum.len(), e, "feedback embedding must stay Talker-width (e), not d_model");
        assert!(res_sum.iter().all(|x| x.is_finite()));
    }

    /// The KV-cached residual generation must reproduce the full-recompute one
    /// it replaced: the codes bit-for-bit (they are argmax/sample decisions,
    /// and a single flipped code changes the audio), and the feedback
    /// embedding to within fp reassociation. Attention is causal, so this is a
    /// theorem about the cache, not a tolerance to tune - if it ever fails,
    /// the cache is wrong, not imprecise.
    ///
    /// Run at BOTH MTP shapes the checkpoint family has: `tiny` (0.6B-like,
    /// `embedding_dim == d_model`, projection Identity) and `tiny_projected`
    /// (1.7B-like, `embedding_dim != d_model`, a real
    /// `small_to_mtp_projection` on every position's input).
    #[test]
    fn kv_cached_residuals_match_the_full_recompute() {
        if gpu_disabled() {
            return;
        }
        for cfg in [MtpConfig::tiny(), MtpConfig::tiny_projected()] {
            let e = cfg.embedding_dim as usize;
            let nres = cfg.num_code_groups as usize - 1;
            let m = MtpModel::new_synthetic_on(gpu_core::testgpu::dev(PIPELINES), cfg, 11);
            let mut rng = data::rng::Rng::new(3);
            let th: Vec<f32> = (0..e).map(|_| rng.next_gaussian() as f32 * 0.5).collect();
            let cb0: Vec<f32> = (0..e).map(|_| rng.next_gaussian() as f32 * 0.5).collect();

            let (c_cached, r_cached) = m.generate_residuals(&th, &cb0);
            let (c_full, r_full) =
                m.generate_residuals_recompute(&th, &cb0, None, &mut data::rng::Rng::new(0));
            assert_eq!(c_cached.len(), nres);
            assert_eq!(c_cached, c_full, "KV cache changed the residual codes");
            let err = r_cached
                .iter()
                .zip(&r_full)
                .fold(0.0f32, |mx, (a, b)| mx.max((a - b).abs()));
            assert!(err < 1e-4, "cached res_sum diverges from the recompute: {err}");

            // Run it a second time: the cache carries no state between frames
            // (each position's tape overwrites its own cache row before
            // attending), so a repeated call must give the identical answer.
            let (c_again, _) = m.generate_residuals(&th, &cb0);
            assert_eq!(c_again, c_cached, "a second frame saw stale KV-cache rows");
        }
    }
}

/// The coalesced RMSNorm this model now selects (`rmsnorm_rows`, via
/// `block::rms_variant` inside `block::rmsnorm_fwd`) is NOT bit-identical to
/// the per-element `rmsnorm` it replaced: 64 partial sums fold in a different
/// order. It was adopted for throughput, so what it computes is gated here,
/// against a HOST reference, at the shapes THIS model's decode tape really
/// dispatches - narrow rows are where the two reduction orders differ most,
/// and they are also the whole reason the swap is worth making.
#[cfg(test)]
mod rmsnorm_variant_agreement {
    use super::*;

    #[test]
    fn the_registered_slot_names_the_coalesced_kernel() {
        assert_eq!(PIPELINES[MtpModel::only_fwd_ids().rmsnorm_rows].0, "rmsnorm_rows");
    }

    #[test]
    fn the_tape_norms_match_the_host_reference() {
        // The MTP tape is the one adopting tape here that is NOT
        // decode-shaped: its rows are the 16 code groups, never 1, and it is
        // replayed 15 times per audio frame. Real MTP: d_model 1024, 16/8
        // heads of 128, num_code_groups 16.
        let shapes = [(16, 1024, "ln1/ln2/final norm"), (256, 128, "q_norm (t*n_heads)"), (128, 128, "k_norm (t*n_kv_heads)")];
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        model::block::assert_rmsnorm_variant_agrees(&gpu, &MtpModel::only_fwd_ids(), &shapes);
    }
}
