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
//! CPU.
//!
//! ## Training
//! [`MtpModel::new_trainable`] adds a real forward+backward over exactly that
//! wiring - the hybrid one, not a second all-device reimplementation of it: the
//! trainable forward calls the SAME [`MtpModel::assemble`] (hence the same
//! `small_to_mtp_projection`), the SAME device decoder tape, and the SAME
//! per-position `lm_head` host math a served run does, so what the gradient
//! check gates is the production forward rather than a parallel copy of it. The
//! decoder half's backward composes the shared `model::block` builders
//! (`rmsnorm_bwd`, `rope_bwd`, `gqa_bwd`, `swiglu_bwd`) over the same
//! activations the forward tape leaves resident; the host half (per-position
//! heads, the projection, the codec-embedding scatter) is a hand-written
//! adjoint. Gated by `mtp_analytic_grads_match_finite_differences` at both
//! checkpoint shapes (`MtpConfig::tiny` and `tiny_projected`).

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

// ---- backward kernel table (present only in TRAIN_PIPELINES) ----
// These slots sit PAST the end of `PIPELINES`, so an inference-built handle
// cannot reach them: `only_fwd_ids` still names `block::UNREGISTERED` in every
// backward slot, and a served MTP compiles exactly the kernels it dispatches.
const RMSNORM_DX: usize = 17;
const RMSNORM_DX_ROWS: usize = 18;
const RMSNORM_DW: usize = 19;
const ROPE_BWD: usize = 20;
const GQA_DSCORES: usize = 21;
const GQA_DV: usize = 22;
const GQA_DQ: usize = 23;
const GQA_DK: usize = 24;
const SILU_DA: usize = 25;
const SILU_DB: usize = 26;
const MATMUL_DX: usize = 27;
const MATMUL_DW: usize = 28;

/// [`PIPELINES`] plus the backward half, for [`MtpModel::new_trainable`].
///
/// A separate table rather than a longer `PIPELINES` because a served MTP
/// dispatches none of these and would otherwise pay their compilation on every
/// load. The first `PIPELINES.len()` entries are `PIPELINES` verbatim and in
/// order (gated by `train_pipelines_extends_the_inference_table`), so every
/// forward-slot const above indexes both tables identically and a trainable
/// model can replay the very same forward tape an inference one records.
pub const TRAIN_PIPELINES: &[(&str, &str)] = &[
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
    // ---- backward half ----
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dx_rows", kernels::RMSNORM_DX_ROWS),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
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
    // `Some` only for a `new_trainable` build; the whole backward half.
    train: Option<Train>,
}

/// One frame's training example, in the reference's own `forward_finetune`
/// alignment: no time shift at all. `targets[k-1]` is the ground-truth code of
/// residual codebook `k` (`1..num_code_groups`) of the SAME frame, and it is
/// also the code fed in at sequence position `k + 1` - teacher forcing, exactly
/// as `Qwen3TTSTalkerCodePredictorModel.forward_finetune` builds its own input
/// embeddings. `crate::sft::MultiCodebookLabels::residual_targets` produces one
/// such row per frame from a `[T, num_q]` codes tensor.
struct Batch {
    talker_hidden: Vec<f32>,
    cb0_embed: Vec<f32>,
    targets: Vec<u32>,
}

/// What one trainable forward hands its backward. Held rather than recomputed,
/// because the backward's host half needs the exact rows the forward consumed
/// and produced; re-deriving them would be a second chance to disagree.
struct Fwd {
    /// The `embedding_dim`-wide input row at each position, BEFORE
    /// `small_to_mtp_projection` - the projection backward's `x`.
    raw: Vec<Vec<f32>>,
    /// Final-norm hidden states, `[num_code_groups, d_model]`.
    hidden: Vec<f32>,
    /// `[(num_code_groups - 1) * vocab]`, from `crate::sft::ce_batch` - already
    /// averaged over the scored rows, so it differentiates exactly the scalar
    /// `MtpModel::forward` returns.
    d_logits: Vec<f32>,
}

/// The backward half of a [`MtpModel::new_trainable`] build: the device
/// gradient buffers plus the prebuilt decoder backward tape, and the host-side
/// gradients of the three parameter families that live off the device (the
/// per-residual `codec_embedding` and `lm_head` tables, and
/// `small_to_mtp_projection`).
struct Train {
    bwd_steps: Vec<Step>,
    /// `d_res[l]` is the gradient w.r.t. layer `l`'s residual INPUT, so
    /// `d_res[0]` is the gradient w.r.t. the assembled
    /// `[num_code_groups, d_model]` input-embedding sequence - the seam where
    /// the host half picks the chain back up.
    d_res: Vec<DeviceBuffer>,
    /// Gradient w.r.t. the final-norm OUTPUT, i.e. w.r.t. what the per-position
    /// output heads read. Host-written at the start of every backward.
    d_hidden: DeviceBuffer,
    inv: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    dq_pre: DeviceBuffer,
    dk_pre: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    g_codec_embedding: Vec<Vec<f32>>,
    g_lm_head: Vec<Vec<f32>>,
    g_projection: Option<(Vec<f32>, Vec<f32>)>,
    batch: Option<Batch>,
    fwd: Option<Fwd>,
}

/// A parameter that lives on the HOST rather than in the `ParamStore` - one of
/// the three families the served forward reads through `model::hostmath`
/// instead of a device dispatch. Parsed from the checkpoint name so
/// `read_weight` / `write_weight` / `read_grad` route by one rule.
enum HostParam {
    Codec(usize),
    Head(usize),
    ProjWeight,
    ProjBias,
}

fn host_param(name: &str) -> Option<HostParam> {
    if let Some(i) = name.strip_prefix("codec_embedding.").and_then(|r| r.strip_suffix(".weight")) {
        return i.parse().ok().map(HostParam::Codec);
    }
    if let Some(i) = name.strip_prefix("lm_head.").and_then(|r| r.strip_suffix(".weight")) {
        return i.parse().ok().map(HostParam::Head);
    }
    match name {
        "small_to_mtp_projection.weight" => Some(HostParam::ProjWeight),
        "small_to_mtp_projection.bias" => Some(HostParam::ProjBias),
        _ => None,
    }
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
        Self::build(gpu, cfg, decoder, codec_embedding, lm_head, small_to_mtp_projection, false)
    }

    /// The one builder. `train` decides the `ParamStore` role of every decoder
    /// weight (`Trainable` allocates the gradient/AdamW buffers, `Frozen`
    /// allocates the weight only) and whether the backward tape and the
    /// host-side gradient tables are built at all - so an inference build pays
    /// nothing for the training half, and a training build shares the forward
    /// tape rather than growing a second one.
    fn build(
        gpu: Gpu,
        cfg: MtpConfig,
        decoder: std::collections::HashMap<String, Vec<f32>>,
        codec_embedding: Vec<Vec<f32>>,
        lm_head: Vec<Vec<f32>>,
        small_to_mtp_projection: Option<(Vec<f32>, Vec<f32>)>,
        train: bool,
    ) -> MtpModel {
        assert!(
            small_to_mtp_projection.is_some() || cfg.embedding_dim == cfg.d_model,
            "embedding_dim ({}) != d_model ({}) requires a small_to_mtp_projection",
            cfg.embedding_dim,
            cfg.d_model
        );
        let t = cfg.num_code_groups;
        let role = if train { paramstore::Role::Trainable } else { paramstore::Role::Frozen };
        let roles = Self::decoder_param_list(&cfg)
            .into_iter()
            .map(|(n, c)| (n, c, role))
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
            train: None,
            gpu,
        };
        m.fwd_steps = m.forward_steps();
        if train {
            m.train = Some(m.build_train());
        }
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

    /// Per-frame residual codebook generation, pinned GREEDY.
    ///
    /// **A convenience wrapper only.** A real decode calls
    /// [`Self::generate_residuals_with`] with the run's resolved subtalker
    /// plan, which the reference (and this checkpoint's
    /// `generation_config.json`) says is SAMPLED. This exists for the parity,
    /// determinism and shape tests that want an argmax they can compare
    /// bit-for-bit, and it says so in a `SamplerCfg::greedy()` rather than in a
    /// `None` that a caller could mistake for "the default".
    pub fn generate_residuals(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let mut rng = data::rng::Rng::new(0);
        self.generate_residuals_with(talker_hidden, cb0_embed, &crate::sampling::SamplerCfg::greedy(), &mut rng)
    }

    /// Per-frame residual codebook generation under an explicit filter chain -
    /// the run's `GenerationPlan::subtalker`, resolved from the checkpoint's
    /// `subtalker_*` keys.
    ///
    /// Given the Talker final hidden state at this frame (`talker_hidden`,
    /// `[embedding_dim]`) and the Talker codebook-0 embedding (`cb0_embed`,
    /// same width, from the Talker's own table), autoregressively predict
    /// residual codebooks `1..=15` and return `(codes, residual_embed_sum)`:
    ///   * `codes` - the 15 residual codebook ids (codebooks 1..15),
    ///   * `residual_embed_sum` - `Σ_{i=1}^{15} codec_embedding[i-1][code_i]`,
    ///     the residual part of the frame's feedback embedding.
    ///
    /// Mirrors `code_predictor.generate` in `modeling_qwen3_tts.py`: position 0
    /// is the Talker hidden, position 1 the cb0 embed, position `i+1` (`i>=1`)
    /// the embedding of codebook `i`; `lm_head[i-1]` reads hidden position `i+1`
    /// to predict codebook `i+1`. Because attention is causal, predicting
    /// codebook `k` only needs positions `0..=k` filled, so we grow the sequence
    /// in place (future positions stay zero and never influence the read
    /// position).
    ///
    /// Every draw goes through [`crate::sampling::sample_residual`], the same
    /// module (and the same `draw_from_logits`) codebook 0 uses; a greedy `cfg`
    /// is an argmax and consumes no `rng`. `rng` is shared with the caller's
    /// codebook-0 stream on purpose: one seed reproduces one clip.
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
        cfg: &crate::sampling::SamplerCfg,
        rng: &mut data::rng::Rng,
    ) -> (Vec<u32>, Vec<f32>) {
        let e = self.cfg.embedding_dim as usize;
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
            let best = crate::sampling::sample_residual(&row, cfg, rng).token as usize;
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
        cfg: &crate::sampling::SamplerCfg,
        rng: &mut data::rng::Rng,
    ) -> (Vec<u32>, Vec<f32>) {
        let d = self.cfg.d_model as usize;
        let e = self.cfg.embedding_dim as usize;
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
            let best = crate::sampling::sample_residual(row, cfg, rng).token as usize;
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
        let (decoder, codec_embedding, lm_head, projection) = Self::synthetic_weights(&cfg, seed);
        MtpModel::build(gpu, cfg, decoder, codec_embedding, lm_head, projection, false)
    }

    /// Randomly-initialised weights for every parameter family this model has:
    /// the decoder block set, the per-residual `codec_embedding` / `lm_head`
    /// tables, and `small_to_mtp_projection` when (and only when) the config
    /// actually needs one.
    #[allow(clippy::type_complexity)]
    fn synthetic_weights(
        cfg: &MtpConfig,
        seed: u64,
    ) -> (
        std::collections::HashMap<String, Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Option<(Vec<f32>, Vec<f32>)>,
    ) {
        use data::rng::Rng;
        let cfg = cfg.clone();
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
        // `small_to_mtp_projection` has to PRESERVE the width of the row it
        // rescales - it is the seam between the Talker's hidden state and this
        // decoder's, not an attenuator - so its std is `1/sqrt(embedding_dim)`
        // rather than the flat 0.02 every other tensor here gets. Those two
        // coincide at the real 1.7B shape (`1/sqrt(2048) = 0.022`) and differ
        // by 10x at a toy `embedding_dim` of 24, where a flat 0.02 would shrink
        // the residual stream to an rms of ~0.05 and leave every downstream
        // RMSNorm running at a ~20x gain - a miniature that does not behave
        // like the model it stands in for.
        let projection = if e != d {
            Some((normal(d * e, 1.0 / (e as f32).sqrt()), normal(d, 0.0)))
        } else {
            None
        };
        (decoder, codec_embedding, lm_head, projection)
    }

    /// Build a randomly-initialised **trainable** MTP (for tests / gradient
    /// checks), the twin of `TalkerModel::new_trainable`.
    ///
    /// The Talker's version delegates wholesale to `qwen3::Qwen`, which already
    /// carries a gradient-checked backward; the MTP has no such inner model -
    /// its decoder is assembled here out of `model::block`, its input
    /// embeddings are gathered and projected on the host, and its output heads
    /// are fifteen separate per-position linears, so this is the first backward
    /// over any of that. `crate::sft`'s `finetune_lora`/`finetune_full` train
    /// the Talker only.
    pub fn new_trainable(cfg: MtpConfig, seed: u64) -> MtpModel {
        Self::new_trainable_on(Gpu::new(TRAIN_PIPELINES), cfg, seed)
    }

    /// [`Self::new_trainable`] on an existing device handle, which MUST have
    /// been built from [`TRAIN_PIPELINES`] - a `PIPELINES` handle has no
    /// backward kernels at all, and a `Step`'s `kind` is an index into the
    /// specific handle's own compiled pipeline vector, so the mismatch is
    /// checked here by name rather than left to fail as an out-of-range
    /// dispatch deep inside the backward tape.
    pub fn new_trainable_on(gpu: Gpu, cfg: MtpConfig, seed: u64) -> MtpModel {
        for (i, (name, _)) in TRAIN_PIPELINES.iter().enumerate() {
            assert_eq!(
                gpu.kernel_name(i),
                Some(*name),
                "MtpModel::new_trainable_on: slot {i} is {:?}, expected {name} - \
                 the handle must be built from mtp::TRAIN_PIPELINES",
                gpu.kernel_name(i)
            );
        }
        let (decoder, codec_embedding, lm_head, projection) = Self::synthetic_weights(&cfg, seed);
        MtpModel::build(gpu, cfg, decoder, codec_embedding, lm_head, projection, true)
    }
}

// ---------------------------------------------------------------------------
// Training: forward loss + backward. Only a `new_trainable` build has any of
// this; every entry point below panics with the same message on an inference
// one rather than silently returning zeros.
// ---------------------------------------------------------------------------

impl MtpModel {
    /// Full `KernelIds` - forward slots plus the backward half [`TRAIN_PIPELINES`]
    /// adds. Deliberately distinct from [`Self::only_fwd_ids`], which the
    /// forward and decode tapes keep using: those two must stay reachable from
    /// an inference-only handle, and their `block::UNREGISTERED` backward slots
    /// are what makes an accidental backward dispatch a panic there.
    fn train_ids() -> KernelIds {
        KernelIds {
            rmsnorm: RMSNORM,
            rms_inv: RMS_INV,
            rmsnorm_dx: RMSNORM_DX,
            rmsnorm_dx_rows: RMSNORM_DX_ROWS,
            rmsnorm_dw: RMSNORM_DW,
            rope: ROPE,
            rope_bwd: ROPE_BWD,
            gqa_scores: GQA_SCORES,
            gqa_apply: GQA_APPLY,
            attn_softmax: ATTN_SOFTMAX,
            gqa_dscores: GQA_DSCORES,
            gqa_dv: GQA_DV,
            gqa_dq: GQA_DQ,
            gqa_dk: GQA_DK,
            silu_mul: SILU_MUL,
            silu_da: SILU_DA,
            silu_db: SILU_DB,
            rmsnorm_rows: RMSNORM_ROWS,
        }
    }

    /// `dW[n,k] += dYᵀ·X` for a linear `out[m,n] = x[m,k] · W[n,k]ᵀ`.
    fn dw_step(&self, d_out: &DeviceBuffer, x: &DeviceBuffer, gw: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        self.gpu.step(MATMUL_DW, &[d_out, x, gw], &[m, k, n], n * k)
    }

    /// `dX[m,k] = dY·W` for the same linear; `acc = 1` adds instead of
    /// overwriting, which is how the three q/k/v projections fold their input
    /// gradients onto one `d_xn`.
    #[allow(clippy::too_many_arguments)]
    fn dx_step(&self, d_out: &DeviceBuffer, w: &DeviceBuffer, dx: &DeviceBuffer, m: u32, k: u32, n: u32, acc: u32) -> Step {
        self.gpu.step(MATMUL_DX, &[d_out, w, dx], &[m, k, n, acc], m * k)
    }

    fn build_train(&self) -> Train {
        let c = &self.cfg;
        let n = self.t as u64;
        let d = c.d_model as u64;
        let ff = c.d_ff as u64;
        let hq = c.q_dim() as u64;
        let hkv = c.kv_dim() as u64;
        let st = |x: u64| self.gpu.storage(x);
        let nres = c.n_residual() as usize;
        let (dm, em, v) = (c.d_model as usize, c.embedding_dim as usize, c.vocab as usize);
        let mut tr = Train {
            bwd_steps: Vec::new(),
            d_res: (0..=c.n_layers).map(|_| st(n * d)).collect(),
            d_hidden: st(n * d),
            // Sized for the WIDEST row count any `rmsnorm_dw` on the tape asks
            // for: the per-head q-norm, at `num_code_groups * n_heads` rows.
            inv: st(n * c.n_heads as u64),
            d_xn: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_ctx: st(n * hq),
            d_scores: st((c.n_heads * self.t * self.t) as u64),
            d_q: st(n * hq),
            d_k: st(n * hkv),
            d_v: st(n * hkv),
            dq_pre: st(n * hq),
            dk_pre: st(n * hkv),
            d_h: st(n * ff),
            d_gate_pre: st(n * ff),
            d_up: st(n * ff),
            g_codec_embedding: (0..nres).map(|_| vec![0.0f32; v * em]).collect(),
            g_lm_head: (0..nres).map(|_| vec![0.0f32; v * dm]).collect(),
            g_projection: self
                .small_to_mtp_projection
                .as_ref()
                .map(|(w, b)| (vec![0.0f32; w.len()], vec![0.0f32; b.len()])),
            batch: None,
            fwd: None,
        };
        tr.bwd_steps = self.backward_steps(&tr);
        tr
    }

    /// The decoder half's backward tape: the exact adjoint of
    /// [`Self::forward_steps`], read bottom-up, composed from the same shared
    /// `model::block` builders the forward uses. Input is `d_hidden` (the
    /// gradient the host-side output heads produce); output is `d_res[0]`, the
    /// gradient w.r.t. the assembled input-embedding sequence.
    fn backward_steps(&self, tr: &Train) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.t;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let theta = c.rope_theta;
        let ids = Self::train_ids();
        let ga = Gqa { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
        let g = &self.gpu;
        let w = |name: &str| self.ps.w(name);
        let gw = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::new();

        let last = c.n_layers as usize;
        s.extend(block::rmsnorm_bwd(g, &ids, &self.res[last], w("norm.weight"), &tr.d_hidden, &tr.d_res[last], &tr.inv, Some(gw("norm.weight")), d, n));

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ---- SwiGLU MLP backward (incoming grad = d_res[l+1]) ----
            s.push(self.dw_step(&tr.d_res[l + 1], &lb.h, gw(&p("mlp.down.weight")), n, ff, d));
            s.push(self.dx_step(&tr.d_res[l + 1], w(&p("mlp.down.weight")), &tr.d_h, n, ff, d, 0));
            s.extend(block::swiglu_bwd(g, &ids, &lb.gate_pre, &lb.up, &tr.d_h, &tr.d_gate_pre, &tr.d_up, n * ff));
            s.push(self.dw_step(&tr.d_up, &lb.xn2, gw(&p("mlp.up.weight")), n, d, ff));
            s.push(self.dx_step(&tr.d_up, w(&p("mlp.up.weight")), &tr.d_xn, n, d, ff, 0));
            s.push(self.dw_step(&tr.d_gate_pre, &lb.xn2, gw(&p("mlp.gate.weight")), n, d, ff));
            s.push(self.dx_step(&tr.d_gate_pre, w(&p("mlp.gate.weight")), &tr.d_xn, n, d, ff, 1));
            s.extend(block::rmsnorm_bwd(g, &ids, &lb.xmid, w(&p("ln2.weight")), &tr.d_xn, &tr.d_tmp, &tr.inv, Some(gw(&p("ln2.weight"))), d, n));
            // The MLP residual: xmid feeds both the norm and the skip.
            s.push(g.step(ADD2, &[&tr.d_res[l + 1], &tr.d_tmp, &tr.dxmid], &[n * d], n * d));

            // ---- attention backward (incoming grad = dxmid) ----
            s.push(self.dw_step(&tr.dxmid, &lb.ctx, gw(&p("attn.wo.weight")), n, hq, d));
            s.push(self.dx_step(&tr.dxmid, w(&p("attn.wo.weight")), &tr.d_ctx, n, hq, d, 0));
            // `lb.q`/`lb.k` hold the ROPED, QK-normed q/k the forward attended
            // with (RoPE is applied in place), which is what `gqa_bwd` wants.
            s.extend(block::gqa_bwd(g, &ids, &ga, &lb.q, &lb.k, &lb.v, &lb.probs, &tr.d_ctx, &tr.d_scores, &tr.d_q, &tr.d_k, &tr.d_v));
            s.push(block::rope_bwd(g, &ids, &tr.d_q, n, nh, hd, hq, n, theta));
            s.push(block::rope_bwd(g, &ids, &tr.d_k, n, nkv, hd, hkv, n, theta));
            s.extend(block::rmsnorm_bwd(g, &ids, &lb.q_pre, w(&p("attn.q_norm.weight")), &tr.d_q, &tr.dq_pre, &tr.inv, Some(gw(&p("attn.q_norm.weight"))), hd, n * nh));
            s.extend(block::rmsnorm_bwd(g, &ids, &lb.k_pre, w(&p("attn.k_norm.weight")), &tr.d_k, &tr.dk_pre, &tr.inv, Some(gw(&p("attn.k_norm.weight"))), hd, n * nkv));
            s.push(self.dw_step(&tr.d_v, &lb.xn1, gw(&p("attn.wv.weight")), n, d, hkv));
            s.push(self.dx_step(&tr.d_v, w(&p("attn.wv.weight")), &tr.d_xn, n, d, hkv, 0));
            s.push(self.dw_step(&tr.dk_pre, &lb.xn1, gw(&p("attn.wk.weight")), n, d, hkv));
            s.push(self.dx_step(&tr.dk_pre, w(&p("attn.wk.weight")), &tr.d_xn, n, d, hkv, 1));
            s.push(self.dw_step(&tr.dq_pre, &lb.xn1, gw(&p("attn.wq.weight")), n, d, hq));
            s.push(self.dx_step(&tr.dq_pre, w(&p("attn.wq.weight")), &tr.d_xn, n, d, hq, 1));
            s.extend(block::rmsnorm_bwd(g, &ids, &self.res[l], w(&p("ln1.weight")), &tr.d_xn, &tr.d_tmp, &tr.inv, Some(gw(&p("ln1.weight"))), d, n));
            s.push(g.step(ADD2, &[&tr.dxmid, &tr.d_tmp, &tr.d_res[l]], &[n * d], n * d));
        }
        s
    }

    fn tr(&self) -> &Train {
        self.train.as_ref().expect("MtpModel: not a new_trainable build")
    }
    fn tr_mut(&mut self) -> &mut Train {
        self.train.as_mut().expect("MtpModel: not a new_trainable build")
    }

    /// Set the fixed one-frame training example [`Self::forward`] scores and
    /// [`Self::backward`] differentiates. `talker_hidden` and `cb0_embed` are
    /// `embedding_dim`-wide (Talker-side rows, projected on the way in);
    /// `targets` are the ground-truth residual codebooks `1..num_code_groups`
    /// of that frame - the same-frame, unshifted alignment
    /// `crate::sft::MultiCodebookLabels` materialises.
    ///
    /// The inputs at positions `2..num_code_groups` are the ground-truth codes
    /// themselves (teacher forcing), so `targets` doubles as the input code
    /// sequence and no target may be `crate::sft::IGNORE`: a masked target
    /// would still have to feed the next position, and there is no meaningful
    /// embedding to feed it.
    pub fn set_frame_batch(&mut self, talker_hidden: &[f32], cb0_embed: &[f32], targets: &[u32]) {
        let e = self.cfg.embedding_dim as usize;
        let nres = self.t as usize - 1;
        assert_eq!(talker_hidden.len(), e, "talker_hidden must be [embedding_dim]");
        assert_eq!(cb0_embed.len(), e, "cb0_embed must be [embedding_dim]");
        assert_eq!(targets.len(), nres, "need one target per residual codebook");
        assert!(targets.iter().all(|&c| c < self.cfg.vocab), "residual target outside the codebook vocab");
        let batch = Batch {
            talker_hidden: talker_hidden.to_vec(),
            cb0_embed: cb0_embed.to_vec(),
            targets: targets.to_vec(),
        };
        let tr = self.tr_mut();
        tr.batch = Some(batch);
        tr.fwd = None;
    }

    /// Every trainable parameter, device and host alike, in one list: the
    /// decoder block set, then the per-residual `codec_embedding` / `lm_head`
    /// tables, then `small_to_mtp_projection` when the config has one.
    pub fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = Self::decoder_param_list(&self.cfg).into_iter().map(|(n, _)| n).collect();
        for i in 0..self.cfg.n_residual() as usize {
            names.push(format!("codec_embedding.{i}.weight"));
            names.push(format!("lm_head.{i}.weight"));
        }
        if self.small_to_mtp_projection.is_some() {
            names.push("small_to_mtp_projection.weight".to_string());
            names.push("small_to_mtp_projection.bias".to_string());
        }
        names
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        match host_param(name) {
            Some(HostParam::Codec(i)) => self.codec_embedding[i].clone(),
            Some(HostParam::Head(i)) => self.lm_head[i].clone(),
            Some(HostParam::ProjWeight) => self.small_to_mtp_projection.as_ref().expect("no projection").0.clone(),
            Some(HostParam::ProjBias) => self.small_to_mtp_projection.as_ref().expect("no projection").1.clone(),
            None => self.ps.read_weight(&self.gpu, name),
        }
    }

    /// `&mut self` because three of the four parameter families are host
    /// `Vec<f32>`s the forward reads directly - the model owns them outright
    /// rather than behind interior mutability, so a caller that needs the
    /// `&self` shape the gradient checker's `CheckModel` asks for wraps the
    /// model in a `RefCell` (see `tests/mtp.rs`).
    pub fn write_weight(&mut self, name: &str, data: &[f32]) {
        match host_param(name) {
            Some(HostParam::Codec(i)) => self.codec_embedding[i].copy_from_slice(data),
            Some(HostParam::Head(i)) => self.lm_head[i].copy_from_slice(data),
            Some(HostParam::ProjWeight) => self.small_to_mtp_projection.as_mut().expect("no projection").0.copy_from_slice(data),
            Some(HostParam::ProjBias) => self.small_to_mtp_projection.as_mut().expect("no projection").1.copy_from_slice(data),
            None => self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data)),
        }
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let tr = self.tr();
        match host_param(name) {
            Some(HostParam::Codec(i)) => tr.g_codec_embedding[i].clone(),
            Some(HostParam::Head(i)) => tr.g_lm_head[i].clone(),
            Some(HostParam::ProjWeight) => tr.g_projection.as_ref().expect("no projection").0.clone(),
            Some(HostParam::ProjBias) => tr.g_projection.as_ref().expect("no projection").1.clone(),
            None => self.ps.read_grad(&self.gpu, name),
        }
    }

    pub fn zero_grads(&mut self) {
        self.ps.zero_grads(&self.gpu);
        let tr = self.tr_mut();
        for g in tr.g_codec_embedding.iter_mut().chain(tr.g_lm_head.iter_mut()) {
            g.fill(0.0);
        }
        if let Some((gw, gb)) = tr.g_projection.as_mut() {
            gw.fill(0.0);
            gb.fill(0.0);
        }
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// Mean cross-entropy of the `num_code_groups - 1` residual codebooks of
    /// the frame set by [`Self::set_frame_batch`].
    ///
    /// This is the PRODUCTION forward, not a training-only twin of it: the
    /// input sequence comes from [`Self::assemble`] (so
    /// `small_to_mtp_projection` is exercised exactly as a served run
    /// exercises it), the decoder is the same `fwd_steps` tape
    /// [`Self::logits`] submits, and each position's logits come from the same
    /// [`Self::head_row`]. That is the whole point of gradient-checking it -
    /// a second, parallel forward would only gate itself.
    pub fn forward(&mut self) -> f32 {
        let d = self.cfg.d_model as usize;
        let e = self.cfg.embedding_dim as usize;
        let v = self.cfg.vocab as usize;
        let t = self.t as usize;
        let (th, cb0, targets) = {
            let b = self.tr().batch.as_ref().expect("MtpModel::forward: call set_frame_batch first");
            (b.talker_hidden.clone(), b.cb0_embed.clone(), b.targets.clone())
        };

        // The unprojected rows the projection backward differentiates against.
        // `assemble` gathers and projects them again on its own; keeping that
        // path untouched is deliberate - it is the served one.
        let mut raw: Vec<Vec<f32>> = Vec::with_capacity(t);
        raw.push(th.clone());
        raw.push(cb0.clone());
        for (i, &code) in targets[..t - 2].iter().enumerate() {
            let s = code as usize * e;
            raw.push(self.codec_embedding[i][s..s + e].to_vec());
        }

        let emb = self.assemble(&th, &cb0, &targets[..t - 2]);
        let hidden = self.hidden(&emb);
        let mut logits = vec![0.0f32; (t - 1) * v];
        for i in 1..t {
            let row = self.head_row(i - 1, &hidden[i * d..(i + 1) * d]);
            logits[(i - 1) * v..i * v].copy_from_slice(&row);
        }
        let (loss, d_logits) = crate::sft::ce_batch(&logits, &targets, v);
        self.tr_mut().fwd = Some(Fwd { raw, hidden, d_logits });
        loss
    }

    /// Accumulate the gradients of the loss [`Self::forward`] returned.
    ///
    /// Three stages, host / device / host: the per-position output heads, then
    /// the decoder tape (which reads the activations the forward left resident
    /// and hands back `d_res[0]`), then `small_to_mtp_projection` and the
    /// codec-embedding scatter.
    pub fn backward(&mut self) {
        let (d, e, v, t) = (
            self.cfg.d_model as usize,
            self.cfg.embedding_dim as usize,
            self.cfg.vocab as usize,
            self.t as usize,
        );
        // Disjoint field borrows: the host parameter tables are read while the
        // training state's gradient tables are written.
        let MtpModel { gpu, lm_head, small_to_mtp_projection, train, .. } = self;
        let tr = train.as_mut().expect("MtpModel: not a new_trainable build");
        let fwd = tr.fwd.take().expect("MtpModel::backward: call forward first");
        let targets = tr.batch.as_ref().expect("no batch").targets.clone();

        // ---- 1. per-position output heads (host) ----
        // `lm_head[i-1]` reads decoder position `i` only, so the head gradients
        // never mix positions and `d_hidden`'s position 0 - the Talker hidden
        // state, which no head reads - stays exactly zero.
        let mut d_hidden = vec![0.0f32; t * d];
        for i in 1..t {
            let dl = &fwd.d_logits[(i - 1) * v..i * v];
            let h = &fwd.hidden[i * d..(i + 1) * d];
            let head = &lm_head[i - 1];
            let ghead = &mut tr.g_lm_head[i - 1];
            let dst = &mut d_hidden[i * d..(i + 1) * d];
            for (o, &go) in dl.iter().enumerate() {
                let wrow = &head[o * d..(o + 1) * d];
                let grow = &mut ghead[o * d..(o + 1) * d];
                for k in 0..d {
                    dst[k] += go * wrow[k];
                    grow[k] += go * h[k];
                }
            }
        }

        // ---- 2. decoder (device) ----
        gpu.write(&tr.d_hidden, bytemuck::cast_slice(&d_hidden));
        gpu.submit(&[], &tr.bwd_steps);
        let d_emb = gpu.read(&tr.d_res[0], t * d);

        // ---- 3. small_to_mtp_projection, then the codec-embedding scatter ----
        // `d_raw[0]`/`d_raw[1]` are the gradients w.r.t. the Talker hidden
        // state and the Talker's own codebook-0 embedding. Neither is an MTP
        // parameter (both are inputs supplied by the Talker), so they are
        // dropped here; a jointly-trained Talker+MTP stack is where they would
        // be the seam this hands back.
        let mut d_raw: Vec<Vec<f32>> = Vec::with_capacity(t);
        match (small_to_mtp_projection.as_ref(), tr.g_projection.as_mut()) {
            (Some((w, _)), Some((gw, gb))) => {
                for (r, x) in fwd.raw.iter().enumerate() {
                    let dy = &d_emb[r * d..(r + 1) * d];
                    let mut dx = vec![0.0f32; e];
                    for (o, &go) in dy.iter().enumerate() {
                        gb[o] += go;
                        let wrow = &w[o * e..(o + 1) * e];
                        let grow = &mut gw[o * e..(o + 1) * e];
                        for j in 0..e {
                            grow[j] += go * x[j];
                            dx[j] += go * wrow[j];
                        }
                    }
                    d_raw.push(dx);
                }
            }
            // Identity projection (the 0.6B family): `embedding_dim == d_model`
            // and `assemble` copied the row straight through.
            _ => {
                for r in 0..t {
                    d_raw.push(d_emb[r * d..(r + 1) * d].to_vec());
                }
            }
        }
        for (i, &code) in targets[..t - 2].iter().enumerate() {
            let s = code as usize * e;
            let dst = &mut tr.g_codec_embedding[i][s..s + e];
            for (a, b) in dst.iter_mut().zip(&d_raw[2 + i]) {
                *a += b;
            }
        }
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

    /// Every forward-slot const in this module indexes BOTH tables, and the
    /// trainable model replays the very forward tape an inference one records,
    /// so `TRAIN_PIPELINES` must begin with `PIPELINES` verbatim and in order.
    /// Appending a forward kernel to one table and not the other would
    /// otherwise shift the backward slots under the training tape and dispatch
    /// whichever kernel happened to land there.
    #[test]
    fn train_pipelines_extends_the_inference_table() {
        assert_eq!(&TRAIN_PIPELINES[..PIPELINES.len()], PIPELINES);
        assert_eq!(TRAIN_PIPELINES.len(), MATMUL_DW + 1);
        for (slot, name) in [
            (RMSNORM_DX, "rmsnorm_dx"),
            (RMSNORM_DX_ROWS, "rmsnorm_dx_rows"),
            (RMSNORM_DW, "rmsnorm_dw"),
            (ROPE_BWD, "rope_base_bwd"),
            (GQA_DSCORES, "gqa_bwd_dscores"),
            (GQA_DV, "gqa_bwd_dv"),
            (GQA_DQ, "gqa_bwd_dq"),
            (GQA_DK, "gqa_bwd_dk"),
            (SILU_DA, "silu_bwd_da"),
            (SILU_DB, "silu_bwd_db"),
            (MATMUL_DX, "matmul_dx"),
            (MATMUL_DW, "matmul_dw"),
        ] {
            assert_eq!(TRAIN_PIPELINES[slot].0, name, "backward slot {slot}");
        }
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

        let (codes, res_sum) =
            m.generate_residuals_with(&th, &cb0, &crate::sampling::SamplerCfg::greedy(), &mut data::rng::Rng::new(1));
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
                m.generate_residuals_recompute(&th, &cb0, &crate::sampling::SamplerCfg::greedy(), &mut data::rng::Rng::new(0));
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
