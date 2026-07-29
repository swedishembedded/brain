// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder — bidirectional hybrid short-conv/attention encoder as WGSL
//! compute dispatches, sharing the engine (`gpu_core`, `paramstore`, `kernels`)
//! and the composable Step-builders (`model::block`, `audio::conv`).
//!
//! Per pre-norm layer (no biases anywhere), mixer chosen by `layer_types`:
//!   h = RMSNorm(x)·ln1                                  (eps from checkpoint)
//!   conv layer:  B,C,X = h·Win (row-thirds) ; y = Wout·(C ⊙ conv1d_dw(B ⊙ X))
//!                (depthwise k=3, symmetric pad — encoder variant)
//!   attn layer:  q,k,v = h·Wq,Wk,Wv ; q,k = RoPE(QKNorm(·))
//!                qkv = [q | expand(k) | expand(v)]      (GQA→MHA, group 2)
//!                y = Wo·bidir-attention(qkv)            (non-causal, all j)
//!   x += y ;  h = RMSNorm(x)·ln2 ;  x += Wdown·(SiLU(Wgate·h) ⊙ (Wup·h))
//!   hidden = RMSNorm(x)·norm ;  logits = tok.weightᵀ·hidden   (tied MLM head)
//!   loss = masked cross-entropy (ignore_index = IGNORE), labels UNshifted.
//!
//! Three regimes behind one layer loop:
//! - **Materialized** (parity gates, short-T training): the bidir trio with
//!   per-layer caches and full `[B,H,T,T]` scores — exact, memory ∝ T².
//! - **Chunked inference**: `block::chunked_bidir_fwd` over the same fused qkv
//!   with a bounded `[H, chunk, T]` slab, shared layer scratch (no caches),
//!   and the MLM head evaluated only at gathered probe rows.
//! - **Chunked training** (`new_train_chunked`): materialized caches EXCEPT
//!   the T×T slabs — attention backward recomputes each chunk's scores/probs
//!   (`block::chunked_bidir_bwd`, accumulating dk/dv) and CE runs on the
//!   gathered supervised rows only — together they fit an 8k training step in
//!   the per-binding budget. Gated bit-tight by `tests/chunked_train_equiv.rs`.

use std::cell::Cell;
use std::collections::HashMap;

use audio::conv::{Conv1d, ConvKernels};
use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{self, Bidir, BidirIds, CrossIds};
use paramstore::ParamStore;

use crate::config::{LayerType, LfmConfig};

/// Cross-entropy ignore index (masked/unsupervised positions).
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED_TILE: usize = 0;
const MATMUL: usize = 1;
const MATMUL_REG2: usize = 2;
const MATMUL_TILE: usize = 3;
const RMSNORM_EPS: usize = 4;
const ROPE: usize = 5;
const ROPE_BWD: usize = 6;
const KV_EXPAND: usize = 7;
const KV_EXPAND_BWD: usize = 8;
const SCORES_BIDIR: usize = 9;
const SOFTMAX_BIDIR: usize = 10;
const APPLY_BIDIR: usize = 11;
const DSCORES_BIDIR: usize = 12;
const DV_BIDIR: usize = 13;
const DQ_BIDIR: usize = 14;
const DK_BIDIR: usize = 15;
const MUL: usize = 16;
const CONV1D: usize = 17;
const CONV1D_DX: usize = 18;
const CONV1D_DW: usize = 19;
const NLC_NCL: usize = 20;
const NCL_NLC: usize = 21;
const ADD2: usize = 22;
const SILU_MUL: usize = 23;
const SILU_DA: usize = 24;
const SILU_DB: usize = 25;
const CE_VALUE: usize = 26;
const SCORES_CROSS: usize = 27;
const SOFTMAX_CROSS: usize = 28;
const APPLY_CROSS: usize = 29;
/// Plain embedding gather — reused as a generic row-gather (indices → rows).
const EMBED: usize = 30;
// Backward / optimizer set (training).
const CE_STATS: usize = 31;
const CE_GRAD_STATS: usize = 32;
const MATMUL_DX: usize = 33;
const MATMUL_DW: usize = 34;
const MATMUL_DX_REG: usize = 35;
const MATMUL_DW_REG: usize = 36;
const EMB_BWD: usize = 37;
const ADAMW: usize = 38;
const GRADNORM_SQ: usize = 39;
const GRAD_SCALE: usize = 40;
const CLIP_COEF: usize = 41;
const GRAD_SCALE_BUF: usize = 42;
const RMSNORM_EPS_INV: usize = 43;
const RMSNORM_DW: usize = 44;
const RMSNORM_EPS_DX: usize = 45;
// Chunked-training set (8k backward + gathered MLM head).
const DSCORES_CROSS: usize = 46;
const DQ_CROSS: usize = 47;
const DK_CROSS_ACC: usize = 48;
const DV_CROSS_ACC: usize = 49;
const ROW_SCATTER: usize = 50;
const FLASH_BIDIR: usize = 51;
const HEAD_PACK: usize = 52;
const HEAD_PACK_T: usize = 53;
const HEAD_UNPACK: usize = 54;
const SOFTMAX_ROWS: usize = 55;

const PIPELINES: &[(&str, &str)] = &[
    ("embed_tile", kernels::EMBED_TILE),
    ("matmul", kernels::MATMUL),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("matmul_tile", kernels::MATMUL_TILE),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("kv_expand", kernels::KV_EXPAND),
    ("kv_expand_bwd", kernels::KV_EXPAND_BWD),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("mul", kernels::MUL),
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("add2", kernels::ADD2),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("embed", kernels::EMBED),
    ("ce_stats", kernels::CE_STATS),
    ("ce_grad_stats", kernels::CE_GRAD_STATS),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("rms_inv_eps", kernels::RMS_INV_EPS),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rmsnorm_dx_eps", kernels::RMSNORM_DX_EPS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
    ("row_scatter", kernels::ROW_SCATTER),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("head_pack", kernels::HEAD_PACK),
    ("head_pack_t", kernels::HEAD_PACK_T),
    ("head_unpack", kernels::HEAD_UNPACK),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
];

fn linear_kernel(m: usize, n: usize) -> (usize, u32) {
    let naive = std::env::var("BRAIN_LFM_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    block::pick_gemm(m, n, MATMUL, MATMUL_REG2, naive)
}
fn dx_kernel(m: u32, k: u32) -> (usize, u32) {
    let naive = std::env::var("BRAIN_LFM_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    block::pick_gemm(m as usize, k as usize, MATMUL_DX, MATMUL_DX_REG, naive)
}
fn dw_kernel(nrows: u32, k: u32) -> (usize, u32) {
    let naive = std::env::var("BRAIN_LFM_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    block::pick_gemm(nrows as usize, k as usize, MATMUL_DW, MATMUL_DW_REG, naive)
}

/// Attention-mixer activation buffers.
struct AttnBufs {
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    /// Fused `[q | k_exp | v_exp]`, `[n, 3*d]` — what attention consumes.
    qkv: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
}

/// Conv-mixer activation buffers.
struct ConvBufs {
    bg: DeviceBuffer,
    cg: DeviceBuffer,
    xg: DeviceBuffer,
    bx: DeviceBuffer,
    bx_ncl: DeviceBuffer,
    conv_ncl: DeviceBuffer,
    conv_out: DeviceBuffer,
    gated: DeviceBuffer,
}

/// Per-layer common activations (norm outputs, residual mid, FFN).
struct CommonBufs {
    xn1: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

enum MixerBufs {
    Attn(AttnBufs),
    Conv(ConvBufs),
}

struct LayerBufs {
    common: CommonBufs,
    mixer: MixerBufs,
}

/// How the layer loop binds buffers + emits attention.
enum Regime {
    /// Per-layer caches, full-score bidir trio (parity + training).
    Materialized { layers: Vec<LayerBufs>, res: Vec<DeviceBuffer>, scores: DeviceBuffer },
    /// One shared scratch set, ping-pong residuals, query-chunked attention,
    /// head evaluated only at `probe_cap` gathered rows.
    Chunked {
        common: CommonBufs,
        attn: AttnBufs,
        conv: ConvBufs,
        res: [DeviceBuffer; 2],
        scores: DeviceBuffer,
        probs: DeviceBuffer,
        chunk: u32,
        /// Per-head context accumulator for GEMM attention (`[H, t, hd]`).
        ctx_pack: DeviceBuffer,
        probe_cap: u32,
        probe_idx: DeviceBuffer,
        probe_h: DeviceBuffer,
        probe_logits: DeviceBuffer,
        n_probes: Cell<u32>,
    },
}

/// Gathered MLM head: CE evaluated only at the supervised rows (≤ cap).
struct HeadGather {
    cap: u32,
    n_sup: Cell<u32>,
    /// Gather indices (pad slots point at row 0 — in-range, harmless: their
    /// targets are IGNORE).
    sup_idx: DeviceBuffer,
    /// Scatter indices (pad slots are u32::MAX — skipped by `row_scatter`).
    sup_idx_scatter: DeviceBuffer,
    sup_targets: DeviceBuffer,
    probe_h: DeviceBuffer,
    logits_g: DeviceBuffer,
    ce_g: DeviceBuffer,
    ce_stats_g: DeviceBuffer,
    d_logits_g: DeviceBuffer,
    d_probe_h: DeviceBuffer,
}

/// Backward scratch (materialized regime only — training asserts it).
struct BwdBufs {
    dres: Vec<DeviceBuffer>,
    d_logits: DeviceBuffer,
    ce_stats: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    dq_pre: DeviceBuffer,
    dk_pre: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    inv: DeviceBuffer,
    /// Conv-mixer grad scratch, reused in strict consumption order (see the
    /// conv backward emission for the aliasing discipline).
    d_mix1: DeviceBuffer,
    d_mix2: DeviceBuffer,
    d_mix3: DeviceBuffer,
    d_mix4: DeviceBuffer,
}

pub struct Lfm {
    pub gpu: Gpu,
    pub cfg: LfmConfig,
    pub ps: ParamStore,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    regime: Regime,
    opt: Option<optim::Optim>,
    bwd: Option<BwdBufs>,
    /// Chunked-training attention (fwd + per-chunk-recompute bwd): the query
    /// chunk; None = full materialized scores (small-T training / parity).
    train_chunk: Option<u32>,
    /// Gathered supervised-row MLM head (8k training: full logits exceed the
    /// binding budget). None = full [n, vocab] head.
    head: Option<HeadGather>,
    ce_grad_uni: DeviceBuffer,
    bwd_steps: Vec<Step>,
    /// Chunked-training attention slabs (`[heads, chunk, t]`; size-1 otherwise).
    slab_scores: DeviceBuffer,
    slab_probs: DeviceBuffer,
    slab_dscores: DeviceBuffer,
    /// Mixer-projection / FFN-down outputs (consumed immediately by their
    /// residual adds — shared across layers, never an activation cache).
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn_final: DeviceBuffer,
    /// Full `[n, vocab]` logits + CE (materialized regime only; size-1 otherwise).
    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,

    fwd_steps: Vec<Step>,
    /// Chunked-inference regime: one forward per group size 1..=b (index
    /// `b_use-1`), so partial scheduler groups run at their true size.
    fwd_variants: Vec<Vec<Step>>,
}

impl Lfm {
    /// Load an inference-only model (frozen weights, materialized attention —
    /// intended for short contexts / parity work).
    pub fn load_inference(path: &str, b: u32, t: u32) -> Lfm {
        let (cfg, init) = Self::load_ckpt(path);
        Lfm::new_impl(cfg, b, t, &init, None, false)
    }

    /// Load an inference-only model on the chunked long-context path: attention
    /// scratch bounded by `slab_budget_bytes`, MLM head evaluated at up to
    /// `probe_cap` gathered rows (0 = hidden-states only).
    pub fn load_inference_chunked(path: &str, b: u32, t: u32, slab_budget_bytes: u64, probe_cap: u32) -> Lfm {
        let (cfg, init) = Self::load_ckpt(path);
        Lfm::new_chunked(cfg, b, t, &init, slab_budget_bytes, probe_cap)
    }

    fn load_ckpt(path: &str) -> (LfmConfig, HashMap<String, Vec<f32>>) {
        let c = checkpoint::load(path);
        let cfg = LfmConfig::from_json(&c.header["config"]);
        (cfg, c.by_role(""))
    }

    /// Materialized-attention model, frozen weights (parity / short-context
    /// inference). For training use [`Lfm::new_train`].
    pub fn new(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Lfm {
        Lfm::new_impl(cfg, b, t, init, None, false)
    }

    /// Trainable model (weights + grads + AdamW moments, forward + backward
    /// graphs). Materialized attention — training seq is bounded by the T²
    /// score memory; [`Lfm::new_train_chunked`] lifts that for long context.
    pub fn new_train(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Lfm {
        Lfm::new_impl(cfg, b, t, init, None, true)
    }

    /// Long-context trainable model: query-chunked attention (fwd + per-chunk
    /// recompute bwd, slab bounded by `slab_budget_bytes`) and the MLM head
    /// evaluated only at the ≤ `head_cap` supervised rows — together these keep
    /// an 8k training step inside the per-binding budget.
    pub fn new_train_chunked(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, slab_budget_bytes: u64, head_cap: u32) -> Lfm {
        let mut m = Lfm::new_impl_chunkedtrain(cfg, b, t, init, slab_budget_bytes, head_cap);
        m.fwd_steps = m.forward_steps();
        m.bwd_steps = m.backward_steps();
        m
    }

    /// Load a trainable model from a checkpoint (fine-tuning).
    pub fn load_train(path: &str, b: u32, t: u32) -> Lfm {
        let (cfg, init) = Self::load_ckpt(path);
        Lfm::new_impl(cfg, b, t, &init, None, true)
    }

    /// Load a long-context trainable model (see [`Lfm::new_train_chunked`]).
    pub fn load_train_chunked(path: &str, b: u32, t: u32, slab_budget_bytes: u64, head_cap: u32) -> Lfm {
        let (cfg, init) = Self::load_ckpt(path);
        Lfm::new_train_chunked(cfg, b, t, &init, slab_budget_bytes, head_cap)
    }

    /// Chunked-attention model from an in-memory init map.
    pub fn new_chunked(
        cfg: LfmConfig,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
        slab_budget_bytes: u64,
        probe_cap: u32,
    ) -> Lfm {
        Lfm::new_impl(cfg, b, t, init, Some((slab_budget_bytes, probe_cap)), false)
    }

    /// Allocation for the long-context training regime: materialized per-layer
    /// caches EXCEPT the T×T score/prob slabs (replaced by `[H, chunk, T]`
    /// shared slabs + per-chunk recompute) and the full-vocab head (replaced by
    /// the gathered supervised-row head). Graphs are built by the caller.
    fn new_impl_chunkedtrain(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, slab_budget_bytes: u64, head_cap: u32) -> Lfm {
        let mut m = Lfm::new_impl_alloc(cfg, b, t, init, true, /*tiny_probs=*/true);
        let heads = m.cfg.n_heads as u64;
        let per_row = heads * t as u64 * 4;
        let chunk = ((slab_budget_bytes / per_row.max(1)) as u32).clamp(64, 4096).min(t);
        let slab = heads * chunk as u64 * t as u64;
        let d = m.cfg.d_model as u64;
        let v = m.cfg.vocab as u64;
        let cap = head_cap.max(1) as u64;
        m.train_chunk = Some(chunk);
        m.slab_scores = m.gpu.storage(slab);
        m.slab_probs = m.gpu.storage(slab);
        m.slab_dscores = m.gpu.storage(slab);
        m.head = Some(HeadGather {
            cap: head_cap.max(1),
            n_sup: Cell::new(0),
            sup_idx: m.gpu.buffer("sup_idx", cap * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
            sup_idx_scatter: m.gpu.buffer("sup_idx_scatter", cap * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
            sup_targets: m.gpu.buffer("sup_targets", cap * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
            probe_h: m.gpu.storage(cap * d),
            logits_g: m.gpu.storage(cap * v),
            ce_g: m.gpu.storage(cap),
            ce_stats_g: m.gpu.storage(cap * 2),
            d_logits_g: m.gpu.storage(cap * v),
            d_probe_h: m.gpu.storage(cap * d),
        });
        m
    }

    fn new_impl(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, chunked: Option<(u64, u32)>, train: bool) -> Lfm {
        assert!(!(train && chunked.is_some()), "chunked-INFERENCE regime is frozen; long-context training uses new_train_chunked");
        let mut m = Lfm::new_impl_alloc_inner(cfg, b, t, init, chunked, train, false);
        m.fwd_steps = m.forward_steps();
        if matches!(m.regime, Regime::Chunked { .. }) && b > 1 {
            m.fwd_variants = (1..=b).map(|bu| m.forward_steps_for(bu)).collect();
        }
        if train {
            m.bwd_steps = m.backward_steps();
        }
        m
    }

    /// Materialized-regime allocation only (no graphs); `tiny_probs` shrinks the
    /// per-layer T×T prob caches to size 1 (the chunked-training slabs replace
    /// them).
    fn new_impl_alloc(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, train: bool, tiny_probs: bool) -> Lfm {
        Lfm::new_impl_alloc_inner(cfg, b, t, init, None, train, tiny_probs)
    }

    fn new_impl_alloc_inner(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, chunked: Option<(u64, u32)>, train: bool, tiny_probs: bool) -> Lfm {
        let gpu = Gpu::new(PIPELINES);
        let role = if train { paramstore::Role::Trainable } else { paramstore::Role::Frozen };
        let roles = cfg
            .param_list()
            .into_iter()
            .map(|(n, c)| (n, c, role))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);
        let opt = train.then(|| optim::Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF));

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let st = |x: u64| gpu.storage(x);

        let tokens = gpu.buffer("tokens", n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        let targets = gpu.buffer("targets", n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);

        let common = |st: &dyn Fn(u64) -> DeviceBuffer| CommonBufs {
            xn1: st(n * d),
            xmid: st(n * d),
            xn2: st(n * d),
            gate_pre: st(n * ff),
            up: st(n * ff),
            h: st(n * ff),
        };
        let attn_bufs = |st: &dyn Fn(u64) -> DeviceBuffer, probs: u64| AttnBufs {
            q_pre: st(n * hq),
            q: st(n * hq),
            k_pre: st(n * hkv),
            k: st(n * hkv),
            v: st(n * hkv),
            qkv: st(n * 3 * d),
            probs: st(probs),
            ctx: st(n * hq),
        };
        let conv_bufs = |st: &dyn Fn(u64) -> DeviceBuffer| ConvBufs {
            bg: st(n * d),
            cg: st(n * d),
            xg: st(n * d),
            bx: st(n * d),
            bx_ncl: st(n * d),
            conv_ncl: st(n * d),
            conv_out: st(n * d),
            gated: st(n * d),
        };

        let (regime, logits, ce_buf) = match chunked {
            None => {
                // tiny_probs (chunked training): the T×T caches shrink to 1 —
                // scores/probs live in the shared [H, chunk, T] slabs instead.
                let bht2 = if tiny_probs {
                    1
                } else {
                    let bht2 = (b as u64) * cfg.n_heads as u64 * (t as u64) * (t as u64);
                    assert!(bht2 <= u32::MAX as u64, "materialized attention exceeds u32 index space (b={b}, t={t}); use the chunked path");
                    bht2
                };
                let mut res = Vec::new();
                for _ in 0..=cfg.n_layers() {
                    res.push(st(n * d));
                }
                let mut layers = Vec::new();
                for ty in &cfg.layer_types {
                    let mixer = match ty {
                        LayerType::Attention => MixerBufs::Attn(attn_bufs(&st, bht2)),
                        LayerType::Conv => MixerBufs::Conv(conv_bufs(&st)),
                    };
                    layers.push(LayerBufs { common: common(&st), mixer });
                }
                // Full logits/CE only when the gathered head is NOT in play.
                let (lg, ce) = if tiny_probs { (st(1), st(1)) } else { (st(n * v), st(n)) };
                (Regime::Materialized { layers, res, scores: st(bht2) }, lg, ce)
            }
            Some((budget, probe_cap)) => {
                let heads = cfg.n_heads as u64;
                let per_row = heads * t as u64 * 4;
                let chunk = ((budget / per_row.max(1)) as u32).clamp(64, 4096).min(t);
                let slab = heads * chunk as u64 * t as u64;
                let cap = probe_cap.max(1) as u64;
                (
                    Regime::Chunked {
                        common: common(&st),
                        attn: attn_bufs(&st, 1),
                        conv: conv_bufs(&st),
                        res: [st(n * d), st(n * d)],
                        scores: st(slab),
                        probs: st(slab),
                        chunk,
                        ctx_pack: st(n * hq),
                        probe_cap,
                        probe_idx: gpu.buffer("probe_idx", cap * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
                        probe_h: st(cap * d),
                        probe_logits: st(cap * v),
                        n_probes: Cell::new(0),
                    },
                    st(1),
                    st(1),
                )
            }
        };

        // Backward scratch (training only; size-0 allocations otherwise).
        let bwd = train.then(|| {
            let bht2 = if tiny_probs { 1 } else { (b as u64) * cfg.n_heads as u64 * (t as u64) * (t as u64) };
            let dlog = if tiny_probs { 1 } else { n * v };
            let mut dres = Vec::new();
            for _ in 0..=cfg.n_layers() {
                dres.push(st(n * d));
            }
            BwdBufs {
                dres,
                d_logits: st(dlog),
                ce_stats: st(if tiny_probs { 1 } else { n * 2 }),
                d_xn: st(n * d),
                d_tmp: st(n * d),
                dxmid: st(n * d),
                d_ctx: st(n * hq),
                d_scores: st(bht2),
                d_qkv: st(n * 3 * d),
                d_q: st(n * hq),
                d_k: st(n * hkv),
                d_v: st(n * hkv),
                dq_pre: st(n * hq),
                dk_pre: st(n * hkv),
                d_h: st(n * ff),
                d_gate_pre: st(n * ff),
                d_up: st(n * ff),
                inv: st(n * cfg.n_heads as u64),
                d_mix1: st(n * d),
                d_mix2: st(n * d),
                d_mix3: st(n * d),
                d_mix4: st(n * d),
            }
        });

        let mut m = Lfm {
            cfg,
            b,
            t,
            count: Cell::new(1.0),
            ps,
            tokens,
            targets,
            regime,
            opt,
            bwd,
            train_chunk: None,
            head: None,
            ce_grad_uni: gpu.uniform_dynamic(4),
            bwd_steps: Vec::new(),
            slab_scores: st(1),
            slab_probs: st(1),
            slab_dscores: st(1),
            proj: st(n * d),
            mlp_out: st(n * d),
            xn_final: st(n * d),
            logits,
            ce_buf,
            fwd_steps: Vec::new(),
            fwd_variants: Vec::new(),
            gpu,
        };
        m
    }

    /// Set the input tokens and (unshifted) targets; `IGNORE` masks a position
    /// out of the loss. Encoder-only inference passes all-`IGNORE` targets.
    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        debug_assert!(x.len() <= (self.b * self.t) as usize);
        self.gpu.write(&self.tokens, x);
        self.gpu.write(&self.targets, y);
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
        if let Some(hg) = &self.head {
            // Gathered head: the supervised rows + their targets, padded to cap
            // with (row 0, IGNORE) so padded slots never enter loss or grads.
            let mut rows: Vec<u32> = Vec::new();
            let mut tgts: Vec<u32> = Vec::new();
            for (i, &t) in y.iter().enumerate() {
                if t != IGNORE {
                    rows.push(i as u32);
                    tgts.push(t);
                }
            }
            assert!(
                rows.len() as u32 <= hg.cap,
                "{} supervised rows > head cap {} (raise head_cap or lower mask_prob)",
                rows.len(),
                hg.cap
            );
            hg.n_sup.set(rows.len() as u32);
            let mut scatter = rows.clone();
            scatter.resize(hg.cap as usize, u32::MAX); // sentinel: pad slots scatter nowhere
            rows.resize(hg.cap as usize, 0); // in-range for the forward gather
            tgts.resize(hg.cap as usize, IGNORE);
            self.gpu.write(&hg.sup_idx, &rows);
            self.gpu.write(&hg.sup_idx_scatter, &scatter);
            self.gpu.write(&hg.sup_targets, &tgts);
        }
    }

    /// Inference convenience: tokens only, loss disabled.
    pub fn set_tokens(&self, x: &[u32]) {
        let y = vec![IGNORE; x.len()];
        self.set_batch(x, &y);
    }

    /// Chunked regime: set the absolute row indices (`b*t` space) whose MLM
    /// logits the next forward should produce (≤ `probe_cap`; e.g. the
    /// `<|mask|>` positions of a fill-mask request).
    pub fn set_probe_rows(&self, rows: &[u32]) {
        let Regime::Chunked { probe_idx, probe_cap, n_probes, .. } = &self.regime else {
            panic!("set_probe_rows: not a chunked-regime model");
        };
        assert!(rows.len() as u32 <= *probe_cap, "{} probe rows > cap {probe_cap}", rows.len());
        // Pad with row 0 so the fixed-size gather stays in range.
        let mut idx = rows.to_vec();
        idx.resize(*probe_cap as usize, 0);
        self.gpu.write(probe_idx, &idx);
        n_probes.set(rows.len() as u32);
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn bidir_ids() -> BidirIds {
        BidirIds {
            scores: SCORES_BIDIR,
            softmax: SOFTMAX_BIDIR,
            apply: APPLY_BIDIR,
            dscores: DSCORES_BIDIR,
            dv: DV_BIDIR,
            dq: DQ_BIDIR,
            dk: DK_BIDIR,
        }
    }

    /// Attention shape over the fused expanded buffer (MHA after kv_expand).
    fn bidir(&self) -> Bidir {
        let d = self.cfg.d_model;
        Bidir {
            b: self.b,
            t: self.t,
            n_heads: self.cfg.n_heads,
            head_dim: self.cfg.head_dim,
            stride: 3 * d,
            q_off: 0,
            k_off: d,
            v_off: 2 * d,
        }
    }

    /// The depthwise symmetric-pad conv shape (`[b, d, t] -> [b, d, t]`).
    fn conv_shape_b(&self, b_use: u32) -> Conv1d {
        let d = self.cfg.d_model;
        let k = self.cfg.conv_k;
        let pad = k / 2;
        Conv1d {
            n: b_use,
            cin: d,
            l: self.t,
            cout: d,
            k,
            stride: 1,
            pad,
            dilation: 1,
            groups: d,
            lo: Conv1d::out_len(self.t, k, 1, pad, pad, 1),
        }
    }

    fn conv_kernels() -> ConvKernels {
        ConvKernels { fwd: CONV1D, dx: CONV1D_DX, dw: CONV1D_DW }
    }

    /// A linear over a row range `[r0, r0+nout)` of a taller `[rows, k]` weight
    /// (row-thirds of the conv mixer's fused `in_proj`).
    fn sliced_linear(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, r0: u32, nout: u32, k: u32, out: &DeviceBuffer, m: u32) {
        let (mk, mt) = linear_kernel(m as usize, nout as usize);
        s.push(self.gpu.step_sliced(
            mk,
            &[x, self.w(wname), out],
            &[(0, 0), (r0 as u64 * k as u64, nout as u64 * k as u64), (0, 0)],
            &[m, k, nout],
            mt,
        ));
    }

    /// Attention-mixer projections + QK-norm + RoPE + GQA→MHA expansion into
    /// the fused qkv (everything before the score/apply kernels).
    fn emit_attn_qkv(&self, s: &mut Vec<Step>, l: usize, xn1: &DeviceBuffer, ab: &AttnBufs, b_use: u32, build_fused: bool) {
        let c = &self.cfg;
        let n = b_use * self.t;
        let (d, hd, hq, hkv) = (c.d_model, c.head_dim, c.q_dim(), c.kv_dim());
        let (nh, nkv, group) = (c.n_heads, c.n_kv_heads, c.group());
        let (eps, theta) = (c.norm_eps, c.rope_theta);
        let p = |name: &str| format!("blocks.{l}.{name}");

        let (mk, mt) = linear_kernel(n as usize, hq as usize);
        s.push(self.gpu.step(mk, &[xn1, self.w(&p("attn.wq.weight")), &ab.q_pre], &[n, d, hq], mt));
        let (mk, mt) = linear_kernel(n as usize, hkv as usize);
        s.push(self.gpu.step(mk, &[xn1, self.w(&p("attn.wk.weight")), &ab.k_pre], &[n, d, hkv], mt));
        let (mk, mt) = linear_kernel(n as usize, hkv as usize);
        s.push(self.gpu.step(mk, &[xn1, self.w(&p("attn.wv.weight")), &ab.v], &[n, d, hkv], mt));
        // Per-head QK-RMSNorm (head_dim rows), then RoPE in place.
        s.push(block::rmsnorm_eps_fwd(&self.gpu, RMSNORM_EPS, &ab.q_pre, self.w(&p("attn.q_norm.weight")), &ab.q, hd, n * nh, eps));
        s.push(block::rmsnorm_eps_fwd(&self.gpu, RMSNORM_EPS, &ab.k_pre, self.w(&p("attn.k_norm.weight")), &ab.k, hd, n * nkv, eps));
        s.push(self.gpu.step(ROPE, &[&ab.q], &[n, nh, hd, hq, 0, self.t, gpu_core::f(theta)], n * nh * (hd / 2)));
        s.push(self.gpu.step(ROPE, &[&ab.k], &[n, nkv, hd, hkv, 0, self.t, gpu_core::f(theta)], n * nkv * (hd / 2)));
        // GQA→MHA fused buffer for the trio consumers; the GEMM-attention path
        // packs straight from q/k/v (folding the replication) and reuses `qkv`
        // as pack space, so it skips these.
        if build_fused {
            s.push(block::kv_expand_fwd(&self.gpu, KV_EXPAND, &ab.q, &ab.qkv, n, nh, 1, hd, 3 * d, 0));
            s.push(block::kv_expand_fwd(&self.gpu, KV_EXPAND, &ab.k, &ab.qkv, n, nh, group, hd, 3 * d, d));
            s.push(block::kv_expand_fwd(&self.gpu, KV_EXPAND, &ab.v, &ab.qkv, n, nh, group, hd, 3 * d, 2 * d));
        }
    }

    /// Conv-mixer body: in_proj thirds, gating, depthwise conv, out into `gated`.
    fn emit_conv_mixer(&self, s: &mut Vec<Step>, l: usize, xn1: &DeviceBuffer, cb: &ConvBufs, b_use: u32) {
        let n = b_use * self.t;
        let d = self.cfg.d_model;
        let p = |name: &str| format!("blocks.{l}.{name}");
        // in_proj row-thirds: B = rows 0..d, C = d..2d, X = 2d..3d.
        self.sliced_linear(s, xn1, &p("conv.in_proj.weight"), 0, d, d, &cb.bg, n);
        self.sliced_linear(s, xn1, &p("conv.in_proj.weight"), d, d, d, &cb.cg, n);
        self.sliced_linear(s, xn1, &p("conv.in_proj.weight"), 2 * d, d, d, &cb.xg, n);
        s.push(self.gpu.step(MUL, &[&cb.bg, &cb.xg, &cb.bx], &[n * d], n * d));
        // NLC -> NCL, depthwise symmetric conv, NCL -> NLC.
        s.push(self.gpu.step(NLC_NCL, &[&cb.bx, &cb.bx_ncl], &[n * d, d, self.t], n * d));
        s.push(audio::conv::conv1d_fwd(&self.gpu, &Self::conv_kernels(), &self.conv_shape_b(b_use), &cb.bx_ncl, self.w(&p("conv.conv.weight")), &cb.conv_ncl));
        s.push(self.gpu.step(NCL_NLC, &[&cb.conv_ncl, &cb.conv_out], &[n * d, d, self.t], n * d));
        s.push(self.gpu.step(MUL, &[&cb.cg, &cb.conv_out, &cb.gated], &[n * d], n * d));
    }

    /// SwiGLU FFN: ln2(xmid) → gate/up → silu·mul → down, residual into `out`.
    fn emit_ffn(&self, s: &mut Vec<Step>, l: usize, cb: &CommonBufs, mlp_out: &DeviceBuffer, out: &DeviceBuffer, b_use: u32) {
        let n = b_use * self.t;
        let (d, ff, eps) = (self.cfg.d_model, self.cfg.d_ff, self.cfg.norm_eps);
        let p = |name: &str| format!("blocks.{l}.{name}");
        s.push(block::rmsnorm_eps_fwd(&self.gpu, RMSNORM_EPS, &cb.xmid, self.w(&p("ln2.weight")), &cb.xn2, d, n, eps));
        let (mk, mt) = linear_kernel(n as usize, ff as usize);
        s.push(self.gpu.step(mk, &[&cb.xn2, self.w(&p("mlp.gate.weight")), &cb.gate_pre], &[n, d, ff], mt));
        let (mk, mt) = linear_kernel(n as usize, ff as usize);
        s.push(self.gpu.step(mk, &[&cb.xn2, self.w(&p("mlp.up.weight")), &cb.up], &[n, d, ff], mt));
        s.push(self.gpu.step(SILU_MUL, &[&cb.gate_pre, &cb.up, &cb.h], &[n * ff], n * ff));
        let (mk, mt) = linear_kernel(n as usize, d as usize);
        s.push(self.gpu.step(mk, &[&cb.h, self.w(&p("mlp.down.weight")), mlp_out], &[n, ff, d], mt));
        s.push(self.gpu.step(ADD2, &[&cb.xmid, mlp_out, out], &[n * d], n * d));
    }

    fn forward_steps(&self) -> Vec<Step> {
        self.forward_steps_for(self.b)
    }

    /// The forward for a row-batch of `b_use` sequences (≤ the built `b`) —
    /// buffers are sized for the max, kernels touch the `b_use·t` prefix. The
    /// chunked-inference regime prebuilds one list per group size so the
    /// scheduler never pays repeat-padding for a partial group.
    fn forward_steps_for(&self, b_use: u32) -> Vec<Step> {
        let c = &self.cfg;
        assert!(b_use >= 1 && b_use <= self.b);
        assert!(b_use == self.b || matches!(self.regime, Regime::Chunked { .. }), "partial batches: chunked regime only");
        let n = b_use * self.t;
        let d = c.d_model;
        let v = c.vocab;
        let hq = c.q_dim();
        let eps = c.norm_eps;
        let dw = d as u64;
        let tiles = block::vocab_tiles(v as u64, dw);
        let mut s: Vec<Step> = Vec::new();

        // The residual-in / residual-out buffer for layer `l`, per regime.
        let res_of = |l: usize| -> &DeviceBuffer {
            match &self.regime {
                Regime::Materialized { res, .. } => &res[l],
                Regime::Chunked { res, .. } => &res[l % 2],
            }
        };

        // Token embedding, vocab-tiled (65536×1024 fp32 = 256 MB per binding
        // otherwise).
        for &(v0, cnt) in &tiles {
            s.push(self.gpu.step_sliced(
                EMBED_TILE,
                &[&self.tokens, self.w("tok.weight"), res_of(0)],
                &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                &[d, n, v0, cnt],
                n * d,
            ));
        }

        for (l, ty) in c.layer_types.iter().enumerate() {
            let p = |name: &str| format!("blocks.{l}.{name}");
            let (common, attn, conv): (&CommonBufs, Option<&AttnBufs>, Option<&ConvBufs>) = match &self.regime {
                Regime::Materialized { layers, .. } => {
                    let lb = &layers[l];
                    match &lb.mixer {
                        MixerBufs::Attn(a) => (&lb.common, Some(a), None),
                        MixerBufs::Conv(cv) => (&lb.common, None, Some(cv)),
                    }
                }
                Regime::Chunked { common, attn, conv, .. } => (common, Some(attn), Some(conv)),
            };
            s.push(block::rmsnorm_eps_fwd(&self.gpu, RMSNORM_EPS, res_of(l), self.w(&p("ln1.weight")), &common.xn1, d, n, eps));

            let mixer_out = &self.proj;
            match ty {
                LayerType::Attention => {
                    let ab = attn.expect("attn bufs");
                    let gemm_attn = matches!(&self.regime, Regime::Chunked { .. }) && self.gpu.caps().workgroup_reductions;
                    self.emit_attn_qkv(&mut s, l, &common.xn1, ab, b_use, !gemm_attn);
                    match &self.regime {
                        Regime::Materialized { scores, .. } => match self.train_chunk {
                            None => {
                                s.extend(block::bidir_fwd(&self.gpu, &Self::bidir_ids(), &self.bidir(), &ab.qkv, scores, &ab.probs, &ab.ctx));
                            }
                            Some(chunk) => {
                                let spans: Vec<(u32, u32)> = (0..self.b).map(|i| (i * self.t, self.t)).collect();
                                let ids = CrossIds { scores: SCORES_CROSS, softmax: SOFTMAX_CROSS, apply: APPLY_CROSS };
                                block::chunked_bidir_fwd(
                                    &self.gpu, &ids, c.n_heads, c.head_dim, hq, &ab.qkv, 3 * d, 0, d, 2 * d,
                                    &ab.ctx, &self.slab_scores, &self.slab_probs, &spans, chunk, &mut s,
                                );
                            }
                        },
                        Regime::Chunked { scores, probs, chunk, ctx_pack, .. } => {
                            let spans: Vec<(u32, u32)> = (0..b_use).map(|i| (i * self.t, self.t)).collect();
                            // GEMM attention on cooperative devices (register-
                            // tiled matmuls over per-head packs, GQA folded into
                            // the pack — measured ~8x over the naive trio and the
                            // flash kernel at 8k on a P40); the chunked cross trio
                            // is the CPU-JIT path (its native fast paths own that
                            // regime and read the fused qkv).
                            if self.gpu.caps().workgroup_reductions {
                                let ids = block::GemmAttnIds {
                                    head_pack: HEAD_PACK,
                                    head_pack_t: HEAD_PACK_T,
                                    head_unpack: HEAD_UNPACK,
                                    softmax_rows: SOFTMAX_ROWS,
                                    matmul: MATMUL,
                                    matmul_reg2: MATMUL_REG2,
                                };
                                block::gemm_bidir_fwd(
                                    &self.gpu, &ids, c.n_heads, c.head_dim, c.group(), &ab.q, hq, (&ab.k, &ab.v),
                                    c.kv_dim(), &ab.ctx, hq, &ab.qkv, ctx_pack, scores, probs, &spans, *chunk, false, &mut s,
                                );
                            } else {
                                let ids = CrossIds { scores: SCORES_CROSS, softmax: SOFTMAX_CROSS, apply: APPLY_CROSS };
                                block::chunked_bidir_fwd(
                                    &self.gpu, &ids, c.n_heads, c.head_dim, hq, &ab.qkv, 3 * d, 0, d, 2 * d,
                                    &ab.ctx, scores, probs, &spans, *chunk, &mut s,
                                );
                            }
                        }
                    }
                    let (mk, mt) = linear_kernel(n as usize, d as usize);
                    s.push(self.gpu.step(mk, &[&ab.ctx, self.w(&p("attn.wo.weight")), mixer_out], &[n, hq, d], mt));
                }
                LayerType::Conv => {
                    let cb = conv.expect("conv bufs");
                    self.emit_conv_mixer(&mut s, l, &common.xn1, cb, b_use);
                    let (mk, mt) = linear_kernel(n as usize, d as usize);
                    s.push(self.gpu.step(mk, &[&cb.gated, self.w(&p("conv.out_proj.weight")), mixer_out], &[n, d, d], mt));
                }
            }
            s.push(self.gpu.step(ADD2, &[res_of(l), mixer_out, &common.xmid], &[n * d], n * d));
            self.emit_ffn(&mut s, l, common, &self.mlp_out, res_of(l + 1), b_use);
        }

        // Final norm.
        let last = c.n_layers() as usize;
        s.push(block::rmsnorm_eps_fwd(&self.gpu, RMSNORM_EPS, res_of(last), self.w("norm.weight"), &self.xn_final, d, n, eps));

        // Head.
        if let Some(hg) = &self.head {
            // Gathered supervised-row head: `embed` row-gather → tiled tied
            // matmul over ≤ cap rows → masked CE (padded rows carry IGNORE).
            let cap = hg.cap;
            s.push(self.gpu.step(EMBED, &[&hg.sup_idx, &self.xn_final, &hg.probe_h], &[d, cap], cap * d));
            let hw = c.head_weight();
            for &(v0, cnt) in &tiles {
                s.push(self.gpu.step_sliced(
                    MATMUL_TILE,
                    &[&hg.probe_h, self.w(hw), &hg.logits_g],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                    &[cap, d, v, v0, cnt],
                    cap * cnt,
                ));
            }
            s.push(self.gpu.step(CE_VALUE, &[&hg.logits_g, &hg.sup_targets, &hg.ce_g], &[cap, v, IGNORE], cap));
            return s;
        }
        match &self.regime {
            Regime::Materialized { .. } => {
                let head = c.head_weight();
                if tiles.len() == 1 {
                    let (mk, mt) = linear_kernel(n as usize, v as usize);
                    s.push(self.gpu.step(mk, &[&self.xn_final, self.w(head), &self.logits], &[n, d, v], mt));
                } else {
                    for &(v0, cnt) in &tiles {
                        s.push(self.gpu.step_sliced(
                            MATMUL_TILE,
                            &[&self.xn_final, self.w(head), &self.logits],
                            &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                            &[n, d, v, v0, cnt],
                            n * cnt,
                        ));
                    }
                }
                s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, v, IGNORE], n));
            }
            Regime::Chunked { probe_cap, probe_idx, probe_h, probe_logits, .. } => {
                if *probe_cap > 0 {
                    // Gather the probe rows of the hidden state (the `embed`
                    // kernel IS a row gather: indices → rows of a table), then
                    // run the tied head only on those rows.
                    s.push(self.gpu.step(EMBED, &[probe_idx, &self.xn_final, probe_h], &[d, *probe_cap], *probe_cap * d));
                    let head = c.head_weight();
                    for &(v0, cnt) in &tiles {
                        s.push(self.gpu.step_sliced(
                            MATMUL_TILE,
                            &[probe_h, self.w(head), probe_logits],
                            &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                            &[*probe_cap, d, v, v0, cnt],
                            *probe_cap * cnt,
                        ));
                    }
                }
            }
        }
        s
    }

    /// Backward for a plain linear `y = x·Wᵀ`: weight grad (accumulating into
    /// the param grad) + input grad (`acc` = add into `dx` vs overwrite).
    #[allow(clippy::too_many_arguments)]
    fn lin_bwd(&self, s: &mut Vec<Step>, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        let (bk, bt) = dw_kernel(nout, k);
        s.push(self.gpu.step(bk, &[d_out, x, self.ps.g(wname)], &[m, k, nout], bt));
        let (bk, bt) = dx_kernel(m, k);
        s.push(self.gpu.step(bk, &[d_out, self.w(wname), dx], &[m, k, nout, acc], bt));
    }

    /// Backward for a row-sliced linear (the conv mixer's in_proj thirds):
    /// grads land in the `[r0, r0+nout)` rows of the weight grad.
    #[allow(clippy::too_many_arguments)]
    fn sliced_lin_bwd(&self, s: &mut Vec<Step>, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, r0: u32, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        let sl = (r0 as u64 * k as u64, nout as u64 * k as u64);
        let (bk, bt) = dw_kernel(nout, k);
        s.push(self.gpu.step_sliced(bk, &[d_out, x, self.ps.g(wname)], &[(0, 0), (0, 0), sl], &[m, k, nout], bt));
        let (bk, bt) = dx_kernel(m, k);
        s.push(self.gpu.step_sliced(bk, &[d_out, self.w(wname), dx], &[(0, 0), sl, (0, 0)], &[m, k, nout, acc], bt));
    }

    /// RMSNorm backward via the shared eps-aware builder.
    #[allow(clippy::too_many_arguments)]
    fn norm_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, inv: &DeviceBuffer, dim: u32, rows: u32) {
        s.extend(block::rmsnorm_eps_bwd(
            &self.gpu,
            RMSNORM_EPS_INV,
            RMSNORM_DW,
            RMSNORM_EPS_DX,
            x,
            self.w(wname),
            dy,
            dx,
            inv,
            Some(self.ps.g(wname)),
            dim,
            rows,
            self.cfg.norm_eps,
        ));
    }

    fn backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let bw = self.bwd.as_ref().expect("train mode");
        let Regime::Materialized { layers, res, .. } = &self.regime else {
            panic!("backward: materialized regime only");
        };
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let (nh, nkv, group) = (c.n_heads, c.n_kv_heads, c.group());
        let theta = c.rope_theta;
        let head = c.head_weight();
        let mut s: Vec<Step> = Vec::new();

        // ---- head + final norm ----
        if let Some(hg) = &self.head {
            // Gathered head backward: CE stats/grad over ≤ cap rows, tied-head
            // dW from the gathered rows, then the input grad scattered back to
            // the supervised rows of d_xn (zeroed via submit `clears` — rows
            // with no supervision contribute nothing).
            let cap = hg.cap;
            s.push(self.gpu.step(CE_STATS, &[&hg.logits_g, &hg.sup_targets, &hg.ce_stats_g], &[cap, v, IGNORE], cap));
            s.push(self.gpu.step_buf(CE_GRAD_STATS, &self.ce_grad_uni, &[&hg.logits_g, &hg.sup_targets, &hg.ce_stats_g, &hg.d_logits_g], cap * v));
            let (bk, bt) = dw_kernel(v, d);
            s.push(self.gpu.step(bk, &[&hg.d_logits_g, &hg.probe_h, self.ps.g(head)], &[cap, d, v], bt));
            let (bk, bt) = dx_kernel(cap, d);
            s.push(self.gpu.step(bk, &[&hg.d_logits_g, self.w(head), &hg.d_probe_h], &[cap, d, v, 0], bt));
            s.push(self.gpu.step(ROW_SCATTER, &[&hg.sup_idx_scatter, &hg.d_probe_h, &bw.d_xn], &[cap, d, n], cap * d));
        } else {
            // Two-pass CE gradient (per-row stats then per-element grad): the
            // 65536-vocab equivalent of qwen's ce_stats path.
            s.push(self.gpu.step(CE_STATS, &[&self.logits, &self.targets, &bw.ce_stats], &[n, v, IGNORE], n));
            s.push(self.gpu.step_buf(CE_GRAD_STATS, &self.ce_grad_uni, &[&self.logits, &self.targets, &bw.ce_stats, &bw.d_logits], n * v));
            let (bk, bt) = dw_kernel(v, d);
            s.push(self.gpu.step(bk, &[&bw.d_logits, &self.xn_final, self.ps.g(head)], &[n, d, v], bt));
            let (bk, bt) = dx_kernel(n, d);
            s.push(self.gpu.step(bk, &[&bw.d_logits, self.w(head), &bw.d_xn], &[n, d, v, 0], bt));
        }
        let last = c.n_layers() as usize;
        self.norm_bwd(&mut s, &res[last], "norm.weight", &bw.d_xn, &bw.dres[last], &bw.inv, d, n);

        for (l, lb) in layers.iter().enumerate().rev() {
            let p = |name: &str| format!("blocks.{l}.{name}");
            let cb = &lb.common;

            // ---- FFN backward (input grad = dres[l+1]) ----
            self.lin_bwd(&mut s, &bw.dres[l + 1], &cb.h, &p("mlp.down.weight"), &bw.d_h, n, ff, d, 0);
            s.push(self.gpu.step(SILU_DA, &[&cb.gate_pre, &cb.up, &bw.d_h, &bw.d_gate_pre], &[n * ff], n * ff));
            s.push(self.gpu.step(SILU_DB, &[&cb.gate_pre, &bw.d_h, &bw.d_up], &[n * ff], n * ff));
            self.lin_bwd(&mut s, &bw.d_up, &cb.xn2, &p("mlp.up.weight"), &bw.d_xn, n, d, ff, 0);
            self.lin_bwd(&mut s, &bw.d_gate_pre, &cb.xn2, &p("mlp.gate.weight"), &bw.d_xn, n, d, ff, 1);
            self.norm_bwd(&mut s, &cb.xmid, &p("ln2.weight"), &bw.d_xn, &bw.d_tmp, &bw.inv, d, n);
            s.push(self.gpu.step(ADD2, &[&bw.dres[l + 1], &bw.d_tmp, &bw.dxmid], &[n * d], n * d));

            // ---- mixer backward (input grad = dxmid) ----
            match &lb.mixer {
                MixerBufs::Attn(ab) => {
                    self.lin_bwd(&mut s, &bw.dxmid, &ab.ctx, &p("attn.wo.weight"), &bw.d_ctx, n, hq, d, 0);
                    match self.train_chunk {
                        None => s.extend(block::bidir_bwd(
                            &self.gpu, &Self::bidir_ids(), &self.bidir(), &ab.qkv, &ab.probs, &bw.d_ctx, &bw.d_scores, &bw.d_qkv,
                        )),
                        Some(chunk) => {
                            let spans: Vec<(u32, u32)> = (0..self.b).map(|i| (i * self.t, self.t)).collect();
                            let fwd_ids = CrossIds { scores: SCORES_CROSS, softmax: SOFTMAX_CROSS, apply: APPLY_CROSS };
                            let bwd_ids = block::CrossBwdIds { dscores: DSCORES_CROSS, dq: DQ_CROSS, dk_acc: DK_CROSS_ACC, dv_acc: DV_CROSS_ACC };
                            block::chunked_bidir_bwd(
                                &self.gpu, &fwd_ids, &bwd_ids, c.n_heads, hd, hq, &ab.qkv, 3 * d, 0, d, 2 * d,
                                &bw.d_ctx, &bw.d_qkv, &self.slab_scores, &self.slab_probs, &self.slab_dscores,
                                &spans, chunk, &mut s,
                            );
                        }
                    }
                    // Fused-region grads back to the narrow projections:
                    // q copies out (group 1), k/v group-sum.
                    s.push(block::kv_expand_bwd(&self.gpu, KV_EXPAND_BWD, &bw.d_qkv, &bw.d_q, n, nh, 1, hd, 3 * d, 0));
                    s.push(block::kv_expand_bwd(&self.gpu, KV_EXPAND_BWD, &bw.d_qkv, &bw.d_k, n, nh, group, hd, 3 * d, d));
                    s.push(block::kv_expand_bwd(&self.gpu, KV_EXPAND_BWD, &bw.d_qkv, &bw.d_v, n, nh, group, hd, 3 * d, 2 * d));
                    // RoPE backward in place, then QK-norm backward.
                    s.push(self.gpu.step(ROPE_BWD, &[&bw.d_q], &[n, nh, hd, hq, 0, self.t, gpu_core::f(theta)], n * nh * (hd / 2)));
                    s.push(self.gpu.step(ROPE_BWD, &[&bw.d_k], &[n, nkv, hd, hkv, 0, self.t, gpu_core::f(theta)], n * nkv * (hd / 2)));
                    self.norm_bwd(&mut s, &ab.q_pre, &p("attn.q_norm.weight"), &bw.d_q, &bw.dq_pre, &bw.inv, hd, n * nh);
                    self.norm_bwd(&mut s, &ab.k_pre, &p("attn.k_norm.weight"), &bw.d_k, &bw.dk_pre, &bw.inv, hd, n * nkv);
                    self.lin_bwd(&mut s, &bw.d_v, &cb.xn1, &p("attn.wv.weight"), &bw.d_xn, n, d, hkv, 0);
                    self.lin_bwd(&mut s, &bw.dk_pre, &cb.xn1, &p("attn.wk.weight"), &bw.d_xn, n, d, hkv, 1);
                    self.lin_bwd(&mut s, &bw.dq_pre, &cb.xn1, &p("attn.wq.weight"), &bw.d_xn, n, d, hq, 1);
                }
                MixerBufs::Conv(cv) => {
                    // out_proj backward: d_gated (d_mix1).
                    self.lin_bwd(&mut s, &bw.dxmid, &cv.gated, &p("conv.out_proj.weight"), &bw.d_mix1, n, d, d, 0);
                    // y = C ⊙ conv_out:  d_C (d_mix2) and d_conv_out (d_mix3).
                    s.push(self.gpu.step(MUL, &[&bw.d_mix1, &cv.conv_out, &bw.d_mix2], &[n * d], n * d));
                    s.push(self.gpu.step(MUL, &[&bw.d_mix1, &cv.cg, &bw.d_mix3], &[n * d], n * d));
                    // Permute adjoint (NLC→NCL), conv backward, permute back.
                    s.push(self.gpu.step(NLC_NCL, &[&bw.d_mix3, &bw.d_mix4], &[n * d, d, self.t], n * d));
                    s.extend(audio::conv::conv1d_bwd(
                        &self.gpu,
                        &Self::conv_kernels(),
                        &self.conv_shape_b(self.b),
                        &bw.d_mix4,
                        &cv.bx_ncl,
                        self.w(&p("conv.conv.weight")),
                        Some(&bw.d_mix3), // d_bx (NCL) — d_mix3's NLC value is consumed
                        Some(self.ps.g(&p("conv.conv.weight"))),
                    ));
                    s.push(self.gpu.step(NCL_NLC, &[&bw.d_mix3, &bw.d_mix4], &[n * d, d, self.t], n * d));
                    // Bx = B ⊙ X:  d_B (d_mix1, reused) and d_X (d_mix3, reused).
                    s.push(self.gpu.step(MUL, &[&bw.d_mix4, &cv.xg, &bw.d_mix1], &[n * d], n * d));
                    s.push(self.gpu.step(MUL, &[&bw.d_mix4, &cv.bg, &bw.d_mix3], &[n * d], n * d));
                    // in_proj row-thirds backward (B rows 0..d, C d..2d, X 2d..3d).
                    self.sliced_lin_bwd(&mut s, &bw.d_mix1, &cb.xn1, &p("conv.in_proj.weight"), 0, &bw.d_xn, n, d, d, 0);
                    self.sliced_lin_bwd(&mut s, &bw.d_mix2, &cb.xn1, &p("conv.in_proj.weight"), d, &bw.d_xn, n, d, d, 1);
                    self.sliced_lin_bwd(&mut s, &bw.d_mix3, &cb.xn1, &p("conv.in_proj.weight"), 2 * d, &bw.d_xn, n, d, d, 1);
                }
            }
            self.norm_bwd(&mut s, &res[l], &p("ln1.weight"), &bw.d_xn, &bw.d_tmp, &bw.inv, d, n);
            s.push(self.gpu.step(ADD2, &[&bw.dxmid, &bw.d_tmp, &bw.dres[l]], &[n * d], n * d));
        }

        // Embedding backward (tied: accumulates onto the head grad in tok.weight).
        s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &bw.dres[0], self.ps.g("tok.weight")], &[n, d, v], v * d));
        s
    }

    pub fn backward(&self) {
        assert!(!self.bwd_steps.is_empty(), "backward: model built without training graphs");
        match &self.head {
            Some(hg) => {
                self.gpu.write(&self.ce_grad_uni, &[hg.cap, self.cfg.vocab, IGNORE, gpu_core::f(self.count.get())]);
                // d_xn is the row_scatter target: unsupervised rows must be zero.
                let bw = self.bwd.as_ref().expect("train mode");
                self.gpu.submit(&[&bw.d_xn], &self.bwd_steps);
            }
            None => {
                let n = self.b * self.t;
                self.gpu.write(&self.ce_grad_uni, &[n, self.cfg.vocab, IGNORE, gpu_core::f(self.count.get())]);
                self.gpu.submit(&[], &self.bwd_steps);
            }
        }
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.as_ref().expect("train mode").step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }
    pub fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name)))
            .collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
    }

    /// Run the forward graph; returns the masked-CE loss (materialized regime;
    /// 0.0 in the chunked regime — read the taps instead).
    /// Chunked-inference regime: run the forward for a partial group of
    /// `b_use` sequences (tokens for exactly `b_use*t` rows must be set).
    pub fn forward_group(&self, b_use: u32) {
        if b_use == self.b || self.fwd_variants.is_empty() {
            self.gpu.submit(&[], &self.fwd_steps);
        } else {
            self.gpu.submit(&[], &self.fwd_variants[(b_use - 1) as usize]);
        }
    }

    /// Final hidden states for the first `rows` sequence rows.
    pub fn read_hidden_rows(&self, rows: usize) -> Vec<f32> {
        self.gpu.read(&self.xn_final, rows * self.cfg.d_model as usize)
    }

    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd_steps);
        if let Some(hg) = &self.head {
            let losses = self.gpu.read(&hg.ce_g, hg.cap as usize);
            return losses.iter().sum::<f32>() / self.count.get();
        }
        match &self.regime {
            Regime::Materialized { .. } => {
                let n = (self.b * self.t) as usize;
                let losses = self.gpu.read(&self.ce_buf, n);
                losses.iter().sum::<f32>() / self.count.get()
            }
            Regime::Chunked { .. } => 0.0,
        }
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    // ---- parity / inference taps ----

    /// Residual stream after `l` layers (materialized regime only).
    pub fn read_res(&self, l: usize) -> Vec<f32> {
        let Regime::Materialized { res, .. } = &self.regime else {
            panic!("read_res: chunked regime keeps no per-layer residuals");
        };
        let n = (self.b * self.t) as usize;
        self.gpu.read(&res[l], n * self.cfg.d_model as usize)
    }

    /// Final hidden states (post `embedding_norm`), `[b*t, d_model]`.
    pub fn read_hidden(&self) -> Vec<f32> {
        let n = (self.b * self.t) as usize;
        self.gpu.read(&self.xn_final, n * self.cfg.d_model as usize)
    }

    /// MLM logits `[b*t, vocab]` (materialized regime only).
    pub fn read_logits(&self) -> Vec<f32> {
        assert!(matches!(self.regime, Regime::Materialized { .. }), "read_logits: use read_probe_logits on the chunked path");
        let n = (self.b * self.t) as usize;
        self.gpu.read(&self.logits, n * self.cfg.vocab as usize)
    }

    /// Chunked regime: logits at the rows set via [`Self::set_probe_rows`],
    /// `[n_probes, vocab]`.
    pub fn read_probe_logits(&self) -> Vec<f32> {
        let Regime::Chunked { probe_logits, n_probes, .. } = &self.regime else {
            panic!("read_probe_logits: not a chunked-regime model");
        };
        let k = n_probes.get() as usize;
        let v = self.cfg.vocab as usize;
        let mut out = self.gpu.read(probe_logits, k.max(1) * v);
        out.truncate(k * v);
        out
    }

    /// The chunk the chunked regime is running with (test introspection).
    pub fn chunk(&self) -> Option<u32> {
        match &self.regime {
            Regime::Chunked { chunk, .. } => Some(*chunk),
            Regime::Materialized { .. } => None,
        }
    }

    /// Full-sequence MLM logits for `tokens` (materialized regime).
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        self.set_tokens(tokens);
        self.forward();
        self.read_logits()
    }
}

impl model::ModelConfig for LfmConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        LfmConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        LfmConfig::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        LfmConfig::from_json(v)
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn block_size(&self) -> u32 {
        self.block_size
    }
    fn finalize_for_dataset(mut self, vocab: u32, block_size: u32) -> Self {
        self.vocab = vocab;
        self.block_size = block_size;
        self
    }
}

impl model::Model for Lfm {
    type Config = LfmConfig;

    /// Trait construction is the TRAINABLE model — what the generic trainer
    /// and the blanket gradcheck need. (The inherent `Lfm::new` stays frozen
    /// for parity/inference work.)
    fn new(cfg: LfmConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Lfm::new_train(cfg, b, t, init)
    }
    fn init_weights(cfg: &LfmConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }
    fn config(&self) -> &LfmConfig {
        &self.cfg
    }
    /// `Batch::Lm` carries UNshifted MLM targets here (the loader shift is a
    /// causal-LM concern; `data::mlm::get_mlm_batch` produces this shape).
    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => Lfm::set_batch(self, tokens, targets),
            _ => panic!("lfm::Lfm only supports Batch::Lm (MLM, unshifted targets)"),
        }
    }
    fn forward(&self) -> f32 {
        Lfm::forward(self)
    }
    fn backward(&self) {
        Lfm::backward(self)
    }
    fn zero_grads(&self) {
        Lfm::zero_grads(self)
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Lfm::adamw_step(self, t, lr, wd, clip, extra_scale)
    }
    fn poll_wait(&self) {
        Lfm::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        self.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Lfm::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Lfm::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Lfm::read_grad(self, name)
    }
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(Lfm::logits_all(self, tokens))
    }
    fn save(&self, path: &str) {
        Lfm::save(self, path)
    }
    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }
}
