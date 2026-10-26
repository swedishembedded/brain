// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recurrent-depth execution - a prelude, a
// shared core re-run with the input re-injected, and a coda - for its clients.
// If your team needs expertise in running a prelude/core/coda decoder or in
// proving it matches its reference then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Huginn's execution graph (forward).
//!
//! Every block is a sandwich with four norms:
//!
//! ```text
//! x = norm_2(attn(norm_1(x)) + x)
//! x = norm_4(mlp(norm_3(x)) + x)
//! ```
//!
//! The attention is a fused QKV projection with a per-head Q/K bias and
//! interleaved-pair RoPE (the `rope_train` kernel, not the half-split
//! `rope_base`); the MLP is a fused gate/up FC into SiLU(gate) * up.
//!
//! The same block is composed three ways, in this order:
//!
//! 1. prelude blocks over `emb * sqrt(width)`; their output replaces the
//!    embeddings for the rest of the run;
//! 2. `num_steps` repetitions of `x = adapter(concat(x, prelude_out))`
//!    followed by the recurrent core blocks (only these weights repeat);
//! 3. the final norm, the coda blocks, the final norm again, and the tied
//!    head.
//!
//! The starting latent state is the caller's (`set_state`): the reference's
//! own `like-init` state is a random draw, so a parity run hands it the same
//! state the golden used, and generation draws it once per request.
//!
//! Training (`new_train`) runs the same graph with a tape: the boundary
//! residual of every block **application** is kept, so the recurrence's
//! `steps · n_recurrent` uses of one shared parameter set each contribute a
//! term to that set's gradient instead of the last use overwriting the others.
//! The backward walks the applications in reverse, rebuilding each block's
//! internal activations from its boundary residual (they are shared scratch in
//! the forward) and differentiating it with the reverse of the very same
//! `block_steps` the forward composed.

use std::cell::Cell;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use model::saved::{Residuals, SavedPolicy};
use model::streamed::{RevisitingTape, StreamedStore};
use model::tape::LayerTape;
use paramstore::ParamStore;

use crate::config::HuginnConfig;

/// Cross-entropy ignore index (unsupervised positions).
pub const IGNORE: u32 = 0xFFFF_FFFF;

const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const ROPE_TRAIN: usize = 3;
const GQA_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const GQA_APPLY: usize = 6;
const SILU_MUL: usize = 7;
const ADD2: usize = 8;
const BIAS_ADD: usize = 9;
const COPY_COLS: usize = 10;
const SCALE_ROW: usize = 11;
const ROPE_AT: usize = 12;
const KV_APPEND: usize = 13;
const ATTN_DECODE_SCORES: usize = 14;
const DECODE_SOFTMAX: usize = 15;
const ATTN_DECODE_APPLY: usize = 16;
const MATMUL_GEMV: usize = 17;
const MATMUL_REG2: usize = 18;
const FLASH_GQA_FWD: usize = 19;
const PAGED_KV_APPEND_BATCHED: usize = 20;
const PAGED_DECODE_SCORES_BATCHED: usize = 21;
const DECODE_SOFTMAX_BATCHED: usize = 22;
const PAGED_DECODE_APPLY_BATCHED: usize = 23;
// Training slots, appended so every index above keeps its meaning.
const CE_VALUE: usize = 24;
const CE_GRAD: usize = 25;
const MATMUL_DX: usize = 26;
const MATMUL_DW: usize = 27;
const EMB_BWD: usize = 28;
const ADAMW: usize = 29;
const GRADNORM_SQ: usize = 30;
const GRAD_SCALE: usize = 31;
const CLIP_COEF: usize = 32;
const GRAD_SCALE_BUF: usize = 33;
const AXPY: usize = 34;
const BIAS_GRAD: usize = 35;
const RMS_INV: usize = 36;
const RMSNORM_DX: usize = 37;
const RMSNORM_DW: usize = 38;
const ROPE_TRAIN_BWD: usize = 39;
const GQA_DSCORES: usize = 40;
const GQA_DV: usize = 41;
const GQA_DQ: usize = 42;
const GQA_DK: usize = 43;
const SILU_DA: usize = 44;
const SILU_DB: usize = 45;

// Slots the decode engine composes its own graph from.
pub(crate) const EMBED_ID: usize = EMBED;
pub(crate) const MATMUL_ID: usize = MATMUL;
pub(crate) const ADD2_ID: usize = ADD2;
pub(crate) const BIAS_ADD_ID: usize = BIAS_ADD;
pub(crate) const COPY_COLS_ID: usize = COPY_COLS;
pub(crate) const ROPE_AT_ID: usize = ROPE_AT;
pub(crate) const SILU_MUL_ID: usize = SILU_MUL;
pub(crate) const SCALE_ROW_ID: usize = SCALE_ROW;
pub(crate) const KV_APPEND_ID: usize = KV_APPEND;
pub(crate) const ATTN_DECODE_SCORES_ID: usize = ATTN_DECODE_SCORES;
pub(crate) const DECODE_SOFTMAX_ID: usize = DECODE_SOFTMAX;
pub(crate) const ATTN_DECODE_APPLY_ID: usize = ATTN_DECODE_APPLY;
pub(crate) const MATMUL_GEMV_ID: usize = MATMUL_GEMV;
pub(crate) const MATMUL_REG2_ID: usize = MATMUL_REG2;
pub(crate) const FLASH_GQA_ID: usize = FLASH_GQA_FWD;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_train", kernels::ROPE_TRAIN),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("bias_add", kernels::BIAS_ADD),
    ("copy_cols", kernels::COPY_COLS),
    ("scale_row", kernels::SCALE_ROW),
    ("rope_at", kernels::ROPE_AT),
    ("kv_append", kernels::KV_APPEND),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("ce_grad", kernels::CE_GRAD_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("axpy", kernels::AXPY),
    ("bias_grad", kernels::BIAS_GRAD),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_train_bwd", kernels::ROPE_TRAIN_BWD),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    // Narrow boundary residuals (`model::saved::Residuals`), resolved by name.
    ("pack_bf16", kernels::PACK_BF16),
    ("unpack_bf16", kernels::UNPACK_BF16),
    ("pack_fp8_rows", kernels::PACK_FP8_ROWS),
    ("unpack_fp8_rows", kernels::UNPACK_FP8_ROWS),
    // A layer-streamed full fine-tune's device side (`model::ops::store_device_kernels`; `copy_cols` is above).
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("gradnorm_part_bf16", kernels::GRADNORM_PART_BF16),
];

pub(crate) fn decode_batched_ids() -> block::GqaDecodeBatchedIds {
    block::GqaDecodeBatchedIds {
        kv_append_batched: PAGED_KV_APPEND_BATCHED,
        scores_batched: PAGED_DECODE_SCORES_BATCHED,
        softmax_batched: DECODE_SOFTMAX_BATCHED,
        apply_batched: PAGED_DECODE_APPLY_BATCHED,
    }
}

pub(crate) fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
        rmsnorm_dw: RMSNORM_DW,
        rope: ROPE_TRAIN,
        rope_bwd: ROPE_TRAIN_BWD,
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
        rmsnorm_rows: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
    }
}

struct Scratch {
    xn1: DeviceBuffer,
    xn2: DeviceBuffer,
    xn3: DeviceBuffer,
    qkv: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    /// The attention residual, `x_in + attn_out`: `norm_2`'s input.
    tmp: DeviceBuffer,
    fc: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    mh: DeviceBuffer,
    mout: DeviceBuffer,
    /// The MLP residual, `norm_2(attention residual) + mlp_out`: `norm_4`'s
    /// input. A separate buffer from [`Scratch::tmp`] because the backward
    /// differentiates both residuals and each needs its own input.
    res: DeviceBuffer,
    concat: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

/// The training tape: the boundary residual of every block application.
///
/// The backward needs all of them - the recurrent core's `steps · n_recurrent`
/// applications are the ones whose shared weights each contribute a gradient
/// term - and a shared buffer could not carry a distinct gradient per
/// application.
struct Tape {
    /// `[embed, scaled embed, prelude.0 … prelude.n-1]` - input and output
    /// boundary of every prelude application, `n_prelude + 2` buffers. The
    /// last one is the injection every recurrent step re-reads.
    embeds: Vec<DeviceBuffer>,
    /// Per recurrent step `[adapter, core.0 … core.n-1]`, so
    /// `steps * (n_recurrent + 1)` boundaries: the part of the tape that grows
    /// with the recurrence, and the part a narrow saved format holds
    /// ([`Residuals`]).
    core: Residuals,
    /// The normed state the coda consumes (`ln_f` applied, coda not).
    latent: DeviceBuffer,
    /// `[coda.0 … coda.n-1, final norm]`.
    coda: Vec<DeviceBuffer>,
    /// Per-row cross-entropy of the supervised positions, `[b*t]`.
    ce_rows: DeviceBuffer,
    /// The LM targets, [`IGNORE`] for unsupervised positions.
    targets: DeviceBuffer,
    /// The tied head's logits `[b*t, vocab]`. Owned by the tape rather than
    /// shared with the inference path's own buffer, so nothing a caller does
    /// between [`model::Model::forward`] and [`model::Model::backward`] can
    /// change the logits the backward's cross-entropy gradient is taken of.
    logits: DeviceBuffer,
}

/// Backward scratch and the gradient of every tape boundary (training builds
/// only). Every buffer here is written by exactly one dispatch except
/// [`Back::d_inject`], which accumulates the injection's share over the
/// recurrent steps and is cleared at the start of each backward.
struct Back {
    /// Gradient of each prelude block's INPUT: entry `i` is the gradient of
    /// `Tape::embeds[1 + i]`.
    d_embeds: Vec<DeviceBuffer>,
    /// Gradient of the prelude output every recurrent step re-injects.
    d_inject: DeviceBuffer,
    /// Gradient of the embedding table's scaled output (the `emb_scale` row
    /// scale's own input).
    d_embed: DeviceBuffer,
    core: Vec<DeviceBuffer>,
    latent: DeviceBuffer,
    coda: Vec<DeviceBuffer>,
    dlogits: DeviceBuffer,
    /// The adapter's input gradient, `[b*t, 2*width]`: its two halves are the
    /// recurrent state's and the injection's.
    d_concat: DeviceBuffer,
    d_half: DeviceBuffer,
    dqkv: DeviceBuffer,
    d_xn1: DeviceBuffer,
    d_xin: DeviceBuffer,
    d_xn2: DeviceBuffer,
    d_xn2_total: DeviceBuffer,
    d_xn3: DeviceBuffer,
    d_gate: DeviceBuffer,
    d_up: DeviceBuffer,
    d_fc: DeviceBuffer,
    d_mh: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_tmp2: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    /// Per-row reciprocal RMS the norm-gain gradients reduce over.
    rms_inv: DeviceBuffer,
    /// Receives each block's recomputed output during the backward (never read).
    recomputed: DeviceBuffer,
    /// The CE-gradient uniform `[rows, vocab, ignore, count]`.
    ce_grad_uniform: DeviceBuffer,
}

/// The layer index a block of stage `stage` (`prelude`, `core` or `coda`) has in the window: the prelude blocks
/// first, then the core blocks, then the coda, each block its own layer.
fn block_layer(cfg: &HuginnConfig, stage: &str, i: usize) -> usize {
    match stage {
        "prelude" => i,
        "core" => cfg.n_prelude as usize + i,
        _ => (cfg.n_prelude + cfg.n_recurrent) as usize + i,
    }
}

/// The layer a tensor belongs to in the window (`prelude.{i}.*`, `core.{i}.*` and `coda.{i}.*` by [`block_layer`]), or
/// `None` for the embedding, the adapter and the final norm, which stay in the store.
fn tensor_layer(cfg: &HuginnConfig, name: &str) -> Option<u32> {
    let (stage, rest) = name.split_once('.')?;
    let i: usize = rest.split_once('.')?.0.parse().ok()?;
    matches!(stage, "prelude" | "core" | "coda").then(|| block_layer(cfg, stage, i) as u32)
}

pub struct Huginn {
    pub gpu: Gpu,
    pub cfg: HuginnConfig,
    pub ps: ParamStore,
    b: u32,
    t: u32,
    train: bool,
    /// Recurrent iterations the graph runs, fixed at construction.
    steps: u32,
    /// Supervised rows the loss is averaged over.
    count: Cell<f32>,
    tokens: DeviceBuffer,
    /// Embedding buffer pair: the prelude rewrites the embeddings in place.
    embeds: [DeviceBuffer; 2],
    /// The caller's starting state, kept apart from the residual buffers: the
    /// recurrence consumes `h[0]` on its first block, so a second `forward`
    /// has to restore this rather than start from a block output.
    init_state: DeviceBuffer,
    /// Residual-stream pair the recurrence ping-pongs through.
    h: [DeviceBuffer; 2],
    /// Residual pair for the coda.
    coda: [DeviceBuffer; 2],
    /// The normed state after the last recurrent step, before the coda.
    latent: DeviceBuffer,
    logits: DeviceBuffer,
    scale: DeviceBuffer,
    /// Buffer index holding the last recurrence's state, for `read_latent`.
    state_buf: Cell<usize>,
    scratch: Scratch,
    tape: Option<Tape>,
    back: Option<Back>,
    /// Training forward: embed → prelude → recurrence → coda → head → CE.
    fwd_steps: LayerTape,
    /// Training backward: head → coda → recurrence → prelude → embedding.
    bwd_steps: LayerTape,
    /// The block tensors held in a device window of layers, when the run streams them
    /// ([`model::Model::set_resident`]); `None` keeps every block in the store. Every core block runs once in each
    /// recurrent step, so its gradient is carried between its visits and it is updated once the backward is complete.
    streamed: Option<StreamedStore>,
    /// A streamed build runs its backward inside the update: set by `backward`, taken by `adamw_step`.
    backward_pending: Cell<bool>,
    opt: Option<optim::Optim>,
}

impl Huginn {
    /// Frozen-weight model over a checkpoint source: inference, parity and
    /// generation.
    pub fn new(cfg: HuginnConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Huginn {
        Huginn::new_impl(cfg, b, t, src, false)
    }

    /// [`Huginn::new`] under its explicit name (what `model::Model` calls it).
    pub fn new_inference(cfg: HuginnConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Huginn {
        Huginn::new_impl(cfg, b, t, src, false)
    }

    /// Trainable model: weights, gradients and AdamW moments for every
    /// parameter, the tape of every block application, and the backward graph.
    pub fn new_train(cfg: HuginnConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Huginn {
        Huginn::new_impl(cfg, b, t, src, true)
    }

    fn new_impl(cfg: HuginnConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool) -> Huginn {
        Huginn::build(cfg, b, t, src, train, None).unwrap_or_else(|e| panic!("huginn: {e}"))
    }

    /// A training build whose block tensors (prelude, core and coda; see [`block_layer`]) stream through a device
    /// window under `resident` from the start (see [`model::streamed`]): they are read from `src` into the window's
    /// host state and never all reach the device, so a model larger than it trains. The embedding (the tied head), the
    /// adapter and the final norm load as usual.
    pub fn new_resident_on(cfg: HuginnConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, resident: &model::resident_ft::ResidentFtConfig) -> Result<Huginn, String> {
        Huginn::build(cfg, b, t, src, true, Some(resident))
    }

    fn build(cfg: HuginnConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool, resident: Option<&model::resident_ft::ResidentFtConfig>) -> Result<Huginn, String> {
        cfg.validate().expect("huginn config");
        assert!(t <= cfg.block_size, "sequence {t} exceeds positional capacity {}", cfg.block_size);
        assert!(b > 0 && t > 0);
        if resident.is_some() && !train {
            return Err("huginn: only a training build streams its blocks".into());
        }
        let gpu = Gpu::new(&optim::with_moment_kernels(PIPELINES));
        let model_cfg = cfg.clone();
        let layer_of = |name: &str| tensor_layer(&model_cfg, name);
        let roles = cfg
            .param_list()
            .into_iter()
            .map(|(n, c)| {
                let role = if !train {
                    paramstore::Role::Frozen
                } else if resident.is_some() && layer_of(&n).is_some() {
                    paramstore::Role::Streamed
                } else {
                    paramstore::Role::Trainable
                };
                (n, c, role)
            })
            .collect();
        let mut ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let streamed = resident.map(|r| StreamedStore::attach(&gpu, &mut ps, r.clone(), src, &layer_of)).transpose()?;
        let opt = train.then(|| optim::Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF));
        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let nh = cfg.n_heads as u64;
        let st = |x: u64| gpu.storage(x);
        let tokens = gpu.buffer("huginn_tokens", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let scale = gpu.buffer("huginn_emb_scale", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        gpu.write_f32(&scale, &vec![cfg.emb_scale; n as usize]);
        let logits = st(n * cfg.vocab as u64);
        let embeds = [st(n * d), st(n * d)];
        let init_state = st(n * d);
        let h = [st(n * d), st(n * d)];
        let coda = [st(n * d), st(n * d)];
        let latent = st(n * d);
        let bht2 = (b as u64) * nh * (t as u64) * (t as u64);
        assert!(bht2 <= u32::MAX as u64, "materialized attention exceeds u32 index space");
        let scratch = Scratch {
            xn1: st(n * d),
            xn2: st(n * d),
            xn3: st(n * d),
            qkv: st(n * 3 * d),
            q: st(n * d),
            k: st(n * d),
            v: st(n * d),
            ctx: st(n * d),
            proj: st(n * d),
            tmp: st(n * d),
            fc: st(n * 2 * ff),
            gate: st(n * ff),
            up: st(n * ff),
            mh: st(n * ff),
            mout: st(n * d),
            res: st(n * d),
            concat: st(n * 2 * d),
            scores: st(nh * n * n),
            probs: st(nh * n * n),
        };
        // The inference forward runs at least one recurrent step; the training
        // graph is recorded for the same count, fixed here because it sizes the
        // tape.
        let steps = cfg.mean_recurrence.max(1);
        let per_step = cfg.n_recurrent as usize + 1;
        let tape = train.then(|| Tape {
            embeds: (0..cfg.n_prelude as usize + 2).map(|_| st(n * d)).collect(),
            core: Residuals::new((0..steps as usize * per_step).map(|_| st(n * d)).collect(), 0..steps as usize * per_step),
            latent: st(n * d),
            coda: (0..cfg.n_coda as usize + 1).map(|_| st(n * d)).collect(),
            ce_rows: st(n),
            targets: gpu.buffer("huginn_targets", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST),
            logits: st(n * cfg.vocab as u64),
        });
        let back = train.then(|| Back {
            d_embeds: (0..cfg.n_prelude as usize).map(|_| st(n * d)).collect(),
            d_inject: st(n * d),
            d_embed: st(n * d),
            core: (0..steps as usize * per_step).map(|_| st(n * d)).collect(),
            latent: st(n * d),
            coda: (0..cfg.n_coda as usize + 1).map(|_| st(n * d)).collect(),
            dlogits: st(n * cfg.vocab as u64),
            d_concat: st(n * 2 * d),
            d_half: st(n * d),
            dqkv: st(n * 3 * d),
            d_xn1: st(n * d),
            d_xin: st(n * d),
            d_xn2: st(n * d),
            d_xn2_total: st(n * d),
            d_xn3: st(n * d),
            d_gate: st(n * ff),
            d_up: st(n * ff),
            d_fc: st(n * 2 * ff),
            d_mh: st(n * ff),
            d_tmp: st(n * d),
            d_tmp2: st(n * d),
            d_ctx: st(n * d),
            d_scores: st(nh * n * n),
            d_q: st(n * d),
            d_k: st(n * d),
            d_v: st(n * d),
            rms_inv: st(n),
            recomputed: st(n * d),
            ce_grad_uniform: gpu.uniform_dynamic(4),
        });
        let mut m = Huginn {
            gpu,
            cfg,
            ps,
            b,
            t,
            train,
            steps,
            count: Cell::new(1.0),
            tokens,
            embeds,
            init_state,
            h,
            coda,
            latent,
            logits,
            scale,
            state_buf: Cell::new(0),
            scratch,
            tape,
            back,
            fwd_steps: LayerTape::default(),
            bwd_steps: LayerTape::default(),
            streamed,
            backward_pending: Cell::new(false),
            opt,
        };
        if train {
            m.fwd_steps = m.training_forward_steps();
            m.bwd_steps = m.training_backward_steps();
        }
        Ok(m)
    }

    pub fn batch(&self) -> u32 {
        self.b
    }

    pub fn seq(&self) -> u32 {
        self.t
    }

    /// Recurrent iterations the model's graph runs.
    pub fn steps(&self) -> u32 {
        self.steps
    }

    pub fn is_training(&self) -> bool {
        self.train
    }

    pub fn set_tokens(&self, tokens: &[u32]) {
        assert_eq!(tokens.len(), (self.b * self.t) as usize);
        self.gpu.write(&self.tokens, bytemuck::cast_slice(tokens));
    }

    /// Upload a causal-LM batch: `targets[r]` is the token position `r` must
    /// predict, [`IGNORE`] where the position is unsupervised.
    pub fn set_tokens_targets(&self, tokens: &[u32], targets: &[u32]) {
        let tape = self.tape.as_ref().expect("set_tokens_targets on an inference-only build");
        let rows = (self.b * self.t) as usize;
        assert_eq!(tokens.len(), rows, "tokens must be b*t");
        assert_eq!(targets.len(), rows, "targets must be b*t");
        self.gpu.write(&self.tokens, bytemuck::cast_slice(tokens));
        self.gpu.write(&tape.targets, bytemuck::cast_slice(targets));
        self.count.set(targets.iter().filter(|v| **v != IGNORE).count().max(1) as f32);
    }

    /// The starting latent state of the recurrence, `[b*t, width]` row-major.
    pub fn set_state(&self, state: &[f32]) {
        assert_eq!(state.len(), (self.b * self.t) as usize * self.cfg.d_model as usize);
        self.gpu.write_f32(&self.init_state, state);
    }

    fn rows(&self) -> u32 {
        self.b * self.t
    }

    /// `out = x @ wᵀ` for `x:[m, k]`, `w:[n, k]`, `out:[m, n]`.
    fn linear(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, w: &str, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
        steps.push(self.gpu.step(MATMUL, &[x, self.ps.w(w), out], &[m, k, n], m * n));
    }

    /// Copy `cnt` columns starting at `src_off` of each of `rows` rows from a
    /// `src_stride`-wide matrix into a `dst_stride`-wide one.
    fn copy_cols(&self, steps: &mut Vec<Step>, src: &DeviceBuffer, src_stride: u32, src_off: u32, dst: &DeviceBuffer, dst_stride: u32, dst_off: u32, cnt: u32) {
        let rows = self.rows();
        steps.push(self.gpu.step(COPY_COLS, &[src, dst], &[rows, cnt, src_stride, src_off, dst_stride, dst_off], rows * cnt));
    }

    fn rope(&self, steps: &mut Vec<Step>, buf: &DeviceBuffer, n_heads: u32, head_dim: u32) {
        let rows = self.rows();
        steps.push(self.gpu.step(
            ROPE_TRAIN,
            &[buf],
            &[rows, n_heads, head_dim, n_heads * head_dim, 0, self.t, gpu_core::f(self.cfg.rope_theta)],
            rows * n_heads * (head_dim / 2),
        ));
    }

    /// One sandwich block: `name` is `prelude.{i}` / `core.{i}` / `coda.{i}`.
    fn block_steps(&self, steps: &mut Vec<Step>, name: &str, x_in: &DeviceBuffer, x_out: &DeviceBuffer) {
        let cfg = &self.cfg;
        let k = kernel_ids();
        let n = self.rows();
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (nh, hd) = (cfg.n_heads, cfg.head_dim);
        let s = &self.scratch;
        let w = |leaf: &str| self.ps.w(&format!("{name}.{leaf}"));
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nh, head_dim: hd };

        steps.push(block::rmsnorm_fwd(&self.gpu, &k, x_in, w("ln1.weight"), &s.xn1, d, n, cfg.norm_eps));
        self.linear(steps, &s.xn1, &format!("{name}.attn.wqkv.weight"), &s.qkv, n, d, 3 * d);
        self.copy_cols(steps, &s.qkv, 3 * d, 0, &s.q, d, 0, d);
        self.copy_cols(steps, &s.qkv, 3 * d, d, &s.k, d, 0, d);
        self.copy_cols(steps, &s.qkv, 3 * d, 2 * d, &s.v, d, 0, d);
        // Per-head Q/K bias (the checkpoint's `[2, 1, heads, dim]` tensor,
        // imported as its two halves).
        for (buf, bias) in [(&s.q, "attn.q_bias"), (&s.k, "attn.k_bias")] {
            steps.push(self.gpu.step(BIAS_ADD, &[buf, w(bias)], &[n, d], n * d));
        }
        self.rope(steps, &s.q, nh, hd);
        self.rope(steps, &s.k, nh, hd);
        steps.extend(block::gqa_fwd(&self.gpu, &k, &ga, &s.q, &s.k, &s.v, &s.scores, &s.probs, &s.ctx));
        self.linear(steps, &s.ctx, &format!("{name}.attn.wo.weight"), &s.proj, n, d, d);
        steps.push(self.gpu.step(ADD2, &[&s.proj, x_in, &s.tmp], &[n * d], n * d));
        steps.push(block::rmsnorm_fwd(&self.gpu, &k, &s.tmp, w("ln2.weight"), &s.xn2, d, n, cfg.norm_eps));
        // Fully connected: norm_3, the fused gate/up FC split into its halves,
        // SiLU(gate) * up, projection, residual, norm_4.
        steps.push(block::rmsnorm_fwd(&self.gpu, &k, &s.xn2, w("ln3.weight"), &s.xn3, d, n, cfg.norm_eps));
        self.linear(steps, &s.xn3, &format!("{name}.mlp.fc.weight"), &s.fc, n, d, 2 * ff);
        self.copy_cols(steps, &s.fc, 2 * ff, 0, &s.gate, ff, 0, ff);
        self.copy_cols(steps, &s.fc, 2 * ff, ff, &s.up, ff, 0, ff);
        steps.push(block::swiglu_fwd(&self.gpu, &k, &s.gate, &s.up, &s.mh, n * ff));
        self.linear(steps, &s.mh, &format!("{name}.mlp.proj.weight"), &s.mout, n, ff, d);
        steps.push(self.gpu.step(ADD2, &[&s.mout, &s.xn2, &s.res], &[n * d], n * d));
        steps.push(block::rmsnorm_fwd(&self.gpu, &k, &s.res, w("ln4.weight"), x_out, d, n, cfg.norm_eps));
    }

    /// Record embed → prelude → recurrence → norm → coda → norm → head.
    fn record(&self, steps: u32) -> Vec<Step> {
        let cfg = &self.cfg;
        let k = kernel_ids();
        let n = self.rows();
        let d = cfg.d_model;
        let mut g: Vec<Step> = Vec::new();

        g.push(self.gpu.step(EMBED, &[&self.tokens, self.ps.w("tok.weight"), &self.embeds[0]], &[d, n], n * d));
        g.push(self.gpu.step(SCALE_ROW, &[&self.embeds[0], &self.scale, &self.embeds[1]], &[n * d, d], n * d));
        let mut cur = 1usize;
        for i in 0..cfg.n_prelude {
            self.block_steps(&mut g, &format!("prelude.{i}"), &self.embeds[cur], &self.embeds[1 - cur]);
            cur = 1 - cur;
        }
        // The prelude's output, not the raw embedding, is what every recurrent
        // step re-injects.
        let inject = &self.embeds[cur];

        // Recurrence. `state` holds the running latent; the injection lands in
        // the other residual buffer, which block 0 consumes, so no state copy
        // is ever needed after the initial one.
        let mut state = 0usize;
        self.copy_cols(&mut g, &self.init_state, d, 0, &self.h[0], d, 0, d);
        for _ in 0..steps.max(1) {
            let mid = 1 - state;
            self.copy_cols(&mut g, &self.h[state], d, 0, &self.scratch.concat, 2 * d, 0, d);
            self.copy_cols(&mut g, inject, d, 0, &self.scratch.concat, 2 * d, d, d);
            self.linear(&mut g, &self.scratch.concat, "adapter.weight", &self.h[mid], n, 2 * d, d);
            self.block_steps(&mut g, "core.0", &self.h[mid], &self.h[state]);
            for i in 1..cfg.n_recurrent {
                self.block_steps(&mut g, &format!("core.{i}"), &self.h[state], &self.h[1 - state]);
                state = 1 - state;
            }
        }
        self.state_buf.set(state);

        // Final norm, then the coda from the normed state, then the norm again.
        g.push(block::rmsnorm_fwd(&self.gpu, &k, &self.h[state], self.ps.w("ln_f.weight"), &self.latent, d, n, cfg.norm_eps));
        let mut cur = 0usize;
        for i in 0..cfg.n_coda {
            let x_in = if i == 0 { &self.latent } else { &self.coda[cur] };
            self.block_steps(&mut g, &format!("coda.{i}"), x_in, &self.coda[1 - cur]);
            cur = 1 - cur;
        }
        g.push(block::rmsnorm_fwd(&self.gpu, &k, &self.coda[cur], self.ps.w("ln_f.weight"), &self.coda[1 - cur], d, n, cfg.norm_eps));
        cur = 1 - cur;
        self.linear(&mut g, &self.coda[cur], "tok.weight", &self.logits, n, d, cfg.vocab);
        g
    }

    /// The training forward: the inference graph's own block composition, but
    /// every block application writes its output to its own tape slot, and the
    /// graph ends in the tied head and the masked cross-entropy rows.
    /// One segment per block application (the layer indices are [`block_layer`]'s): a core block appears once
    /// in every recurrent step, the prelude and coda blocks once.
    fn training_forward_steps(&self) -> LayerTape {
        let cfg = &self.cfg;
        let k = kernel_ids();
        let n = self.rows();
        let d = cfg.d_model;
        let tape = self.tape.as_ref().expect("training graph on an inference build");
        let mut passes = LayerTape::default();
        let mut g: Vec<Step> = Vec::new();

        g.push(self.gpu.step(EMBED, &[&self.tokens, self.ps.w("tok.weight"), &tape.embeds[0]], &[d, n], n * d));
        g.push(self.gpu.step(SCALE_ROW, &[&tape.embeds[0], &self.scale, &tape.embeds[1]], &[n * d, d], n * d));
        for i in 0..cfg.n_prelude as usize {
            passes.segment(&mut g, None);
            self.block_steps(&mut g, &format!("prelude.{i}"), &tape.embeds[1 + i], &tape.embeds[2 + i]);
            passes.segment(&mut g, Some(block_layer(&self.cfg, "prelude", i)));
        }
        // The prelude's output, not the raw embedding, is what every recurrent
        // step re-injects - so its gradient is the sum over all of them.
        let inject = &tape.embeds[cfg.n_prelude as usize + 1];

        let per_step = cfg.n_recurrent as usize + 1;
        let mut state = &self.init_state;
        for s in 0..self.steps as usize {
            let b = s * per_step;
            self.copy_cols(&mut g, state, d, 0, &self.scratch.concat, 2 * d, 0, d);
            self.copy_cols(&mut g, inject, d, 0, &self.scratch.concat, 2 * d, d, d);
            self.linear(&mut g, &self.scratch.concat, "adapter.weight", &tape.core[b], n, 2 * d, d);
            tape.core.round_trip(&self.gpu, &mut g, b);
            let mut prev = &tape.core[b];
            for i in 0..cfg.n_recurrent as usize {
                let out = &tape.core[b + 1 + i];
                passes.segment(&mut g, None);
                self.block_steps(&mut g, &format!("core.{i}"), prev, out);
                tape.core.round_trip(&self.gpu, &mut g, b + 1 + i);
                passes.segment(&mut g, Some(block_layer(&self.cfg, "core", i)));
                prev = out;
            }
            state = prev;
        }

        g.push(block::rmsnorm_fwd(&self.gpu, &k, state, self.ps.w("ln_f.weight"), &tape.latent, d, n, cfg.norm_eps));
        let n_coda = cfg.n_coda as usize;
        let mut cur = &tape.latent;
        for i in 0..n_coda {
            passes.segment(&mut g, None);
            self.block_steps(&mut g, &format!("coda.{i}"), cur, &tape.coda[i]);
            passes.segment(&mut g, Some(block_layer(&self.cfg, "coda", i)));
            cur = &tape.coda[i];
        }
        g.push(block::rmsnorm_fwd(&self.gpu, &k, cur, self.ps.w("ln_f.weight"), &tape.coda[n_coda], d, n, cfg.norm_eps));
        self.linear(&mut g, &tape.coda[n_coda], "tok.weight", &tape.logits, n, d, cfg.vocab);
        g.push(self.gpu.step(CE_VALUE, &[&tape.logits, &tape.targets, &tape.ce_rows], &[n, cfg.vocab, IGNORE], n));
        passes.segment(&mut g, None);
        passes
    }

    /// The backward graph. Every block application is differentiated where the
    /// forward ran it, in reverse: the head, the final norm, the coda, the
    /// recurrence (each step's blocks then its adapter), the prelude, and
    /// finally the embedding scatter. A shared parameter's gradient buffer is
    /// never assigned - `matmul_dw`, `rmsnorm_dw`, `bias_grad` and `emb_bwd`
    /// all accumulate - so the gradient is the sum over every application.
    fn training_backward_steps(&self) -> LayerTape {
        let cfg = &self.cfg;
        let k = kernel_ids();
        let n = self.rows();
        let (d, v) = (cfg.d_model, cfg.vocab);
        let tape = self.tape.as_ref().expect("training graph on an inference build");
        let g = self.back.as_ref().expect("training graph on an inference build");
        let n_coda = cfg.n_coda as usize;
        let per_step = cfg.n_recurrent as usize + 1;
        let mut passes = LayerTape::default();
        // `d_inject` is the one boundary gradient with several producers (every recurrent step's adapter); everything
        // else is written by one. Each submission of the pass, however many an update makes, starts it from zero.
        passes.clear_on_submit(&g.d_inject);
        let mut steps: Vec<Step> = Vec::new();

        // Tied head: logits = normed @ tok.weightᵀ.
        steps.push(self.gpu.step_buf(CE_GRAD, &g.ce_grad_uniform, &[&tape.logits, &tape.targets, &g.dlogits], n * v));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.dlogits, self.ps.w("tok.weight"), &g.coda[n_coda]], &[n, d, v, 0], n * d));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.dlogits, &tape.coda[n_coda], self.ps.g("tok.weight")], &[n, d, v], v * d));

        // The second use of `ln_f`, then the coda in reverse.
        steps.extend(block::rmsnorm_bwd(
            &self.gpu,
            &k,
            &tape.coda[n_coda - 1],
            self.ps.w("ln_f.weight"),
            &g.coda[n_coda],
            &g.coda[n_coda - 1],
            &g.rms_inv,
            Some(self.ps.g("ln_f.weight")),
            d,
            n,
            cfg.norm_eps,
        ));
        for i in (0..n_coda).rev() {
            let x_in = if i == 0 { &tape.latent } else { &tape.coda[i - 1] };
            let dest = if i == 0 { &g.latent } else { &g.coda[i - 1] };
            passes.segment(&mut steps, None);
            self.block_bwd(&mut steps, &format!("coda.{i}"), x_in, &g.coda[i], dest);
            passes.segment(&mut steps, Some(block_layer(&self.cfg, "coda", i)));
        }
        // The first use of `ln_f`: its input is the last recurrent block's
        // output, so its input gradient lands on that block's output gradient.
        let last_core = self.steps as usize * per_step - 1;
        tape.core.restore(&self.gpu, &mut steps, last_core);
        steps.extend(block::rmsnorm_bwd(
            &self.gpu,
            &k,
            &tape.core[last_core],
            self.ps.w("ln_f.weight"),
            &g.latent,
            &g.core[last_core],
            &g.rms_inv,
            Some(self.ps.g("ln_f.weight")),
            d,
            n,
            cfg.norm_eps,
        ));

        // The recurrence in reverse. Each step's state gradient is written by
        // the step after it (or by `ln_f` above), so the blocks see it complete.
        let inject = &tape.embeds[cfg.n_prelude as usize + 1];
        for s in (0..self.steps as usize).rev() {
            let b = s * per_step;
            for i in (0..cfg.n_recurrent as usize).rev() {
                passes.segment(&mut steps, None);
                tape.core.restore(&self.gpu, &mut steps, b + i);
                self.block_bwd(&mut steps, &format!("core.{i}"), &tape.core[b + i], &g.core[b + i + 1], &g.core[b + i]);
                passes.segment(&mut steps, Some(block_layer(&self.cfg, "core", i)));
            }
            // Adapter: out = concat(state, inject) @ adapter.weightᵀ.
            if s > 0 {
                tape.core.restore(&self.gpu, &mut steps, b - 1);
            }
            let h_prev = if s == 0 { &self.init_state } else { &tape.core[b - 1] };
            self.copy_cols(&mut steps, h_prev, d, 0, &self.scratch.concat, 2 * d, 0, d);
            self.copy_cols(&mut steps, inject, d, 0, &self.scratch.concat, 2 * d, d, d);
            steps.push(self.gpu.step(MATMUL_DW, &[&g.core[b], &self.scratch.concat, self.ps.g("adapter.weight")], &[n, 2 * d, d], d * 2 * d));
            steps.push(self.gpu.step(MATMUL_DX, &[&g.core[b], self.ps.w("adapter.weight"), &g.d_concat], &[n, 2 * d, d, 0], n * 2 * d));
            // The state half only reaches a parameter through the step before
            // it - step 0's state is the caller's, which has no gradient.
            if s > 0 {
                self.copy_cols(&mut steps, &g.d_concat, 2 * d, 0, &g.core[b - 1], d, 0, d);
            }
            self.copy_cols(&mut steps, &g.d_concat, 2 * d, d, &g.d_half, d, 0, d);
            steps.push(self.gpu.step(AXPY, &[&g.d_inject, &g.d_half], &[n * d, gpu_core::f(1.0)], n * d));
        }

        // The prelude in reverse, then the row scale and the scatter into the
        // tied table (which already holds the head's contribution).
        for i in (0..cfg.n_prelude as usize).rev() {
            let d_out = if i + 1 == cfg.n_prelude as usize { &g.d_inject } else { &g.d_embeds[i + 1] };
            passes.segment(&mut steps, None);
            self.block_bwd(&mut steps, &format!("prelude.{i}"), &tape.embeds[1 + i], d_out, &g.d_embeds[i]);
            passes.segment(&mut steps, Some(block_layer(&self.cfg, "prelude", i)));
        }
        steps.push(self.gpu.step(SCALE_ROW, &[&g.d_embeds[0], &self.scale, &g.d_embed], &[n * d, d], n * d));
        steps.push(self.gpu.step(EMB_BWD, &[&self.tokens, &g.d_embed, self.ps.g("tok.weight")], &[n, d, v], v * d));
        passes.segment(&mut steps, None);
        passes
    }

    /// Submit a training forward; a layer-streamed build loads each block into its window slot at every application.
    fn submit_forward(&self, tape: &LayerTape) {
        match &self.streamed {
            Some(stream) => stream.forward(&self.gpu, tape).unwrap_or_else(|e| panic!("huginn layer-streamed forward: {e}")),
            None => tape.submit(&self.gpu),
        }
    }

    /// Whether the blocks stream through a device window (the inference forward and incremental decode then read
    /// blocks it does not hold).
    pub(crate) fn is_streamed(&self) -> bool {
        self.streamed.is_some()
    }

    fn holds_streamed(&self, name: &str) -> bool {
        self.streamed.as_ref().is_some_and(|s| s.holds(name))
    }

    /// One block's backward. `d_out` is the gradient of the block's output and
    /// `dest` receives the gradient of its input; `x_in` is the boundary
    /// residual the block consumed, from which its shared internal scratch is
    /// rebuilt before it is differentiated.
    fn block_bwd(&self, steps: &mut Vec<Step>, name: &str, x_in: &DeviceBuffer, d_out: &DeviceBuffer, dest: &DeviceBuffer) {
        let cfg = &self.cfg;
        let k = kernel_ids();
        let n = self.rows();
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (nh, hd) = (cfg.n_heads, cfg.head_dim);
        let eps = cfg.norm_eps;
        let s = &self.scratch;
        let g = self.back.as_ref().expect("training graph on an inference build");
        let w = |leaf: &str| format!("{name}.{leaf}");
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nh, head_dim: hd };

        // Rebuild the block's internals exactly as the forward ran them.
        self.block_steps(steps, name, x_in, &g.recomputed);

        // x_out = norm_4(res), res = mout + xn2.
        steps.extend(block::rmsnorm_bwd(&self.gpu, &k, &s.res, self.ps.w(&w("ln4.weight")), d_out, &g.d_tmp2, &g.rms_inv, Some(self.ps.g(&w("ln4.weight"))), d, n, eps));
        // mout = mh @ mlp.proj.weightᵀ; mh = SiLU(gate) * up.
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_tmp2, &s.mh, self.ps.g(&w("mlp.proj.weight"))], &[n, ff, d], d * ff));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_tmp2, self.ps.w(&w("mlp.proj.weight")), &g.d_mh], &[n, ff, d, 0], n * ff));
        steps.extend(block::swiglu_bwd(&self.gpu, &k, &s.gate, &s.up, &g.d_mh, &g.d_gate, &g.d_up, n * ff));
        // The fused gate/up FC's two column halves, reassembled into one row.
        self.copy_cols(steps, &g.d_gate, ff, 0, &g.d_fc, 2 * ff, 0, ff);
        self.copy_cols(steps, &g.d_up, ff, 0, &g.d_fc, 2 * ff, ff, ff);
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_fc, &s.xn3, self.ps.g(&w("mlp.fc.weight"))], &[n, d, 2 * ff], 2 * ff * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_fc, self.ps.w(&w("mlp.fc.weight")), &g.d_xn3], &[n, d, 2 * ff, 0], n * d));
        // xn3 = norm_3(xn2); xn2 feeds both the MLP residual and norm_3.
        steps.extend(block::rmsnorm_bwd(&self.gpu, &k, &s.xn2, self.ps.w(&w("ln3.weight")), &g.d_xn3, &g.d_xn2, &g.rms_inv, Some(self.ps.g(&w("ln3.weight"))), d, n, eps));
        steps.push(self.gpu.step(ADD2, &[&g.d_tmp2, &g.d_xn2, &g.d_xn2_total], &[n * d], n * d));
        // xn2 = norm_2(tmp), tmp = proj + x_in.
        steps.extend(block::rmsnorm_bwd(&self.gpu, &k, &s.tmp, self.ps.w(&w("ln2.weight")), &g.d_xn2_total, &g.d_tmp, &g.rms_inv, Some(self.ps.g(&w("ln2.weight"))), d, n, eps));
        // proj = ctx @ attn.wo.weightᵀ, and the attention below it.
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_tmp, &s.ctx, self.ps.g(&w("attn.wo.weight"))], &[n, d, d], d * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_tmp, self.ps.w(&w("attn.wo.weight")), &g.d_ctx], &[n, d, d, 0], n * d));
        steps.extend(block::gqa_bwd(&self.gpu, &k, &ga, &s.q, &s.k, &s.v, &s.probs, &g.d_ctx, &g.d_scores, &g.d_q, &g.d_k, &g.d_v));
        steps.push(block::rope_bwd(&self.gpu, &k, &g.d_q, n, nh, hd, nh * hd, self.t, cfg.rope_theta));
        steps.push(block::rope_bwd(&self.gpu, &k, &g.d_k, n, nh, hd, nh * hd, self.t, cfg.rope_theta));
        // The per-head Q/K bias is added to every row, so its gradient is the
        // column sum of the post-RoPE gradient. Both pass it through unchanged.
        steps.push(self.gpu.step(BIAS_GRAD, &[&g.d_q, self.ps.g(&w("attn.q_bias"))], &[n, d], d));
        steps.push(self.gpu.step(BIAS_GRAD, &[&g.d_k, self.ps.g(&w("attn.k_bias"))], &[n, d], d));
        // The fused QKV projection's three column ranges.
        self.copy_cols(steps, &g.d_q, d, 0, &g.dqkv, 3 * d, 0, d);
        self.copy_cols(steps, &g.d_k, d, 0, &g.dqkv, 3 * d, d, d);
        self.copy_cols(steps, &g.d_v, d, 0, &g.dqkv, 3 * d, 2 * d, d);
        steps.push(self.gpu.step(MATMUL_DW, &[&g.dqkv, &s.xn1, self.ps.g(&w("attn.wqkv.weight"))], &[n, d, 3 * d], 3 * d * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.dqkv, self.ps.w(&w("attn.wqkv.weight")), &g.d_xn1], &[n, d, 3 * d, 0], n * d));
        // xn1 = norm_1(x_in): the input gradient is the attention path plus the
        // residual that bypassed the whole attention half.
        steps.extend(block::rmsnorm_bwd(&self.gpu, &k, x_in, self.ps.w(&w("ln1.weight")), &g.d_xn1, &g.d_xin, &g.rms_inv, Some(self.ps.g(&w("ln1.weight"))), d, n, eps));
        steps.push(self.gpu.step(ADD2, &[&g.d_tmp, &g.d_xin, dest], &[n * d], n * d));
    }

    /// Logits after `steps` recurrent iterations, `[vocab]` of the last token.
    pub fn forward(&self, steps: u32) -> Vec<f32> {
        self.forward_rows(steps).split_off((self.rows() as usize - 1) * self.cfg.vocab as usize)
    }

    /// Every row's logits, `[b*t, vocab]`.
    pub fn forward_rows(&self, steps: u32) -> Vec<f32> {
        assert!(!self.is_streamed(), "huginn: a layer-streamed build holds a window of its blocks; save it and load the checkpoint to run the inference forward");
        let g = self.record(steps);
        self.gpu.submit(&[], &g);
        self.gpu.read(&self.logits, self.rows() as usize * self.cfg.vocab as usize)
    }

    /// The normed latent state after the last recurrent step (`ln_f` applied,
    /// coda not), `[b*t, width]`.
    pub fn read_latent(&self) -> Vec<f32> {
        self.gpu.read(&self.latent, self.rows() as usize * self.cfg.d_model as usize)
    }

    /// The raw (norm-free) state after the last recurrent step.
    pub fn read_state(&self) -> Vec<f32> {
        self.gpu.read(&self.h[self.state_buf.get()], self.rows() as usize * self.cfg.d_model as usize)
    }
}

impl model::OffDeviceState for Huginn {
    fn commit(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let stream = self.streamed.as_ref().ok_or_else(|| std::io::Error::other("huginn: no streamed state to commit"))?;
        self.gpu.flush();
        stream.commit(&self.gpu, dir, step)
    }

    fn restore(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let stream = self.streamed.as_ref().ok_or_else(|| std::io::Error::other("huginn: no streamed state to restore"))?;
        self.gpu.flush();
        stream.restore(&self.gpu, dir, step)
    }
}

impl model::Model for Huginn {
    fn optimizer_state(&mut self) -> Option<model::OptimizerState<'_>> {
        Some(model::OptimizerState { gpu: &self.gpu, store: &mut self.ps, optim: self.opt.as_ref()? })
    }

    fn optimizer_parts(&self) -> Option<model::OptimizerParts<'_>> {
        Some(model::OptimizerParts { gpu: &self.gpu, store: &self.ps, optim: self.opt.as_ref()? })
    }

    /// Huginn keeps only the residual between block applications and rebuilds each block before its backward (one
    /// scratch set serves every application): it runs recompute, with the recurrence's residuals in f32 or, once
    /// narrowed, in bf16 or FP8 ([`Residuals`]; the prelude's and coda's few stay f32). Keeping every application is
    /// refused by name.
    ///
    /// Stream every block's tensors (prelude, core and coda; each block is a layer of its own, see
    /// [`block_layer`]) through a device window under `cfg` (see [`model::streamed`]): a core block runs once
    /// in every recurrent step, so each application loads it, its gradient is carried on the host between its visits,
    /// and it is updated once the backward is complete. The embedding (the tied head), the adapter and the final norm
    /// stay in the store under its own optimizer.
    fn set_resident(&mut self, cfg: &model::resident_ft::ResidentFtConfig) -> Result<(), String> {
        if !self.train {
            return Err("huginn: an inference build has nothing to stream".into());
        }
        if self.streamed.is_some() {
            return Err("huginn: the blocks already stream".into());
        }
        let model_cfg = self.cfg.clone();
        let layer_of = |name: &str| tensor_layer(&model_cfg, name);
        let stream = StreamedStore::convert(&self.gpu, &mut self.ps, cfg.clone(), &layer_of).map_err(|e| format!("huginn: {e}"))?;
        self.opt.as_ref().expect("a training build has an optimizer").invalidate();
        self.streamed = Some(stream);
        // The recorded passes bound the store's buffers as they were: record them again over the window's slots.
        self.fwd_steps = self.training_forward_steps();
        self.bwd_steps = self.training_backward_steps();
        Ok(())
    }

    fn set_saved_policy(&mut self, policy: SavedPolicy) -> Result<(), String> {
        let held = self.tape.as_ref().map_or(model::saved::SavedFormat::F32, |t| t.core.format());
        let current = SavedPolicy::recompute_with(held);
        if !current.narrows_to(policy) {
            return Err(format!("huginn recomputes each block application from its {held:?} residual in the backward; {policy:?} is not available"));
        }
        if let SavedPolicy::Quantized(format) = policy {
            if policy != current {
                let (rows, width) = (self.rows(), self.cfg.d_model);
                let tape = self.tape.as_mut().ok_or("huginn: an inference build keeps no residuals to narrow")?;
                tape.core.quantize(&self.gpu, format, rows, width).map_err(|e| format!("huginn: {e}"))?;
                self.fwd_steps = self.training_forward_steps();
                self.bwd_steps = self.training_backward_steps();
            }
        }
        Ok(())
    }

    type Config = HuginnConfig;

    fn new(cfg: Self::Config, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>) -> Self {
        Huginn::new_train(cfg, b, t, init)
    }

    fn new_resident(cfg: Self::Config, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>, resident: &model::resident_ft::ResidentFtConfig) -> Result<Self, String> {
        Huginn::new_resident_on(cfg, b, t, init, resident)
    }

    fn new_inference(cfg: Self::Config, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>) -> Self {
        Huginn::new_inference(cfg, b, t, init)
    }

    fn init_weights(cfg: &Self::Config, seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => self.set_tokens_targets(tokens, targets),
            other => panic!("huginn: unsupported batch shape {:?}", std::mem::discriminant(&other)),
        }
    }

    /// The mean cross-entropy over the supervised rows, through the CE kernels:
    /// the scalar [`model::Model::backward`] differentiates.
    fn forward(&self) -> f32 {
        assert!(self.train, "forward on an inference-only build has no objective; read logits instead");
        self.submit_forward(&self.fwd_steps);
        let ce = self.gpu.read(&self.tape.as_ref().expect("training graph on an inference build").ce_rows, self.rows() as usize);
        let loss: f64 = ce.iter().map(|v| *v as f64).sum();
        (loss / self.count.get() as f64) as f32
    }

    fn backward(&self) {
        assert!(self.train, "backward on an inference-only build");
        let (n, v) = (self.rows(), self.cfg.vocab);
        let g = self.back.as_ref().expect("training graph on an inference build");
        self.gpu.write(&g.ce_grad_uniform, &[n, v, IGNORE, gpu_core::f(self.count.get())]);
        if self.streamed.as_ref().is_some_and(StreamedStore::is_layer_streamed) {
            // The weights of a layer-streamed build are not all on the device: the backward runs block by block in
            // `adamw_step`, where the blocks it loads are the ones it updates.
            assert!(!self.backward_pending.replace(true), "huginn layer-streamed full fine-tune: a second backward before the update (gradient accumulation) is not supported; every micro-batch is its own step");
            return;
        }
        self.bwd_steps.submit(&self.gpu);
    }

    fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        let opt = self.opt.as_ref().expect("adamw_step on an inference-only build");
        match &self.streamed {
            Some(stream) => {
                if stream.is_layer_streamed() {
                    assert!(self.backward_pending.replace(false), "huginn layer-streamed full fine-tune: adamw_step without a backward");
                }
                let args = model::resident_ft::StepArgs { step: t, lr, wd, adam, clip, extra_scale };
                stream.adamw_step(&self.gpu, &self.ps, opt, &RevisitingTape(&self.bwd_steps), &args).unwrap_or_else(|e| panic!("huginn layer-streamed update: {e}"));
            }
            None => opt.step(&self.gpu, &self.ps, t, lr, wd, adam, clip, extra_scale),
        }
    }

    fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    fn last_grad_norm(&self) -> Option<f32> {
        self.opt.as_ref().and_then(|o| o.last_grad_norm(&self.gpu, &self.ps))
    }

    fn pin_rounding_seed(&self, seed: u64) {
        if let Some(o) = &self.opt {
            o.set_rounding_seed(seed);
        }
    }

    fn param_names(&self) -> Vec<String> {
        self.cfg.param_list().into_iter().map(|(n, _)| n).collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        match &self.streamed {
            Some(stream) if stream.holds(name) => stream.read_weight(&self.gpu, name).unwrap_or_else(|e| panic!("huginn: reading {name}: {e}")),
            _ => self.ps.read_weight(&self.gpu, name),
        }
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        match &self.streamed {
            Some(stream) if stream.holds(name) => stream.write_weight(&self.gpu, name, data).unwrap_or_else(|e| panic!("huginn: writing {name}: {e}")),
            _ => self.gpu.write_f32(self.ps.w(name), data),
        }
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        assert!(!self.holds_streamed(name), "huginn: {name} streams through a layer window; no gradient outlives its layer");
        self.ps.read_grad(&self.gpu, name)
    }

    fn off_device_state(&self) -> Option<&dyn model::OffDeviceState> {
        self.streamed.as_ref().map(|_| self as &dyn model::OffDeviceState)
    }

    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        assert_eq!(tokens.len(), self.rows() as usize, "huginn: logits_all takes one token per row of the built shape");
        self.set_tokens(tokens);
        Some(self.forward_rows(self.steps))
    }

    fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            self.ps.params.iter().map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name))).collect();
        let card = checkpoint::st::ModelCard::new("brain/huginn", "huginn");
        checkpoint::save_carded(path, self.cfg.to_json(), &tensors, &card);
    }
}

impl model::ModelConfig for HuginnConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        HuginnConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        HuginnConfig::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        HuginnConfig::from_json(v).expect("huginn config")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The slots are hand-numbered against `PIPELINES`; a kernel inserted
    /// without renumbering would dispatch the wrong WGSL silently on the CPU
    /// backend, which has no binding check. Pin every slot the graphs use.
    #[test]
    fn kernel_slots_match_the_pipeline_list() {
        for (slot, want) in [
            (EMBED, "embed"),
            (MATMUL, "matmul"),
            (RMSNORM, "rmsnorm"),
            (ROPE_TRAIN, "rope_train"),
            (GQA_SCORES, "gqa_scores"),
            (ATTN_SOFTMAX, "attn_softmax"),
            (GQA_APPLY, "gqa_apply"),
            (SILU_MUL, "silu_mul"),
            (ADD2, "add2"),
            (BIAS_ADD, "bias_add"),
            (COPY_COLS, "copy_cols"),
            (SCALE_ROW, "scale_row"),
            (ROPE_AT, "rope_at"),
            (KV_APPEND, "kv_append"),
            (ATTN_DECODE_SCORES, "attn_decode_scores"),
            (DECODE_SOFTMAX, "decode_softmax"),
            (ATTN_DECODE_APPLY, "attn_decode_apply"),
            (MATMUL_GEMV, "matmul_gemv"),
            (MATMUL_REG2, "matmul_reg2"),
            (FLASH_GQA_FWD, "flash_attn_causal_gqa"),
            (PAGED_KV_APPEND_BATCHED, "paged_kv_append_batched"),
            (PAGED_DECODE_SCORES_BATCHED, "paged_decode_scores_batched"),
            (DECODE_SOFTMAX_BATCHED, "decode_softmax_batched"),
            (PAGED_DECODE_APPLY_BATCHED, "paged_decode_apply_batched"),
            (CE_VALUE, "ce_value"),
            (CE_GRAD, "ce_grad"),
            (MATMUL_DX, "matmul_dx"),
            (MATMUL_DW, "matmul_dw"),
            (EMB_BWD, "emb_bwd"),
            (ADAMW, "adamw"),
            (GRADNORM_SQ, "gradnorm_sq"),
            (GRAD_SCALE, "grad_scale"),
            (CLIP_COEF, "clip_coef"),
            (GRAD_SCALE_BUF, "grad_scale_buf"),
            (AXPY, "axpy"),
            (BIAS_GRAD, "bias_grad"),
            (RMS_INV, "rms_inv"),
            (RMSNORM_DX, "rmsnorm_dx"),
            (RMSNORM_DW, "rmsnorm_dw"),
            (ROPE_TRAIN_BWD, "rope_train_bwd"),
            (GQA_DSCORES, "gqa_bwd_dscores"),
            (GQA_DV, "gqa_bwd_dv"),
            (GQA_DQ, "gqa_bwd_dq"),
            (GQA_DK, "gqa_bwd_dk"),
            (SILU_DA, "silu_bwd_da"),
            (SILU_DB, "silu_bwd_db"),
        ] {
            assert_eq!(PIPELINES[slot].0, want, "slot {slot} is not {want}");
        }
    }

    /// The training path around the gradient: `zero_grads` → `forward` →
    /// `backward` → `adamw_step`. The gradient check proves each parameter's
    /// analytic gradient on its own; this proves the pieces are wired together
    /// and that the shared core actually moves under them.
    #[test]
    fn a_few_training_steps_lower_the_loss() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let cfg = HuginnConfig::tiny();
        let init = crate::init::init_weights(&cfg, 3);
        let m = Huginn::new_train(cfg, 2, 4, &init);
        let x: Vec<u32> = (0..8).map(|i| (i * 5 + 2) % 29).collect();
        let y: Vec<u32> = (0..8).map(|i| (i * 5 + 3) % 29).collect();
        model::Model::set_batch(&m, model::Batch::Lm { tokens: &x, targets: &y });
        let before = model::Model::read_weight(&m, "core.0.attn.wqkv.weight");
        let first = model::Model::forward(&m);
        for step in 1..=16 {
            model::Model::zero_grads(&m);
            let _ = model::Model::forward(&m);
            model::Model::backward(&m);
            model::Model::adamw_step(&m, step, 2e-2, 0.0, Default::default(), None, 1.0);
        }
        let last = model::Model::forward(&m);
        assert!(last < first, "16 AdamW steps raised the loss: {first} -> {last}");
        assert_ne!(before, model::Model::read_weight(&m, "core.0.attn.wqkv.weight"), "the shared core's weights never moved");
    }
}
