// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.8-27B dense hybrid decoder - forward AND backward, text-only,
//! no LoRA/int8/decode-step/vision-splice/sharding yet (each lands as its
//! own later milestone, mirroring `qwen35moe::model`'s own structure once it
//! does). See `crate::config` for the architecture summary.
//!
//! **Scope, strictly, matching the M5+M6 milestones**: a single
//! whole-sequence prefill forward/backward (`t` must already be a multiple
//! of the derived GDN chunk size - asserted loudly in [`Qwen35::new_on`],
//! see [`gdn_chunk_size`]). Two construction paths, mirroring
//! `qwen35moe::model` exactly: [`Qwen35::new_on`] builds a frozen
//! (`Role::Frozen`), forward-only instance (`backward`/`zero_grads`/
//! `adamw_step` all assert and panic on such an instance); [`Qwen35::new_train_on`]
//! builds a fully trainable (`Role::Trainable` everywhere, full-parameter -
//! no LoRA-specific subset yet, that is M8) instance whose `forward()`
//! additionally saves the activation cache `backward()` reads (the
//! `train_acts` field's own doc has the exact "one forward, one backward,
//! then the cache is gone" contract). The MTP head (M7),
//! LoRA/full finetune beyond this full-parameter path (M8), the vision
//! splice (M9), and int8/decode/sharding land as their own milestones, each
//! mirroring the matching piece of `qwen35moe::model` the way this file's
//! forward/backward already does.
//!
//! ## Layer forward, verified against the installed
//! `transformers.models.qwen3_5` reference (not a secondhand description -
//! see `tools/goldens/qwen35_dump_reference.py`, which hand-replays every
//! step below against the real reference module and reports the manual-vs-
//! real diff per layer)
//!
//! Every layer, regardless of token-mixer type: `xn1 = rmsnorm(res)`, mix
//! (GDN or GQA, below), `xmid = res + mix_out`, `xn2 = rmsnorm(xmid)`, a
//! plain dense SwiGLU MLP (`down(silu(gate(xn2)) * up(xn2))` - every layer,
//! no router, no experts, unlike `qwen35moe`), `res' = xmid + mlp_out`.
//!
//! **Gated DeltaNet** and **GQA** mixer mechanics (chunked delta-rule
//! recurrence, per-head-interleaved doubled `q_proj` gate split, partial
//! M-RoPE) are byte-identical to `qwen35moe`'s own - this file's
//! `layer_gdn_fwd`/`layer_gqa_fwd` are close copies of
//! `qwen35moe::model`'s, differing only in using this crate's own
//! `Qwen35Config` (dense) instead of qwen35moe's (MoE). The `(1+w)` RMSNorm
//! fold happens once, at import time (`crate::import`) - this file's
//! `rmsnorm_fwd` calls assume the stored weight already IS the final
//! multiplier, exactly like every other model in this engine.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use paramstore::{ParamStore, Role};

use audio::conv::{conv1d_bwd, conv1d_fwd, Conv1d, ConvKernels};
use model::block::{
    gqa_bwd, gqa_fwd, kv_expand_bwd, kv_expand_fwd, rmsnorm_bwd, rmsnorm_fwd, rope2d_partial_bwd, rope2d_partial_fwd, swiglu_bwd, Gqa, KernelIds,
};
use model::gdn::{
    gdn_chunk_bwd, gdn_chunk_fwd, gdn_chunk_fwd_train, GdnBwdIds, GdnBwdScratchBufs, GdnIds, GdnScratchBufs, GdnScratchTrainBufs, GdnShape,
};
pub use model::gdn::gdn_chunk_size;
use optim::Optim;

use crate::config::{LayerType, Qwen35Config};

// ---- kernel pipeline (order fixes the indices below) -----------------------
// Forward + backward subset of qwen35moe::model::PIPELINES (no MoE - this
// model has none; no int8/decode/LoRA/splice tiers yet - each lands with
// its own milestone).

pub const PIPELINES: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM), // 0
    ("matmul", kernels::MATMUL), // 1
    ("embed", kernels::EMBED), // 2
    ("sigmoid", kernels::SIGMOID), // 3
    ("silu", kernels::SILU), // 4
    ("silu_mul", kernels::SILU_MUL), // 5
    ("mul", kernels::MUL), // 6
    ("add2", kernels::ADD2), // 7
    ("l2norm_scale", kernels::L2NORM_SCALE), // 8
    ("concat_split", kernels::CONCAT_SPLIT), // 9
    ("nlc_nchw", kernels::NLC_NCHW), // 10
    ("nchw_nlc", kernels::NCHW_NLC), // 11
    ("conv1d", kernels::CONV1D), // 12
    ("gdn_decay_gate", kernels::GDN_DECAY_GATE), // 13
    ("gdn_layout_permute", kernels::GDN_LAYOUT_PERMUTE), // 14
    ("rope2d_partial", kernels::ROPE2D_PARTIAL), // 15
    ("gqa_scores", kernels::GQA_SCORES), // 16
    ("attn_softmax", kernels::ATTN_SOFTMAX), // 17
    ("gqa_apply", kernels::GQA_APPLY), // 18
    ("kv_expand", kernels::KV_EXPAND), // 19
    ("scale_row", kernels::SCALE_ROW), // 20
    ("bmm", kernels::BMM), // 21
    ("bmm_acc", kernels::BMM_ACC), // 22
    ("gdn_chunk_cumsum_step", kernels::GDN_CHUNK_CUMSUM_STEP), // 23
    ("gdn_decay_mask", kernels::GDN_DECAY_MASK), // 24
    ("gdn_mask_strict_lower", kernels::GDN_MASK_STRICT_LOWER), // 25
    ("gdn_ut_step", kernels::GDN_UT_STEP), // 26
    ("gdn_add_identity", kernels::GDN_ADD_IDENTITY), // 27
    ("gdn_row_scale_off", kernels::GDN_ROW_SCALE_OFF), // 28
    ("gdn_decay_scale", kernels::GDN_DECAY_SCALE), // 29
    ("gdn_state_decay", kernels::GDN_STATE_DECAY), // 30
    ("exp", kernels::EXP), // 31
    ("sub", kernels::SUB), // 32
    ("region_copy", kernels::REGION_COPY), // 33
    ("ce_value", kernels::CE_VALUE_MASKED), // 34
    // -- training (backward + AdamW) tier -- see `Qwen35::new_train_on`/`backward`.
    ("rms_inv", kernels::RMS_INV), // 35
    ("rmsnorm_dx", kernels::RMSNORM_DX), // 36
    ("rmsnorm_dw", kernels::RMSNORM_DW), // 37
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES), // 38
    ("gqa_bwd_dv", kernels::GQA_BWD_DV), // 39
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ), // 40
    ("gqa_bwd_dk", kernels::GQA_BWD_DK), // 41
    ("silu_bwd_da", kernels::SILU_BWD_DA), // 42
    ("silu_bwd_db", kernels::SILU_BWD_DB), // 43
    ("sigmoid_bwd", kernels::SIGMOID_BWD), // 44
    ("silu_bwd", kernels::SILU_BWD), // 45
    ("concat2", kernels::CONCAT2), // 46
    ("bias_grad", kernels::BIAS_GRAD), // 47
    ("kv_expand_bwd", kernels::KV_EXPAND_BWD), // 48
    ("matmul_dx", kernels::MATMUL_DX), // 49
    ("matmul_dw", kernels::MATMUL_DW), // 50
    ("conv1d_dx", kernels::CONV1D_DX), // 51
    ("conv1d_dw", kernels::CONV1D_DW), // 52
    ("gdn_decay_gate_bwd", kernels::GDN_DECAY_GATE_BWD), // 53
    ("splice_add", kernels::SPLICE_ADD), // 54
    ("row_dot", kernels::ROW_DOT), // 55
    ("gdn_chunk_reverse_cumsum_step", kernels::GDN_CHUNK_REVERSE_CUMSUM_STEP), // 56
    ("gdn_ut_bwd_dattn0", kernels::GDN_UT_BWD_DATTN0), // 57
    ("gdn_ut_bwd_dtmat", kernels::GDN_UT_BWD_DTMAT), // 58
    ("gdn_mask_strict_lower_bwd", kernels::GDN_MASK_STRICT_LOWER_BWD), // 59
    ("gdn_decay_mask_bwd", kernels::GDN_DECAY_MASK_BWD), // 60
    ("gdn_decay_scale_bwd", kernels::GDN_DECAY_SCALE_BWD), // 61
    ("gdn_decay_scale_bwd_last", kernels::GDN_DECAY_SCALE_BWD_LAST), // 62
    ("gdn_state_decay_bwd_dscale", kernels::GDN_STATE_DECAY_BWD_DSCALE), // 63
    ("adamw", kernels::ADAMW), // 64
    ("gradnorm_sq", kernels::GRADNORM_SQ), // 65
    ("grad_scale", kernels::GRAD_SCALE), // 66
    ("clip_coef", kernels::CLIP_COEF), // 67
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF), // 68
    ("emb_bwd", kernels::EMB_BWD), // 69
    ("ce_grad", kernels::CE_GRAD_MASKED), // 70
    ("scale_add", kernels::SCALE_ADD), // 71
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX), // 72
];

const RMSNORM: usize = 0;
const MATMUL: usize = 1;
const EMBED: usize = 2;
const SIGMOID: usize = 3;
const SILU: usize = 4;
const SILU_MUL: usize = 5;
const MUL: usize = 6;
const ADD2: usize = 7;
const L2NORM_SCALE: usize = 8;
const CONCAT_SPLIT: usize = 9;
const NLC_NCHW: usize = 10;
const NCHW_NLC: usize = 11;
const CONV1D: usize = 12;
const GDN_DECAY_GATE: usize = 13;
const GDN_LAYOUT_PERMUTE: usize = 14;
const ROPE2D_PARTIAL: usize = 15;
const GQA_SCORES: usize = 16;
const ATTN_SOFTMAX: usize = 17;
const GQA_APPLY: usize = 18;
const KV_EXPAND: usize = 19;
const SCALE_ROW: usize = 20;
const BMM: usize = 21;
const BMM_ACC: usize = 22;
const GDN_CHUNK_CUMSUM_STEP: usize = 23;
const GDN_DECAY_MASK: usize = 24;
const GDN_MASK_STRICT_LOWER: usize = 25;
const GDN_UT_STEP: usize = 26;
const GDN_ADD_IDENTITY: usize = 27;
const GDN_ROW_SCALE_OFF: usize = 28;
const GDN_DECAY_SCALE: usize = 29;
const GDN_STATE_DECAY: usize = 30;
const EXP: usize = 31;
const SUB: usize = 32;
const REGION_COPY: usize = 33;
const CE_VALUE: usize = 34;
const RMS_INV: usize = 35;
const RMSNORM_DX: usize = 36;
const RMSNORM_DW: usize = 37;
const GQA_BWD_DSCORES: usize = 38;
const GQA_BWD_DV: usize = 39;
const GQA_BWD_DQ: usize = 40;
const GQA_BWD_DK: usize = 41;
const SILU_BWD_DA: usize = 42;
const SILU_BWD_DB: usize = 43;
const SIGMOID_BWD: usize = 44;
const SILU_BWD: usize = 45;
const CONCAT2: usize = 46;
const BIAS_GRAD: usize = 47;
const KV_EXPAND_BWD: usize = 48;
const MATMUL_DX: usize = 49;
const MATMUL_DW: usize = 50;
const CONV1D_DX: usize = 51;
const CONV1D_DW: usize = 52;
const GDN_DECAY_GATE_BWD: usize = 53;
const SPLICE_ADD: usize = 54;
const ROW_DOT: usize = 55;
const GDN_CHUNK_REVERSE_CUMSUM_STEP: usize = 56;
const GDN_UT_BWD_DATTN0: usize = 57;
const GDN_UT_BWD_DTMAT: usize = 58;
const GDN_MASK_STRICT_LOWER_BWD: usize = 59;
const GDN_DECAY_MASK_BWD: usize = 60;
const GDN_DECAY_SCALE_BWD: usize = 61;
const GDN_DECAY_SCALE_BWD_LAST: usize = 62;
const GDN_STATE_DECAY_BWD_DSCALE: usize = 63;
const ADAMW: usize = 64;
const GRADNORM_SQ: usize = 65;
const GRAD_SCALE: usize = 66;
const CLIP_COEF: usize = 67;
const GRAD_SCALE_BUF: usize = 68;
const EMB_BWD: usize = 69;
const CE_GRAD: usize = 70;
const SCALE_ADD: usize = 71;
const L2NORM_SCALE_DX: usize = 72;

fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
        rmsnorm_dw: RMSNORM_DW,
        rope: RMSNORM,
        rope_bwd: RMSNORM,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: GQA_BWD_DSCORES,
        gqa_dv: GQA_BWD_DV,
        gqa_dq: GQA_BWD_DQ,
        gqa_dk: GQA_BWD_DK,
        silu_mul: SILU_MUL,
        silu_da: SILU_BWD_DA,
        silu_db: SILU_BWD_DB,
    }
}

/// Backward-only kernel ids [`gdn_chunk_bwd`]/[`gdn_chunk_fwd_train`]
/// dispatch, beyond [`gdn_ids`] (shared with the forward path).
fn gdn_bwd_ids() -> GdnBwdIds {
    GdnBwdIds {
        splice_add: SPLICE_ADD,
        row_dot: ROW_DOT,
        scale_add: SCALE_ADD,
        reverse_cumsum_step: GDN_CHUNK_REVERSE_CUMSUM_STEP,
        ut_bwd_dattn0: GDN_UT_BWD_DATTN0,
        ut_bwd_dtmat: GDN_UT_BWD_DTMAT,
        mask_strict_lower_bwd: GDN_MASK_STRICT_LOWER_BWD,
        decay_mask_bwd: GDN_DECAY_MASK_BWD,
        decay_scale_bwd: GDN_DECAY_SCALE_BWD,
        decay_scale_bwd_last: GDN_DECAY_SCALE_BWD_LAST,
        state_decay_bwd_dscale: GDN_STATE_DECAY_BWD_DSCALE,
    }
}

fn gdn_ids() -> GdnIds {
    GdnIds {
        bmm: BMM,
        bmm_acc: BMM_ACC,
        cumsum_step: GDN_CHUNK_CUMSUM_STEP,
        decay_mask: GDN_DECAY_MASK,
        mask_strict_lower: GDN_MASK_STRICT_LOWER,
        ut_step: GDN_UT_STEP,
        add_identity: GDN_ADD_IDENTITY,
        row_scale: SCALE_ROW,
        row_scale_off: GDN_ROW_SCALE_OFF,
        decay_scale: GDN_DECAY_SCALE,
        state_decay: GDN_STATE_DECAY,
        exp: EXP,
        sub: SUB,
        mul: MUL,
        region_copy: REGION_COPY,
    }
}

fn conv_kernels() -> ConvKernels {
    ConvKernels { fwd: CONV1D, dx: CONV1D_DX, dw: CONV1D_DW }
}

/// Everything [`Qwen35::layer_gdn_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GDN mixer arm - the SSA activation-cache
/// convention this module's doc describes, at the per-buffer granularity
/// backward actually reads. Mirrors `qwen35moe::model::GdnLayerActs` field
/// for field (the mixer math is identical between the two archs).
struct GdnLayerActs {
    shape: GdnShape,
    ncl_in: DeviceBuffer,
    ncl_out: DeviceBuffer,
    query: DeviceBuffer,
    key: DeviceBuffer,
    bproj: DeviceBuffer,
    aproj: DeviceBuffer,
    g_decay: DeviceBuffer,
    query_cm: DeviceBuffer,
    key_cm: DeviceBuffer,
    value_cm: DeviceBuffer,
    beta_cm: DeviceBuffer,
    scratch_train: GdnScratchTrainBufs,
    out_tok: DeviceBuffer,
    normed: DeviceBuffer,
    z: DeviceBuffer,
    z_silu: DeviceBuffer,
    gated: DeviceBuffer,
}

/// Everything [`Qwen35::layer_gqa_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GQA mixer arm. Mirrors
/// `qwen35moe::model::GqaLayerActs` field for field.
struct GqaLayerActs {
    q_normed: DeviceBuffer,
    k_normed: DeviceBuffer,
    v: DeviceBuffer,
    q_value: DeviceBuffer,
    k: DeviceBuffer,
    q_gate: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    gate: DeviceBuffer,
    ctx_gated: DeviceBuffer,
}

/// Everything [`Qwen35::mlp_fwd`]'s training branch saves - universal, every
/// layer, both mixer types. The dense-MLP analogue of `qwen35moe::model::
/// MoeLayerActs` (this model has no MoE at all, so no router/expert acts).
struct MlpLayerActs {
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

/// Saved mixer activations for one layer's backward pass. `Gdn` is several
/// times wider than `Gqa` and one of these is kept per layer for the whole
/// step, so the wide variant is boxed rather than padding every `Gqa` layer
/// out to match it - mirrors `qwen35moe::model::MixerActs`'s own reasoning.
enum MixerActs {
    Gdn(Box<GdnLayerActs>),
    Gqa(GqaLayerActs),
}

struct LayerTrainActs {
    xn1: DeviceBuffer,
    mixer: MixerActs,
    xmid: DeviceBuffer,
    mlp: MlpLayerActs,
}

/// The full backward activation cache for one `forward()` call on a
/// [`Qwen35::new_train_on`] instance. `Some` only right after a `forward()`
/// call (populated by `run_forward`'s train branch; read and taken by
/// `backward()`) - mirrors the engine-wide "forward reallocates fresh
/// buffers every call" convention, so `backward()` MUST run against the
/// same `forward()` call whose gradient it computes.
struct TrainActs {
    layers: Vec<LayerTrainActs>,
    xn_final: DeviceBuffer,
}

pub struct Qwen35 {
    pub gpu: Gpu,
    pub cfg: Qwen35Config,
    ps: ParamStore,
    b: u32,
    t: u32,
    /// The GDN chunk size this instance was built for - see [`gdn_chunk_size`].
    chunk: u32,
    /// `true` for a [`Self::new_train_on`] build: every weight is
    /// `Role::Trainable` (see [`Self::new_impl_on`]'s role filter), `forward()`
    /// saves the activation cache `backward()` reads, and `backward`/
    /// `zero_grads`/`adamw_step` are live instead of panicking. `false` (the
    /// `new_on` path) keeps inference-only behaviour byte-for-byte with M5.
    is_train: bool,
    opt: Optim,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    count: Cell<f32>,

    /// All-ones buffer of width `linear_key_head_dim`, bound as
    /// `l2norm_scale.wgsl`'s per-dim scale so its learnably-scaled L2-norm
    /// computes the reference's bare `l2norm(x)` (GDN's q/k norm has no
    /// learnable scale).
    ones_khd: DeviceBuffer,
    /// M-RoPE `cos`/`sin` tables, built once at construction for the fixed
    /// `(b,t)` this instance decodes: text-only, so every axis carries the
    /// same plain sequential position per sequence.
    cos: DeviceBuffer,
    sin: DeviceBuffer,

    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,
    /// CE-gradient uniform (`[n, vocab, IGNORE, count]`), written once per
    /// `backward()` call (`count` is only known after `set_batch`).
    ce_grad_uni: DeviceBuffer,

    /// Residual stream, one entry per layer boundary (`res[0]` = embeddings,
    /// `res[n_layers]` = input to the final norm) - the SSA activation-cache
    /// convention `qwen35moe::model` uses, kept even though nothing
    /// backprops through it yet: useful for parity debugging, any layer's
    /// residual output is independently readable via [`Self::debug_res`].
    res: RefCell<Vec<DeviceBuffer>>,

    /// Backward's activation cache - see [`TrainActs`]'s own doc.
    train_acts: RefCell<Option<TrainActs>>,
}

impl Qwen35 {
    pub fn new_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_impl_on(gpu, cfg, b, t, init, false)
    }

    /// Build a fully trainable instance (every weight `Role::Trainable`,
    /// full-parameter - no LoRA-specific subset yet, that is M8). See the
    /// struct's own `is_train` doc.
    pub fn new_train_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_impl_on(gpu, cfg, b, t, init, true)
    }

    fn new_impl_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, train: bool) -> Qwen35 {
        let chunk = gdn_chunk_size(t);
        assert_eq!(
            t % chunk,
            0,
            "qwen35: t={t} is not a multiple of the derived GDN chunk size {chunk} -- \
             model::gdn is prefill-only (no T-padding support, see its module doc); \
             gdn_chunk_size always returns a value that divides t by construction, so \
             this assert failing would mean a logic error in gdn_chunk_size itself"
        );

        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> = cfg.param_list().into_iter().map(|(n, c)| (n, c, role)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let ones_khd = gpu.storage_init("qwen35.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);

        // Text-only: every axis of the M-RoPE table carries the same plain
        // sequential position, reset per sequence (row = batch*t + pos).
        let positions: Vec<[u32; 3]> = (0..b).flat_map(|_| (0..t).map(|ti| [ti, ti, ti])).collect();
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("qwen35.rope_cos", &cos);
        let sin = gpu.storage_init("qwen35.rope_sin", &sin);

        let n = (b * t) as u64;
        let tokens = gpu.storage(n);
        let targets = gpu.storage(n);
        let logits = gpu.storage(n * cfg.vocab as u64);
        let ce_buf = gpu.storage(n);
        let ce_grad_uni = gpu.uniform_dynamic(4);
        let d = cfg.d_model as u64;
        let res = RefCell::new((0..=cfg.n_layers).map(|_| gpu.storage(n * d)).collect());

        Qwen35 {
            gpu,
            cfg,
            ps,
            b,
            t,
            chunk,
            is_train: train,
            opt,
            tokens,
            targets,
            count: Cell::new(1.0),
            ones_khd,
            cos,
            sin,
            logits,
            ce_buf,
            ce_grad_uni,
            res,
            train_acts: RefCell::new(None),
        }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    /// True if `name` has a gradient buffer (i.e. is optimised). Frozen
    /// parameters have none, so their weight-gradient dispatches must be
    /// skipped. Mirrors `qwen35moe::model::Qwen35::trainable` exactly (this
    /// model has no LoRA-frozen-base case yet, M8, so today it is simply
    /// `self.is_train`, but kept as a named method for the same reason
    /// qwen35moe keeps it - the reader shouldn't need to know that).
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    /// The gradient buffer for a trainable weight - only valid on a
    /// [`Self::new_train_on`] instance.
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }

    /// RMSNorm backward via the shared builder: input grad always, gain grad
    /// only when the gain is trainable.
    fn rmsnorm_bwd_step(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        let inv = self.gpu.storage(rows as u64);
        let gw = self.trainable(wname).then(|| self.g(wname));
        steps.extend(rmsnorm_bwd(&self.gpu, &kernel_ids(), x, self.w(wname), dy, dx, &inv, gw, dim, rows));
    }

    pub fn set_batch(&self, tokens: &[u32], targets: &[u32]) {
        self.gpu.write(&self.tokens, tokens);
        self.gpu.write(&self.targets, targets);
        let c = targets.iter().filter(|&&v| v != model::IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    /// Backward for a linear `y = x*Wt`. Accumulates the input gradient into
    /// `dx` (flag `acc`); `dW += d_out^t*x` (skipped when `wname` is Frozen).
    /// No LoRA branch yet (M8) - mirrors `qwen35moe::model::proj_bwd`'s own
    /// `None` (non-LoRA) arm exactly.
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, steps: &mut Vec<Step>, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        let g = &self.gpu;
        if self.trainable(wname) {
            steps.push(g.step(MATMUL_DW, &[d_out, x, self.g(wname)], &[m, k, nout], nout * k));
        }
        steps.push(g.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
    }

    // ---- one Gated DeltaNet (Linear) layer --------------------------------

    fn layer_gdn_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32) -> (DeviceBuffer, Option<GdnLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let conv_dim = c.linear_conv_dim();
        let key_dim = c.linear_key_dim();
        let value_dim = c.linear_value_dim();
        let nkh = c.linear_num_key_heads;
        let nvh = c.linear_num_value_heads;
        let khd = c.linear_key_head_dim;
        let vhd = c.linear_value_head_dim;
        let group = c.linear_group();
        let kw = c.linear_conv_kernel_dim;
        let (b, t, chunk) = (self.b, self.t, self.chunk);
        let n_chunks = t / chunk;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // 1. mixed_qkv = in_proj_qkv(xn1).
        let mixed_qkv = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(MATMUL, &[xn1, self.w(&p("in_proj_qkv.weight")), &mixed_qkv], &[n, d, conv_dim], n * conv_dim)]);

        // 2. Depthwise causal conv1d + SiLU (activation AFTER the conv).
        // conv1d.wgsl is NCL ([N,Cin,L]); mixed_qkv is token-major ([B,T,C]).
        let ncl_in = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(NLC_NCHW, &[&mixed_qkv, &ncl_in], &[n * conv_dim, conv_dim, t], n * conv_dim)]);
        let conv_shape =
            Conv1d { n: b, cin: conv_dim, l: t, cout: conv_dim, k: kw, stride: 1, pad: kw - 1, dilation: 1, groups: conv_dim, lo: t };
        let ncl_out = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[conv1d_fwd(g, &conv_kernels(), &conv_shape, &ncl_in, self.w(&p("conv1d.weight")), &ncl_out)]);
        let ncl_act = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(SILU, &[&ncl_out, &ncl_act], &[n * conv_dim], n * conv_dim)]);
        let mixed_act = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(NCHW_NLC, &[&ncl_act, &mixed_act], &[n * conv_dim, conv_dim, t], n * conv_dim)]);

        // 3. Split into query/key/value - ONE whole-row contiguous split.
        let query = g.storage((n * key_dim) as u64);
        let key = g.storage((n * key_dim) as u64);
        let value = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(CONCAT_SPLIT, &[&mixed_act, &query], &[n, conv_dim, key_dim, 0, 1, 1], n * key_dim),
                g.step(CONCAT_SPLIT, &[&mixed_act, &key], &[n, conv_dim, key_dim, key_dim, 1, 1], n * key_dim),
                g.step(CONCAT_SPLIT, &[&mixed_act, &value], &[n, conv_dim, value_dim, 2 * key_dim, 1, 1], n * value_dim),
            ],
        );

        // 4. L2-normalize query/key - bare l2norm (no learnable scale).
        let query_n = g.storage((n * key_dim) as u64);
        let key_n = g.storage((n * key_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(L2NORM_SCALE, &[&query, &self.ones_khd, &query_n], &[n * nkh, khd, f(1e-6)], n * key_dim),
                g.step(L2NORM_SCALE, &[&key, &self.ones_khd, &key_n], &[n * nkh, khd, f(1e-6)], n * key_dim),
            ],
        );

        // 5. beta = sigmoid(in_proj_b(xn1)); g = -exp(A_log)*softplus(in_proj_a(xn1)+dt_bias);
        // z = in_proj_z(xn1) (feeds the gated RMSNorm at the end, step 10).
        let bproj = g.storage((n * nvh) as u64);
        let aproj = g.storage((n * nvh) as u64);
        let z = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_b.weight")), &bproj], &[n, d, nvh], n * nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_a.weight")), &aproj], &[n, d, nvh], n * nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_z.weight")), &z], &[n, d, value_dim], n * value_dim),
            ],
        );
        let beta = g.storage((n * nvh) as u64);
        let g_decay = g.storage((n * nvh) as u64);
        g.submit(
            &[],
            &[
                g.step(SIGMOID, &[&bproj, &beta], &[n * nvh], n * nvh),
                g.step(GDN_DECAY_GATE, &[&aproj, self.w(&p("A_log")), self.w(&p("dt_bias")), &g_decay], &[n, nvh], n * nvh),
            ],
        );

        // 6. Repeat query/key from linear_num_key_heads to linear_num_value_heads.
        let query_w = g.storage((n * nvh * khd) as u64);
        let key_w = g.storage((n * nvh * khd) as u64);
        g.submit(
            &[],
            &[
                kv_expand_fwd(g, KV_EXPAND, &query_n, &query_w, n, nvh, group, khd, nvh * khd, 0),
                kv_expand_fwd(g, KV_EXPAND, &key_n, &key_w, n, nvh, group, khd, nvh * khd, 0),
            ],
        );

        // 7. Chunk-major permute (token-major -> chunk-major) for gdn_chunk_fwd.
        let shape = GdnShape { b, h: nvh, t, dk: khd, dv: vhd, chunk };
        let permute_fwd = |src: &DeviceBuffer, dim: u32| -> DeviceBuffer {
            let dst = g.storage(b as u64 * nvh as u64 * n_chunks as u64 * chunk as u64 * dim as u64);
            g.submit(&[], &[g.step(GDN_LAYOUT_PERMUTE, &[src, &dst], &[b, nvh, n_chunks, chunk, dim, 1], b * nvh * n_chunks * chunk * dim)]);
            dst
        };
        let query_cm = permute_fwd(&query_w, khd);
        let key_cm = permute_fwd(&key_w, khd);
        let value_cm = permute_fwd(&value, vhd);
        let g_cm = permute_fwd(&g_decay, 1);
        let beta_cm = permute_fwd(&beta, 1);

        // 8. gdn_chunk_fwd - the chunked-recurrence forward itself. Training
        // builds use gdn_chunk_fwd_train instead: bit-identical out/final_state
        // but additionally saves the per-chunk history gdn_chunk_bwd needs.
        let bh = shape.bh() as u64;
        let initial_state = g.storage(bh * khd as u64 * vhd as u64);
        let final_state = g.storage(bh * khd as u64 * vhd as u64);
        let out_cm = g.storage(shape.bhc() as u64 * chunk as u64 * vhd as u64);
        let scratch_train = if self.is_train { Some(GdnScratchTrainBufs::new(g, &shape)) } else { None };
        if let Some(strain) = &scratch_train {
            let steps = gdn_chunk_fwd_train(
                g,
                &gdn_ids(),
                &gdn_bwd_ids(),
                &shape,
                &query_cm,
                &key_cm,
                &value_cm,
                &g_cm,
                &beta_cm,
                &initial_state,
                &strain.as_ref(),
                &out_cm,
                &final_state,
            );
            g.submit(&strain.clears(), &steps);
        } else {
            let scratch = GdnScratchBufs::new(g, &shape);
            let steps = gdn_chunk_fwd(
                g,
                &gdn_ids(),
                &shape,
                &query_cm,
                &key_cm,
                &value_cm,
                &g_cm,
                &beta_cm,
                &initial_state,
                &scratch.as_ref(),
                &out_cm,
                &final_state,
            );
            g.submit(&scratch.clears(), &steps);
        }

        // 9. Permute back to token-major.
        let out_tok = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[g.step(GDN_LAYOUT_PERMUTE, &[&out_cm, &out_tok], &[b, nvh, n_chunks, chunk, vhd, 0], b * nvh * n_chunks * chunk * vhd)],
        );

        // 10. Gated RMSNorm ("norm before gate"): normed = RMSNorm(out_tok)*weight,
        // THEN * SiLU(z).
        let normed = g.storage((n * value_dim) as u64);
        let z_silu = g.storage((n * value_dim) as u64);
        let gated = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[
                rmsnorm_fwd(g, &kernel_ids(), &out_tok, self.w(&p("norm.weight")), &normed, vhd, n * nvh),
                g.step(SILU, &[&z, &z_silu], &[n * value_dim], n * value_dim),
                g.step(MUL, &[&normed, &z_silu, &gated], &[n * value_dim], n * value_dim),
            ],
        );

        // 11. out_proj.
        let out = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(MATMUL, &[&gated, self.w(&p("out_proj.weight")), &out], &[n, value_dim, d], n * d)]);

        let acts = scratch_train.map(|scratch_train| GdnLayerActs {
            shape,
            ncl_in,
            ncl_out,
            query,
            key,
            bproj,
            aproj,
            g_decay,
            query_cm,
            key_cm,
            value_cm,
            beta_cm,
            scratch_train,
            out_tok,
            normed,
            z,
            z_silu,
            gated,
        });
        (out, acts)
    }

    // ---- one GQA (Full) layer ----------------------------------------------

    fn layer_gqa_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32) -> (DeviceBuffer, Option<GqaLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, qd, kvd) = (c.q_proj_dim(), c.q_dim(), c.kv_dim());
        let p = |s: &str| format!("blocks.{l}.self_attn.{s}");

        let q_full = g.storage((n * qpd) as u64);
        let k = g.storage((n * kvd) as u64);
        let v = g.storage((n * kvd) as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn1, self.w(&p("q_proj.weight")), &q_full], &[n, d, qpd], n * qpd),
                g.step(MATMUL, &[xn1, self.w(&p("k_proj.weight")), &k], &[n, d, kvd], n * kvd),
                g.step(MATMUL, &[xn1, self.w(&p("v_proj.weight")), &v], &[n, d, kvd], n * kvd),
            ],
        );

        // Per-head de-interleaved split of q_full's [query|gate] halves -
        // NOT a whole-row split. Fold n_heads into concat_split's own N so
        // each head's 2*head_dim block splits into its own first/second half.
        let q_value = g.storage((n * qd) as u64);
        let q_gate = g.storage((n * qd) as u64);
        g.submit(
            &[],
            &[
                g.step(CONCAT_SPLIT, &[&q_full, &q_value], &[n * nh, 2 * hd, hd, 0, 1, 1], n * nh * hd),
                g.step(CONCAT_SPLIT, &[&q_full, &q_gate], &[n * nh, 2 * hd, hd, hd, 1, 1], n * nh * hd),
            ],
        );

        let q_normed = g.storage((n * qd) as u64);
        let k_normed = g.storage((n * kvd) as u64);
        g.submit(
            &[],
            &[
                rmsnorm_fwd(g, &kernel_ids(), &q_value, self.w(&p("q_norm.weight")), &q_normed, hd, n * nh),
                rmsnorm_fwd(g, &kernel_ids(), &k, self.w(&p("k_norm.weight")), &k_normed, hd, n * nkv),
            ],
        );

        let half = c.rotary_dim() / 2;
        g.submit(
            &[],
            &[
                rope2d_partial_fwd(g, ROPE2D_PARTIAL, &q_normed, &self.cos, &self.sin, n, nh, half, qd, 0, hd),
                rope2d_partial_fwd(g, ROPE2D_PARTIAL, &k_normed, &self.cos, &self.sin, n, nkv, half, kvd, 0, hd),
            ],
        );

        let scores = g.storage(self.b as u64 * nh as u64 * self.t as u64 * self.t as u64);
        let probs = g.storage(self.b as u64 * nh as u64 * self.t as u64 * self.t as u64);
        let ctx = g.storage((n * qd) as u64);
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
        g.submit(&[], &gqa_fwd(g, &kernel_ids(), &ga, &q_normed, &k_normed, &v, &scores, &probs, &ctx));

        let gate = g.storage((n * qd) as u64);
        let ctx_gated = g.storage((n * qd) as u64);
        let out = g.storage((n * d) as u64);
        g.submit(
            &[],
            &[
                g.step(SIGMOID, &[&q_gate, &gate], &[n * qd], n * qd),
                g.step(MUL, &[&ctx, &gate, &ctx_gated], &[n * qd], n * qd),
                g.step(MATMUL, &[&ctx_gated, self.w(&p("o_proj.weight")), &out], &[n, qd, d], n * d),
            ],
        );

        let acts = self.is_train.then(|| GqaLayerActs { q_normed, k_normed, v, q_value, k, q_gate, probs, ctx, gate, ctx_gated });
        (out, acts)
    }

    // ---- dense SwiGLU MLP, universal for every layer -----------------------

    fn mlp_fwd(&self, l: usize, xn2: &DeviceBuffer, n: u32) -> (DeviceBuffer, Option<MlpLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.intermediate_size;
        let p = |s: &str| format!("blocks.{l}.mlp.{s}");

        let gate_pre = g.storage((n * ff) as u64);
        let up = g.storage((n * ff) as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn2, self.w(&p("gate.weight")), &gate_pre], &[n, d, ff], n * ff),
                g.step(MATMUL, &[xn2, self.w(&p("up.weight")), &up], &[n, d, ff], n * ff),
            ],
        );
        let h = g.storage((n * ff) as u64);
        g.submit(&[], &[g.step(SILU_MUL, &[&gate_pre, &up, &h], &[n * ff], n * ff)]);
        let down = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(MATMUL, &[&h, self.w(&p("down.weight")), &down], &[n, ff, d], n * d)]);

        let acts = self.is_train.then(|| MlpLayerActs { xn2: xn2.clone(), gate_pre, up, h });
        (down, acts)
    }

    pub(crate) fn run_forward(&self) {
        let g = &self.gpu;
        let n = self.b * self.t;
        let d = self.cfg.d_model;
        let res = self.res.borrow();
        let mut layer_acts: Vec<LayerTrainActs> = Vec::new();

        g.submit(&[], &[g.step(EMBED, &[&self.tokens, self.w("tok.weight"), &res[0]], &[d, n], n * d)]);

        let types = self.cfg.layer_types();
        for (l, ty) in types.iter().enumerate() {
            let xres = &res[l];
            let xn1 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), xres, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n)]);

            let (mixer_out, mixer_acts) = match ty {
                LayerType::Linear => {
                    let (o, a) = self.layer_gdn_fwd(l, &xn1, n);
                    (o, a.map(|a| MixerActs::Gdn(Box::new(a))))
                }
                LayerType::Full => {
                    let (o, a) = self.layer_gqa_fwd(l, &xn1, n);
                    (o, a.map(MixerActs::Gqa))
                }
            };

            let xmid = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

            let xn2 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &xmid, self.w(&format!("blocks.{l}.ln2.weight")), &xn2, d, n)]);

            let (mlp_out, mlp_acts) = self.mlp_fwd(l, &xn2, n);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &res[l + 1]], &[n * d], n * d)]);

            if self.is_train {
                layer_acts.push(LayerTrainActs {
                    xn1,
                    mixer: mixer_acts.expect("qwen35: is_train but layer_gdn_fwd/layer_gqa_fwd returned no acts"),
                    xmid,
                    mlp: mlp_acts.expect("qwen35: is_train but mlp_fwd returned no acts"),
                });
            }
        }

        let xn_final = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res[self.cfg.n_layers as usize], self.w("norm.weight"), &xn_final, d, n)]);
        let v = self.cfg.vocab;
        g.submit(&[], &[g.step(MATMUL, &[&xn_final, self.w(self.cfg.head_weight()), &self.logits], &[n, d, v], n * v)]);

        if self.is_train {
            *self.train_acts.borrow_mut() = Some(TrainActs { layers: layer_acts, xn_final });
        }
    }

    // ---- backward (training builds only) -----------------------------------

    /// Reverse of [`Self::layer_gdn_fwd`]'s 11 steps. `d_out` is the upstream
    /// gradient into this layer's mixer output; accumulates into `d_xn1`
    /// (already zero-fresh - the FIRST touch below is a plain overwrite,
    /// `acc=0`). Mirrors `qwen35moe::model::Qwen35::gdn_mixer_bwd` exactly
    /// (the mixer math is identical between the two archs).
    fn gdn_mixer_bwd(&self, l: usize, xn1: &DeviceBuffer, la: &GdnLayerActs, d_out: &DeviceBuffer, d_xn1: &DeviceBuffer, n: u32) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let conv_dim = c.linear_conv_dim();
        let key_dim = c.linear_key_dim();
        let value_dim = c.linear_value_dim();
        let nvh = c.linear_num_value_heads;
        let khd = c.linear_key_head_dim;
        let vhd = c.linear_value_head_dim;
        let group = c.linear_group();
        let kw = c.linear_conv_kernel_dim;
        let (b, t, chunk) = (self.b, self.t, self.chunk);
        let n_chunks = t / chunk;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");
        let shape = la.shape;

        // ---- 11. out_proj backward ----
        let d_gated = g.storage((n * value_dim) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, d_out, &la.gated, &p("out_proj.weight"), &d_gated, n, value_dim, d, 0);
            g.submit(&[], &s);
        }

        // ---- 10. gated RMSNorm backward: gated = normed*z_silu; z_silu = silu(z); normed = rmsnorm(out_tok) ----
        let d_normed = g.storage((n * value_dim) as u64);
        let d_z_silu = g.storage((n * value_dim) as u64);
        let d_z = g.storage((n * value_dim) as u64);
        let d_out_tok = g.storage((n * value_dim) as u64);
        {
            let mut s = vec![
                g.step(MUL, &[&d_gated, &la.z_silu, &d_normed], &[n * value_dim], n * value_dim),
                g.step(MUL, &[&d_gated, &la.normed, &d_z_silu], &[n * value_dim], n * value_dim),
                g.step(SILU_BWD, &[&la.z, &d_z_silu, &d_z], &[n * value_dim], n * value_dim),
            ];
            self.rmsnorm_bwd_step(&mut s, &la.out_tok, &p("norm.weight"), &d_normed, &d_out_tok, vhd, n * nvh);
            g.submit(&[], &s);
        }

        // ---- 9. permute back to chunk-major (forward used to_chunk_major=0; backward flips it) ----
        let d_out_cm = g.storage(shape.bhc() as u64 * shape.chunk as u64 * vhd as u64);
        g.submit(
            &[],
            &[g.step(GDN_LAYOUT_PERMUTE, &[&d_out_tok, &d_out_cm], &[b, nvh, n_chunks, chunk, vhd, 1], b * nvh * n_chunks * chunk * vhd)],
        );

        // ---- 8. gdn_chunk_bwd - the chunked-recurrence backward itself ----
        let bh = shape.bh() as u64;
        let bhc = shape.bhc() as u64;
        let cw = shape.chunk as u64;
        let dk = shape.dk as u64;
        let dv = shape.dv as u64;
        let d_final_state = g.storage(bh * dk * dv); // no incremental decode continuation -> zero
        let d_initial_state = g.storage(bh * dk * dv); // discarded (no earlier chunk upstream)
        let d_query_cm = g.storage(bhc * cw * dk);
        let d_key_cm = g.storage(bhc * cw * dk);
        let d_value_cm = g.storage(bhc * cw * dv);
        let d_g_cm = g.storage(bhc * cw);
        let d_beta_cm = g.storage(bhc * cw);
        let bwd_scratch = GdnBwdScratchBufs::new(g, &shape);
        {
            let steps = gdn_chunk_bwd(
                g,
                &gdn_ids(),
                &gdn_bwd_ids(),
                &shape,
                &la.query_cm,
                &la.key_cm,
                &la.value_cm,
                &la.beta_cm,
                &la.scratch_train.as_ref(),
                &d_out_cm,
                &d_final_state,
                &bwd_scratch.as_ref(),
                &d_query_cm,
                &d_key_cm,
                &d_value_cm,
                &d_g_cm,
                &d_beta_cm,
                &d_initial_state,
            );
            let mut clears = bwd_scratch.clears();
            clears.push(&d_final_state);
            clears.push(&d_query_cm);
            clears.push(&d_key_cm);
            clears.push(&d_beta_cm);
            g.submit(&clears, &steps);
        }

        // ---- 7. permute back to token-major (forward used to_chunk_major=1; backward flips it) ----
        let permute_bwd = |src_cm: &DeviceBuffer, dim: u32| -> DeviceBuffer {
            let dst = g.storage(n as u64 * nvh as u64 * dim as u64);
            g.submit(
                &[],
                &[g.step(GDN_LAYOUT_PERMUTE, &[src_cm, &dst], &[b, nvh, n_chunks, chunk, dim, 0], b * nvh * n_chunks * chunk * dim)],
            );
            dst
        };
        let d_query_w = permute_bwd(&d_query_cm, khd);
        let d_key_w = permute_bwd(&d_key_cm, khd);
        let d_value = permute_bwd(&d_value_cm, vhd);
        let d_g_decay = permute_bwd(&d_g_cm, 1);
        let d_beta = permute_bwd(&d_beta_cm, 1);

        // ---- 6. kv_expand backward (group-sum, overwrite -- no accumulate needed) ----
        let d_query_n = g.storage((n * key_dim) as u64);
        let d_key_n = g.storage((n * key_dim) as u64);
        g.submit(
            &[],
            &[
                kv_expand_bwd(g, KV_EXPAND_BWD, &d_query_w, &d_query_n, n, nvh, group, khd, nvh * khd, 0),
                kv_expand_bwd(g, KV_EXPAND_BWD, &d_key_w, &d_key_n, n, nvh, group, khd, nvh * khd, 0),
            ],
        );

        // ---- 5. beta/g_decay backward into bproj/aproj, A_log/dt_bias reductions, in_proj_{b,a,z} ----
        let d_bproj = g.storage((n * nvh) as u64);
        let d_aproj = g.storage((n * nvh) as u64);
        {
            let mut s = vec![
                g.step(SIGMOID_BWD, &[&la.bproj, &d_beta, &d_bproj], &[n * nvh], n * nvh),
                g.step(GDN_DECAY_GATE_BWD, &[&la.aproj, self.w(&p("A_log")), self.w(&p("dt_bias")), &d_g_decay, &d_aproj], &[n, nvh], n * nvh),
            ];
            let mul_tmp = g.storage((n * nvh) as u64);
            s.push(g.step(MUL, &[&d_g_decay, &la.g_decay, &mul_tmp], &[n * nvh], n * nvh));
            if self.trainable(&p("A_log")) {
                s.push(g.step(BIAS_GRAD, &[&mul_tmp, self.g(&p("A_log"))], &[n, nvh], nvh));
            }
            if self.trainable(&p("dt_bias")) {
                s.push(g.step(BIAS_GRAD, &[&d_aproj, self.g(&p("dt_bias"))], &[n, nvh], nvh));
            }
            // FIRST touch to d_xn1 in this function (acc=0) -- in_proj_a/z below
            // accumulate on top; in_proj_qkv (processed last here) accumulates last of all.
            self.proj_bwd(&mut s, &d_bproj, xn1, &p("in_proj_b.weight"), d_xn1, n, d, nvh, 0);
            self.proj_bwd(&mut s, &d_aproj, xn1, &p("in_proj_a.weight"), d_xn1, n, d, nvh, 1);
            self.proj_bwd(&mut s, &d_z, xn1, &p("in_proj_z.weight"), d_xn1, n, d, value_dim, 1);
            g.submit(&[], &s);
        }

        // ---- 4. L2-norm backward ----
        let nkh = c.linear_num_key_heads;
        let d_query = g.storage((n * key_dim) as u64);
        let d_key = g.storage((n * key_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(L2NORM_SCALE_DX, &[&la.query, &self.ones_khd, &d_query_n, &d_query], &[n * nkh, khd, f(1e-6)], n * key_dim),
                g.step(L2NORM_SCALE_DX, &[&la.key, &self.ones_khd, &d_key_n, &d_key], &[n * nkh, khd, f(1e-6)], n * key_dim),
            ],
        );

        // ---- 3. qkv split backward (concat2 x2: the 3-way split's adjoint) ----
        let d_qk = g.storage((n * 2 * key_dim) as u64);
        let d_mixed_act = g.storage((n * conv_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(CONCAT2, &[&d_query, &d_key, &d_qk], &[n, key_dim, key_dim, 1, 1], n * 2 * key_dim),
                g.step(CONCAT2, &[&d_qk, &d_value, &d_mixed_act], &[n, 2 * key_dim, value_dim, 1, 1], n * conv_dim),
            ],
        );

        // ---- 2. conv1d + SiLU backward ----
        let d_ncl_act = g.storage((n * conv_dim) as u64);
        let d_ncl_out = g.storage((n * conv_dim) as u64);
        let d_ncl_in = g.storage((n * conv_dim) as u64);
        let d_mixed_qkv = g.storage((n * conv_dim) as u64);
        let conv_shape =
            Conv1d { n: b, cin: conv_dim, l: t, cout: conv_dim, k: kw, stride: 1, pad: kw - 1, dilation: 1, groups: conv_dim, lo: t };
        {
            let mut s = vec![
                g.step(NLC_NCHW, &[&d_mixed_act, &d_ncl_act], &[n * conv_dim, conv_dim, t], n * conv_dim),
                g.step(SILU_BWD, &[&la.ncl_out, &d_ncl_act, &d_ncl_out], &[n * conv_dim], n * conv_dim),
            ];
            let conv_dw = self.trainable(&p("conv1d.weight")).then(|| self.g(&p("conv1d.weight")));
            s.extend(conv1d_bwd(g, &conv_kernels(), &conv_shape, &d_ncl_out, &la.ncl_in, self.w(&p("conv1d.weight")), Some(&d_ncl_in), conv_dw));
            s.push(g.step(NCHW_NLC, &[&d_ncl_in, &d_mixed_qkv], &[n * conv_dim, conv_dim, t], n * conv_dim));
            g.submit(&[], &s);
        }

        // ---- 1. in_proj_qkv backward (last accumulate into d_xn1) ----
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, &d_mixed_qkv, xn1, &p("in_proj_qkv.weight"), d_xn1, n, d, conv_dim, 1);
            g.submit(&[], &s);
        }
    }

    /// Reverse of [`Self::layer_gqa_fwd`]'s 7 steps. Mirrors
    /// `qwen35moe::model::Qwen35::gqa_mixer_bwd` exactly.
    fn gqa_mixer_bwd(&self, l: usize, xn1: &DeviceBuffer, la: &GqaLayerActs, d_out: &DeviceBuffer, d_xn1: &DeviceBuffer, n: u32) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, qd, kvd) = (c.q_proj_dim(), c.q_dim(), c.kv_dim());
        let p = |s: &str| format!("blocks.{l}.self_attn.{s}");

        // ---- 7. o_proj backward ----
        let d_ctx_gated = g.storage((n * qd) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, d_out, &la.ctx_gated, &p("o_proj.weight"), &d_ctx_gated, n, qd, d, 0);
            g.submit(&[], &s);
        }

        // ---- 6. ctx*gate backward, sigmoid backward ----
        let d_ctx = g.storage((n * qd) as u64);
        let d_gate = g.storage((n * qd) as u64);
        let d_q_gate = g.storage((n * qd) as u64);
        g.submit(
            &[],
            &[
                g.step(MUL, &[&d_ctx_gated, &la.gate, &d_ctx], &[n * qd], n * qd),
                g.step(MUL, &[&d_ctx_gated, &la.ctx, &d_gate], &[n * qd], n * qd),
                g.step(SIGMOID_BWD, &[&la.q_gate, &d_gate, &d_q_gate], &[n * qd], n * qd),
            ],
        );

        // ---- 5. gqa_bwd ----
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
        let d_scores = g.storage(self.b as u64 * nh as u64 * self.t as u64 * self.t as u64);
        let d_q_normed = g.storage((n * qd) as u64);
        let d_k_normed = g.storage((n * kvd) as u64);
        let d_v = g.storage((n * kvd) as u64);
        g.submit(&[], &gqa_bwd(g, &kernel_ids(), &ga, &la.q_normed, &la.k_normed, &la.v, &la.probs, &d_ctx, &d_scores, &d_q_normed, &d_k_normed, &d_v));

        // ---- 4. RoPE backward (in place, sign=-1) ----
        let half = c.rotary_dim() / 2;
        g.submit(
            &[],
            &[
                rope2d_partial_bwd(g, ROPE2D_PARTIAL, &d_q_normed, &self.cos, &self.sin, n, nh, half, qd, 0, hd),
                rope2d_partial_bwd(g, ROPE2D_PARTIAL, &d_k_normed, &self.cos, &self.sin, n, nkv, half, kvd, 0, hd),
            ],
        );

        // ---- 3. per-head QK-norm backward ----
        let d_q_value = g.storage((n * qd) as u64);
        let d_k = g.storage((n * kvd) as u64);
        {
            let mut s = Vec::new();
            self.rmsnorm_bwd_step(&mut s, &la.q_value, &p("q_norm.weight"), &d_q_normed, &d_q_value, hd, n * nh);
            self.rmsnorm_bwd_step(&mut s, &la.k, &p("k_norm.weight"), &d_k_normed, &d_k, hd, n * nkv);
            g.submit(&[], &s);
        }

        // ---- 2. q_full [value|gate] split backward (concat2, per-head interleaved) ----
        let d_q_full = g.storage((n * qpd) as u64);
        g.submit(&[], &[g.step(CONCAT2, &[&d_q_value, &d_q_gate, &d_q_full], &[n * nh, hd, hd, 1, 1], n * nh * 2 * hd)]);

        // ---- 1. q/k/v proj backward ----
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, &d_q_full, xn1, &p("q_proj.weight"), d_xn1, n, d, qpd, 0);
            self.proj_bwd(&mut s, &d_k, xn1, &p("k_proj.weight"), d_xn1, n, d, kvd, 1);
            self.proj_bwd(&mut s, &d_v, xn1, &p("v_proj.weight"), d_xn1, n, d, kvd, 1);
            g.submit(&[], &s);
        }
    }

    /// Reverse of [`Self::mlp_fwd`]. Returns `d_xn2` (the gradient into the
    /// pre-MLP-norm hidden state, i.e. `ln2`'s output) - the caller still owes
    /// `ln2`'s own backward to fold that into `d_xmid`. Dense-MLP analogue of
    /// `qwen35moe::model::Qwen35::moe_sublayer_bwd` (no router/expert phases -
    /// this is just `qwen3::model.rs`'s own dense-MLP backward pattern:
    /// `down`'s proj_bwd, `swiglu_bwd`, then `up`/`gate`'s proj_bwd
    /// accumulating into one `d_xn2`).
    fn mlp_bwd(&self, l: usize, la: &MlpLayerActs, d_mlp_out: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.intermediate_size;
        let p = |s: &str| format!("blocks.{l}.mlp.{s}");

        let d_h = g.storage((n * ff) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, d_mlp_out, &la.h, &p("down.weight"), &d_h, n, ff, d, 0);
            g.submit(&[], &s);
        }

        let d_gate_pre = g.storage((n * ff) as u64);
        let d_up = g.storage((n * ff) as u64);
        g.submit(&[], &swiglu_bwd(g, &kernel_ids(), &la.gate_pre, &la.up, &d_h, &d_gate_pre, &d_up, n * ff));

        let d_xn2 = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            // FIRST touch to d_xn2 (acc=0); gate accumulates on top.
            self.proj_bwd(&mut s, &d_up, &la.xn2, &p("up.weight"), &d_xn2, n, d, ff, 0);
            self.proj_bwd(&mut s, &d_gate_pre, &la.xn2, &p("gate.weight"), &d_xn2, n, d, ff, 1);
            g.submit(&[], &s);
        }
        d_xn2
    }

    /// Reverse of [`Self::run_forward`]. Mirrors `qwen35moe::model::Qwen35::
    /// backward` (minus the sharding/vision-splice seams - this model has
    /// neither yet).
    pub fn backward(&self) {
        assert!(self.is_train, "qwen35: backward() requires a Qwen35::new_train_on build");
        let ta = self.train_acts.borrow_mut().take().expect(
            "qwen35: backward() called without an immediately preceding forward() -- \
             every forward() call reallocates its activation cache fresh (this file's \
             own convention throughout), so backward() must run against the SAME call",
        );
        let g = &self.gpu;
        let n = self.b * self.t;
        let d = self.cfg.d_model;
        let v = self.cfg.vocab;

        // ---- head epilogue backward: CE-grad, lm_head, final norm ----
        g.write(&self.ce_grad_uni, &[n, v, model::IGNORE, f(self.count.get())]);
        let d_logits = g.storage((n * v) as u64);
        g.submit(&[], &[g.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &d_logits], n * v)]);

        let d_xn_final = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, &d_logits, &ta.xn_final, self.cfg.head_weight(), &d_xn_final, n, d, v, 0);
            g.submit(&[], &s);
        }

        let mut d_res_next = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.rmsnorm_bwd_step(&mut s, &self.res.borrow()[self.cfg.n_layers as usize], "norm.weight", &d_xn_final, &d_res_next, d, n);
            g.submit(&[], &s);
        }

        let res = self.res.borrow();
        for l in (0..self.cfg.n_layers as usize).rev() {
            let la = &ta.layers[l];

            // ---- second residual add backward: res[l+1] = xmid + mlp_out ----
            let d_mlp_out = &d_res_next;
            let d_xn2 = self.mlp_bwd(l, &la.mlp, d_mlp_out, n);

            let d_ln2_dx = g.storage((n * d) as u64);
            let d_xmid = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.rmsnorm_bwd_step(&mut s, &la.xmid, &format!("blocks.{l}.ln2.weight"), &d_xn2, &d_ln2_dx, d, n);
                s.push(g.step(ADD2, &[&d_res_next, &d_ln2_dx, &d_xmid], &[n * d], n * d));
                g.submit(&[], &s);
            }

            // ---- first residual add backward: xmid = res[l] + mixer_out ----
            let d_xn1 = g.storage((n * d) as u64);
            match &la.mixer {
                MixerActs::Gdn(acts) => self.gdn_mixer_bwd(l, &la.xn1, acts, &d_xmid, &d_xn1, n),
                MixerActs::Gqa(acts) => self.gqa_mixer_bwd(l, &la.xn1, acts, &d_xmid, &d_xn1, n),
            }

            // ---- ln1 backward: xn1 = rmsnorm(res[l]) -> d_res[l] = d_xmid + d_tmp ----
            let d_ln1_dx = g.storage((n * d) as u64);
            let d_res_l = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.rmsnorm_bwd_step(&mut s, &res[l], &format!("blocks.{l}.ln1.weight"), &d_xn1, &d_ln1_dx, d, n);
                s.push(g.step(ADD2, &[&d_xmid, &d_ln1_dx, &d_res_l], &[n * d], n * d));
                g.submit(&[], &s);
            }
            d_res_next = d_res_l;
        }
        drop(res);

        // ---- embedding backward (tok.weight) ----
        if self.trainable("tok.weight") {
            g.submit(&[], &[g.step(EMB_BWD, &[&self.tokens, &d_res_next, self.g("tok.weight")], &[n, d, v], v * d)]);
        }
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    pub fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    /// Run the forward graph and return the scalar loss.
    pub fn forward(&self) -> f32 {
        self.run_forward();
        let n = self.b * self.t;
        self.gpu.submit(&[], &[self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, self.cfg.vocab, model::IGNORE], n)]);
        let vals = self.gpu.read(&self.ce_buf, n as usize);
        vals.iter().sum::<f32>() / self.count.get()
    }

    /// The residual stream at layer boundary `l` (`0` = embeddings, `l+1` =
    /// layer `l`'s own output, `cfg.n_layers` = input to the final norm) -
    /// parity-debugging introspection only, valid after a `run_forward()`
    /// call (via [`Self::logits_all`]).
    pub fn debug_res(&self, l: usize) -> Vec<f32> {
        let n = (self.b * self.t) as usize;
        let d = self.cfg.d_model as usize;
        self.gpu.read(&self.res.borrow()[l], n * d)
    }

    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        assert_eq!(self.b, 1, "qwen35::logits_all requires b==1 (single sequence)");
        assert_eq!(
            tokens.len() as u32,
            self.t,
            "qwen35::logits_all requires tokens.len() == the configured t (no partial-length prefill in this pass)"
        );
        self.gpu.write(&self.tokens, tokens);
        self.run_forward();
        self.gpu.read(&self.logits, (self.t * self.cfg.vocab) as usize)
    }

    pub fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            self.ps.params.iter().map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name))).collect();
        let config = self.cfg.to_json();
        checkpoint::save_carded(path, config, &tensors, &checkpoint::st::ModelCard::new("brain/qwen35", "qwen35"));
    }
}

// ---- architecture-agnostic Model seam ---------------------------------------

impl model::ModelConfig for Qwen35Config {
    fn param_list(&self) -> Vec<(String, usize)> {
        Qwen35Config::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        Qwen35Config::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        Qwen35Config::from_json(v)
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

impl model::Model for Qwen35 {
    type Config = Qwen35Config;

    fn new(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Qwen35::new_on(Gpu::new(PIPELINES), cfg, b, t, init)
    }
    fn init_weights(cfg: &Qwen35Config, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }
    fn config(&self) -> &Qwen35Config {
        &self.cfg
    }
    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => Qwen35::set_batch(self, tokens, targets),
            _ => panic!("qwen35::Qwen35 only supports Batch::Lm"),
        }
    }
    fn forward(&self) -> f32 {
        Qwen35::forward(self)
    }
    fn backward(&self) {
        Qwen35::backward(self)
    }
    fn zero_grads(&self) {
        Qwen35::zero_grads(self)
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Qwen35::adamw_step(self, t, lr, wd, clip, extra_scale)
    }
    fn poll_wait(&self) {
        Qwen35::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        Qwen35::param_names(self)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Qwen35::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Qwen35::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Qwen35::read_grad(self, name)
    }
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(Qwen35::logits_all(self, tokens))
    }
    fn save(&self, path: &str) {
        Qwen35::save(self, path)
    }
    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }
}
