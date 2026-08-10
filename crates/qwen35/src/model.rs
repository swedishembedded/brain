// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5-35B-A3B hybrid decoder forward assembly — device [`Step`]s wired
//! to the [`model::Model`] trait.
//!
//! **Scope, strictly**: text-only (no vision splice — that is separate
//! follow-on work), forward-only/inference-only (frozen weights,
//! `Role::Frozen` throughout — see [`Qwen35::backward`]'s panic), no
//! incremental/KV-cache decode (a full-sequence prefill-shaped forward over a
//! fixed `t` only, matching [`model::gdn`]'s own "chunked/prefill only"
//! scope), no T-padding (`t` must already be a multiple of the derived GDN
//! chunk size — asserted loudly in [`Qwen35::new_on`], see
//! [`gdn_chunk_size`]).
//!
//! **Honest scope note on numerical parity**: this environment has no
//! `torch`/`transformers` installed (see `docs/models/qwen35/status.md`'s
//! "environment gap" note about `tools/goldens/qwen35_dump_reference.py`), so
//! bit-exact parity against the real HF reference is **not achievable or
//! claimed here**. Every op below was checked line-for-line against the real
//! `/data/workspace/resources/qwen3.5/modeling_qwen3_5_moe.py` (not a
//! secondhand description), but the achievable and required bar for this
//! pass is *structural* correctness: compiles, runs, produces finite output,
//! deterministic across repeated runs at the same seed. See this module's
//! final report (delivered alongside this change) for the specific spots
//! that are least certain.
//!
//! One assumption worth flagging up front: `Qwen3_5MoeRMSNorm.forward`
//! (attention/MLP layer norms) computes `output * (1.0 + weight)`, not a
//! plain `output * weight` — the reference's own comment says so ("We
//! initialize with 0s to be 1 centered as the RMSNorm here does"). This
//! engine's shared `rmsnorm.wgsl` (used by every model, not just this one)
//! assumes the plain-multiply form, i.e. that a checkpoint's stored weight is
//! already the FINAL per-channel multiplier — which is exactly what
//! llama.cpp's GGUF conversion typically bakes in for this style of norm (the
//! `+1` folded into the stored value at conversion time), and is also what
//! `crates/qwen35/src/import.rs` (unmodified by this change) assumes. If that
//! assumption is wrong for some future checkpoint source, RMSNorm output
//! would be off by a `(x+1)` vs `x` factor — a real, if unlikely, gap, called
//! out here rather than silently assumed away. `Qwen3_5MoeRMSNormGated` (the
//! Gated DeltaNet output norm) is a genuinely different class with no such
//! `+1` (`hidden_states = self.weight * hidden_states...`, verified directly
//! against the reference), so `rmsnorm_fwd` is exactly right there.
//!
//! ## Layer forward, per the real reference (`Qwen3_5MoeDecoderLayer.forward`)
//!
//! Every layer, regardless of token-mixer type: `xn1 = rmsnorm(res)`, mix
//! (GDN or GQA, below), `xmid = res + mix_out`, `xn2 = rmsnorm(xmid)`, MoE
//! (universal — every layer, no dense fallback), `res' = xmid + moe_out`.
//!
//! **Gated DeltaNet** (`Qwen3_5MoeGatedDeltaNet.forward`): `mixed_qkv =
//! in_proj_qkv(xn1)` → depthwise causal conv1d (`causal_conv1d_fn`, SiLU
//! activation AFTER the conv — confirmed from `self.activation =
//! config.hidden_act` and every Qwen family config using `"silu"`, not
//! assumed) → split into `query,key,value` (one whole-row contiguous split —
//! confirmed via `torch.split(mixed_qkv, [key_dim,key_dim,value_dim],
//! dim=-1)`, i.e. NOT per-head) → L2-normalize `query`/`key` (no learnable
//! scale, confirmed `use_qk_l2norm_in_kernel=True` calls the bare `l2norm`
//! helper) → `beta=sigmoid(in_proj_b(xn1))`,
//! `g=-exp(A_log)*softplus(in_proj_a(xn1)+dt_bias)` (confirmed verbatim) →
//! repeat `query`/`key` from `linear_num_key_heads` to `linear_num_value_heads`
//! (`repeat_interleave`, i.e. `model::block::kv_expand_fwd`'s exact
//! `repeat_kv` semantics — confirmed, no new kernel needed) → chunk-major
//! permute (new `gdn_layout_permute.wgsl`, see its own header) →
//! `model::gdn::gdn_chunk_fwd` → permute back → gated RMSNorm
//! (`Qwen3_5MoeRMSNormGated`: norm computed on the UNGATED value first — "#
//! Norm before gate" in the reference — THEN `* weight`, THEN `*
//! SiLU(in_proj_z(xn1))`; confirmed this is exactly `rmsnorm_fwd` composed
//! with `silu.wgsl` + `mul.wgsl`, no new kernel) → `out_proj`.
//!
//! **GQA (`Qwen3_5MoeAttention.forward`)**: `q_proj` emits a DOUBLED width
//! (`num_heads*head_dim*2`) whose split into `query`/`gate` is **per-head
//! interleaved**, not a single whole-row split — confirmed:
//! `torch.chunk(q_proj(x).view(*shape,-1,head_dim*2), 2, dim=-1)` chunks the
//! LAST axis of a `[...,n_heads,2*head_dim]` view, so head `h`'s own
//! `2*head_dim` slice splits into its own first/second half. `concat_split
//! .wgsl` (existing — see this module's kernel-choice note below) handles
//! this by folding `n_heads` into its own batch axis (`N = rows*n_heads`,
//! `Ctot = 2*head_dim`, `Csrc = head_dim`). Then per-head QK-RMSNorm, partial
//! M-RoPE (`rope2d_partial_fwd`, `partial_rotary_factor` fraction rotated),
//! GQA attention, `ctx * sigmoid(gate)`, `o_proj`.
//!
//! ## Kernel-reuse notes (deviations from a literal reading of the task spec)
//!
//! - **qkv / q-gate splits**: the task's own text suggested `region_copy
//!   .wgsl` for these. Read closely, `region_copy` requires `src` and `dst`
//!   to share the SAME `row_stride`/`off` addressing (`dst[i] = src[i]` for
//!   the identical flat `i`) — it copies a sub-REGION between two
//!   same-shaped buffers, it cannot project a wide strided row into a fresh
//!   COMPACT narrower buffer (which is what a real split needs: downstream
//!   consumers like `l2norm_scale`/`gdn_chunk_fwd`/`gqa_fwd` all require
//!   compact operands, and none of them accept an extra `row_stride`
//!   parameter to work around it). `concat_split.wgsl` (existing — originally
//!   for `concat2`'s backward channel-slice) does exactly the needed gather:
//!   `da[n,c,h,w] = dy[n, c+c_off, h, w]` — setting `H=W=1` makes it a plain
//!   compact channel-slice-into-a-fresh-buffer copy, and folding a repeated
//!   axis (e.g. per-head) into its `N` handles the interleaved case too. Used
//!   for both splits; no new kernel.
//! - **chunk-major permute**: genuinely new (`gdn_layout_permute.wgsl`) — see
//!   its own header for why `nlc_nchw`/`nchw_nlc` (the only existing
//!   layout-permute kernels) don't cover a 5-index permute that also SPLITS
//!   the token axis into `(chunk, c)`.
//! - **GDN q/k head repeat**: `model::block::kv_expand_fwd` is exactly
//!   `repeat_kv`-shaped and shape-generic (any `hd`, not GQA-specific) — used
//!   as-is, no variant needed.
//! - **conv1d layout**: `conv1d.wgsl` is NCL (`[N,Cin,L]`); every other
//!   buffer in this engine is token-major (`[rows, C]` = `[B,T,C]` row-major,
//!   equivalently NLC with `N=B,L=T,C=C`). `nlc_nchw`/`nchw_nlc` (existing)
//!   convert between exactly these two layouts with `hw=T`; no new kernel.

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};

use audio::conv::{conv1d_fwd, Conv1d, ConvKernels};
use model::block::{gqa_fwd, kv_expand_fwd, rmsnorm_fwd, rope2d_partial_fwd, Gqa, KernelIds};
use model::gdn::{gdn_chunk_fwd, GdnIds, GdnScratch, GdnShape};
use model::moe::{
    expert_fwd, expert_fwd_i8, router_fwd_kind, shared_expert_fwd, ExpertScratch, ExpertScratch8, MoeIds, MoeIds8,
    MoeShape, RouterKind, SharedExpertIds, SharedExpertScratch,
};

use crate::config::{LayerType, Qwen35Config};
use crate::q8::{Q8Mixer, Qwen35Q8};

// ---- kernel pipeline (order fixes the indices below) -----------------------

pub const PIPELINES: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),                             // 0
    ("matmul", kernels::MATMUL),                                // 1
    ("embed", kernels::EMBED),                                  // 2
    ("sigmoid", kernels::SIGMOID),                               // 3
    ("silu", kernels::SILU),                                     // 4
    ("silu_mul", kernels::SILU_MUL),                             // 5
    ("mul", kernels::MUL),                                       // 6
    ("add2", kernels::ADD2),                                     // 7
    ("l2norm_scale", kernels::L2NORM_SCALE),                     // 8
    ("concat_split", kernels::CONCAT_SPLIT),                     // 9
    ("nlc_nchw", kernels::NLC_NCHW),                             // 10
    ("nchw_nlc", kernels::NCHW_NLC),                             // 11
    ("conv1d", kernels::CONV1D),                                 // 12
    ("gdn_decay_gate", kernels::GDN_DECAY_GATE),                 // 13
    ("gdn_layout_permute", kernels::GDN_LAYOUT_PERMUTE),         // 14
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),                 // 15
    ("gqa_scores", kernels::GQA_SCORES),                         // 16
    ("attn_softmax", kernels::ATTN_SOFTMAX),                     // 17
    ("gqa_apply", kernels::GQA_APPLY),                           // 18
    ("kv_expand", kernels::KV_EXPAND),                           // 19
    ("router_gate", kernels::ROUTER_GATE),                       // 20
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),             // 21
    ("scale_add", kernels::SCALE_ADD),                           // 22
    ("scale_row", kernels::SCALE_ROW),                           // 23
    ("bmm", kernels::BMM),                                       // 24
    ("bmm_acc", kernels::BMM_ACC),                               // 25
    ("gdn_chunk_cumsum_step", kernels::GDN_CHUNK_CUMSUM_STEP),   // 26
    ("gdn_decay_mask", kernels::GDN_DECAY_MASK),                 // 27
    ("gdn_mask_strict_lower", kernels::GDN_MASK_STRICT_LOWER),   // 28
    ("gdn_ut_step", kernels::GDN_UT_STEP),                       // 29
    ("gdn_add_identity", kernels::GDN_ADD_IDENTITY),             // 30
    ("gdn_row_scale_off", kernels::GDN_ROW_SCALE_OFF),           // 31
    ("gdn_decay_scale", kernels::GDN_DECAY_SCALE),               // 32
    ("gdn_state_decay", kernels::GDN_STATE_DECAY),               // 33
    ("exp", kernels::EXP),                                       // 34
    ("sub", kernels::SUB),                                       // 35
    ("region_copy", kernels::REGION_COPY),                       // 36
    ("ce_value", kernels::CE_VALUE_MASKED),                      // 37
    // -- int8 (DP4A) inference tier -- see `crate::q8`'s own module doc.
    ("max_abs_row", kernels::MAX_ABS_ROW),                       // 38
    ("quant_pack", kernels::QUANT_PACK),                         // 39
    // NOTE: `kernels::MATMUL_I8` (no suffix) is the STATIC per-tensor-scale
    // variant (`sx`/`sw` baked into the uniform, see its own doc); `Qwen35Q8`
    // needs the DYNAMIC per-token/per-channel variant (`sx`/`sw` as buffers,
    // indexed `sx[row]`/`sw[col]`) that `Q8::quant`/`Q8::mm8` actually
    // produce -- `kernels::MATMUL_I8_DYN`, exactly as `qwen::model.rs`'s own
    // `Q8` pipeline registers under this same local name (`qwen/src/model.rs:159`).
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),                   // 40
    ("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8),       // 41
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
const ROUTER_GATE: usize = 20;
const MOE_LINEAR_GATED: usize = 21;
const SCALE_ADD: usize = 22;
const SCALE_ROW: usize = 23;
const BMM: usize = 24;
const BMM_ACC: usize = 25;
const GDN_CHUNK_CUMSUM_STEP: usize = 26;
const GDN_DECAY_MASK: usize = 27;
const GDN_MASK_STRICT_LOWER: usize = 28;
const GDN_UT_STEP: usize = 29;
const GDN_ADD_IDENTITY: usize = 30;
const GDN_ROW_SCALE_OFF: usize = 31;
const GDN_DECAY_SCALE: usize = 32;
const GDN_STATE_DECAY: usize = 33;
const EXP: usize = 34;
const SUB: usize = 35;
const REGION_COPY: usize = 36;
const CE_VALUE: usize = 37;
const MAX_ABS_ROW: usize = 38;
const QUANT_PACK: usize = 39;
const MATMUL_I8: usize = 40;
const MOE_LINEAR_GATED_I8: usize = 41;

/// Forward-only: the backward-only slots [`KernelIds`] carries are never
/// dispatched, so they point at `rmsnorm` (index 0) — harmless, matching
/// `omni::thinker::kernel_ids`'s own convention.
fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMSNORM,
        rmsnorm_dx: RMSNORM,
        rmsnorm_dw: RMSNORM,
        rope: RMSNORM,
        rope_bwd: RMSNORM,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: RMSNORM,
        gqa_dv: RMSNORM,
        gqa_dq: RMSNORM,
        gqa_dk: RMSNORM,
        silu_mul: SILU_MUL,
        silu_da: RMSNORM,
        silu_db: RMSNORM,
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

fn moe_ids() -> MoeIds {
    MoeIds { router_gate: ROUTER_GATE, linear_gated: MOE_LINEAR_GATED, silu_mul: SILU_MUL, scale_add: SCALE_ADD }
}

/// int8 counterpart of [`moe_ids`], for [`model::moe::expert_fwd_i8`].
fn moe_ids8() -> MoeIds8 {
    MoeIds8 {
        linear_gated_i8: MOE_LINEAR_GATED_I8,
        silu_mul: SILU_MUL,
        scale_add: SCALE_ADD,
        quant: [MAX_ABS_ROW, QUANT_PACK],
    }
}

fn shared_expert_ids() -> SharedExpertIds {
    SharedExpertIds { matmul: MATMUL, silu_mul: SILU_MUL, sigmoid: SIGMOID, scale_row: SCALE_ROW, add2: ADD2 }
}

fn conv_kernels() -> ConvKernels {
    // Forward-only: dx/dw are never dispatched, point at fwd (harmless).
    ConvKernels { fwd: CONV1D, dx: CONV1D, dw: CONV1D }
}

/// The Gated DeltaNet chunk size this forward uses. The reference default is
/// 64 (`torch_chunk_gated_delta_rule`'s `chunk_size=64`); `model::gdn` is
/// prefill-only and asserts `t % chunk == 0` itself (no padding support), so
/// rather than force every caller's `t` to be a multiple of 64, this picks
/// the LARGEST candidate in `[64,32,16,8,4,2,1]` that divides `t` exactly —
/// landing on exactly 64 at the real 35B-A3B scale (`block_size=4096`,
/// `4096%64==0`) while still giving a tiny test config (`tiny()`:
/// `block_size=24`) a genuinely multi-chunk exercise (`24%8==0` -> chunk 8,
/// 3 chunks) instead of silently collapsing to one giant chunk covering the
/// whole sequence (which `t%t==0` would always pick first and would never
/// exercise the cross-chunk recurrence at all).
pub fn gdn_chunk_size(t: u32) -> u32 {
    for c in [64, 32, 16, 8, 4, 2, 1] {
        if t % c == 0 {
            return c;
        }
    }
    1
}

/// Owned scratch buffers for one [`model::gdn::gdn_chunk_fwd`] call — see
/// that module's [`GdnScratch`] doc for what each buffer holds. Freshly
/// allocated per Gated-DeltaNet layer call (never shared/reused across
/// layers or across forward passes), so every buffer here is zero-fresh —
/// `t_mat` is still passed through `Gpu::submit`'s `clears` list at the call
/// site per [`gdn_chunk_fwd`]'s own documented contract, defensively.
struct GdnScratchBufs {
    g_cs: DeviceBuffer,
    exp_g_cs: DeviceBuffer,
    k_beta: DeviceBuffer,
    v_beta: DeviceBuffer,
    k_beta_decay: DeviceBuffer,
    decay_mask: DeviceBuffer,
    raw_attn0: DeviceBuffer,
    attn0: DeviceBuffer,
    t_mat: DeviceBuffer,
    u: DeviceBuffer,
    w: DeviceBuffer,
    raw_intra: DeviceBuffer,
    intra_scores: DeviceBuffer,
    q_scaled: DeviceBuffer,
    decay_scale: DeviceBuffer,
    decayed_k: DeviceBuffer,
    v_prime: DeviceBuffer,
    v_new: DeviceBuffer,
}

impl GdnScratchBufs {
    fn new(g: &Gpu, shape: &GdnShape) -> GdnScratchBufs {
        let bhc = shape.bhc() as u64;
        let bh = shape.bh() as u64;
        let c = shape.chunk as u64;
        let dk = shape.dk as u64;
        let dv = shape.dv as u64;
        GdnScratchBufs {
            g_cs: g.storage(bhc * c),
            exp_g_cs: g.storage(bhc * c),
            k_beta: g.storage(bhc * c * dk),
            v_beta: g.storage(bhc * c * dv),
            k_beta_decay: g.storage(bhc * c * dk),
            decay_mask: g.storage(bhc * c * c),
            raw_attn0: g.storage(bhc * c * c),
            attn0: g.storage(bhc * c * c),
            t_mat: g.storage(bhc * c * c),
            u: g.storage(bhc * c * dv),
            w: g.storage(bhc * c * dk),
            raw_intra: g.storage(bhc * c * c),
            intra_scores: g.storage(bhc * c * c),
            q_scaled: g.storage(bh * c * dk),
            decay_scale: g.storage(bh * c),
            decayed_k: g.storage(bh * c * dk),
            v_prime: g.storage(bh * c * dv),
            v_new: g.storage(bh * c * dv),
        }
    }

    fn as_ref(&self) -> GdnScratch<'_> {
        GdnScratch {
            g_cs: &self.g_cs,
            exp_g_cs: &self.exp_g_cs,
            k_beta: &self.k_beta,
            v_beta: &self.v_beta,
            k_beta_decay: &self.k_beta_decay,
            decay_mask: &self.decay_mask,
            raw_attn0: &self.raw_attn0,
            attn0: &self.attn0,
            t_mat: &self.t_mat,
            u: &self.u,
            w: &self.w,
            raw_intra: &self.raw_intra,
            intra_scores: &self.intra_scores,
            q_scaled: &self.q_scaled,
            decay_scale: &self.decay_scale,
            decayed_k: &self.decayed_k,
            v_prime: &self.v_prime,
            v_new: &self.v_new,
        }
    }
}

/// Qwen3.5-35B-A3B hybrid decoder — forward/inference only (see module doc).
pub struct Qwen35 {
    pub gpu: Gpu,
    pub cfg: Qwen35Config,
    ps: ParamStore,
    /// `Some` selects the int8 (DP4A) inference tier for the linears
    /// `Qwen35Q8::is_i8_linear` names (`ps` excludes those names entirely —
    /// see [`Qwen35::new_impl_on`]'s role filter); `None` is the plain fp32
    /// path. See `crate::q8`'s module doc for exactly which linears that is
    /// and why.
    q8: Option<Qwen35Q8>,
    b: u32,
    t: u32,
    /// The GDN chunk size this instance was built for — see [`gdn_chunk_size`].
    chunk: u32,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    count: Cell<f32>,

    /// Residual stream, one entry per layer boundary (`res[0]` = embeddings,
    /// `res[n_layers]` = input to the final norm) — the SSA activation-cache
    /// convention `crates/glm/src/model.rs` uses, kept here even though
    /// nothing backprops through it (useful for parity debugging: any layer's
    /// residual output is independently readable).
    res: Vec<DeviceBuffer>,

    /// All-ones buffer of width `linear_key_head_dim`, bound as `l2norm_scale
    /// .wgsl`'s per-dim scale so its learnably-scaled L2-norm computes the
    /// reference's bare `l2norm(x)` (GDN's q/k norm has no learnable scale).
    ones_khd: DeviceBuffer,
    /// M-RoPE `cos`/`sin` tables (`qwenvl::mrope::mrope_tables`), built once
    /// at construction for the fixed `(b,t)` this instance decodes: text-only,
    /// so every axis carries the same plain sequential position per sequence
    /// (`qwenvl::mrope`'s own tests prove this collapses exactly to ordinary
    /// half-split RoPE).
    cos: DeviceBuffer,
    sin: DeviceBuffer,

    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,
}

impl Qwen35 {
    pub fn new(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, init, false)
    }

    /// Build on an existing device handle (test fixtures share one `Gpu` per
    /// binary — see `gpu_core::testgpu`).
    pub fn new_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_impl_on(gpu, cfg, b, t, init, false)
    }

    /// [`Self::new`] with the int8 (DP4A) inference tier: the attention/GDN
    /// mixer projections and every routed expert's gate/up/down are
    /// quantized (`crate::q8::Qwen35Q8::is_i8_linear`); the router, shared
    /// expert, embeddings and norms stay fp32. See `crate::q8`'s module doc
    /// for the full rationale. Inference-only, same as the fp32 path
    /// (`Qwen35::backward` panics regardless).
    pub fn new_i8(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, init, true)
    }

    /// [`Self::new_i8`] on an existing device handle — see [`Self::new_on`].
    pub fn new_on_i8(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_impl_on(gpu, cfg, b, t, init, true)
    }

    fn new_impl_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, i8: bool) -> Qwen35 {
        let chunk = gdn_chunk_size(t);
        assert_eq!(
            t % chunk,
            0,
            "qwen35: t={t} is not a multiple of the derived GDN chunk size {chunk} -- \
             model::gdn is prefill-only (no T-padding support, see its module doc); \
             gdn_chunk_size always returns a value that divides t by construction, so \
             this assert failing would mean a logic error in gdn_chunk_size itself"
        );

        // Inference-only pass: every weight is Role::Frozen (no grad/Adam
        // buffers allocated at all -- see paramstore::ParamStore::new_with_roles_src).
        // In int8 mode the linears `Qwen35Q8::is_i8_linear` names live in
        // `q8` (packed int8), NOT the fp32 store -- filter them out here so
        // no redundant fp32 copy is ever uploaded (mirrors `qwen::model.rs`'s
        // own `Q8::is_i8_linear` filter, `model.rs:504-507` in that crate).
        let roles: Vec<(String, usize, Role)> = cfg
            .param_list()
            .into_iter()
            .filter(|(n, _)| !(i8 && Qwen35Q8::is_i8_linear(n)))
            .map(|(n, c)| (n, c, Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);

        // Quantize+upload the int8 linears from the SAME source, streaming
        // one tensor at a time (see `Qwen35Q8::build`'s own doc).
        let q8 = if i8 { Some(Qwen35Q8::build(&gpu, src, &cfg, b * t, MAX_ABS_ROW, QUANT_PACK, MATMUL_I8)) } else { None };

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let mut res = Vec::with_capacity(cfg.n_layers as usize + 1);
        for _ in 0..=cfg.n_layers {
            res.push(gpu.storage(n * d));
        }

        let ones_khd = gpu.storage_init("qwen35.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);

        // Text-only: every axis of the M-RoPE table carries the same plain
        // sequential position, reset per sequence (row = batch*t + pos).
        let positions: Vec<[u32; 3]> = (0..b).flat_map(|_| (0..t).map(|ti| [ti, ti, ti])).collect();
        let (cos, sin) = qwenvl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("qwen35.rope_cos", &cos);
        let sin = gpu.storage_init("qwen35.rope_sin", &sin);

        let tokens = gpu.storage(n);
        let targets = gpu.storage(n);
        let logits = gpu.storage(n * cfg.vocab as u64);
        let ce_buf = gpu.storage(n);

        Qwen35 { gpu, cfg, ps, q8, b, t, chunk, tokens, targets, count: Cell::new(1.0), res, ones_khd, cos, sin, logits, ce_buf }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    pub fn set_batch(&self, tokens: &[u32], targets: &[u32]) {
        self.gpu.write(&self.tokens, tokens);
        self.gpu.write(&self.targets, targets);
        let c = targets.iter().filter(|&&v| v != model::IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    // ---- one Gated DeltaNet (Linear) layer --------------------------------

    fn layer_gdn_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32) -> DeviceBuffer {
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

        // int8 linears for this layer, if any -- see `crate::q8`'s module doc.
        let q8l = self.q8.as_ref().map(|q8| match &q8.mixers[l] {
            Q8Mixer::Gdn(ql) => (q8, ql),
            Q8Mixer::Gqa(_) => panic!("qwen35 q8: layer {l} expected a GDN mixer, found GQA (layer_types() drift)"),
        });

        // 1. mixed_qkv = in_proj_qkv(xn1).
        let mixed_qkv = g.storage((n * conv_dim) as u64);
        if let Some((q8, ql)) = q8l {
            // xn1 quantized once here; reused unchanged by step 5's in_proj_b/a/z
            // below (no quant() call happens on any OTHER buffer in between).
            let mut s = Vec::new();
            q8.quant(g, &mut s, xn1, d, n);
            q8.mm8(g, &mut s, &ql.in_proj_qkv, &mixed_qkv, n);
            g.submit(&[], &s);
        } else {
            g.submit(&[], &[g.step(MATMUL, &[xn1, self.w(&p("in_proj_qkv.weight")), &mixed_qkv], &[n, d, conv_dim], n * conv_dim)]);
        }

        // 2. Depthwise causal conv1d + SiLU (activation AFTER the conv --
        // `causal_conv1d_fn(..., activation=self.activation)`, `self.activation
        // = config.hidden_act`, assumed "silu" per every Qwen-family config).
        // conv1d.wgsl is NCL ([N,Cin,L]); mixed_qkv is token-major ([B,T,C] =
        // NLC with N=B,L=T,C=conv_dim) -- convert both ways with the existing
        // nlc_nchw/nchw_nlc kernels (see module doc).
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

        // 3. Split into query/key/value -- ONE whole-row contiguous split
        // (`torch.split(mixed_qkv, [key_dim,key_dim,value_dim], dim=-1)`),
        // not per-head. `concat_split.wgsl` with H=W=1 is a plain compact
        // channel-slice gather (see module doc's kernel-choice note).
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

        // 4. L2-normalize query/key -- bare l2norm (no learnable scale): bind
        // the all-ones buffer as l2norm_scale.wgsl's per-dim scale.
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
        if let Some((q8, ql)) = q8l {
            // Reuses step 1's xn1 quantization already resident in q8.xq/sx
            // (no intervening quant() call -- see step 1's own comment).
            let mut s = Vec::new();
            q8.mm8(g, &mut s, &ql.in_proj_b, &bproj, n);
            q8.mm8(g, &mut s, &ql.in_proj_a, &aproj, n);
            q8.mm8(g, &mut s, &ql.in_proj_z, &z, n);
            g.submit(&[], &s);
        } else {
            g.submit(
                &[],
                &[
                    g.step(MATMUL, &[xn1, self.w(&p("in_proj_b.weight")), &bproj], &[n, d, nvh], n * nvh),
                    g.step(MATMUL, &[xn1, self.w(&p("in_proj_a.weight")), &aproj], &[n, d, nvh], n * nvh),
                    g.step(MATMUL, &[xn1, self.w(&p("in_proj_z.weight")), &z], &[n, d, value_dim], n * value_dim),
                ],
            );
        }
        let beta = g.storage((n * nvh) as u64);
        let g_decay = g.storage((n * nvh) as u64);
        g.submit(
            &[],
            &[
                g.step(SIGMOID, &[&bproj, &beta], &[n * nvh], n * nvh),
                g.step(GDN_DECAY_GATE, &[&aproj, self.w(&p("A_log")), self.w(&p("dt_bias")), &g_decay], &[n, nvh], n * nvh),
            ],
        );

        // 6. Repeat query/key from linear_num_key_heads to linear_num_value_heads
        // (repeat_interleave -- exactly kv_expand_fwd's repeat_kv semantics).
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

        // 8. gdn_chunk_fwd -- the chunked-recurrence forward itself.
        let bh = shape.bh() as u64;
        let scratch = GdnScratchBufs::new(g, &shape);
        let initial_state = g.storage(bh * khd as u64 * vhd as u64);
        let final_state = g.storage(bh * khd as u64 * vhd as u64);
        let out_cm = g.storage(shape.bhc() as u64 * chunk as u64 * vhd as u64);
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
        g.submit(&[&scratch.t_mat], &steps);

        // 9. Permute back to token-major.
        let out_tok = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[g.step(GDN_LAYOUT_PERMUTE, &[&out_cm, &out_tok], &[b, nvh, n_chunks, chunk, vhd, 0], b * nvh * n_chunks * chunk * vhd)],
        );

        // 10. Gated RMSNorm (`Qwen3_5MoeRMSNormGated`, "norm before gate"):
        // normed = RMSNorm(out_tok)*weight, THEN * SiLU(z).
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
        if let Some((q8, ql)) = q8l {
            // Fresh quant() call: `gated` is a different activation from
            // xn1 (steps 1/5 above), safe to overwrite q8.xq/sx now that
            // every earlier mm8 reading the old contents is already queued.
            let mut s = Vec::new();
            q8.quant(g, &mut s, &gated, value_dim, n);
            q8.mm8(g, &mut s, &ql.out_proj, &out, n);
            g.submit(&[], &s);
        } else {
            g.submit(&[], &[g.step(MATMUL, &[&gated, self.w(&p("out_proj.weight")), &out], &[n, value_dim, d], n * d)]);
        }
        out
    }

    // ---- one GQA (Full) layer ----------------------------------------------

    fn layer_gqa_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, qd, kvd) = (c.q_proj_dim(), c.q_dim(), c.kv_dim());
        let p = |s: &str| format!("blocks.{l}.self_attn.{s}");

        // int8 linears for this layer, if any -- see `crate::q8`'s module doc.
        let q8l = self.q8.as_ref().map(|q8| match &q8.mixers[l] {
            Q8Mixer::Gqa(ql) => (q8, ql),
            Q8Mixer::Gdn(_) => panic!("qwen35 q8: layer {l} expected a GQA mixer, found GDN (layer_types() drift)"),
        });

        let q_full = g.storage((n * qpd) as u64);
        let k = g.storage((n * kvd) as u64);
        let v = g.storage((n * kvd) as u64);
        if let Some((q8, ql)) = q8l {
            // xn1 quantized once, shared by q/k/v (DP4A GEMM per projection).
            let mut s = Vec::new();
            q8.quant(g, &mut s, xn1, d, n);
            q8.mm8(g, &mut s, &ql.q_proj, &q_full, n);
            q8.mm8(g, &mut s, &ql.k_proj, &k, n);
            q8.mm8(g, &mut s, &ql.v_proj, &v, n);
            g.submit(&[], &s);
        } else {
            g.submit(
                &[],
                &[
                    g.step(MATMUL, &[xn1, self.w(&p("q_proj.weight")), &q_full], &[n, d, qpd], n * qpd),
                    g.step(MATMUL, &[xn1, self.w(&p("k_proj.weight")), &k], &[n, d, kvd], n * kvd),
                    g.step(MATMUL, &[xn1, self.w(&p("v_proj.weight")), &v], &[n, d, kvd], n * kvd),
                ],
            );
        }

        // Per-head de-interleaved split of q_full's [query|gate] halves --
        // NOT a whole-row split (see module doc). Fold n_heads into
        // concat_split's own N so each head's 2*head_dim block splits into
        // its own first/second half.
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
        if let Some((q8, ql)) = q8l {
            let mut s = vec![
                g.step(SIGMOID, &[&q_gate, &gate], &[n * qd], n * qd),
                g.step(MUL, &[&ctx, &gate, &ctx_gated], &[n * qd], n * qd),
            ];
            // Fresh quant() call: ctx_gated is a different activation from
            // xn1 above, safe to overwrite q8.xq/sx now that every earlier
            // mm8 reading the old contents is already queued ahead of it.
            q8.quant(g, &mut s, &ctx_gated, qd, n);
            q8.mm8(g, &mut s, &ql.o_proj, &out, n);
            g.submit(&[], &s);
        } else {
            g.submit(
                &[],
                &[
                    g.step(SIGMOID, &[&q_gate, &gate], &[n * qd], n * qd),
                    g.step(MUL, &[&ctx, &gate, &ctx_gated], &[n * qd], n * qd),
                    g.step(MATMUL, &[&ctx_gated, self.w(&p("o_proj.weight")), &out], &[n, qd, d], n * d),
                ],
            );
        }
        out
    }

    // ---- MoE sublayer, universal for every layer ---------------------------

    fn moe_sublayer(&self, l: usize, xmid: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let e = c.n_experts;
        let moe_ff = c.moe_intermediate_size;
        let shared_ff = c.shared_expert_intermediate_size;
        let p = |s: &str| format!("blocks.{l}.{s}");

        let xn2 = g.storage((n * d) as u64);
        let router_logits = g.storage((n * e) as u64);
        let mut steps = vec![
            rmsnorm_fwd(g, &kernel_ids(), xmid, self.w(&p("ln2.weight")), &xn2, d, n),
            g.step(MATMUL, &[&xn2, self.w(&p("mlp.router.weight")), &router_logits], &[n, d, e], n * e),
        ];

        let shape = MoeShape { rows: n, d_model: d, moe_ff, n_experts: e, top_k: c.top_k };
        let gate = g.storage((n * e) as u64);
        // aux_coef/z_coef only affect router_bwd (never reached here -- see
        // model::moe::router_fwd_kind's forward-only call path), so 0.0 is a
        // pure "unused" value, not a behaviour change to the forward gate math.
        steps.push(router_fwd_kind(g, &moe_ids(), RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0 }, &shape, &router_logits, None, &gate, None));

        let moe_acc = g.storage((n * d) as u64);
        // Router and gate above are ALWAYS fp32 (see `crate::q8`'s module
        // doc for why); only the routed experts' gate/up/down switch tier.
        if let Some(q8) = &self.q8 {
            // xn2 quantized once, shared by every expert's gate/up (the
            // down-projection's input `h` is expert-specific and quantized
            // separately inside `expert_fwd_i8`'s own scratch).
            q8.quant(g, &mut steps, &xn2, d, n);
            let ml = &q8.moe[l];
            let ids8 = moe_ids8();
            let scratch8 = ExpertScratch8 {
                gate_pre: &g.storage((n * moe_ff) as u64),
                up: &g.storage((n * moe_ff) as u64),
                h: &g.storage((n * moe_ff) as u64),
                hq: &g.storage((n * moe_ff / 4) as u64),
                sh: &g.storage(n as u64),
                expert_out: &g.storage((n * d) as u64),
            };
            for ei in 0..e {
                let ex = &ml.experts[ei as usize];
                steps.extend(expert_fwd_i8(
                    g,
                    &ids8,
                    &shape,
                    &q8.xq,
                    &q8.sx,
                    &gate,
                    ex.gate.as_moe(),
                    ex.up.as_moe(),
                    ex.down.as_moe(),
                    &scratch8,
                    &moe_acc,
                    ei,
                    ei != 0,
                ));
            }
        } else {
            let scratch = ExpertScratch {
                gate_pre: &g.storage((n * moe_ff) as u64),
                up: &g.storage((n * moe_ff) as u64),
                h: &g.storage((n * moe_ff) as u64),
                expert_out: &g.storage((n * d) as u64),
            };
            for ei in 0..e {
                let ep = |s: &str| format!("blocks.{l}.mlp.experts.{ei}.{s}");
                steps.extend(expert_fwd(
                    g,
                    &moe_ids(),
                    &shape,
                    &xn2,
                    &gate,
                    self.w(&ep("gate.weight")),
                    self.w(&ep("up.weight")),
                    self.w(&ep("down.weight")),
                    &scratch,
                    &moe_acc,
                    ei,
                    ei != 0,
                ));
            }
        }

        let moe_out = g.storage((n * d) as u64);
        let sh_scratch = SharedExpertScratch {
            gate_pre: &g.storage((n * shared_ff) as u64),
            up: &g.storage((n * shared_ff) as u64),
            h: &g.storage((n * shared_ff) as u64),
            mlp_out: &g.storage((n * d) as u64),
            gate_logits: &g.storage(n as u64),
            gate_scalar: &g.storage(n as u64),
            scaled: &g.storage((n * d) as u64),
        };
        steps.extend(shared_expert_fwd(
            g,
            &shared_expert_ids(),
            n,
            d,
            shared_ff,
            &xn2,
            self.w(&p("mlp.shared_expert.gate.weight")),
            self.w(&p("mlp.shared_expert.up.weight")),
            self.w(&p("mlp.shared_expert.down.weight")),
            Some(self.w(&p("mlp.shared_expert_gate.weight"))),
            &sh_scratch,
            &moe_acc,
            &moe_out,
        ));

        g.submit(&[], &steps);
        moe_out
    }

    // ---- full stack ----------------------------------------------------------

    fn run_forward(&self) {
        let g = &self.gpu;
        let n = self.b * self.t;
        let d = self.cfg.d_model;

        g.submit(&[], &[g.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, n], n * d)]);

        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            let xres = &self.res[l];
            let xn1 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), xres, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n)]);

            let attn_out = match ty {
                LayerType::Linear => self.layer_gdn_fwd(l, &xn1, n),
                LayerType::Full => self.layer_gqa_fwd(l, &xn1, n),
            };

            let xmid = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[xres, &attn_out, &xmid], &[n * d], n * d)]);

            let moe_out = self.moe_sublayer(l, &xmid, n);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &moe_out, &self.res[l + 1]], &[n * d], n * d)]);
        }

        let xn_final = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &self.res[self.cfg.n_layers as usize], self.w("norm.weight"), &xn_final, d, n)]);

        let v = self.cfg.vocab;
        g.submit(&[], &[g.step(MATMUL, &[&xn_final, self.w(self.cfg.head_weight()), &self.logits], &[n, d, v], n * v)]);
    }

    pub fn forward(&self) -> f32 {
        self.run_forward();
        let n = self.b * self.t;
        self.gpu.submit(&[], &[self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, self.cfg.vocab, model::IGNORE], n)]);
        let vals = self.gpu.read(&self.ce_buf, n as usize);
        vals.iter().sum::<f32>() / self.count.get()
    }

    /// Per-position logits for one sequence (`b` must be 1, `tokens.len()`
    /// must equal the configured `t` -- this pass has no partial-length
    /// prefill, see module doc).
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

const BACKWARD_PANIC: &str =
    "qwen35: backward not implemented -- model::gdn has no backward yet, see crates/model/tests/gdn_chunk_bwd.rs";

impl model::Model for Qwen35 {
    type Config = Qwen35Config;

    fn new(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Qwen35::new(cfg, b, t, init)
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
        panic!("{BACKWARD_PANIC}")
    }
    fn zero_grads(&self) {
        panic!("{BACKWARD_PANIC}")
    }
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra_scale: f32) {
        panic!("{BACKWARD_PANIC}")
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
    fn read_grad(&self, _name: &str) -> Vec<f32> {
        panic!("qwen35: no gradients exist -- every weight is Role::Frozen (see backward()'s panic)")
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
