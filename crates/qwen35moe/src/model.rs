// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5-35B-A3B hybrid decoder forward AND backward assembly — device
//! [`Step`]s wired to the [`model::Model`] trait.
//!
//! **Scope, strictly**: no incremental/KV-cache decode with an image splice
//! (single-sequence text decode via [`Qwen35::step`] is separate follow-on
//! work; see `crate::vl`'s own doc) — a prefill-shaped vision-language
//! embedding splice IS wired here ([`Qwen35::enable_mm_splice`]/
//! [`Qwen35::write_img_embeds`], driven by `crate::vl::Qwen35Vl`), mirroring
//! `qwen3::Qwen`'s own seam. No incremental/KV-cache decode (a full-sequence
//! prefill-shaped forward over a fixed `t` only, matching [`model::gdn`]'s own
//! "chunked/prefill only" scope), no T-padding (`t` must already be a
//! multiple of the derived GDN chunk size — asserted loudly in
//! [`Qwen35::new_on`], see [`gdn_chunk_size`]). **Two construction paths**:
//! [`Qwen35::new`]/[`Qwen35::new_i8`] build a frozen (`Role::Frozen`),
//! forward-only instance (`backward`/`zero_grads`/`adamw_step` all assert and
//! panic on such an instance — see [`Qwen35::backward`]'s own assert);
//! [`Qwen35::new_train`] builds a fully trainable (`Role::Trainable`
//! everywhere, full-parameter — no LoRA-specific subset) instance whose
//! `forward()` additionally saves the activation cache `backward()` reads
//! (see [`Qwen35::train_acts`]'s own doc for the exact "one forward, one
//! backward, then the cache is gone" contract). Int8 and training are
//! mutually exclusive.
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
//! `crates/qwen35moe/src/import.rs` (unmodified by this change) assumes. If that
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

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::Shard;
use paramstore::{ParamStore, Role};

use audio::conv::{conv1d_bwd, conv1d_fwd, Conv1d, ConvKernels};
use model::block::{
    gqa_bwd, gqa_decode_step, gqa_fwd, kv_expand_bwd, kv_expand_fwd, rmsnorm_bwd, rmsnorm_fwd, rope2d_partial_bwd, rope2d_partial_fwd,
    swiglu_bwd, Gqa, GqaDecodeIds, KernelIds,
};
use model::gdn::{
    gdn_causal_conv1d_step, gdn_chunk_bwd, gdn_chunk_fwd, gdn_chunk_fwd_train, gdn_recurrent_step, GdnBwdIds, GdnBwdScratch, GdnConvIds,
    GdnConvShape, GdnIds, GdnRecurrentScratch, GdnScratch, GdnScratchTrain, GdnShape,
};
use model::moe::{
    expert_fwd, expert_fwd_i8, moe_layer_bwd, router_fwd_kind, shared_expert_fwd, ExpertBwdScratch, ExpertGrads,
    ExpertScratch, ExpertScratch8, MoeActs, MoeIds, MoeIds8, MoeIdsBwd, MoeShape, RouterBwdIds, RouterKind,
    SharedExpertIds, SharedExpertScratch,
};
use optim::Optim;

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
    // produce -- `kernels::MATMUL_I8_DYN`, exactly as `qwen3::model.rs`'s own
    // `Q8` pipeline registers under this same local name (`qwen/src/model.rs:159`).
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),                   // 40
    ("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8),       // 41
    // -- training (backward + AdamW) tier -- see `Qwen35::new_train`/`backward`.
    ("rms_inv", kernels::RMS_INV),                               // 42
    ("rmsnorm_dx", kernels::RMSNORM_DX),                         // 43
    ("rmsnorm_dw", kernels::RMSNORM_DW),                         // 44
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),               // 45
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),                         // 46
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),                         // 47
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),                         // 48
    ("silu_bwd_da", kernels::SILU_BWD_DA),                       // 49
    ("silu_bwd_db", kernels::SILU_BWD_DB),                       // 50
    ("sigmoid_bwd", kernels::SIGMOID_BWD),                       // 51
    ("silu_bwd", kernels::SILU_BWD),                             // 52
    ("concat2", kernels::CONCAT2),                               // 53
    ("bias_grad", kernels::BIAS_GRAD),                           // 54
    ("kv_expand_bwd", kernels::KV_EXPAND_BWD),                   // 55
    ("matmul_dx", kernels::MATMUL_DX),                           // 56
    ("matmul_dw", kernels::MATMUL_DW),                           // 57
    ("conv1d_dx", kernels::CONV1D_DX),                           // 58
    ("conv1d_dw", kernels::CONV1D_DW),                           // 59
    ("gdn_decay_gate_bwd", kernels::GDN_DECAY_GATE_BWD),         // 60
    ("splice_add", kernels::SPLICE_ADD),                         // 61
    ("row_dot", kernels::ROW_DOT),                               // 62
    ("gdn_chunk_reverse_cumsum_step", kernels::GDN_CHUNK_REVERSE_CUMSUM_STEP), // 63
    ("gdn_ut_bwd_dattn0", kernels::GDN_UT_BWD_DATTN0),           // 64
    ("gdn_ut_bwd_dtmat", kernels::GDN_UT_BWD_DTMAT),             // 65
    ("gdn_mask_strict_lower_bwd", kernels::GDN_MASK_STRICT_LOWER_BWD), // 66
    ("gdn_decay_mask_bwd", kernels::GDN_DECAY_MASK_BWD),         // 67
    ("gdn_decay_scale_bwd", kernels::GDN_DECAY_SCALE_BWD),       // 68
    ("gdn_decay_scale_bwd_last", kernels::GDN_DECAY_SCALE_BWD_LAST), // 69
    ("gdn_state_decay_bwd_dscale", kernels::GDN_STATE_DECAY_BWD_DSCALE), // 70
    ("router_bwd", kernels::ROUTER_BWD),                         // 71
    ("expert_counts", kernels::EXPERT_COUNTS),                   // 72
    ("scale_add_dexp", kernels::SCALE_ADD_DEXP),                 // 73
    ("scale_add_dgate", kernels::SCALE_ADD_DGATE),               // 74
    ("moe_linear_gated_dx", kernels::MOE_LINEAR_GATED_DX),       // 75
    ("moe_linear_gated_dw", kernels::MOE_LINEAR_GATED_DW),       // 76
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),               // 77
    ("adamw", kernels::ADAMW),                                   // 78
    ("gradnorm_sq", kernels::GRADNORM_SQ),                       // 79
    ("grad_scale", kernels::GRAD_SCALE),                         // 80
    ("clip_coef", kernels::CLIP_COEF),                           // 81
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),                 // 82
    ("emb_bwd", kernels::EMB_BWD),                               // 83
    ("ce_grad", kernels::CE_GRAD_MASKED),                        // 84
    // -- single-sequence incremental decode tier -- see `Qwen35::step`.
    ("causal_conv1d_step", kernels::CAUSAL_CONV1D_STEP),         // 85
    ("kv_append", kernels::KV_APPEND),                           // 86
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),         // 87
    ("decode_softmax", kernels::DECODE_SOFTMAX),                 // 88
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),           // 89
    // -- vision-language embedding splice tier -- see `Qwen35::enable_mm_splice`
    // / `crate::vl::Qwen35Vl`. Appended at the true end per this file's own
    // "local PIPELINES indices are position-dependent" convention (matching
    // `qwen3::model.rs`'s own `SPLICE_ADD_OFFSET_SRC` addition) -- inserting
    // anywhere else would silently shift every constant below out of sync.
    ("splice", kernels::SPLICE),                                 // 90
    ("splice_bwd", kernels::SPLICE_BWD),                         // 91
    // -- LoRA tier -- see `Qwen35::lora_fwd`/`Qwen35::proj_bwd`'s LoRA branch.
    // Appended at the true end, same convention as the splice tier above.
    ("axpy", kernels::AXPY),                                     // 92
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
const RMS_INV: usize = 42;
const RMSNORM_DX: usize = 43;
const RMSNORM_DW: usize = 44;
const GQA_BWD_DSCORES: usize = 45;
const GQA_BWD_DV: usize = 46;
const GQA_BWD_DQ: usize = 47;
const GQA_BWD_DK: usize = 48;
const SILU_BWD_DA: usize = 49;
const SILU_BWD_DB: usize = 50;
const SIGMOID_BWD: usize = 51;
const SILU_BWD: usize = 52;
const CONCAT2: usize = 53;
const BIAS_GRAD: usize = 54;
const KV_EXPAND_BWD: usize = 55;
const MATMUL_DX: usize = 56;
const MATMUL_DW: usize = 57;
const CONV1D_DX: usize = 58;
const CONV1D_DW: usize = 59;
const GDN_DECAY_GATE_BWD: usize = 60;
const SPLICE_ADD: usize = 61;
const ROW_DOT: usize = 62;
const GDN_CHUNK_REVERSE_CUMSUM_STEP: usize = 63;
const GDN_UT_BWD_DATTN0: usize = 64;
const GDN_UT_BWD_DTMAT: usize = 65;
const GDN_MASK_STRICT_LOWER_BWD: usize = 66;
const GDN_DECAY_MASK_BWD: usize = 67;
const GDN_DECAY_SCALE_BWD: usize = 68;
const GDN_DECAY_SCALE_BWD_LAST: usize = 69;
const GDN_STATE_DECAY_BWD_DSCALE: usize = 70;
const ROUTER_BWD: usize = 71;
const EXPERT_COUNTS: usize = 72;
const SCALE_ADD_DEXP: usize = 73;
const SCALE_ADD_DGATE: usize = 74;
const MOE_LINEAR_GATED_DX: usize = 75;
const MOE_LINEAR_GATED_DW: usize = 76;
const L2NORM_SCALE_DX: usize = 77;
const ADAMW: usize = 78;
const GRADNORM_SQ: usize = 79;
const GRAD_SCALE: usize = 80;
const CLIP_COEF: usize = 81;
const GRAD_SCALE_BUF: usize = 82;
const EMB_BWD: usize = 83;
const CE_GRAD: usize = 84;
const CAUSAL_CONV1D_STEP: usize = 85;
const KV_APPEND: usize = 86;
const ATTN_DECODE_SCORES: usize = 87;
const DECODE_SOFTMAX: usize = 88;
const ATTN_DECODE_APPLY: usize = 89;
const SPLICE: usize = 90;
const SPLICE_BWD: usize = 91;
const AXPY: usize = 92;

/// Every slot is a REAL kernel now (backward is wired, see [`Qwen35::backward`]):
/// `rope`/`rope_bwd` still point at `rmsnorm` (index 0) because qwen35 never
/// dispatches `block::rope_fwd`/`rope_bwd` (it uses the M-RoPE table-driven
/// `rope2d_partial_{fwd,bwd}` instead, which take their own kernel index, not
/// a [`KernelIds`] field) — harmless, matching `omni::thinker::kernel_ids`'s
/// own convention for a slot this model genuinely never dispatches.
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

/// [`model::gdn::gdn_causal_conv1d_step`]'s kernel id -- the streaming
/// causal-conv decode step, dispatched by [`Qwen35::layer_gdn_decode_step`]
/// in place of `layer_gdn_fwd`'s whole-sequence `conv1d_fwd`.
fn gdn_conv_ids() -> GdnConvIds {
    GdnConvIds { causal_conv1d_step: CAUSAL_CONV1D_STEP }
}

/// [`model::block::gqa_decode_step`]'s kernel ids -- the incremental
/// KV-cache-append-and-attend decode step, dispatched by
/// [`Qwen35::layer_gqa_decode_step`] in place of `layer_gqa_fwd`'s
/// whole-sequence `gqa_fwd`. Same four kernels `qwen3::Qwen::decode_ids`
/// resolves, hoisted through `model::block` for exactly this reuse.
fn gqa_decode_ids() -> GqaDecodeIds {
    GqaDecodeIds { kv_append: KV_APPEND, attn_decode_scores: ATTN_DECODE_SCORES, decode_softmax: DECODE_SOFTMAX, attn_decode_apply: ATTN_DECODE_APPLY }
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

/// `dx`/`dw` are real kernels now (see [`Qwen35::backward`]'s GDN conv1d
/// backward); an inference-only build never dispatches them.
fn conv_kernels() -> ConvKernels {
    ConvKernels { fwd: CONV1D, dx: CONV1D_DX, dw: CONV1D_DW }
}

/// Backward-only kernel ids [`model::gdn::gdn_chunk_bwd`]/[`gdn_chunk_fwd_train`]
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

/// [`model::moe::router_bwd`]'s kernel ids — `Softmax` router (qwen35's own,
/// see `moe_sublayer`'s `RouterKind::Softmax` choice), so `expert_counts` is
/// required (aux-loss usage fractions), unused by the returned scalar loss
/// (`aux_coef=0.0`, see `moe_sublayer`'s own comment) but still dispatched —
/// `router_bwd.wgsl`'s own interface requires the `fe` buffer to exist.
fn router_bwd_ids() -> RouterBwdIds {
    RouterBwdIds { router_bwd: ROUTER_BWD, expert_counts: Some(EXPERT_COUNTS) }
}

/// [`model::moe::expert_dgate`]/[`expert_bwd`]'s kernel ids (composed by
/// [`moe_layer_bwd`]) — `linear_gated: true` selects the row-skipping
/// backward kernels, matching `moe_sublayer`'s own gated forward
/// (`MOE_LINEAR_GATED`).
fn moe_bwd_ids() -> MoeIdsBwd {
    MoeIdsBwd {
        scale_add_dexp: SCALE_ADD_DEXP,
        scale_add_dgate: SCALE_ADD_DGATE,
        silu_da: SILU_BWD_DA,
        silu_db: SILU_BWD_DB,
        linear_dx: MOE_LINEAR_GATED_DX,
        linear_dw: MOE_LINEAR_GATED_DW,
        linear_gated: true,
    }
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

/// Owned scratch for one [`gdn_chunk_fwd_train`] call — the training-mode
/// sibling of [`GdnScratchBufs`] that additionally saves the per-chunk
/// history [`gdn_chunk_bwd`] reads back (see [`GdnScratchTrain`]'s own doc for
/// exactly which field is which). `clears()` lists every field
/// [`gdn_chunk_fwd_train`]'s own doc says MUST be zeroed before submit
/// (`t_mat`, every `_hist` field, `state_history`).
struct GdnScratchTrainBufs {
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
    q_scaled_hist: DeviceBuffer,
    decay_scale_hist: DeviceBuffer,
    decayed_k_hist: DeviceBuffer,
    v_prime_hist: DeviceBuffer,
    v_new_hist: DeviceBuffer,
    state_history: DeviceBuffer,
}

impl GdnScratchTrainBufs {
    fn new(g: &Gpu, shape: &GdnShape) -> GdnScratchTrainBufs {
        let bhc = shape.bhc() as u64;
        let bh = shape.bh() as u64;
        let c = shape.chunk as u64;
        let dk = shape.dk as u64;
        let dv = shape.dv as u64;
        let n_chunks = shape.n_chunks() as u64;
        GdnScratchTrainBufs {
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
            q_scaled_hist: g.storage(bhc * c * dk),
            decay_scale_hist: g.storage(bhc * c),
            decayed_k_hist: g.storage(bhc * c * dk),
            v_prime_hist: g.storage(bhc * c * dv),
            v_new_hist: g.storage(bhc * c * dv),
            state_history: g.storage((n_chunks + 1) * bh * dk * dv),
        }
    }

    fn as_ref(&self) -> GdnScratchTrain<'_> {
        GdnScratchTrain {
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
            q_scaled_hist: &self.q_scaled_hist,
            decay_scale_hist: &self.decay_scale_hist,
            decayed_k_hist: &self.decayed_k_hist,
            v_prime_hist: &self.v_prime_hist,
            v_new_hist: &self.v_new_hist,
            state_history: &self.state_history,
        }
    }

    fn clears(&self) -> Vec<&DeviceBuffer> {
        vec![
            &self.t_mat,
            &self.q_scaled_hist,
            &self.decay_scale_hist,
            &self.decayed_k_hist,
            &self.v_prime_hist,
            &self.v_new_hist,
            &self.state_history,
        ]
    }
}

/// Owned scratch for one [`gdn_chunk_bwd`] call — see [`GdnBwdScratch`]'s own
/// doc for what each field holds and which MUST be zeroed (`clears()` below:
/// `d_g_cs`, `d_exp_g_cs`, `d_u`, `d_decay_mask`).
struct GdnBwdScratchBufs {
    d_decayed_k: DeviceBuffer,
    d_q_scaled: DeviceBuffer,
    d_v_new: DeviceBuffer,
    d_decay_scale: DeviceBuffer,
    d_query_chunk: DeviceBuffer,
    d_key_chunk: DeviceBuffer,
    state_a: DeviceBuffer,
    state_b: DeviceBuffer,
    d_raw_intra: DeviceBuffer,
    d_k_beta_decay: DeviceBuffer,
    d_v_beta: DeviceBuffer,
    d_raw_attn0: DeviceBuffer,
    d_attn0: DeviceBuffer,
    d_g_cs: DeviceBuffer,
    d_exp_g_cs: DeviceBuffer,
    d_t_mat: DeviceBuffer,
    d_u: DeviceBuffer,
    d_w: DeviceBuffer,
    d_intra_scores: DeviceBuffer,
    d_decay_mask: DeviceBuffer,
    d_k_beta: DeviceBuffer,
    dot_scratch: DeviceBuffer,
    mul_scratch: DeviceBuffer,
    mul_scratch_cc: DeviceBuffer,
}

impl GdnBwdScratchBufs {
    fn new(g: &Gpu, shape: &GdnShape) -> GdnBwdScratchBufs {
        let bhc = shape.bhc() as u64;
        let bh = shape.bh() as u64;
        let c = shape.chunk as u64;
        let dk = shape.dk as u64;
        let dv = shape.dv as u64;
        GdnBwdScratchBufs {
            d_decayed_k: g.storage(bh * c * dk),
            d_q_scaled: g.storage(bh * c * dk),
            d_v_new: g.storage(bh * c * dv),
            d_decay_scale: g.storage(bh * c),
            d_query_chunk: g.storage(bh * c * dk),
            d_key_chunk: g.storage(bh * c * dk),
            state_a: g.storage(bh * dk * dv),
            state_b: g.storage(bh * dk * dv),
            d_raw_intra: g.storage(bhc * c * c),
            d_k_beta_decay: g.storage(bhc * c * dk),
            d_v_beta: g.storage(bhc * c * dv),
            d_raw_attn0: g.storage(bhc * c * c),
            d_attn0: g.storage(bhc * c * c),
            d_g_cs: g.storage(bhc * c),
            d_exp_g_cs: g.storage(bhc * c),
            d_t_mat: g.storage(bhc * c * c),
            d_u: g.storage(bhc * c * dv),
            d_w: g.storage(bhc * c * dk),
            d_intra_scores: g.storage(bhc * c * c),
            d_decay_mask: g.storage(bhc * c * c),
            d_k_beta: g.storage(bhc * c * dk),
            dot_scratch: g.storage(bhc * c),
            mul_scratch: g.storage(bhc * c),
            mul_scratch_cc: g.storage(bhc * c * c),
        }
    }

    fn as_ref(&self) -> GdnBwdScratch<'_> {
        GdnBwdScratch {
            d_decayed_k: &self.d_decayed_k,
            d_q_scaled: &self.d_q_scaled,
            d_v_new: &self.d_v_new,
            d_decay_scale: &self.d_decay_scale,
            d_query_chunk: &self.d_query_chunk,
            d_key_chunk: &self.d_key_chunk,
            state_a: &self.state_a,
            state_b: &self.state_b,
            d_raw_intra: &self.d_raw_intra,
            d_k_beta_decay: &self.d_k_beta_decay,
            d_v_beta: &self.d_v_beta,
            d_raw_attn0: &self.d_raw_attn0,
            d_attn0: &self.d_attn0,
            d_g_cs: &self.d_g_cs,
            d_exp_g_cs: &self.d_exp_g_cs,
            d_t_mat: &self.d_t_mat,
            d_u: &self.d_u,
            d_w: &self.d_w,
            d_intra_scores: &self.d_intra_scores,
            d_decay_mask: &self.d_decay_mask,
            d_k_beta: &self.d_k_beta,
            dot_scratch: &self.dot_scratch,
            mul_scratch: &self.mul_scratch,
            mul_scratch_cc: &self.mul_scratch_cc,
        }
    }

    fn clears(&self) -> Vec<&DeviceBuffer> {
        vec![&self.d_g_cs, &self.d_exp_g_cs, &self.d_u, &self.d_decay_mask]
    }
}

// ---- backward activation cache (training builds only) ----------------------

/// Everything [`Qwen35::layer_gdn_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GDN mixer arm — the SSA activation-cache convention
/// this module's doc describes, at the per-buffer granularity backward
/// actually reads. Field names match the local variable they were assigned
/// from in `layer_gdn_fwd`'s body.
struct GdnLayerActs {
    shape: GdnShape,
    // step 2: conv1d + SiLU.
    ncl_in: DeviceBuffer,  // conv1d's own `x` (dw needs it)
    ncl_out: DeviceBuffer, // conv1d's output, pre-SiLU (silu_bwd needs it)
    // step 4: L2-norm (needs the PRE-norm query/key).
    query: DeviceBuffer,
    key: DeviceBuffer,
    // step 5: bproj (pre-sigmoid, for sigmoid_bwd), aproj (for
    // gdn_decay_gate_bwd), g_decay (gdn_decay_gate's OWN output value --
    // needed for d_A_log = bias_grad(d_g_decay * g_decay), see that
    // gradient's own derivation in `gdn_decay_gate_bwd.wgsl`'s header).
    bproj: DeviceBuffer,
    aproj: DeviceBuffer,
    g_decay: DeviceBuffer,
    // step 7: chunk-major inputs gdn_chunk_bwd itself reads.
    query_cm: DeviceBuffer,
    key_cm: DeviceBuffer,
    value_cm: DeviceBuffer,
    beta_cm: DeviceBuffer,
    // step 8: gdn_chunk_fwd_train's saved history.
    scratch_train: GdnScratchTrainBufs,
    // step 9: token-major output (rmsnorm's `x`).
    out_tok: DeviceBuffer,
    // step 10: gated RMSNorm ("norm before gate").
    normed: DeviceBuffer,
    z: DeviceBuffer,      // pre-SiLU (silu_bwd)
    z_silu: DeviceBuffer, // post-SiLU (mul_bwd's other operand)
    // step 11: out_proj's input.
    gated: DeviceBuffer,
}

/// Everything [`Qwen35::layer_gqa_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GQA mixer arm.
struct GqaLayerActs {
    q_normed: DeviceBuffer, // post QK-norm AND post-RoPE (rope is in-place; this is gqa_bwd's own `q`)
    k_normed: DeviceBuffer, // post QK-norm AND post-RoPE (gqa_bwd's own `kbuf`)
    v: DeviceBuffer,        // raw v projection (gqa_bwd's own `v`)
    q_value: DeviceBuffer,  // pre q_norm (q_norm's rmsnorm_bwd `x`)
    k: DeviceBuffer,        // pre k_norm (k_norm's rmsnorm_bwd `x`)
    q_gate: DeviceBuffer,   // pre-sigmoid
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    gate: DeviceBuffer, // post-sigmoid (mul_bwd's other operand)
    ctx_gated: DeviceBuffer,
}

/// Everything [`Qwen35::moe_sublayer`]'s training branch saves — universal
/// (every layer, both mixer types).
struct MoeLayerActs {
    xn2: DeviceBuffer,
    router_logits: DeviceBuffer,
    gate: DeviceBuffer,
    fe: DeviceBuffer,
    acts: MoeActs,
    // Note: `moe_acc`'s own VALUE is never read by backward (only its
    // gradient, `d_moe_out` itself — no `model::moe` backward primitive reads
    // the pre-shared-expert-add accumulator back), so it is deliberately NOT
    // saved here despite being a real forward intermediate.
    // shared expert (sigmoid-gated: Qwen3.5's `shared_expert_gate`).
    sh_gate_pre: DeviceBuffer,
    sh_up: DeviceBuffer,
    sh_h: DeviceBuffer,
    sh_mlp_out: DeviceBuffer,
    sh_gate_logits: DeviceBuffer,
    sh_gate_scalar: DeviceBuffer,
}

enum MixerActs {
    Gdn(GdnLayerActs),
    Gqa(GqaLayerActs),
}

struct LayerTrainActs {
    xn1: DeviceBuffer,
    mixer: MixerActs,
    xmid: DeviceBuffer,
    moe: MoeLayerActs,
}

/// The full backward activation cache for one `forward()` call on a
/// [`Qwen35::new_train`] instance — see [`Qwen35::train_acts`]'s own doc.
struct TrainActs {
    layers: Vec<LayerTrainActs>,
    xn_final: DeviceBuffer,
}

/// Qwen3.5-35B-A3B hybrid decoder — forward/inference only (see module doc).
pub struct Qwen35 {
    pub gpu: Gpu,
    pub cfg: Qwen35Config,
    /// Pipeline shard this instance owns (whole model, `embed && head`, by
    /// default — see [`Shard::whole`]). Layer indices stay ABSOLUTE
    /// throughout (`res`/`ps`/weight names are all indexed/named by the real
    /// `0..cfg.n_layers` layer number); only the forward/backward loop bounds
    /// and the embed/head gates are shard-relative. Mirrors `qwen3::Qwen`'s
    /// own `shard` field exactly.
    pub shard: Shard,
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
    /// `true` for a [`Self::new_train`] build: every weight is `Role::Trainable`
    /// (see [`Self::new_impl_on`]'s role filter), `forward()` saves the
    /// activation cache [`Self::backward`] needs (`layer_gdn_fwd`'s
    /// `gdn_chunk_fwd_train` branch, `layer_gqa_fwd`'s and `moe_sublayer`'s own
    /// saved buffers), and `backward`/`zero_grads`/`adamw_step` are live instead
    /// of panicking. `false` (the `new`/`new_i8` paths) keeps today's
    /// inference-only behaviour byte-for-byte.
    is_train: bool,
    opt: Optim,

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

    /// Vision-language embedding splice (off = `None`). When set to `(row0,
    /// n_rows)`, `run_forward` overwrites residual rows `[row0, row0+n_rows)`
    /// with `img_embeds` (written by the vision front-end via
    /// [`Qwen35::write_img_embeds`]) right after the token-embedding gather,
    /// and — on a `new_train` build — `backward` routes those rows' gradient
    /// into `d_img_embeds` (read via [`Qwen35::read_d_img_embeds`]) instead of
    /// `tok.weight`. Mirrors `qwen3::Qwen`'s own seam exactly, except no
    /// fwd/bwd step-list rebuild is needed: `run_forward`/`backward` already
    /// build their step lists fresh on every call (see this module's own doc).
    mm_splice: Cell<Option<(u32, u32)>>,
    img_embeds: DeviceBuffer,
    d_img_embeds: DeviceBuffer,

    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,

    /// Backward's activation cache — `Some` only right after a `forward()`
    /// call on a [`Self::new_train`] instance (populated by `run_forward`'s
    /// train branch; read by `backward()`). This mirrors the engine-wide
    /// "forward reallocates fresh buffers every call" convention this file
    /// already uses everywhere else, so `backward()` MUST run against the
    /// same `forward()` call whose gradient it computes — exactly the
    /// `zero_grads(); forward(); backward();` sequencing every caller
    /// (`gradcheck`, a real training loop) already uses.
    train_acts: RefCell<Option<TrainActs>>,
    /// CE-gradient uniform (`[n, vocab, IGNORE, count]`), written once per
    /// `backward()` call (`count` is only known after `set_batch`).
    ce_grad_uni: DeviceBuffer,

    // ---- single-sequence (batch=1) incremental decode state ---------------
    // See `Qwen35::step`'s doc for the overall contract. Everything below is
    // persistent, threaded across `step` calls, and disjoint from the
    // prefill-only buffers above (`res`, `logits`, ...) -- decode allocates
    // its own fresh `[d_model]`-shaped scratch per call (this file's own
    // "reallocate every call" convention), the same way `layer_gdn_fwd`/
    // `layer_gqa_fwd` do for prefill; only the buffers below need to survive
    // between calls.
    /// The next absolute position [`Self::step`] will decode (the cache fill
    /// level) -- `qwen3::Qwen`'s own `dec_pos` convention.
    dec_pos: Cell<u32>,
    /// Decode KV-cache / GDN-state capacity. Reuses this instance's own fixed
    /// `t` (the prefill length it was constructed for) rather than a second,
    /// independent "max decode length" constructor parameter -- a deliberate
    /// simplification for this pass (single fixed sequence length shared by
    /// `logits_all` and `step`), not a hard limitation of the per-layer decode
    /// math itself, which only needs `dec_cap` as an upper bound on `pos`.
    dec_cap: u32,
    /// One-token input buffer for the decode-path `EMBED` gather.
    dec_tokens: DeviceBuffer,
    /// Decode-path M-RoPE: a single-row `[rotary_dim/2]` cos/sin table,
    /// rewritten every [`Self::layer_gqa_decode_step`] call for that step's
    /// absolute position -- see that function's own doc for why a fresh
    /// 1-row table (not a slice into `Self::cos`/`Self::sin`) is required.
    dec_cos: DeviceBuffer,
    dec_sin: DeviceBuffer,
    /// Per-layer plain (non-paged) KV cache for GQA layers, `[dec_cap,
    /// kv_dim]`; a size-1 dummy at GDN layer indices (never dispatched into,
    /// mirrors `qwen3::Qwen::new_impl`'s own `dummy_layer`/`hd_or_dummy`
    /// convention for "this slot doesn't apply to this layer type" rather
    /// than an `Option`, so every layer index still has a plain buffer to
    /// index by `l`).
    gqa_kcache: Vec<DeviceBuffer>,
    gqa_vcache: Vec<DeviceBuffer>,
    /// Per-layer persistent Gated DeltaNet recurrent state, `[bh, dk, dv]`
    /// (`bh = linear_num_value_heads`, single sequence) for GDN layers; a
    /// size-1 dummy at GQA layer indices. Threaded across `step` calls by
    /// [`gdn_recurrent_step`]; zeroed by [`Self::reset_decode_cache`].
    gdn_state: Vec<DeviceBuffer>,
    /// Per-layer persistent causal-conv history ring buffer, `[1, conv_dim,
    /// K-1]`, for GDN layers; a size-1 dummy at GQA layer indices. Threaded
    /// across `step` calls by [`gdn_causal_conv1d_step`]; zeroed by
    /// [`Self::reset_decode_cache`].
    gdn_hist: Vec<DeviceBuffer>,

    // ---- LoRA scratch (persistent, reused across every targeted linear) ----
    // Sized once at construction for `cfg.lora`'s rank and the widest output
    // dimension across the 9 targetable leaves (GDN's `in_proj_qkv`/
    // `in_proj_z`/`in_proj_b`/`in_proj_a`/`out_proj`, GQA's `q_proj`/
    // `k_proj`/`v_proj`/`o_proj`) — mirrors `qwen3::Qwen`'s own
    // `lora_a`/`lora_da`/`lora_out` fields exactly (see [`Self::lora_fwd`]/
    // [`Self::proj_bwd`]'s LoRA branch). Size-1 dummies when `cfg.lora` is
    // `None` (rank forced to 1 in [`Self::new_impl_on`], never read).
    /// `[n*r]` : `a = x @ Aᵀ`.
    lora_a: DeviceBuffer,
    /// `[n*r]` : grad wrt `a`.
    lora_da: DeviceBuffer,
    /// `[n*max_out]` : `delta = a @ Bᵀ`.
    lora_out: DeviceBuffer,

    // ---- pipeline-parallel cross-stage seam (`model::Shardable`) ----------
    // Unlike `qwen3::Qwen`, this file carries no persistent per-layer `dres`
    // array (backward's residual grad is a plain carried local, `d_res_next`
    // -- see `Self::backward`'s own doc); these two boundary buffers stand in
    // for `dres[shard.end]` (read in) / `dres[shard.start]` (written out) so
    // a non-head/non-embed stage still has somewhere stable to receive/expose
    // its cross-stage gradient. Always allocated at `res_numel()` (cheap: one
    // `[b·t·d_model]` slab); unused on a whole/head/embed-only build.
    /// This stage's upstream gradient at `res[shard.end]`, written externally
    /// by [`Self::write_out_dres`] before a non-head stage's `backward()`.
    dres_boundary_in: DeviceBuffer,
    /// This stage's gradient at `res[shard.start]`, refreshed by every
    /// `backward()` call, read externally by [`Self::read_in_dres`].
    dres_boundary_out: RefCell<DeviceBuffer>,
}

/// Which per-sequence GQA cache / GDN recurrent state one
/// [`Qwen35::run_decode_step`] call reads and updates -- introduced so that
/// ONE `run_decode_step` implementation composes with either:
///   - [`Qwen35::step`]'s own single persistent sequence (`self.gqa_kcache`/
///     `self.gqa_vcache`/`self.gdn_state`/`self.gdn_hist`, threaded across
///     calls exactly as before this struct existed), or
///   - `crate::serve::Engine`'s paged multi-sequence decode, which owns a
///     SEPARATE GQA cache + GDN slot per admitted request and must be able to
///     say "run one decode step, but against THIS request's own state, not
///     whichever one happens to live on the model struct" -- the real design
///     problem a paged serving engine adds on top of P11b's single-sequence
///     `step` (see `crate::serve`'s module doc for the full design).
///
/// Every field is indexed by absolute layer index `l` (length
/// `cfg.n_layers`), with a size-1 dummy buffer at the layer indices that
/// don't apply to that field -- the SAME "every layer index has a plain
/// buffer, dummy where irrelevant" convention `Qwen35`'s own
/// `gqa_kcache`/`gdn_state` fields already use (mirroring
/// `qwen3::Qwen::new_impl`'s `dummy_layer` idea), so a caller building one of
/// these for a new sequence can reuse that same construction loop.
pub(crate) struct DecodeCaches<'a> {
    /// Per-layer `[cap, kv_dim]` KV cache for GQA layers (dummy at GDN
    /// indices) -- `model::block::gqa_decode_step`'s own `kcache`/`vcache`.
    pub gqa_kcache: &'a [DeviceBuffer],
    pub gqa_vcache: &'a [DeviceBuffer],
    /// Cache row capacity, shared by every GQA layer's cache in this call
    /// (one per-sequence capacity, not a per-layer one).
    pub gqa_cap: u32,
    /// Per-layer Gated-DeltaNet recurrent `state`/conv `hist` for GDN layers
    /// (dummy at GQA indices) -- `gdn_recurrent_step`'s `state`,
    /// `gdn_causal_conv1d_step`'s `hist`.
    pub gdn_state: &'a [DeviceBuffer],
    pub gdn_hist: &'a [DeviceBuffer],
}

/// The parameter subset a shard holds. A whole shard returns `cfg.param_list()`
/// verbatim (so the single-device store is byte-identical). A partial shard
/// keeps only its layers' weights, plus `tok.weight` when it embeds and/or
/// carries the tied head, and `norm.weight`+head when it is the head stage.
/// Mirrors `qwen3::model::shard_param_list` exactly, adapted for this
/// config's `"blocks.{l}."`-prefixed naming.
fn shard_param_list(cfg: &Qwen35Config, shard: &Shard) -> Vec<(String, usize)> {
    let full = cfg.param_list();
    if shard.is_whole(cfg.n_layers as usize) {
        return full;
    }
    let head = cfg.head_weight(); // "tok.weight" (tied) or "lm_head.weight"
    let tied = head == "tok.weight";
    full.into_iter()
        .filter(|(name, _)| {
            if let Some(rest) = name.strip_prefix("blocks.") {
                let l: usize = rest.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
                return shard.owns(l);
            }
            match name.as_str() {
                "tok.weight" => shard.embed || (shard.head && tied),
                "norm.weight" => shard.head,
                _ if name == head => shard.head, // untied lm_head
                _ => false,
            }
        })
        .collect()
}

impl Qwen35 {
    pub fn new(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, init, false, false, shard)
    }

    /// Build on an existing device handle (test fixtures share one `Gpu` per
    /// binary — see `gpu_core::testgpu`).
    pub fn new_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, false, false, shard)
    }

    /// [`Self::new`] with the int8 (DP4A) inference tier: the attention/GDN
    /// mixer projections and every routed expert's gate/up/down are
    /// quantized (`crate::q8::Qwen35Q8::is_i8_linear`); the router, shared
    /// expert, embeddings and norms stay fp32. See `crate::q8`'s module doc
    /// for the full rationale. Inference-only, same as the fp32 path
    /// (`Qwen35::backward` panics regardless).
    pub fn new_i8(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, init, true, false, shard)
    }

    /// [`Self::new_i8`] on an existing device handle — see [`Self::new_on`].
    pub fn new_on_i8(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, true, false, shard)
    }

    /// Build a TRAINABLE model: every weight `Role::Trainable` (full-parameter
    /// backward — no LoRA-specific plumbing here, per this task's scope note),
    /// `forward()` additionally saves the activation cache `backward()` reads.
    /// int8 and training are mutually exclusive (mirrors `qwen3::Qwen`'s own
    /// `assert!(!(i8 && train))`).
    pub fn new_train(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(Gpu::new(PIPELINES), cfg, b, t, init, false, true, shard)
    }

    /// [`Self::new_train`] on an existing device handle — see [`Self::new_on`].
    pub fn new_train_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, false, true, shard)
    }

    /// Build a single pipeline **stage**: only the layers (and endpoint
    /// weights) in `shard` are allocated on this device, as a TRAINABLE
    /// build (`Role::Trainable` full-parameter, or — when `cfg.lora` is
    /// `Some` — frozen base + trainable LoRA adapters). `shard.gpu_index`
    /// names the canonical physical card (device registry); `Shard::ANY_GPU`
    /// keeps the ambient selection. Mirrors `qwen3::Qwen::new_shard` exactly
    /// (see `crate::shard`'s [`model::Shardable`] impl, the only caller this
    /// is meant for outside tests).
    pub fn new_shard(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> Qwen35 {
        let gpu = if shard.gpu_index == Shard::ANY_GPU {
            Gpu::new(PIPELINES)
        } else {
            Gpu::new_on_index(shard.gpu_index as u32, PIPELINES).unwrap_or_else(|e| panic!("qwen35 shard placement: {e}"))
        };
        Qwen35::new_impl_on(gpu, cfg, b, t, init, false, true, shard)
    }

    fn new_impl_on(
        gpu: Gpu,
        cfg: Qwen35Config,
        b: u32,
        t: u32,
        src: &dyn checkpoint::TensorSource,
        i8: bool,
        train: bool,
        shard: Shard,
    ) -> Qwen35 {
        assert!(!(i8 && train), "qwen35: int8 path is inference-only (Qwen35::new_train is fp32-only)");
        let chunk = gdn_chunk_size(t);
        assert_eq!(
            t % chunk,
            0,
            "qwen35: t={t} is not a multiple of the derived GDN chunk size {chunk} -- \
             model::gdn is prefill-only (no T-padding support, see its module doc); \
             gdn_chunk_size always returns a value that divides t by construction, so \
             this assert failing would mean a logic error in gdn_chunk_size itself"
        );

        // Role assignment:
        //  - inference (`!train`): every weight Role::Frozen (no grad/Adam
        //    buffers allocated at all -- see
        //    paramstore::ParamStore::new_with_roles_src).
        //  - LoRA training (`train && cfg.lora.is_some()`): only the
        //    `.lora_a`/`.lora_b` adapter tensors `Qwen35Config::param_list`
        //    added for each targeted leaf are Trainable; every other weight
        //    (including a LoRA-targeted leaf's own frozen base) is Frozen --
        //    mirrors `qwen3::model.rs`'s own LoRA role-assignment branch
        //    exactly (`model.rs:516-528` in that crate).
        //  - full training (`train && cfg.lora.is_none()`): every weight
        //    Role::Trainable (full-parameter backward).
        // In int8 mode the linears `Qwen35Q8::is_i8_linear` names live in
        // `q8` (packed int8), NOT the fp32 store -- filter them out here so
        // no redundant fp32 copy is ever uploaded (mirrors
        // `qwen3::model.rs`'s own `Q8::is_i8_linear` filter,
        // `model.rs:504-507` in that crate). int8 and LoRA/training are
        // mutually exclusive (the `assert!` above), so `i8` and
        // `cfg.lora.is_some()` never both hold here.
        let roles: Vec<(String, usize, Role)> = shard_param_list(&cfg, &shard)
            .into_iter()
            .filter(|(n, _)| !(i8 && Qwen35Q8::is_i8_linear(n)))
            .map(|(n, c)| {
                let role = if !train {
                    Role::Frozen
                } else if cfg.lora.is_some() {
                    if n.ends_with(".lora_a") || n.ends_with(".lora_b") { Role::Trainable } else { Role::Frozen }
                } else {
                    Role::Trainable
                };
                (n, c, role)
            })
            .collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        // Quantize+upload the int8 linears from the SAME source, streaming
        // one tensor at a time (see `Qwen35Q8::build`'s own doc).
        let q8 = if i8 { Some(Qwen35Q8::build(&gpu, src, &cfg, b * t, MAX_ABS_ROW, QUANT_PACK, MATMUL_I8)) } else { None };

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let mut res = Vec::with_capacity(cfg.n_layers as usize + 1);
        for _ in 0..=cfg.n_layers {
            res.push(gpu.storage(n * d));
        }
        // Pipeline-parallel cross-stage boundary gradient buffers -- see the
        // struct fields' own doc.
        let dres_boundary_in = gpu.storage(n * d);
        let dres_boundary_out = RefCell::new(gpu.storage(n * d));

        // LoRA scratch (rank r; max projection output across all 9 targetable
        // leaves -- GDN's in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj,
        // GQA's q_proj/k_proj/v_proj/o_proj -- mirrors `qwen3::model.rs`'s own
        // sizing exactly). `.max(1)` so a `cfg.lora: None` build still gets a
        // valid (unused) 1-element rank.
        let lora_r = cfg.lora.as_ref().map(|l| l.rank as u64).unwrap_or(0).max(1);
        let lora_max_out = cfg
            .linear_conv_dim()
            .max(cfg.linear_value_dim())
            .max(cfg.linear_num_value_heads)
            .max(d as u32)
            .max(cfg.q_proj_dim())
            .max(cfg.kv_dim()) as u64;
        let lora_a = gpu.storage(n * lora_r);
        let lora_da = gpu.storage(n * lora_r);
        let lora_out = gpu.storage(n * lora_max_out);

        let ones_khd = gpu.storage_init("qwen35.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);

        // Text-only: every axis of the M-RoPE table carries the same plain
        // sequential position, reset per sequence (row = batch*t + pos).
        let positions: Vec<[u32; 3]> = (0..b).flat_map(|_| (0..t).map(|ti| [ti, ti, ti])).collect();
        let (cos, sin) = qwenvl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("qwen35.rope_cos", &cos);
        let sin = gpu.storage_init("qwen35.rope_sin", &sin);

        let tokens = gpu.storage(n);
        let targets = gpu.storage(n);
        // Vision-language splice: off by default (size-1 dummies), real sizes
        // allocated by `enable_mm_splice`.
        let img_embeds = gpu.storage(1);
        let d_img_embeds = gpu.storage(1);
        let logits = gpu.storage(n * cfg.vocab as u64);
        let ce_buf = gpu.storage(n);
        let ce_grad_uni = gpu.uniform_dynamic(4);

        // Single-sequence incremental decode state -- see `Qwen35::step`'s
        // doc and the struct fields' own docs for what each buffer holds.
        // `dec_cap = t`: this pass's decode capacity is this instance's own
        // fixed prefill length (see `dec_cap`'s own doc).
        let dec_tokens = gpu.storage(1);
        let dec_half = (cfg.rotary_dim() / 2).max(1) as u64;
        let dec_cos = gpu.storage(dec_half);
        let dec_sin = gpu.storage(dec_half);
        let kv_dim = cfg.kv_dim() as u64;
        let mut gqa_kcache = Vec::with_capacity(cfg.n_layers as usize);
        let mut gqa_vcache = Vec::with_capacity(cfg.n_layers as usize);
        let mut gdn_state = Vec::with_capacity(cfg.n_layers as usize);
        let mut gdn_hist = Vec::with_capacity(cfg.n_layers as usize);
        let gdn_bh = cfg.linear_num_value_heads as u64;
        let gdn_state_len = gdn_bh * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
        let gdn_hist_len = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
        for ty in cfg.layer_types() {
            match ty {
                LayerType::Full => {
                    gqa_kcache.push(gpu.storage(t as u64 * kv_dim));
                    gqa_vcache.push(gpu.storage(t as u64 * kv_dim));
                    gdn_state.push(gpu.storage(1));
                    gdn_hist.push(gpu.storage(1));
                }
                LayerType::Linear => {
                    gqa_kcache.push(gpu.storage(1));
                    gqa_vcache.push(gpu.storage(1));
                    gdn_state.push(gpu.storage(gdn_state_len));
                    gdn_hist.push(gpu.storage(gdn_hist_len));
                }
            }
        }

        Qwen35 {
            gpu,
            cfg,
            shard,
            ps,
            q8,
            b,
            t,
            chunk,
            is_train: train,
            opt,
            tokens,
            targets,
            count: Cell::new(1.0),
            res,
            ones_khd,
            cos,
            sin,
            mm_splice: Cell::new(None),
            img_embeds,
            d_img_embeds,
            logits,
            ce_buf,
            train_acts: RefCell::new(None),
            ce_grad_uni,
            dec_pos: Cell::new(0),
            dec_cap: t,
            dec_tokens,
            dec_cos,
            dec_sin,
            gqa_kcache,
            gqa_vcache,
            gdn_state,
            gdn_hist,
            lora_a,
            lora_da,
            lora_out,
            dres_boundary_in,
            dres_boundary_out,
        }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    /// True if `name` has a gradient buffer (i.e. is optimised). Frozen
    /// parameters (LoRA base, inference) have none, so their weight-gradient
    /// dispatches must be skipped — only the input-gradient (dX) path runs to
    /// keep backprop flowing to lower-layer adapters. Mirrors
    /// `qwen3::model.rs`'s own `trainable` helper exactly.
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    /// The gradient buffer for a trainable weight — only valid on a
    /// [`Self::new_train`] instance (every weight is `Role::Trainable` there,
    /// see [`Self::new_impl_on`]'s role filter).
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }

    /// True if a LoRA adapter is configured for the given projection leaf
    /// (one of the 9 targetable leaf names — never an MoE expert leaf).
    /// Mirrors `qwen3::model.rs`'s own `lora_for` exactly.
    fn lora_for(&self, leaf: &str) -> Option<(u32, f32)> {
        self.cfg.lora.as_ref().filter(|lc| lc.targets_leaf(leaf)).map(|lc| (lc.rank, lc.alpha / lc.rank as f32))
    }

    /// Forward LoRA delta for a targeted linear: `y += (alpha/r)·(x·Aᵀ)·Bᵀ`.
    /// No-op for an untargeted leaf. `m`×`k` is the input, `nout` the output —
    /// mirrors `qwen3::model.rs`'s own `lora_fwd` exactly (same two-matmul +
    /// `AXPY` fusion, using this file's own persistent `lora_a`/`lora_out`
    /// scratch).
    fn lora_fwd(&self, s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, wname: &str, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let Some((r, scale)) = self.lora_for(leaf) else { return };
        let g = &self.gpu;
        let a = format!("{wname}.lora_a");
        let bnm = format!("{wname}.lora_b");
        s.push(g.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
        s.push(g.step(MATMUL, &[&self.lora_a, self.w(&bnm), &self.lora_out], &[m, r, nout], m * nout));
        s.push(g.step(AXPY, &[y, &self.lora_out], &[m * nout, f(scale)], m * nout));
    }

    /// Backward for a (possibly-LoRA) linear `y = x·Wᵀ`. Accumulates the input
    /// gradient into `dx` (flag `acc`). For a full weight: `dW += d_outᵀ·x`
    /// (skipped when `wname` is Frozen — a LoRA-mode base, or an untargeted
    /// weight under a LoRA build, e.g. `mlp.router.weight`), `dx = d_out·W`.
    /// For a LoRA-targeted leaf: the base weight is always frozen (dX only, no
    /// dW), and the adapter grads `gA`/`gB` are produced (scale folded into
    /// the private `lora_a`/`lora_da` scratch) — naive `matmul_dx`/`matmul_dw`
    /// only, no tiled-GEMM selection, matching a correctness-first tiny
    /// gradcheck config per `docs/porting-playbook.md` §10. Mirrors
    /// `qwen3::model.rs`'s own `proj_bwd` exactly.
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, steps: &mut Vec<Step>, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        let g = &self.gpu;
        match self.lora_for(leaf) {
            Some((r, scale)) => {
                // base: dx += d_out·W (frozen weight — no dW).
                steps.push(g.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
                let a = format!("{wname}.lora_a");
                let bnm = format!("{wname}.lora_b");
                // a = (alpha/r)·(x·Aᵀ)  -> gB += d_outᵀ·a
                steps.push(g.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
                steps.push(g.step(GRAD_SCALE, &[&self.lora_a], &[m * r, f(scale)], m * r));
                steps.push(g.step(MATMUL_DW, &[d_out, &self.lora_a, self.g(&bnm)], &[m, r, nout], nout * r));
                // da = (alpha/r)·(d_out·B) -> gA += daᵀ·x ; dx += da·A
                steps.push(g.step(MATMUL_DX, &[d_out, self.w(&bnm), &self.lora_da], &[m, r, nout, 0], m * r));
                steps.push(g.step(GRAD_SCALE, &[&self.lora_da], &[m * r, f(scale)], m * r));
                steps.push(g.step(MATMUL_DW, &[&self.lora_da, x, self.g(&a)], &[m, k, r], r * k));
                steps.push(g.step(MATMUL_DX, &[&self.lora_da, self.w(&a), dx], &[m, k, r, 1], m * k));
            }
            None => {
                if self.trainable(wname) {
                    steps.push(g.step(MATMUL_DW, &[d_out, x, self.g(wname)], &[m, k, nout], nout * k));
                }
                steps.push(g.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
            }
        }
    }

    /// RMSNorm backward via the shared builder: input grad always, gain grad
    /// only when the gain is trainable (frozen under a LoRA build — no norm
    /// gain is ever a LoRA target, so this mirrors `qwen3::Qwen`'s own
    /// LoRA-base-frozen branch, applied here to every norm rather than to a
    /// projection weight).
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

    // ---- vision-language embedding splice seam (see `crate::vl::Qwen35Vl`) --

    /// Enable the VLM embedding splice at residual rows `[row0, row0+n_rows)`:
    /// after the token-embedding gather, `run_forward` overwrites those rows
    /// with the image tokens written via [`Self::write_img_embeds`], and — on
    /// a `new_train` build — `backward` routes their gradient to
    /// [`Self::read_d_img_embeds`] (zeroing them in the residual grad first so
    /// `EMB_BWD` never trains the image-placeholder token id). Unlike
    /// `qwen3::Qwen::enable_mm_splice`, this needs no fwd/bwd step-list
    /// rebuild: `run_forward`/`backward` already build their step lists fresh
    /// on every call (see this module's top-of-file doc), so enabling the
    /// splice is pure buffer allocation + a flag — call once after
    /// construction, before the first `forward()`.
    pub fn enable_mm_splice(&mut self, row0: u32, n_rows: u32) {
        let sz = (n_rows * self.cfg.d_model) as u64;
        self.img_embeds = self.gpu.storage(sz);
        self.d_img_embeds = self.gpu.storage(sz);
        self.mm_splice.set(Some((row0, n_rows)));
    }

    /// Write the projected image tokens `[n_rows, d_model]` (row-major) to
    /// splice into the residual stream on the next `forward()`.
    pub fn write_img_embeds(&self, data: &[f32]) {
        self.gpu.write_f32(&self.img_embeds, data);
    }

    /// Number of spliced image-embedding elements (`n_rows·d_model`); 0 if off.
    fn img_numel(&self) -> usize {
        self.mm_splice.get().map_or(0, |(_, n)| (n * self.cfg.d_model) as usize)
    }

    /// Read the gradient of the spliced image embeddings after `backward()` —
    /// feeds the vision tower/connector backward. Requires a `new_train` build
    /// (see [`Self::backward`]'s splice-gradient step).
    pub fn read_d_img_embeds(&self) -> Vec<f32> {
        self.gpu.read(&self.d_img_embeds, self.img_numel())
    }

    /// Overwrite the M-RoPE `cos`/`sin` tables (`[b·t, rotary_dim/2]` row-major
    /// — see the `cos`/`sin` fields' own doc, and
    /// `qwenvl::mrope::{get_rope_index, mrope_tables}` for how to build them
    /// from real 2-D image-grid positions) for the next `forward()`. RoPE here
    /// is unconditionally table-driven already (no `enable_mrope` gating
    /// needed, unlike `qwen3::Qwen`) — this simply replaces the plain-
    /// sequential-position table built at construction.
    pub fn write_mrope_tables(&self, cos: &[f32], sin: &[f32]) {
        self.gpu.write_f32(&self.cos, cos);
        self.gpu.write_f32(&self.sin, sin);
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
            let mut s = vec![g.step(MATMUL, &[xn1, self.w(&p("in_proj_qkv.weight")), &mixed_qkv], &[n, d, conv_dim], n * conv_dim)];
            self.lora_fwd(&mut s, "in_proj_qkv", xn1, &p("in_proj_qkv.weight"), &mixed_qkv, n, d, conv_dim);
            g.submit(&[], &s);
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
            let mut s = vec![
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_b.weight")), &bproj], &[n, d, nvh], n * nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_a.weight")), &aproj], &[n, d, nvh], n * nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_z.weight")), &z], &[n, d, value_dim], n * value_dim),
            ];
            self.lora_fwd(&mut s, "in_proj_b", xn1, &p("in_proj_b.weight"), &bproj, n, d, nvh);
            self.lora_fwd(&mut s, "in_proj_a", xn1, &p("in_proj_a.weight"), &aproj, n, d, nvh);
            self.lora_fwd(&mut s, "in_proj_z", xn1, &p("in_proj_z.weight"), &z, n, d, value_dim);
            g.submit(&[], &s);
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

        // 8. gdn_chunk_fwd -- the chunked-recurrence forward itself. Training
        // builds use `gdn_chunk_fwd_train` instead: bit-identical `out`/
        // `final_state` (see that function's own doc) but additionally saves
        // the per-chunk history `gdn_chunk_bwd` needs -- `layer_gdn_fwd`'s own
        // "is this a training build" branch, mirroring the `q8l` int8 branch
        // above.
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
            g.submit(&[&scratch.t_mat], &steps);
        }

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
            let mut s = vec![g.step(MATMUL, &[&gated, self.w(&p("out_proj.weight")), &out], &[n, value_dim, d], n * d)];
            self.lora_fwd(&mut s, "out_proj", &gated, &p("out_proj.weight"), &out, n, value_dim, d);
            g.submit(&[], &s);
        }

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
            let mut s = vec![
                g.step(MATMUL, &[xn1, self.w(&p("q_proj.weight")), &q_full], &[n, d, qpd], n * qpd),
                g.step(MATMUL, &[xn1, self.w(&p("k_proj.weight")), &k], &[n, d, kvd], n * kvd),
                g.step(MATMUL, &[xn1, self.w(&p("v_proj.weight")), &v], &[n, d, kvd], n * kvd),
            ];
            self.lora_fwd(&mut s, "q_proj", xn1, &p("q_proj.weight"), &q_full, n, d, qpd);
            self.lora_fwd(&mut s, "k_proj", xn1, &p("k_proj.weight"), &k, n, d, kvd);
            self.lora_fwd(&mut s, "v_proj", xn1, &p("v_proj.weight"), &v, n, d, kvd);
            g.submit(&[], &s);
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
            let mut s = vec![
                g.step(SIGMOID, &[&q_gate, &gate], &[n * qd], n * qd),
                g.step(MUL, &[&ctx, &gate, &ctx_gated], &[n * qd], n * qd),
                g.step(MATMUL, &[&ctx_gated, self.w(&p("o_proj.weight")), &out], &[n, qd, d], n * d),
            ];
            self.lora_fwd(&mut s, "o_proj", &ctx_gated, &p("o_proj.weight"), &out, n, qd, d);
            g.submit(&[], &s);
        }

        let acts = self.is_train.then(|| GqaLayerActs {
            q_normed,
            k_normed,
            v,
            q_value,
            k,
            q_gate,
            probs,
            ctx,
            gate,
            ctx_gated,
        });
        (out, acts)
    }

    // ---- MoE sublayer, universal for every layer ---------------------------

    fn moe_sublayer(&self, l: usize, xmid: &DeviceBuffer, n: u32) -> (DeviceBuffer, Option<MoeLayerActs>) {
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
        // Training builds additionally need EVERY expert's OWN gate_pre/up/h/
        // expert_out (not a shared scratch reused across experts -- see
        // `model::moe::MoeActs`'s own doc for why forward's per-call-reused
        // `ExpertScratch` cannot serve backward), so `moe_acts` is `Some` only
        // for a training, non-int8 build (asserted mutually exclusive at
        // construction).
        let moe_acts: Option<MoeActs> = if let Some(q8) = &self.q8 {
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
            None
        } else if self.is_train {
            let acts = MoeActs::new(g, &shape);
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
                    &acts.at(ei as usize),
                    &moe_acc,
                    ei,
                    ei != 0,
                ));
            }
            Some(acts)
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
            None
        };

        let moe_out = g.storage((n * d) as u64);
        let sh_gate_pre = g.storage((n * shared_ff) as u64);
        let sh_up = g.storage((n * shared_ff) as u64);
        let sh_h = g.storage((n * shared_ff) as u64);
        let sh_mlp_out = g.storage((n * d) as u64);
        let sh_gate_logits = g.storage(n as u64);
        let sh_gate_scalar = g.storage(n as u64);
        let sh_scaled = g.storage((n * d) as u64);
        let sh_scratch = SharedExpertScratch {
            gate_pre: &sh_gate_pre,
            up: &sh_up,
            h: &sh_h,
            mlp_out: &sh_mlp_out,
            gate_logits: &sh_gate_logits,
            gate_scalar: &sh_gate_scalar,
            scaled: &sh_scaled,
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

        let acts = moe_acts.map(|acts| MoeLayerActs {
            xn2,
            router_logits,
            gate,
            fe: g.storage(e as u64),
            acts,
            sh_gate_pre,
            sh_up,
            sh_h,
            sh_mlp_out,
            sh_gate_logits,
            sh_gate_scalar,
        });
        (moe_out, acts)
    }

    // ---- full stack ----------------------------------------------------------

    /// Run this stage's forward graph over its own layer range
    /// (`self.shard.start..self.shard.end`, ABSOLUTE layer indices — `res`
    /// stays indexed by the real layer number, only the loop bounds are
    /// shard-relative). The embedding gather (+ vision splice) runs only on
    /// the embed stage; the final norm + lm_head/logits only on the head
    /// stage. A non-embed stage's `res[shard.start]` must already hold the
    /// previous stage's output (written via [`Self::write_in_res`]) before
    /// this call; a non-head stage's `res[shard.end]` is this stage's output
    /// for the next one (read via [`Self::read_out_res`]). Mirrors
    /// `qwen3::Qwen::forward_steps`'s own shard gating exactly, adapted to
    /// this file's "build and submit inline" convention (no separate
    /// step-list rebuild is needed here).
    pub(crate) fn run_forward(&self) {
        let g = &self.gpu;
        let n = self.b * self.t;
        let d = self.cfg.d_model;
        let mut layer_acts: Vec<LayerTrainActs> = Vec::new();

        if self.shard.embed {
            g.submit(&[], &[g.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, n], n * d)]);

            // Vision-language splice: overwrite the image-placeholder rows of
            // the freshly-gathered residual stream with the projected image
            // tokens (see `Self::enable_mm_splice`'s doc). No-op unless
            // enabled. Only meaningful on the embed stage (it operates on
            // `res[0]`, right after the gather above).
            if let Some((row0, n_rows)) = self.mm_splice.get() {
                g.submit(&[], &[model::vlm::splice_fwd(g, SPLICE, &self.img_embeds, &self.res[0], row0 * d, n_rows * d)]);
            }
        }

        let types = self.cfg.layer_types();
        // `l` is the ABSOLUTE layer index (into `types`/`self.res`/the
        // `blocks.{l}.*` weight names below), not just a `types` index --
        // clippy's `needless_range_loop` heuristic only sees the first use.
        #[allow(clippy::needless_range_loop)]
        for l in self.shard.start..self.shard.end {
            let ty = types[l];
            let xres = &self.res[l];
            let xn1 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), xres, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n)]);

            let (attn_out, mixer_acts) = match ty {
                LayerType::Linear => {
                    let (o, a) = self.layer_gdn_fwd(l, &xn1, n);
                    (o, a.map(MixerActs::Gdn))
                }
                LayerType::Full => {
                    let (o, a) = self.layer_gqa_fwd(l, &xn1, n);
                    (o, a.map(MixerActs::Gqa))
                }
            };

            let xmid = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[xres, &attn_out, &xmid], &[n * d], n * d)]);

            let (moe_out, moe_acts) = self.moe_sublayer(l, &xmid, n);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &moe_out, &self.res[l + 1]], &[n * d], n * d)]);

            if self.is_train {
                layer_acts.push(LayerTrainActs {
                    xn1,
                    mixer: mixer_acts.expect("qwen35: is_train but layer_gdn_fwd/layer_gqa_fwd returned no acts"),
                    xmid,
                    moe: moe_acts.expect("qwen35: is_train but moe_sublayer returned no acts"),
                });
            }
        }

        // Head epilogue (final norm + lm_head/logits): only the head stage.
        // On a non-head stage `xn_final` is never read (`self.shard.head` is
        // `false` in `Self::forward`'s CE step too — see `Qwen35::forward`'s
        // own doc), so a size-1 dummy stands in, matching this file's
        // "size-1 dummy where a value doesn't apply" convention used
        // elsewhere (`gqa_kcache`/`gdn_state`).
        let xn_final = if self.shard.head {
            let xn_final = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &self.res[self.cfg.n_layers as usize], self.w("norm.weight"), &xn_final, d, n)]);
            let v = self.cfg.vocab;
            g.submit(&[], &[g.step(MATMUL, &[&xn_final, self.w(self.cfg.head_weight()), &self.logits], &[n, d, v], n * v)]);
            xn_final
        } else {
            g.storage(1)
        };

        if self.is_train {
            *self.train_acts.borrow_mut() = Some(TrainActs { layers: layer_acts, xn_final });
        }
    }

    // ---- backward (training builds only) ----------------------------------

    /// Reverse of [`Self::layer_gdn_fwd`]'s 11 steps. `d_out` is the upstream
    /// gradient into this layer's mixer output (`attn_out`); accumulates into
    /// `d_xn1` (already zero-fresh -- the FIRST touch below is a plain
    /// overwrite, `acc=0`, establishing its base value, exactly like every
    /// other multi-source accumulator in this file).
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
            self.proj_bwd(&mut s, "out_proj", d_out, &la.gated, &p("out_proj.weight"), &d_gated, n, value_dim, d, 0);
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

        // ---- 8. gdn_chunk_bwd -- the chunked-recurrence backward itself ----
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
            // `d_final_state` (external gradient, none), plus `d_query`/`d_key`/
            // `d_beta` (2/4/2-source accumulators this function's own doc lists
            // as the CALLER's responsibility, distinct from `GdnBwdScratch`'s
            // own MUST-zero list) -- see `gdn_chunk_bwd`'s doc, "Every output
            // with more than one contributing forward use is explicitly
            // zeroed by the caller".
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
            // d_A_log[h] = sum_row d_g_decay[row,h]*g_decay[row,h]; d_dt_bias[h] = sum_row d_aproj[row,h].
            // A_log/dt_bias are never a LoRA target -- Frozen under a LoRA
            // build, same as any other non-targeted weight -- so skip these
            // reductions entirely when frozen (no grad buffer to write into).
            let mul_tmp = g.storage((n * nvh) as u64);
            s.push(g.step(MUL, &[&d_g_decay, &la.g_decay, &mul_tmp], &[n * nvh], n * nvh));
            if self.trainable(&p("A_log")) {
                s.push(g.step(BIAS_GRAD, &[&mul_tmp, self.g(&p("A_log"))], &[n, nvh], nvh));
            }
            if self.trainable(&p("dt_bias")) {
                s.push(g.step(BIAS_GRAD, &[&d_aproj, self.g(&p("dt_bias"))], &[n, nvh], nvh));
            }
            // FIRST touch to d_xn1 in this function (acc=0) -- in_proj_a/z below
            // accumulate on top; in_proj_qkv (step 1, processed last here)
            // accumulates last of all.
            self.proj_bwd(&mut s, "in_proj_b", &d_bproj, xn1, &p("in_proj_b.weight"), d_xn1, n, d, nvh, 0);
            self.proj_bwd(&mut s, "in_proj_a", &d_aproj, xn1, &p("in_proj_a.weight"), d_xn1, n, d, nvh, 1);
            self.proj_bwd(&mut s, "in_proj_z", &d_z, xn1, &p("in_proj_z.weight"), d_xn1, n, d, value_dim, 1);
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
            // conv1d.weight is never a LoRA target -- Frozen under a LoRA
            // build, so its dW argument is `None` there (dX is unconditional).
            let conv_dw = self.trainable(&p("conv1d.weight")).then(|| self.g(&p("conv1d.weight")));
            s.extend(conv1d_bwd(g, &conv_kernels(), &conv_shape, &d_ncl_out, &la.ncl_in, self.w(&p("conv1d.weight")), Some(&d_ncl_in), conv_dw));
            s.push(g.step(NCHW_NLC, &[&d_ncl_in, &d_mixed_qkv], &[n * conv_dim, conv_dim, t], n * conv_dim));
            g.submit(&[], &s);
        }

        // ---- 1. in_proj_qkv backward (last accumulate into d_xn1) ----
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "in_proj_qkv", &d_mixed_qkv, xn1, &p("in_proj_qkv.weight"), d_xn1, n, d, conv_dim, 1);
            g.submit(&[], &s);
        }
    }

    /// Reverse of [`Self::layer_gqa_fwd`]'s 7 steps.
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
            self.proj_bwd(&mut s, "o_proj", d_out, &la.ctx_gated, &p("o_proj.weight"), &d_ctx_gated, n, qd, d, 0);
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
            self.proj_bwd(&mut s, "q_proj", &d_q_full, xn1, &p("q_proj.weight"), d_xn1, n, d, qpd, 0);
            self.proj_bwd(&mut s, "k_proj", &d_k, xn1, &p("k_proj.weight"), d_xn1, n, d, kvd, 1);
            self.proj_bwd(&mut s, "v_proj", &d_v, xn1, &p("v_proj.weight"), d_xn1, n, d, kvd, 1);
            g.submit(&[], &s);
        }
    }

    /// Reverse of [`Self::moe_sublayer`]. Returns `d_xn2` (the gradient into
    /// the pre-MoE-norm hidden state, i.e. `ln2`'s output) -- the caller still
    /// owes `ln2`'s own backward to fold that into `d_xmid`.
    ///
    /// **Ordering, matching [`model::moe::moe_layer_bwd`]'s own documented
    /// phase contract exactly** (Phase A: every expert's `d_gate` column ->
    /// Phase B: router backward, kernel-level, THEN the router weight's own
    /// dense-linear backward (`router_weight_bwd`, supplied here as the
    /// FIRST touch to `d_xn2`, `acc=0`) -> Phase C: every expert's SwiGLU
    /// backward, accumulating into `d_xn2`) runs FIRST, fully establishing
    /// `d_xn2`'s value; the shared expert's OWN backward (no composed helper
    /// exists for it in `model::moe` -- hand-derived here) runs SECOND, its
    /// three `d_xn2` touches all `acc=1` on top of the routed-MoE total.
    fn moe_sublayer_bwd(&self, l: usize, la: &MoeLayerActs, d_moe_out: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let e = c.n_experts;
        let moe_ff = c.moe_intermediate_size;
        let shared_ff = c.shared_expert_intermediate_size;
        let p = |s: &str| format!("blocks.{l}.{s}");
        let shape = MoeShape { rows: n, d_model: d, moe_ff, n_experts: e, top_k: c.top_k };

        let d_xn2 = g.storage((n * d) as u64);

        // ---- Phase A/B/C: routed experts + router (model::moe::moe_layer_bwd) ----
        let d_router_logits = g.storage((n * e) as u64);
        let d_gate = g.storage((n * e) as u64);
        let router_weight_bwd_steps = {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "router", &d_router_logits, &la.xn2, &p("mlp.router.weight"), &d_xn2, n, d, e, 0);
            s
        };
        let expert_weights: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)> = (0..e)
            .map(|ei| {
                let ep = |s: &str| format!("blocks.{l}.mlp.experts.{ei}.{s}");
                (self.w(&ep("gate.weight")).clone(), self.w(&ep("up.weight")).clone(), self.w(&ep("down.weight")).clone())
            })
            .collect();
        // Never a LoRA target (per the standing LoRA task's own scope note:
        // the 256-expert MoE linears are out of scope) -- Frozen under a LoRA
        // build, so each field is `None` there (`ExpertGrads`' own contract).
        let expert_grads: Vec<ExpertGrads> = (0..e)
            .map(|ei| {
                let ep = |s: &str| format!("blocks.{l}.mlp.experts.{ei}.{s}");
                ExpertGrads {
                    gate_w: self.trainable(&ep("gate.weight")).then(|| self.g(&ep("gate.weight"))),
                    up_w: self.trainable(&ep("up.weight")).then(|| self.g(&ep("up.weight"))),
                    down_w: self.trainable(&ep("down.weight")).then(|| self.g(&ep("down.weight"))),
                }
            })
            .collect();
        let d_expert_out = g.storage((n * d) as u64);
        let d_h = g.storage((n * moe_ff) as u64);
        let d_gate_pre = g.storage((n * moe_ff) as u64);
        let d_up = g.storage((n * moe_ff) as u64);
        let sb = ExpertBwdScratch { d_expert_out: &d_expert_out, d_h: &d_h, d_gate_pre: &d_gate_pre, d_up: &d_up };
        let moe_steps = moe_layer_bwd(
            g,
            &router_bwd_ids(),
            &moe_bwd_ids(),
            RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0 },
            &shape,
            &la.router_logits,
            &la.gate,
            Some(&la.fe),
            &d_gate,
            &d_router_logits,
            &router_weight_bwd_steps,
            &la.xn2,
            &expert_weights,
            &expert_grads,
            &la.acts,
            &sb,
            d_moe_out,
            &d_xn2,
        );
        g.submit(&[], &moe_steps);

        // ---- shared expert backward (hand-derived -- no `model::moe` helper
        // exists for a SIGMOID-GATED shared expert; accumulates onto d_xn2,
        // whose base value the routed-MoE backward above already established) ----
        let d_mlp_out = g.storage((n * d) as u64);
        let d_gate_scalar = g.storage(n as u64);
        let d_gate_logits = g.storage(n as u64);
        let d_sh_h = g.storage((n * shared_ff) as u64);
        let d_sh_gate_pre = g.storage((n * shared_ff) as u64);
        let d_sh_up = g.storage((n * shared_ff) as u64);
        {
            let mut s = vec![
                // scaled = mlp_out * gate_scalar (scale_row.wgsl is its own
                // backward w.r.t. its `x` operand -- see that kernel's own doc).
                g.step(SCALE_ROW, &[d_moe_out, &la.sh_gate_scalar, &d_mlp_out], &[n * d, d], n * d),
                g.step(ROW_DOT, &[d_moe_out, &la.sh_mlp_out, &d_gate_scalar], &[n, d, 0, 0, f(1.0)], n),
                g.step(SIGMOID_BWD, &[&la.sh_gate_logits, &d_gate_scalar, &d_gate_logits], &[n], n),
            ];
            self.proj_bwd(&mut s, "shared_expert_gate", &d_gate_logits, &la.xn2, &p("mlp.shared_expert_gate.weight"), &d_xn2, n, d, 1, 1);
            self.proj_bwd(&mut s, "shared_expert_down", &d_mlp_out, &la.sh_h, &p("mlp.shared_expert.down.weight"), &d_sh_h, n, shared_ff, d, 0);
            s.extend(swiglu_bwd(g, &kernel_ids(), &la.sh_gate_pre, &la.sh_up, &d_sh_h, &d_sh_gate_pre, &d_sh_up, n * shared_ff));
            self.proj_bwd(&mut s, "shared_expert_up", &d_sh_up, &la.xn2, &p("mlp.shared_expert.up.weight"), &d_xn2, n, d, shared_ff, 1);
            self.proj_bwd(&mut s, "shared_expert_gate_proj", &d_sh_gate_pre, &la.xn2, &p("mlp.shared_expert.gate.weight"), &d_xn2, n, d, shared_ff, 1);
            g.submit(&[], &s);
        }

        d_xn2
    }

    /// Full backward pass — mirrors [`Self::run_forward`]'s layer loop in
    /// REVERSE, threading `d_res[l+1] -> d_res[l]` the same way forward
    /// threads `res[l] -> res[l+1]`. Requires an immediately preceding
    /// `forward()` call on a [`Self::new_train`] instance (see
    /// [`Self::train_acts`]'s own doc).
    /// Run this stage's backward graph. On the head stage (`self.shard.head`)
    /// this starts from the CE gradient (as before sharding existed); on any
    /// other stage it instead starts from the upstream gradient this stage's
    /// OUTPUT boundary already carries (`self.dres_boundary_in`, written by
    /// [`Self::write_out_dres`] before this call). At the end of the reversed
    /// layer loop, this stage's INPUT-boundary gradient (`dres[shard.start]`)
    /// is stashed in `self.dres_boundary_out` for [`Self::read_in_dres`] to
    /// read — mirrors `qwen3::Qwen::build_backward_steps`'s own shard gating,
    /// adapted to this file's "no persistent per-layer `dres` array" design
    /// (a single carried local, `d_res_next`, plays that role here).
    pub fn backward(&self) {
        assert!(self.is_train, "qwen35: backward() requires a Qwen35::new_train build");
        let ta = self.train_acts.borrow_mut().take().expect(
            "qwen35: backward() called without an immediately preceding forward() -- \
             every forward() call reallocates its activation cache fresh (this file's \
             own convention throughout), so backward() must run against the SAME call",
        );
        let g = &self.gpu;
        let n = self.b * self.t;
        let d = self.cfg.d_model;
        let v = self.cfg.vocab;

        // ---- head epilogue backward (head stage only): CE-grad, lm_head,
        // final norm -- a non-head stage starts instead from the externally
        // supplied gradient at `res[shard.end]` (`Self::write_out_dres`).
        let mut d_res_next = if self.shard.head {
            g.write(&self.ce_grad_uni, &[n, v, model::IGNORE, f(self.count.get())]);
            let d_logits = g.storage((n * v) as u64);
            g.submit(&[], &[g.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &d_logits], n * v)]);

            // ---- lm_head backward ----
            let d_xn_final = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.proj_bwd(&mut s, "lm_head", &d_logits, &ta.xn_final, self.cfg.head_weight(), &d_xn_final, n, d, v, 0);
                g.submit(&[], &s);
            }

            // ---- final norm backward ----
            let d_res_next = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.rmsnorm_bwd_step(&mut s, &self.res[self.cfg.n_layers as usize], "norm.weight", &d_xn_final, &d_res_next, d, n);
                g.submit(&[], &s);
            }
            d_res_next
        } else {
            self.dres_boundary_in.clone()
        };

        for l in (self.shard.start..self.shard.end).rev() {
            let la = &ta.layers[l - self.shard.start];

            // ---- second residual add backward: res[l+1] = xmid + moe_out ----
            // Both branches receive the FULL upstream gradient (d_res_next):
            // `d_moe_out` is passed straight through (read-only reuse of the
            // same buffer, never mutated in place downstream); `d_xmid`'s own
            // base value is `d_res_next` too, ADD2'd with ln2's own dx below
            // (matching `qwen3::model.rs::build_backward_steps`'s own idiom of
            // computing a norm's dx into a private temp then ADD2-combining
            // with the residual branch, never accumulating in place).
            let d_moe_out = &d_res_next;
            let d_xn2 = self.moe_sublayer_bwd(l, &la.moe, d_moe_out, n);

            let d_ln2_dx = g.storage((n * d) as u64);
            let d_xmid = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.rmsnorm_bwd_step(&mut s, &la.xmid, &format!("blocks.{l}.ln2.weight"), &d_xn2, &d_ln2_dx, d, n);
                s.push(g.step(ADD2, &[&d_res_next, &d_ln2_dx, &d_xmid], &[n * d], n * d));
                g.submit(&[], &s);
            }

            // ---- first residual add backward: xmid = res[l] + attn_out ----
            // `d_attn_out` is `d_xmid` itself (read-only reuse, same reasoning
            // as `d_moe_out` above); `d_xn1` accumulates the mixer's own
            // weight-gradient chain (its first touch is `acc=0`, see each
            // mixer backward's own doc).
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
                self.rmsnorm_bwd_step(&mut s, &self.res[l], &format!("blocks.{l}.ln1.weight"), &d_xn1, &d_ln1_dx, d, n);
                s.push(g.step(ADD2, &[&d_xmid, &d_ln1_dx, &d_res_l], &[n * d], n * d));
                g.submit(&[], &s);
            }
            d_res_next = d_res_l;
        }

        // This stage's INPUT-boundary gradient (`dres[shard.start]`), for the
        // previous stage to read via `Self::read_in_dres`. Stashed
        // unconditionally (cheap: one buffer handle) -- a whole/embed-stage
        // build simply never has this read, mirroring the `res_numel`-sized
        // boundary buffers `qwen3::Qwen` always keeps live too.
        *self.dres_boundary_out.borrow_mut() = d_res_next.clone();

        // ---- vision-language splice backward: route the image rows' grad to
        // `d_img_embeds` and ZERO them in `d_res_next` BEFORE `EMB_BWD`, so the
        // image-placeholder token id never accumulates a spurious `tok.weight`
        // gradient from those rows (mirrors `qwen3::Qwen::backward`'s own
        // `mm_splice` case exactly). No-op unless `enable_mm_splice` was called.
        // Only meaningful on the embed stage (operates on `res[0]`/`dres[0]`).
        if self.shard.embed {
            if let Some((row0, n_rows)) = self.mm_splice.get() {
                g.submit(&[], &[model::vlm::splice_bwd(g, SPLICE_BWD, &d_res_next, &self.d_img_embeds, row0 * d, n_rows * d)]);
            }

            // ---- embedding backward (tok.weight; untied per this task's tiny
            // config -- lm_head.weight already got its own dW above). tok.weight
            // is never a LoRA target -- Frozen under a LoRA build, so skip this
            // dispatch entirely then (no grad buffer to write into; d_res_next's
            // own gradient has nowhere further to go, which is correct -- the
            // embedding IS the start of the graph).
            if self.trainable("tok.weight") {
                g.submit(&[], &[g.step(EMB_BWD, &[&self.tokens, &d_res_next, self.g("tok.weight")], &[n, d, v], v * d)]);
            }
        }
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    /// Run the forward graph and return the scalar loss. Only meaningful on
    /// a whole (single-device) instance or a pipeline stage that owns the
    /// head (`self.shard.head`) — `self.logits`/`self.ce_buf` are only
    /// written on the head stage (see [`Self::run_forward`]'s own gate); a
    /// non-head stage's forward step is driven through
    /// [`Self::run_forward`] directly by [`model::Shardable::run_forward_stage`]
    /// instead of this method, exactly mirroring `qwen3::Qwen::forward`'s own
    /// contract.
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
        assert_eq!(self.b, 1, "qwen35moe::logits_all requires b==1 (single sequence)");
        assert_eq!(
            tokens.len() as u32,
            self.t,
            "qwen35moe::logits_all requires tokens.len() == the configured t (no partial-length prefill in this pass)"
        );
        self.gpu.write(&self.tokens, tokens);
        self.run_forward();
        self.gpu.read(&self.logits, (self.t * self.cfg.vocab) as usize)
    }

    // =========================================================================
    // Single-sequence (batch=1) incremental decode -- the per-token twin of
    // `run_forward`/`logits_all` above. Text-only, fp32 only (no int8 decode
    // in this pass -- see `crate::q8`'s module doc for the separate int8
    // tier), single persistent sequence (no paging/continuous batching --
    // that is `model::serve::PagedDecoder`, separate later work built on top
    // of this). Structure mirrors `qwen3::Qwen`'s own `reset_cache`/`step`/
    // `decode_at`/`decode_submit`/`decode_steps` (`crates/qwen3/src/model.rs`)
    // at `n=1`, adjusted for this model's own per-layer math:
    // `layer_gdn_decode_step`/`layer_gqa_decode_step` are the single-token
    // siblings of `layer_gdn_fwd`/`layer_gqa_fwd` above, composed by
    // `run_decode_step` the same way `run_forward` composes the batched pair.
    // =========================================================================

    /// Reset decode state for a fresh sequence: the position counter and
    /// every GDN layer's persistent recurrent `state`/conv `hist` (both must
    /// start at zero for a fresh sequence -- see `model::gdn::gdn_recurrent_step`
    /// and `gdn_causal_conv1d_step`'s own docs). GQA layers' KV caches are
    /// deliberately left untouched: `layer_gqa_decode_step` only ever reads
    /// cache rows `0..=pos` (`model::block::gqa_decode_step`'s own doc), so
    /// stale rows beyond the new sequence's own length are never read -- the
    /// same reasoning `qwen3::Qwen::reset_cache` relies on to not re-zero its
    /// own `kcache`/`vcache`.
    pub fn reset_decode_cache(&self) {
        self.dec_pos.set(0);
        let mut clears: Vec<&DeviceBuffer> = Vec::new();
        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            if *ty == LayerType::Linear {
                clears.push(&self.gdn_state[l]);
                clears.push(&self.gdn_hist[l]);
            }
        }
        self.gpu.submit(&clears, &[]);
    }

    /// The absolute position the next [`Self::step`] will decode.
    pub fn decode_pos(&self) -> u32 {
        self.dec_pos.get()
    }

    /// **Incremental decode** of a single new token id at the current decode
    /// position, returning the final-norm hidden state (`[d_model]`) for that
    /// token -- the same return contract as `qwen3::Qwen::step`: apply this
    /// instance's head (`Self::cfg.head_weight()`) to it on the host to get
    /// logits, exactly as `logits_all`'s own caller would from a row of its
    /// output.
    pub fn step(&self, token_id: u32) -> Vec<f32> {
        assert_eq!(self.b, 1, "qwen35moe::step requires b==1 (single sequence)");
        assert!(self.q8.is_none(), "qwen35moe::step: fp32 decode only in this pass (int8 decode is out of scope)");
        assert!(
            (token_id as usize) < self.cfg.vocab as usize,
            "decode token id {token_id} exceeds vocab {} (checkpoint/tokenizer mismatch?)",
            self.cfg.vocab
        );
        let pos = self.dec_pos.get();
        assert!(pos < self.dec_cap, "qwen35moe::step: decode position {pos} exceeds capacity {}", self.dec_cap);
        // This instance's OWN persistent decode state -- see `DecodeCaches`'s
        // own doc for why `run_decode_step` takes it as an explicit parameter
        // rather than reading `self.gqa_kcache`/`self.gdn_state` directly.
        let caches = DecodeCaches {
            gqa_kcache: &self.gqa_kcache,
            gqa_vcache: &self.gqa_vcache,
            gqa_cap: self.dec_cap,
            gdn_state: &self.gdn_state,
            gdn_hist: &self.gdn_hist,
        };
        let hidden = self.run_decode_step(token_id, pos, &caches);
        self.dec_pos.set(pos + 1);
        self.gpu.read(&hidden, self.cfg.d_model as usize)
    }

    /// One incremental decode step's full layer stack -- the decode-shaped
    /// (`n=1`) sibling of [`Self::run_forward`]. Returns the final-norm
    /// hidden state buffer (unread). `caches` selects WHICH sequence's
    /// per-layer GQA cache / GDN state this call reads and updates -- see
    /// [`DecodeCaches`]'s own doc. `pub(crate)` (not `pub`) because the only
    /// caller outside this module is `crate::serve::Engine`, which drives
    /// this exact function per admitted request against its own paged/GdnSlot
    /// resources instead of a single instance-wide decode state.
    pub(crate) fn run_decode_step(&self, token_id: u32, pos: u32, caches: &DecodeCaches) -> DeviceBuffer {
        let g = &self.gpu;
        let d = self.cfg.d_model;

        g.write(&self.dec_tokens, &[token_id]);
        let mut res = g.storage(d as u64);
        g.submit(&[], &[g.step(EMBED, &[&self.dec_tokens, self.w("tok.weight"), &res], &[d, 1], d)]);

        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            let xn1 = g.storage(d as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, 1)]);

            let attn_out = match ty {
                LayerType::Linear => self.layer_gdn_decode_step(l, &xn1, &caches.gdn_state[l], &caches.gdn_hist[l]),
                LayerType::Full => self.layer_gqa_decode_step(l, &xn1, pos, &caches.gqa_kcache[l], &caches.gqa_vcache[l], caches.gqa_cap),
            };

            let xmid = g.storage(d as u64);
            g.submit(&[], &[g.step(ADD2, &[&res, &attn_out, &xmid], &[d], d)]);

            // Same `moe_sublayer` this file's batched path uses -- only the
            // row count (`n=1`) differs, so no decode-specific MoE function
            // is needed at all (per this change's own scope note).
            let (moe_out, _) = self.moe_sublayer(l, &xmid, 1);
            let res_next = g.storage(d as u64);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &moe_out, &res_next], &[d], d)]);
            res = res_next;
        }

        let xn_final = g.storage(d as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res, self.w("norm.weight"), &xn_final, d, 1)]);
        xn_final
    }

    /// One Gated DeltaNet layer's decode step -- the single-token sibling of
    /// [`Self::layer_gdn_fwd`]. Same 11-step math at `n=1`, EXCEPT: step 2
    /// dispatches [`gdn_causal_conv1d_step`] directly on the token-major
    /// `[1, conv_dim]` buffer (no `nlc_nchw`/`nchw_nlc` round trip -- that
    /// conversion exists only because `conv1d_fwd` is NCL-shaped;
    /// `gdn_causal_conv1d_step`'s own `x`/`y` are already `[N,C]`, see its
    /// doc), and steps 6-9 (kv_expand, chunk-major permute, `gdn_chunk_fwd`,
    /// permute back) become: `kv_expand_fwd` (STILL needed -- repeats
    /// `linear_num_key_heads` up to `linear_num_value_heads`, exactly as in
    /// `layer_gdn_fwd`) followed directly by [`gdn_recurrent_step`] on the
    /// already `[bh,...]`-shaped buffers -- no chunk-major permute at all,
    /// since `gdn_recurrent_step` takes no chunk axis. `query`/`key` are
    /// passed UNSCALED (`gdn_recurrent_step` applies `1/sqrt(dk)` itself, see
    /// its doc).
    ///
    /// `state`/`hist` are THIS call's recurrent state / conv history buffers
    /// (layer `l`'s slice of whichever [`DecodeCaches`] the caller is
    /// driving) -- not necessarily `self.gdn_state[l]`/`self.gdn_hist[l]`,
    /// see [`DecodeCaches`]'s own doc for why.
    fn layer_gdn_decode_step(&self, l: usize, xn1: &DeviceBuffer, state: &DeviceBuffer, hist: &DeviceBuffer) -> DeviceBuffer {
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
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // 1. mixed_qkv = in_proj_qkv(xn1).
        let mixed_qkv = g.storage(conv_dim as u64);
        g.submit(&[], &[g.step(MATMUL, &[xn1, self.w(&p("in_proj_qkv.weight")), &mixed_qkv], &[1, d, conv_dim], conv_dim)]);

        // 2. Streaming causal conv1d + SiLU (activation after the conv, same
        // as prefill) -- gdn_causal_conv1d_step's x/y are already [N=1,C],
        // exactly mixed_qkv's own layout, so no NLC/NCL conversion is needed.
        let conv_out = g.storage(conv_dim as u64);
        let conv_shape = GdnConvShape { n: 1, c: conv_dim, k: kw };
        g.submit(&[], &[gdn_causal_conv1d_step(g, &gdn_conv_ids(), &conv_shape, &mixed_qkv, self.w(&p("conv1d.weight")), hist, &conv_out)]);
        let mixed_act = g.storage(conv_dim as u64);
        g.submit(&[], &[g.step(SILU, &[&conv_out, &mixed_act], &[conv_dim], conv_dim)]);

        // 3. Split into query/key/value -- same whole-row split as prefill, n=1.
        let query = g.storage(key_dim as u64);
        let key = g.storage(key_dim as u64);
        let value = g.storage(value_dim as u64);
        g.submit(
            &[],
            &[
                g.step(CONCAT_SPLIT, &[&mixed_act, &query], &[1, conv_dim, key_dim, 0, 1, 1], key_dim),
                g.step(CONCAT_SPLIT, &[&mixed_act, &key], &[1, conv_dim, key_dim, key_dim, 1, 1], key_dim),
                g.step(CONCAT_SPLIT, &[&mixed_act, &value], &[1, conv_dim, value_dim, 2 * key_dim, 1, 1], value_dim),
            ],
        );

        // 4. L2-normalize query/key -- bare l2norm, same as prefill.
        let query_n = g.storage(key_dim as u64);
        let key_n = g.storage(key_dim as u64);
        g.submit(
            &[],
            &[
                g.step(L2NORM_SCALE, &[&query, &self.ones_khd, &query_n], &[nkh, khd, f(1e-6)], key_dim),
                g.step(L2NORM_SCALE, &[&key, &self.ones_khd, &key_n], &[nkh, khd, f(1e-6)], key_dim),
            ],
        );

        // 5. beta = sigmoid(in_proj_b(xn1)); g = decay-gate(in_proj_a(xn1));
        // z = in_proj_z(xn1) -- same as prefill.
        let bproj = g.storage(nvh as u64);
        let aproj = g.storage(nvh as u64);
        let z = g.storage(value_dim as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_b.weight")), &bproj], &[1, d, nvh], nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_a.weight")), &aproj], &[1, d, nvh], nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_z.weight")), &z], &[1, d, value_dim], value_dim),
            ],
        );
        let beta = g.storage(nvh as u64);
        let g_decay = g.storage(nvh as u64);
        g.submit(
            &[],
            &[
                g.step(SIGMOID, &[&bproj, &beta], &[nvh], nvh),
                g.step(GDN_DECAY_GATE, &[&aproj, self.w(&p("A_log")), self.w(&p("dt_bias")), &g_decay], &[1, nvh], nvh),
            ],
        );

        // 6. Repeat query/key from linear_num_key_heads to linear_num_value_heads.
        let query_w = g.storage((nvh * khd) as u64);
        let key_w = g.storage((nvh * khd) as u64);
        g.submit(
            &[],
            &[
                kv_expand_fwd(g, KV_EXPAND, &query_n, &query_w, 1, nvh, group, khd, nvh * khd, 0),
                kv_expand_fwd(g, KV_EXPAND, &key_n, &key_w, 1, nvh, group, khd, nvh * khd, 0),
            ],
        );

        // 7. gdn_recurrent_step -- the persistent single-token state update,
        // in place of gdn_chunk_fwd (no chunk-major permute either side).
        let shape = GdnShape { b: 1, h: nvh, t: 1, dk: khd, dv: vhd, chunk: 1 };
        let kv_mem = g.storage((nvh * vhd) as u64);
        let sub_out = g.storage((nvh * vhd) as u64);
        let scratch = GdnRecurrentScratch { kv_mem: &kv_mem, sub_out: &sub_out };
        let out_bh = g.storage((nvh * vhd) as u64);
        g.submit(
            &[],
            &gdn_recurrent_step(g, &gdn_ids(), &shape, &query_w, &key_w, &value, &g_decay, &beta, state, &scratch, &out_bh),
        );

        // 8. Gated RMSNorm (norm before gate, same as prefill).
        let normed = g.storage(value_dim as u64);
        let z_silu = g.storage(value_dim as u64);
        let gated = g.storage(value_dim as u64);
        g.submit(
            &[],
            &[
                rmsnorm_fwd(g, &kernel_ids(), &out_bh, self.w(&p("norm.weight")), &normed, vhd, nvh),
                g.step(SILU, &[&z, &z_silu], &[value_dim], value_dim),
                g.step(MUL, &[&normed, &z_silu, &gated], &[value_dim], value_dim),
            ],
        );

        // 9. out_proj.
        let out = g.storage(d as u64);
        g.submit(&[], &[g.step(MATMUL, &[&gated, self.w(&p("out_proj.weight")), &out], &[1, value_dim, d], d)]);
        out
    }

    /// One GQA layer's decode step -- the single-token sibling of
    /// [`Self::layer_gqa_fwd`]: q/k/v-proj, per-head QK-norm, single-position
    /// partial M-RoPE, append this token's k/v into the persistent per-layer
    /// KV cache and attend over `0..=pos` (`model::block::gqa_decode_step`,
    /// the same primitive `qwen3::Qwen::decode_steps` calls), sigmoid output
    /// gate, `o_proj`.
    ///
    /// M-RoPE at a single position: `rope2d_partial_fwd`'s table lookup is
    /// `row % tmod` with `tmod` always equal to the dispatch's own row count
    /// (`model::block::rope2d_fwd`'s doc, "tmod = rows: an exact per-token
    /// table, no frame-repeat") -- at `rows=1` that is always table row 0, so
    /// a slice into the construction-time whole-sequence `Self::cos`/`Self
    /// ::sin` table at row `pos` cannot be addressed this way. Instead this
    /// recomputes a fresh 1-row table for `pos` (`qwenvl::mrope::mrope_tables`
    /// with `[pos,pos,pos]`, mirroring this file's own text-only construction
    /// convention) into the persistent `Self::dec_cos`/`Self::dec_sin`
    /// buffers and rewrites it every call -- exactly `qwen3::Qwen::step_mrope`
    /// / `write_decode_mrope_table`'s own pattern for the identical structural
    /// reason. This is the one piece of this change with no exact
    /// qwen35moe-internal precedent to copy verbatim (qwen3's `Qwen` has no
    /// GDN layers and qwen35moe's own prefill path never decodes one position
    /// at a time) -- see this change's final report for that call-out.
    ///
    /// `kcache`/`vcache`/`cap` are THIS call's KV cache buffers and capacity
    /// (layer `l`'s slice of whichever [`DecodeCaches`] the caller is
    /// driving, and that same call's shared per-sequence capacity) -- not
    /// necessarily `self.gqa_kcache[l]`/`self.gqa_vcache[l]`/`self.dec_cap`,
    /// see [`DecodeCaches`]'s own doc for why.
    fn layer_gqa_decode_step(&self, l: usize, xn1: &DeviceBuffer, pos: u32, kcache: &DeviceBuffer, vcache: &DeviceBuffer, cap: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, qd, kvd) = (c.q_proj_dim(), c.q_dim(), c.kv_dim());
        let p = |s: &str| format!("blocks.{l}.self_attn.{s}");

        let q_full = g.storage(qpd as u64);
        let k = g.storage(kvd as u64);
        let v = g.storage(kvd as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn1, self.w(&p("q_proj.weight")), &q_full], &[1, d, qpd], qpd),
                g.step(MATMUL, &[xn1, self.w(&p("k_proj.weight")), &k], &[1, d, kvd], kvd),
                g.step(MATMUL, &[xn1, self.w(&p("v_proj.weight")), &v], &[1, d, kvd], kvd),
            ],
        );

        // Per-head de-interleaved [query|gate] split -- same as prefill, n=1.
        let q_value = g.storage(qd as u64);
        let q_gate = g.storage(qd as u64);
        g.submit(
            &[],
            &[
                g.step(CONCAT_SPLIT, &[&q_full, &q_value], &[nh, 2 * hd, hd, 0, 1, 1], nh * hd),
                g.step(CONCAT_SPLIT, &[&q_full, &q_gate], &[nh, 2 * hd, hd, hd, 1, 1], nh * hd),
            ],
        );

        let q_normed = g.storage(qd as u64);
        let k_normed = g.storage(kvd as u64);
        g.submit(
            &[],
            &[
                rmsnorm_fwd(g, &kernel_ids(), &q_value, self.w(&p("q_norm.weight")), &q_normed, hd, nh),
                rmsnorm_fwd(g, &kernel_ids(), &k, self.w(&p("k_norm.weight")), &k_normed, hd, nkv),
            ],
        );

        // Single-position partial M-RoPE -- see this function's own doc.
        let half = c.rotary_dim() / 2;
        let (cos_row, sin_row) = qwenvl::mrope::mrope_tables(&[[pos, pos, pos]], c.mrope_section, c.rotary_dim(), c.rope_theta);
        g.write_f32(&self.dec_cos, &cos_row);
        g.write_f32(&self.dec_sin, &sin_row);
        g.submit(
            &[],
            &[
                rope2d_partial_fwd(g, ROPE2D_PARTIAL, &q_normed, &self.dec_cos, &self.dec_sin, 1, nh, half, qd, 0, hd),
                rope2d_partial_fwd(g, ROPE2D_PARTIAL, &k_normed, &self.dec_cos, &self.dec_sin, 1, nkv, half, kvd, 0, hd),
            ],
        );

        // Append k/v into this layer's persistent cache and attend over 0..=pos.
        let scores = g.storage((nh * cap) as u64);
        let probs = g.storage((nh * cap) as u64);
        let ctx = g.storage(qd as u64);
        g.submit(
            &[],
            &gqa_decode_step(
                g,
                &gqa_decode_ids(),
                nh,
                nkv,
                hd,
                pos,
                cap,
                &q_normed,
                &k_normed,
                &v,
                kcache,
                vcache,
                &scores,
                &probs,
                &ctx,
            ),
        );

        let gate = g.storage(qd as u64);
        let ctx_gated = g.storage(qd as u64);
        let out = g.storage(d as u64);
        g.submit(
            &[],
            &[
                g.step(SIGMOID, &[&q_gate, &gate], &[qd], qd),
                g.step(MUL, &[&ctx, &gate, &ctx_gated], &[qd], qd),
                g.step(MATMUL, &[&ctx_gated, self.w(&p("o_proj.weight")), &out], &[1, qd, d], d),
            ],
        );
        out
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    // ---- pipeline-parallel cross-stage seam (`model::Shardable`) ----------

    /// Residual-stream element count at a stage boundary (`b·t·d_model`).
    /// Mirrors `qwen3::Qwen::res_numel` exactly.
    fn res_numel(&self) -> usize {
        (self.b * self.t) as usize * self.cfg.d_model as usize
    }
    /// Read this stage's OUTPUT residual `res[shard.end]` (input to the next
    /// stage's [`Self::write_in_res`]).
    pub fn read_out_res(&self) -> Vec<f32> {
        self.gpu.read(&self.res[self.shard.end], self.res_numel())
    }
    /// Write this stage's INPUT residual `res[shard.start]` (from the
    /// previous stage's [`Self::read_out_res`]).
    pub fn write_in_res(&self, data: &[f32]) {
        self.gpu.write(&self.res[self.shard.start], bytemuck::cast_slice(data));
    }
    /// Read this stage's INPUT-boundary residual gradient `dres[shard.start]`
    /// (for the previous stage's [`Self::write_out_dres`]) — populated by the
    /// preceding `backward()` call (see [`Self::backward`]'s own doc for why
    /// this is a stashed buffer rather than a `self.dres[..]` array index).
    pub fn read_in_dres(&self) -> Vec<f32> {
        self.gpu.read(&self.dres_boundary_out.borrow(), self.res_numel())
    }
    /// Write this stage's OUTPUT-boundary residual gradient `dres[shard.end]`
    /// (from the next stage's [`Self::read_in_dres`]) — consumed by the next
    /// `backward()` call on a non-head stage.
    pub fn write_out_dres(&self, data: &[f32]) {
        self.gpu.write(&self.dres_boundary_in, bytemuck::cast_slice(data));
    }

    /// Every fp32-store name for an inference or full-training build (`self.ps
    /// .params`, unchanged behaviour -- see
    /// `int8_model_excludes_quantized_names_from_the_fp32_param_store`, which
    /// depends on this listing every Frozen inference weight). A LoRA
    /// training build (`self.is_train && cfg.lora.is_some()`) instead lists
    /// only the trainable `.lora_a`/`.lora_b` adapter tensors (`self.ps
    /// .trainable`) -- the frozen base has no gradient buffer (see
    /// [`Self::trainable`]), so listing it here would make any `read_grad`
    /// caller (gradcheck's `directional_check`, `crate::lora::save_adapter`)
    /// panic. Mirrors `qwen3::model.rs`'s own `param_names` filter.
    pub fn param_names(&self) -> Vec<String> {
        if self.is_train && self.cfg.lora.is_some() {
            self.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
        } else {
            self.ps.params.iter().map(|(n, _)| n.clone()).collect()
        }
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
            _ => panic!("qwen35moe::Qwen35 only supports Batch::Lm"),
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
