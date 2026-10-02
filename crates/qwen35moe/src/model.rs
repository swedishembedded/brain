// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5-35B-A3B hybrid decoder forward AND backward assembly - device
//! [`Step`]s wired to the [`model::Model`] trait.
//!
//! **Scope, strictly**: no incremental/KV-cache decode with an image splice
//! (single-sequence text decode via [`Qwen35::step`] is separate follow-on
//! work; see `crate::vl`'s own doc) - a prefill-shaped vision-language
//! embedding splice IS wired here ([`Qwen35::enable_mm_splice`]/
//! [`Qwen35::write_img_embeds`], driven by `crate::vl::Qwen35Vl`), mirroring
//! `qwen3::Qwen`'s own seam. No incremental/KV-cache decode (a full-sequence
//! prefill-shaped forward over a fixed `t` only, matching [`model::gdn`]'s own
//! "chunked/prefill only" scope), no T-padding (`t` must already be a
//! multiple of the derived GDN chunk size - asserted loudly in
//! [`Qwen35::new_on`], see [`gdn_chunk_size`]). **Two construction paths**:
//! [`Qwen35::new`]/[`Qwen35::new_i8`] build a frozen (`Role::Frozen`),
//! forward-only instance (`backward`/`zero_grads`/`adamw_step` all assert and
//! panic on such an instance - see [`Qwen35::backward`]'s own assert);
//! [`Qwen35::new_train`] builds a fully trainable (`Role::Trainable`
//! everywhere, full-parameter - no LoRA-specific subset) instance whose
//! `forward()` additionally saves the activation cache `backward()` reads
//! (see [`Qwen35::train_acts`]'s own doc for the exact "one forward, one
//! backward, then the cache is gone" contract). Int8 and training are
//! mutually exclusive.
//!
//! **Honest scope note on numerical parity**: this environment has no
//! `torch`/`transformers` installed, so bit-exact parity against the real
//! HF reference is **not achievable or
//! claimed here**. Every op below was checked line-for-line against the real
//! `modeling_qwen3_5_moe.py` from the released checkpoint (not a secondhand
//! description), resolved under the model store root - see
//! `BRAIN_QWEN35_DIR`. The achievable and required bar for this pass is
//! *structural* correctness: compiles, runs, produces finite output,
//! deterministic across repeated runs at the same seed.
//!
//! One assumption worth flagging up front: `Qwen3_5MoeRMSNorm.forward`
//! (attention/MLP layer norms) computes `output * (1.0 + weight)`, not a
//! plain `output * weight` - the reference's own comment says so ("We
//! initialize with 0s to be 1 centered as the RMSNorm here does"). This
//! engine's shared `rmsnorm.wgsl` (used by every model, not just this one)
//! assumes the plain-multiply form, i.e. that a checkpoint's stored weight is
//! already the FINAL per-channel multiplier - which is exactly what
//! llama.cpp's GGUF conversion typically bakes in for this style of norm (the
//! `+1` folded into the stored value at conversion time), and is also what
//! `crates/qwen35moe/src/import.rs` (unmodified by this change) assumes. If that
//! assumption is wrong for some future checkpoint source, RMSNorm output
//! would be off by a `(x+1)` vs `x` factor - a real, if unlikely, gap, called
//! out here rather than silently assumed away. `Qwen3_5MoeRMSNormGated` (the
//! Gated DeltaNet output norm) is a genuinely different class with no such
//! `+1` (`hidden_states = self.weight * hidden_states...`, verified directly
//! against the reference), so `rmsnorm_fwd` is exactly right there.
//!
//! ## Layer forward, per the real reference (`Qwen3_5MoeDecoderLayer.forward`)
//!
//! Every layer, regardless of token-mixer type: `xn1 = rmsnorm(res)`, mix
//! (GDN or GQA, below), `xmid = res + mix_out`, `xn2 = rmsnorm(xmid)`, MoE
//! (universal - every layer, no dense fallback), `res' = xmid + moe_out`.
//!
//! **Gated DeltaNet** (`Qwen3_5MoeGatedDeltaNet.forward`): `mixed_qkv =
//! in_proj_qkv(xn1)` → depthwise causal conv1d (`causal_conv1d_fn`, SiLU
//! activation AFTER the conv - confirmed from `self.activation =
//! config.hidden_act` and every Qwen family config using `"silu"`, not
//! assumed) → split into `query,key,value` (one whole-row contiguous split -
//! confirmed via `torch.split(mixed_qkv, [key_dim,key_dim,value_dim],
//! dim=-1)`, i.e. NOT per-head) → L2-normalize `query`/`key` (no learnable
//! scale, confirmed `use_qk_l2norm_in_kernel=True` calls the bare `l2norm`
//! helper) → `beta=sigmoid(in_proj_b(xn1))`,
//! `g=-exp(A_log)*softplus(in_proj_a(xn1)+dt_bias)` (confirmed verbatim) →
//! repeat `query`/`key` from `linear_num_key_heads` to `linear_num_value_heads`
//! (`repeat_interleave`, i.e. `model::block::kv_expand_fwd`'s exact
//! `repeat_kv` semantics - confirmed, no new kernel needed) → chunk-major
//! permute (new `gdn_layout_permute.wgsl`, see its own header) →
//! `model::gdn::gdn_chunk_fwd` → permute back → gated RMSNorm
//! (`Qwen3_5MoeRMSNormGated`: norm computed on the UNGATED value first - "#
//! Norm before gate" in the reference - THEN `* weight`, THEN `*
//! SiLU(in_proj_z(xn1))`; confirmed this is exactly `rmsnorm_fwd` composed
//! with `silu.wgsl` + `mul.wgsl`, no new kernel) → `out_proj`.
//!
//! **GQA (`Qwen3_5MoeAttention.forward`)**: `q_proj` emits a DOUBLED width
//! (`num_heads*head_dim*2`) whose split into `query`/`gate` is **per-head
//! interleaved**, not a single whole-row split - confirmed:
//! `torch.chunk(q_proj(x).view(*shape,-1,head_dim*2), 2, dim=-1)` chunks the
//! LAST axis of a `[...,n_heads,2*head_dim]` view, so head `h`'s own
//! `2*head_dim` slice splits into its own first/second half. `concat_split
//! .wgsl` (existing - see this module's kernel-choice note below) handles
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
//!   the identical flat `i`) - it copies a sub-REGION between two
//!   same-shaped buffers, it cannot project a wide strided row into a fresh
//!   COMPACT narrower buffer (which is what a real split needs: downstream
//!   consumers like `l2norm_scale`/`gdn_chunk_fwd`/`gqa_fwd` all require
//!   compact operands, and none of them accept an extra `row_stride`
//!   parameter to work around it). `concat_split.wgsl` (existing - originally
//!   for `concat2`'s backward channel-slice) does exactly the needed gather:
//!   `da[n,c,h,w] = dy[n, c+c_off, h, w]` - setting `H=W=1` makes it a plain
//!   compact channel-slice-into-a-fresh-buffer copy, and folding a repeated
//!   axis (e.g. per-head) into its `N` handles the interleaved case too. Used
//!   for both splits; no new kernel.
//! - **chunk-major permute**: genuinely new (`gdn_layout_permute.wgsl`) - see
//!   its own header for why `nlc_nchw`/`nchw_nlc` (the only existing
//!   layout-permute kernels) don't cover a 5-index permute that also SPLITS
//!   the token axis into `(chunk, c)`.
//! - **GDN q/k head repeat**: `model::block::kv_expand_fwd` is exactly
//!   `repeat_kv`-shaped and shape-generic (any `hd`, not GQA-specific) - used
//!   as-is, no variant needed.
//! - **conv1d layout**: `conv1d.wgsl` is NCL (`[N,Cin,L]`); every other
//!   buffer in this engine is token-major (`[rows, C]` = `[B,T,C]` row-major,
//!   equivalently NLC with `N=B,L=T,C=C`). `nlc_nchw`/`nchw_nlc` (existing)
//!   convert between exactly these two layouts with `hw=T`; no new kernel.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use gpu_core::select::Dtype;
use gpu_core::{f, DeviceBuffer, Dispatch, Gpu, Step};
use model::ops::{Act, Ops, TierPolicy, Weight};
use model::Shard;
use paramstore::{ParamStore, Role};

use audio::conv::ConvKernels;
use model::block::{self, rmsnorm_bwd, rmsnorm_fwd, swiglu_bwd, KernelIds};
use model::gdn::{GdnBwdIds, GdnConvIds, GdnIds, GdnShape};
// Re-exported (not just imported): `qwen35moe::model::gdn_chunk_size` is part
// of this crate's own public API (`crates/npu`'s topology export and several
// test crates call it as such), unlike the scratch-buffer types above, which
// were never used outside this module even before they lived here.
pub use model::gdn::gdn_chunk_size;
use model::kv_tier::{KvKernels, KvLayer, KvTier};
use model::moe::{
    expert_fwd, expert_fwd_grouped, moe_layer_bwd, router_fwd_kind, shared_expert_fwd, ExpertBwdScratch,
    ExpertGrads, ExpertScratch, ExpertWeights, GroupedExpertFwdIds, GroupedExpertScratch, MoeActs,
    MoeIds, MoeIdsBwd, MoeShape, RouterBwdIds, RouterKind, SharedExpertIds, SharedExpertScratch,
};
use optim::Optim;

use crate::config::{LayerType, Qwen35Config};
use crate::q8::{Bank8, Qwen35Q8};

// ---- kernel pipeline (order fixes the indices below) -----------------------

const STATIC_PIPELINES: &[(&str, &str)] = &[
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
    // variant (`sx`/`sw` baked into the uniform, see its own doc); the mixer
    // linears' `model::ops::Ops` façade and `Qwen35Q8`'s own MoE-expert
    // path both need the DYNAMIC per-token/per-channel variant (`sx`/`sw` as
    // buffers, indexed `sx[row]`/`sw[col]`) -- `kernels::MATMUL_I8_DYN`,
    // registered under this name so `Ops::bind` can resolve it (see
    // `model::ops`'s own doc), exactly as `qwen3::model.rs`'s own `Q8`
    // pipeline registers under this same local name (`qwen/src/model.rs:159`).
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),                   // 40
    // Slot 41 was the per-expert int8 expert kernel; the gather-layout tier at the
    // end of this list replaced it. The entry stays so every index below holds.
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
    // -- decode-only sparse MoE dispatch tier -- see
    // `Qwen35::moe_sublayer_decode_sparse`. Appended at the true end, same
    // convention as the splice/LoRA tiers above.
    ("router_topk_compact", kernels::ROUTER_TOPK_COMPACT),       // 93
    // -- coalesced RMSNorm -- the throughput twin of index 0, selected by
    // `block::rms_variant` inside `block::rmsnorm_fwd`. Appended at the true
    // end, same convention as the tiers above, so every const stays put.
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),                     // 94
    // -- coalesced RMSNorm BACKWARD-x -- the throughput twin of `rmsnorm_dx`,
    // selected by the SAME `block::rms_variant` policy inside
    // `block::rmsnorm_bwd`. Appended at the true end, same convention as the
    // tiers above, so every const stays put.
    ("rmsnorm_dx_rows", kernels::RMSNORM_DX_ROWS),               // 95
    // -- prefill/inference grouped-GEMM MoE dispatch (M5.12) -- device-side
    // top-k routing + histogram/scan/permute + one grouped GEMM per
    // projection, replacing the `n_experts`-long dense per-expert loop for
    // every non-training, non-int8 forward with `rows > 1` (decode's `rows
    // == 1` path above already has its own, cheaper, host-readback-based
    // sparse dispatch). See `Self::moe_sublayer`'s grouped-forward branch and
    // `model::moe::expert_fwd_grouped`'s own module doc (M5.10, first landed
    // for `crates/deepseek2`) for the full pipeline. Appended at the true
    // end, same convention as every tier above.
    ("moe_group_counts", kernels::MOE_GROUP_COUNTS),             // 96
    ("scan_block", kernels::SCAN_BLOCK),                         // 97
    ("scan_add", kernels::SCAN_ADD),                             // 98
    ("moe_group_perm_emit", kernels::MOE_GROUP_PERM_EMIT),       // 99
    ("matmul_reg3_grouped", kernels::MATMUL_REG3_GROUPED),       // 100
    ("moe_group_combine", kernels::MOE_GROUP_COMBINE),           // 101
    // -- batched (cross-sequence) paged decode tier -- see
    // `model::block::gqa_decode_batched_step`, dispatched by
    // `Qwen35::run_decode_batch`. The first three used to be appended by name
    // in `pipelines()` below purely to satisfy `Ops::REQUIRED_KERNELS`; the
    // batched decode path dispatches them directly, so they get hand-numbered
    // consts like every other kernel it uses. Appended at the true end, same
    // convention as every tier above.
    // 102-104: the f32 KV tier's append/scores/apply, resolved by NAME through
    // `model::kv_tier::KvKernels` (the compact tiers' siblings ride in
    // `pipelines()`), never by these positions.
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),   // 102
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED), // 103
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),   // 104
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),     // 105
    // -- gather-layout int8 sparse-MoE tier -- see `Qwen35::moe_sublayer_i8`.
    // Appended at the true end, same convention as every tier above.
    ("moe_router_topk", kernels::MOE_ROUTER_TOPK),               // 106
    ("moe_i8_gemv_gather", kernels::MOE_I8_GEMV_GATHER),         // 107
    ("moe_swiglu_quant", kernels::MOE_SWIGLU_QUANT),             // 108
    ("moe_slot_combine", kernels::MOE_SLOT_COMBINE),             // 109
    // -- device-side head (greedy / top-k) tier -- see `Qwen35::head_*_dev`.
    ("argmax_part", kernels::ARGMAX_PART),                       // 110
    ("argmax_final", kernels::ARGMAX_FINAL),                     // 111
    ("topk_extract_step", kernels::TOPK_EXTRACT_STEP),           // 112
    ("pool_rows_gather2", kernels::POOL_ROWS_GATHER2),           // 113
    ("pool_rows_scatter2", kernels::POOL_ROWS_SCATTER2),         // 114
    ("moe_route_count", kernels::MOE_ROUTE_COUNT),               // 115
    ("moe_route_scan", kernels::MOE_ROUTE_SCAN),                 // 116
    ("moe_route_emit", kernels::MOE_ROUTE_EMIT),                 // 117
    ("moe_i8_grouped", kernels::MOE_I8_GROUPED),                 // 118
    // Prefill fast path of the Gated-DeltaNet chunk forward, see
    // `model::gdn::use_fast_kernels`.
    ("gdn_ut_fwd", kernels::GDN_UT_FWD),                         // 119
    ("bmm_tiled", kernels::BMM_TILED),                           // 120
];

/// This model's FULL kernel set: `STATIC_PIPELINES` (every hand-numbered
/// const above indexes into this -- unchanged positions 0..93) followed by
/// the `model::ops::Ops` façade's own required kernels, appended with
/// NO named consts of their own -- resolved by `Ops::new` purely BY NAME
/// (`Gpu::kernel_index`), never by position. Mirrors `qwen3::model::
/// pipelines`'s own recipe (including the `matmul_reg2` -> `matmul_reg3`
/// bit-identical-faster-twin registration) and its own doc comment's
/// rationale for why this must be one combined list on `self.gpu`'s own `Gpu`
/// handle rather than a second one built via `Gpu::new_like` for `self.ops`.
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = STATIC_PIPELINES.to_vec();
        v.push(("matmul_gemv", kernels::MATMUL_GEMV));
        v.push(("matmul_reg2", kernels::MATMUL_REG3));
        v.push(("matmul_i8_gemv", kernels::MATMUL_I8_GEMV));
        v.push(("matmul_q4_dyn", kernels::MATMUL_Q4_DYN));
        // M5.5's register-tiled Q4 GEMM - `Ops::bind`'s `(PackedInt8, Q4)`
        // arm now resolves to this name instead of the naive `matmul_q4_dyn`
        // above (which stays registered for `model::dispatch`'s own bespoke
        // selector, unaffected by this facade-only change).
        v.push(("matmul_q4_dyn_reg", kernels::MATMUL_Q4_DYN_REG));
        v.push(("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV));
        // `Ops::REQUIRED_KERNELS` also demands the bf16/f16 storage-tier
        // variants even though this crate never builds a `Weight::BF16`/
        // `Weight::F16` and never dispatches the generic `paged_*_batched`
        // family -- see `Ops::new`'s own doc comment ("every model that
        // builds an `Ops` must register the full façade kernel set, not just
        // the tiers it plans to use"). Compiled, never dispatched.
        for dt in [Dtype::BF16, Dtype::F16] {
            v.push(kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", dt).unwrap());
            v.push(kernels::template::dtype_variant("embed", kernels::EMBED, "emb", dt).unwrap());
            v.push(kernels::template::dtype_variant("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", dt).unwrap());
        }
        v.push(
            kernels::template::dtype_variant_store(
                "paged_kv_append_batched_word",
                kernels::PAGED_KV_APPEND_BATCHED_WORD,
                "pool",
                Dtype::BF16,
            )
            .unwrap(),
        );
        v.push(
            kernels::template::dtype_variant(
                "paged_decode_scores_batched",
                kernels::PAGED_DECODE_SCORES_BATCHED,
                "pool_k",
                Dtype::BF16,
            )
            .unwrap(),
        );
        v.push(
            kernels::template::dtype_variant(
                "paged_decode_apply_batched",
                kernels::PAGED_DECODE_APPLY_BATCHED,
                "pool_v",
                Dtype::BF16,
            )
            .unwrap(),
        );
        v.push(kernels::template::dtype_variant("matmul_dx", kernels::MATMUL_DX, "w", Dtype::BF16).unwrap());
        // M12: affine K-quant (Q4_K/Q5_K) kernels plus the group=16 (Q6_K)
        // reuse of the existing symmetric kernels via template knobs -
        // `Ops::REQUIRED_KERNELS` demands these too (see `model::ops::
        // kernel_list`'s own doc comment for why these are `kernels::
        // template::interned` specialisations, not separate `.wgsl` files).
        // This crate never builds a `Weight::KQuant` (no GGUF K-quant loader
        // here). Compiled, never dispatched - same "REQUIRED_KERNELS demands
        // it, this crate never uses it" precedent as the bf16/f16 storage
        // tiers above.
        v.push(("quant_group_sum", kernels::QUANT_GROUP_SUM));
        v.push(kernels::template::interned("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", 4)]).unwrap());
        v.push(kernels::template::interned("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", 8)]).unwrap());
        v.push(kernels::template::interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", 4)]).unwrap());
        v.push(kernels::template::interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", 8)]).unwrap());
        v.push(kernels::template::interned("matmul_i8_dyn", kernels::MATMUL_I8_DYN, &[("QPG", 1)]).unwrap());
        v.push(kernels::template::interned("matmul_i8_gemv", kernels::MATMUL_I8_GEMV, &[("WPG", 4)]).unwrap());
        // The compact KV tiers (`model::kv_tier`) are resolved BY NAME when a
        // cache of that tier is used, so their kernels ride along here.
        let kv = model::kv_tier::kernel_list(&v);
        v.extend(kv);
        v
    })
}

const RMSNORM: usize = 0;
const RMSNORM_ROWS: usize = 94;
const RMSNORM_DX_ROWS: usize = 95;
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
const SPLICE: usize = 90;
const SPLICE_BWD: usize = 91;
const AXPY: usize = 92;
const ROUTER_TOPK_COMPACT: usize = 93;
const MOE_GROUP_COUNTS: usize = 96;
const SCAN_BLOCK: usize = 97;
const SCAN_ADD: usize = 98;
const MOE_GROUP_PERM_EMIT: usize = 99;
const MATMUL_REG3_GROUPED: usize = 100;
const MOE_GROUP_COMBINE: usize = 101;
const DECODE_SOFTMAX_BATCHED: usize = 105;
const MOE_ROUTER_TOPK: usize = 106;
const MOE_I8_GEMV_GATHER: usize = 107;
const MOE_SWIGLU_QUANT: usize = 108;
const MOE_SLOT_COMBINE: usize = 109;
const ARGMAX_PART: usize = 110;
const ARGMAX_FINAL: usize = 111;
const TOPK_EXTRACT_STEP: usize = 112;
const POOL_ROWS_GATHER2: usize = 113;
const POOL_ROWS_SCATTER2: usize = 114;
const MOE_ROUTE_COUNT: usize = 115;
const MOE_ROUTE_SCAN: usize = 116;
const MOE_ROUTE_EMIT: usize = 117;
const MOE_I8_GROUPED: usize = 118;
const GDN_UT_FWD: usize = 119;
const BMM_TILED: usize = 120;
/// Partial-argmax chunks the device head splits a `[vocab]` row into
/// (`argmax_part` then `argmax_final`) - the same split `qwen35` uses.
const HEAD_ARGMAX_CHUNKS: u32 = 256;
/// Rows at or above which a chunk round draws its per-layer temporaries from
/// [`gpu_core::scratch::Arena`] - and so pays that arena's drain, a blocking
/// `poll_wait` per layer. See `qwen35::model::CHUNK_ARENA_MIN_ROWS` for the
/// measured crossover (the trade depends on the row count and nothing else).
const CHUNK_ARENA_MIN_ROWS: u32 = 16;
/// Weight rows one `moe_i8_gemv_gather` workgroup covers (64 threads, 16 lanes
/// per row) - the kernel's own constant, restated here only to size its grid.
const MOE_GATHER_COLS: u32 = 4;
/// Slots per tile and weight rows per workgroup of `moe_i8_grouped` - that
/// kernel's own constants, restated to size its grid and the routing tables.
const MOE_GROUPED_MR: u32 = 8;
const MOE_GROUPED_COLS: u32 = 4;
/// Rows at which the expert GEMMs switch from one slot per weight pass
/// (`moe_i8_gemv_gather`) to the grouped kernel. Below it a token's experts are
/// mostly distinct matrices and there is nothing to share. Swept on a GH200 on
/// the real 35B-A3B decode step, device-timed (`qwen35moe_decode_profile`): the
/// gather GEMV leads through 5 rows (13.8 ms against 14.2 at 5), the grouped
/// kernel from 7 (15.4 against 14.4), and by 16 rows it is 18.0 ms against 28.1.
pub const MOE_GROUPED_MIN_ROWS: u32 = 7;

/// Every slot is a REAL kernel now (backward is wired, see [`Qwen35::backward`]):
/// `rope`/`rope_bwd` still point at `rmsnorm` (index 0) because qwen35 never
/// dispatches `block::rope_fwd`/`rope_bwd` (it uses the M-RoPE table-driven
/// `rope2d_partial_{fwd,bwd}` instead, which take their own kernel index, not
/// a [`KernelIds`] field) - harmless, matching `qwen3omnimoe::thinker::kernel_ids`'s
/// own convention for a slot this model genuinely never dispatches.
fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
        rmsnorm_dx_rows: RMSNORM_DX_ROWS,
        rmsnorm_dw: RMSNORM_DW,
        // Rotation here is table-driven M-RoPE (`rope2d`, via the mixer id
        // sets), never `block::rope_fwd`/`rope_bwd` - so these two slots are
        // UNREGISTERED rather than standing in for `rmsnorm`, which is a live
        // kernel and would misroute instead of failing.
        rope: block::UNREGISTERED,
        rope_bwd: block::UNREGISTERED,
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
        rmsnorm_rows: RMSNORM_ROWS,
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
        fast: Some(model::gdn::GdnFastIds { ut_fwd: GDN_UT_FWD, bmm_tiled: BMM_TILED }),
    }
}

fn moe_ids() -> MoeIds {
    MoeIds { router_gate: ROUTER_GATE, linear_gated: MOE_LINEAR_GATED, silu_mul: SILU_MUL, scale_add: SCALE_ADD }
}

/// Kernel ids for [`model::moe::expert_fwd_grouped`] -- the `rows > 1`
/// prefill/inference sibling of [`moe_ids`]'s per-expert dense loop, wired
/// exactly like `crates/deepseek2/src/model.rs::grouped_expert_ids` (same
/// kernel set, same field-for-field mapping -- no new kernel this crate
/// adds).
fn grouped_expert_ids() -> GroupedExpertFwdIds {
    GroupedExpertFwdIds {
        router_topk_compact: ROUTER_TOPK_COMPACT,
        group_counts: MOE_GROUP_COUNTS,
        scan_block: SCAN_BLOCK,
        scan_add: SCAN_ADD,
        perm_emit: MOE_GROUP_PERM_EMIT,
        gather: EMBED,
        gemm_grouped: MATMUL_REG3_GROUPED,
        silu_mul: SILU_MUL,
        combine: MOE_GROUP_COMBINE,
    }
}

/// [`model::gdn::gdn_causal_conv1d_step`]'s kernel id -- the streaming
/// causal-conv decode step, dispatched by
/// [`model::gdn_mixer::gdn_mixer_decode_fwd`] in place of a whole-sequence
/// `conv1d_fwd`.
fn gdn_conv_ids() -> GdnConvIds {
    GdnConvIds { causal_conv1d_step: CAUSAL_CONV1D_STEP }
}

/// [`model::gdn_mixer::gdn_mixer_decode_fwd`]'s decode-only kernel ids.
fn gdn_mixer_decode_ids() -> model::gdn_mixer::GdnMixerDecodeIds {
    model::gdn_mixer::GdnMixerDecodeIds { conv: gdn_conv_ids(), splice: SPLICE }
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

/// [`model::gdn_mixer`]'s kernel ids - the mixer's own non-chunk kernels,
/// bundling [`kernel_ids`]/[`conv_kernels`]/[`gdn_ids`]/[`gdn_bwd_ids`] as
/// sub-fields (same convention as `model::block::GqaAttnIds`).
fn gdn_mixer_ids() -> model::gdn_mixer::GdnMixerIds {
    model::gdn_mixer::GdnMixerIds {
        kernels: kernel_ids(),
        conv: conv_kernels(),
        chunk: gdn_ids(),
        chunk_bwd: gdn_bwd_ids(),
        nlc_nchw: NLC_NCHW,
        nchw_nlc: NCHW_NLC,
        silu: SILU,
        silu_bwd: SILU_BWD,
        concat_split: CONCAT_SPLIT,
        concat2: CONCAT2,
        l2norm_scale: L2NORM_SCALE,
        l2norm_scale_dx: L2NORM_SCALE_DX,
        sigmoid: SIGMOID,
        sigmoid_bwd: SIGMOID_BWD,
        gdn_decay_gate: GDN_DECAY_GATE,
        gdn_decay_gate_bwd: GDN_DECAY_GATE_BWD,
        kv_expand: KV_EXPAND,
        kv_expand_bwd: KV_EXPAND_BWD,
        gdn_layout_permute: GDN_LAYOUT_PERMUTE,
        mul: MUL,
        bias_grad: BIAS_GRAD,
    }
}

/// [`model::gqa_mixer`]'s kernel ids.
fn gqa_mixer_ids() -> model::gqa_mixer::GqaMixerIds {
    model::gqa_mixer::GqaMixerIds {
        kernels: kernel_ids(),
        concat_split: CONCAT_SPLIT,
        concat2: CONCAT2,
        sigmoid: SIGMOID,
        sigmoid_bwd: SIGMOID_BWD,
        mul: MUL,
        rope2d_partial: ROPE2D_PARTIAL,
    }
}

/// [`model::moe::router_bwd`]'s kernel ids - `Softmax` router (qwen35's own,
/// see `moe_sublayer`'s `RouterKind::Softmax` choice), so `expert_counts` is
/// required (aux-loss usage fractions), unused by the returned scalar loss
/// (`aux_coef=0.0`, see `moe_sublayer`'s own comment) but still dispatched -
/// `router_bwd.wgsl`'s own interface requires the `fe` buffer to exist.
fn router_bwd_ids() -> RouterBwdIds {
    RouterBwdIds { router_bwd: ROUTER_BWD, expert_counts: Some(EXPERT_COUNTS) }
}

/// [`model::moe::expert_dgate`]/[`expert_bwd`]'s kernel ids (composed by
/// [`moe_layer_bwd`]) - `linear_gated: true` selects the row-skipping
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

// ---- backward activation cache (training builds only) ----------------------

/// Everything [`Qwen35::layer_gdn_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GDN mixer arm. `internals` is the LoRA/dtype-
/// agnostic mixer math's own saved state (`model::gdn_mixer`, shared with
/// `crates/qwen35`); `gated` is this crate's own `out_proj` input (the
/// hoisted forward's return value), kept here since only the LOCAL `out_proj`
/// backward reads it - see `model::gdn_mixer`'s own module doc for the
/// projections-stay-in-the-model boundary this split follows.
struct GdnLayerActs {
    internals: model::gdn_mixer::GdnMixerActs,
    gated: DeviceBuffer,
}

/// Everything [`Qwen35::layer_gqa_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GQA mixer arm. `internals`/`ctx_gated` split the
/// same way as [`GdnLayerActs`] - see that struct's own doc.
struct GqaLayerActs {
    internals: model::gqa_mixer::GqaMixerActs,
    ctx_gated: DeviceBuffer,
}

/// The shared expert's per-call scratch (see `Qwen35::shared_expert_steps`),
/// kept by the training branch for backward and dropped by every other.
struct SharedBufs {
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    mlp_out: DeviceBuffer,
    gate_logits: DeviceBuffer,
    gate_scalar: DeviceBuffer,
    scaled: DeviceBuffer,
}

/// Everything [`Qwen35::moe_sublayer`]'s training branch saves - universal
/// (every layer, both mixer types).
struct MoeLayerActs {
    xn2: DeviceBuffer,
    router_logits: DeviceBuffer,
    gate: DeviceBuffer,
    fe: DeviceBuffer,
    acts: MoeActs,
    // Note: `moe_acc`'s own VALUE is never read by backward (only its
    // gradient, `d_moe_out` itself - no `model::moe` backward primitive reads
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

/// Saved mixer activations for one layer's backward pass. `Gdn` is several
/// times wider than `Gqa` and one of these is kept per layer for the whole
/// step, so the wide variant is boxed rather than padding every `Gqa` layer
/// out to match it.
enum MixerActs {
    Gdn(Box<GdnLayerActs>),
    Gqa(GqaLayerActs),
}

struct LayerTrainActs {
    xn1: DeviceBuffer,
    mixer: MixerActs,
    xmid: DeviceBuffer,
    moe: MoeLayerActs,
}

/// The full backward activation cache for one `forward()` call on a
/// [`Qwen35::new_train`] instance - see [`Qwen35::train_acts`]'s own doc.
struct TrainActs {
    layers: Vec<LayerTrainActs>,
    xn_final: DeviceBuffer,
}

/// Per-layer fused expert-weight banks plus the shared scratch
/// [`model::moe::expert_fwd_grouped`] needs - built ONCE in
/// [`Qwen35::new_impl_on`] by concatenating the per-expert `blocks.{l}.mlp.
/// experts.{ei}.{gate,up,down}.weight` buffers [`Qwen35::moe_expert_names`]
/// already names (a host round-trip, done once at construction, not per
/// forward call: this crate's import keeps the per-expert fan-out layout
/// `model::moe::expert_fwd`/[`Qwen35::moe_sublayer_decode_sparse`] both still
/// need, unlike `crates/deepseek2`, which imports the fused bank straight off
/// the GGUF `ffn_*_exps` tensor and so needs no such copy - see the
/// `Qwen35::moe_grouped` field's own doc for exactly which instances get
/// one). `banks[l]` is `Some` only for a layer this shard owns
/// ([`Shard::owns`]); `scratch` is sized for `shape.rows = b*t` - the only
/// row count [`Qwen35::run_forward`] ever drives `moe_sublayer` with for
/// `n > 1` (decode's `n==1` step never reaches the grouped branch).
struct MoeGrouped {
    banks: Vec<Option<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>>,
    scratch: GroupedExpertScratch,
}

/// Qwen3.5-35B-A3B hybrid decoder - forward/inference only (see module doc).
pub struct Qwen35 {
    pub gpu: Gpu,
    pub cfg: Qwen35Config,
    /// Pipeline shard this instance owns (whole model, `embed && head`, by
    /// default - see [`Shard::whole`]). Layer indices stay ABSOLUTE
    /// throughout (`res`/`ps`/weight names are all indexed/named by the real
    /// `0..cfg.n_layers` layer number); only the forward/backward loop bounds
    /// and the embed/head gates are shard-relative. Mirrors `qwen3::Qwen`'s
    /// own `shard` field exactly.
    pub shard: Shard,
    ps: ParamStore,
    /// `Some` selects the int8 (DP4A) inference tier for the MoE experts'
    /// linears (`ps` excludes those names entirely - see [`Qwen35::
    /// new_impl_on`]'s role filter); `None` is the plain fp32 path, which is
    /// what an int8-requested build ALSO gets when the device lacks
    /// `caps.numeric.int8_dot` (`new_impl_on`'s `i8_on` gate) - see
    /// [`Qwen35::moe_int8_active`]. The 9
    /// GDN/GQA mixer linears `Qwen35Q8::is_i8_linear` ALSO names live on
    /// `self.ops`/`self.weights` below instead - `model::ops::Ops::moe_linear`
    /// has a documented, different buffer/param shape for I8/Q4 than
    /// `model::moe::expert_fwd_i8` uses, so the MoE-expert quantization
    /// stays on its own existing, already-correct path rather than being
    /// force-unified with the façade. See `crate::q8`'s module doc for the
    /// current (narrower) scope.
    q8: Option<Qwen35Q8>,
    /// `[layer][expert] -> (gate.weight, up.weight, down.weight)` name
    /// triples, built ONCE here rather than `format!`ed by [`Self::
    /// moe_sublayer`]/[`Self::moe_sublayer_bwd`]'s per-expert dispatch loops
    /// on every forward/backward pass. Those loops run `n_experts` (256 at
    /// the real 35B-A3B scale) times per layer per pass; re-formatting the
    /// SAME three strings that many times, only to immediately hash them
    /// again in [`Self::w`]/[`Self::g`], was pure host-side allocation with
    /// no behavioural purpose since a weight's name never changes after
    /// construction.
    moe_expert_names: Vec<Vec<(String, String, String)>>,
    /// Prefill/inference (`rows > 1`, fp32, non-training) grouped-GEMM MoE
    /// infra - `Some` on exactly the instances [`Self::moe_sublayer`]'s
    /// grouped branch runs on (`!is_train && q8.is_none()`, the SAME
    /// condition [`Self::new_impl_on`] builds this under), `None` for a
    /// training or int8 build, which keep the per-expert dense/int8 loops
    /// unchanged. See [`MoeGrouped`]'s own doc.
    moe_grouped: Option<MoeGrouped>,
    /// The `model::ops` façade - see [`Qwen35::ops_linear`]'s own doc.
    ops: Ops,
    /// Per-layer GDN/GQA mixer linears (the 9 leaves `Qwen35Q8::
    /// is_i8_linear` names, excluding MoE experts) as `model::ops::Weight` -
    /// `F32` unless this instance was built int8 AND the device caps support
    /// the DP4A path, in which case `Weight::upload` promotes it to
    /// `Weight::I8` (see [`Qwen35::ops_linear`]).
    weights: HashMap<String, Weight>,
    b: u32,
    t: u32,
    /// The GDN chunk size this instance was built for - see [`gdn_chunk_size`].
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
    /// `res[n_layers]` = input to the final norm) - the SSA activation-cache
    /// convention `crates/glm/src/model.rs` uses, kept here even though
    /// nothing backprops through it (useful for parity debugging: any layer's
    /// residual output is independently readable).
    res: Vec<DeviceBuffer>,

    /// All-ones buffer of width `linear_key_head_dim`, bound as `l2norm_scale
    /// .wgsl`'s per-dim scale so its learnably-scaled L2-norm computes the
    /// reference's bare `l2norm(x)` (GDN's q/k norm has no learnable scale).
    ones_khd: DeviceBuffer,
    /// M-RoPE `cos`/`sin` tables (`qwen3vl::mrope::mrope_tables`), built once
    /// at construction for the fixed `(b,t)` this instance decodes: text-only,
    /// so every axis carries the same plain sequential position per sequence
    /// (`qwen3vl::mrope`'s own tests prove this collapses exactly to ordinary
    /// half-split RoPE).
    cos: DeviceBuffer,
    sin: DeviceBuffer,

    /// Vision-language embedding splice (off = `None`). When set to `(row0,
    /// n_rows)`, `run_forward` overwrites residual rows `[row0, row0+n_rows)`
    /// with `img_embeds` (written by the vision front-end via
    /// [`Qwen35::write_img_embeds`]) right after the token-embedding gather,
    /// and - on a `new_train` build - `backward` routes those rows' gradient
    /// into `d_img_embeds` (read via [`Qwen35::read_d_img_embeds`]) instead of
    /// `tok.weight`. Mirrors `qwen3::Qwen`'s own seam exactly, except no
    /// fwd/bwd step-list rebuild is needed: `run_forward`/`backward` already
    /// build their step lists fresh on every call (see this module's own doc).
    mm_splice: Cell<Option<(u32, u32)>>,
    img_embeds: DeviceBuffer,
    d_img_embeds: DeviceBuffer,

    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,

    /// Backward's activation cache - `Some` only right after a `forward()`
    /// call on a [`Self::new_train`] instance (populated by `run_forward`'s
    /// train branch; read by `backward()`). This mirrors the engine-wide
    /// "forward reallocates fresh buffers every call" convention this file
    /// already uses everywhere else, so `backward()` MUST run against the
    /// same `forward()` call whose gradient it computes - exactly the
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
    /// Per-layer plain (non-paged) KV cache for GQA layers, `[dec_cap,
    /// kv_dim]`; a size-1 dummy at GDN layer indices (never dispatched into,
    /// mirrors `qwen3::Qwen::new_impl`'s own `dummy_layer`/`hd_or_dummy`
    /// convention for "this slot doesn't apply to this layer type" rather
    /// than an `Option`, so every layer index still has a plain buffer to
    /// index by `l`).
    gqa_kv: Vec<KvLayer>,
    /// This instance's [`CHUNK_ARENA_MIN_ROWS`] - see
    /// [`Self::set_chunk_arena_min_rows`].
    chunk_arena_min_rows: Cell<u32>,
    /// Whether a decode step takes the fused native kernels where the device is
    /// offered them - see [`Self::set_decode_fusion`].
    decode_fusion: Cell<bool>,
    /// Rows from which the int8 expert GEMMs run grouped - see
    /// [`Self::set_moe_grouped_min_rows`].
    moe_grouped_min_rows: Cell<u32>,
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
    // `k_proj`/`v_proj`/`o_proj`) - mirrors `qwen3::Qwen`'s own
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
/// What one BATCHED DECODE step hands [`Qwen35::layer_gqa_fwd`]. `paged`/
/// `cos`/`sin` are built ONCE per step and shared unchanged by every GQA layer;
/// the layer's KV pool is per layer, so it is rebound each time round the layer
/// loop. The `qwen35moe` twin of `qwen35::model::GqaDecodeCtx` - both are thin
/// holders for the same shared `model::gqa_mixer::PagedDecodeBatch`.
pub(crate) struct GqaDecodeCtx<'a> {
    pub paged: &'a model::gqa_mixer::PagedDecodeBatch<'a>,
    /// This layer's K and V pool, in the pool's [`KvTier`].
    pub layer: &'a KvLayer,
    /// `[bsz, rotary_dim/2]` M-RoPE tables - row `b` is sequence `b`'s OWN
    /// decode position, and the rows of one batch are unrelated positions.
    pub cos: &'a DeviceBuffer,
    pub sin: &'a DeviceBuffer,
}

/// What one CHUNKED-prefill round hands [`Qwen35::layer_gqa_fwd`] so its GQA
/// layers attend the sequence's persistent KV cache instead of an isolated
/// `[T,T]` causal block over the round alone. Every field is built ONCE per
/// round by [`Qwen35::run_prefill_chunk_stage`] and shared unchanged by every GQA
/// layer in it (`layer` excepted - that is per layer). The twin of
/// `qwen35::model::GqaChunkCtx`.
pub(crate) struct GqaChunkCtx<'a> {
    /// Where this sequence's KV rows START in the bound pool: `0` for a flat
    /// dedicated cache, `phys * cap` for a window of a shared pool
    /// ([`DecodeCaches::gqa_base_row`]).
    pub base_row: u32,
    /// Absolute position of this round's FIRST token (`0` on round 1).
    pub start: u32,
    /// The per-sequence KV cache row capacity ([`DecodeCaches::gqa_cap`]).
    pub cap: u32,
    pub layer: &'a KvLayer,
    /// `[n]` u32, every entry `base_row / cap` - the single-block table this
    /// sequence's KV window is.
    pub block_ids: DeviceBuffer,
    /// `[n]` u32 with `offsets[i] == start+i`: the row each chunk token's K/V
    /// is appended at in that block.
    pub offsets: DeviceBuffer,
    /// `[n]` u32 with `seq_lens[i] == start+i+1`: this round's causal mask.
    pub seq_lens: DeviceBuffer,
    /// This round's own `[n, rotary_dim/2]` M-RoPE tables, for absolute
    /// positions `start..start+n`.
    pub cos: &'a DeviceBuffer,
    pub sin: &'a DeviceBuffer,
}

/// Which cached-KV shape a [`Qwen35::layer_gqa_fwd`] call is running in. `None`
/// is the isolated `[T,T]` causal block a training/whole-sequence forward wants;
/// the two `Some` arms are the two ways a SERVING pass attends a persistent
/// cache - one sequence's many tokens ([`GqaChunkCtx`]) or many sequences' one
/// token each ([`GqaDecodeCtx`]).
pub(crate) enum GqaCached<'a> {
    Chunk(&'a GqaChunkCtx<'a>),
    Decode(&'a GqaDecodeCtx<'a>),
}

/// What a fused front end ([`Qwen35::rms_quant_front`]) hands the next linear: the
/// normalised rows and their int8 packing, produced by one launch.
pub(crate) struct Front {
    xn: DeviceBuffer,
    xq: DeviceBuffer,
    sx: DeviceBuffer,
}

/// A decode step's final-normed `[rows, d_model]` hidden block and, when a fused
/// front already packed it, the int8 activation the int8 head reads.
pub(crate) struct Hidden {
    pub(crate) xn: DeviceBuffer,
    act: Option<Act>,
}

impl Hidden {
    /// A hidden block nothing has quantised.
    pub(crate) fn plain(xn: DeviceBuffer) -> Hidden {
        Hidden { xn, act: None }
    }
}

/// Which recurrent-state shape a [`Qwen35::layer_gdn_fwd`] call is running in:
/// the Gated-DeltaNet counterpart of [`GqaCached`] - a whole-sequence forward
/// that starts from zero state, one sequence's prefill round continuing its own
/// state, or one decode token for each of several independent sequences.
pub(crate) enum GdnCall<'a> {
    Whole,
    Chunk(model::gdn_mixer::GdnStream<'a>),
    Decode(model::gdn_mixer::GdnDecodeState<'a>),
}

/// The device buffers a decode step reads for everything that varies from one
/// token to the next - see `Qwen35::alloc_decode_meta`.
pub(crate) struct DecodeMeta {
    tokens: DeviceBuffer,
    /// `[bsz, rotary_dim/2]` M-RoPE tables - row `b` is sequence `b`'s OWN
    /// decode position.
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    blocks: DeviceBuffer,
    offsets: DeviceBuffer,
    block_tables: DeviceBuffer,
    seq_lens: DeviceBuffer,
}

/// What a recorded decode step's head produces.
pub(crate) enum DecodeHead {
    /// The greedy token of every row.
    Greedy,
    /// Every row's top-`cap` (token, logit) candidates.
    TopK(u32),
}

enum TapeOut {
    Greedy(DeviceBuffer),
    TopK { vals: DeviceBuffer, idx: DeviceBuffer, cap: u32 },
}

/// One decode step recorded whole (`Qwen35::record_decode`): the dispatches, the
/// buffers they bind and the head's output buffer. Holds its working set alive.
pub(crate) struct DecodeTape {
    tape: gpu_core::tape::Tape,
    meta: DecodeMeta,
    bsz: u32,
    out: TapeOut,
}

impl DecodeTape {
    /// The recording, for profiling it.
    pub(crate) fn tape(&self) -> &gpu_core::tape::Tape {
        &self.tape
    }
}

/// What a replayed decode step read back.
pub(crate) enum DecodeOut {
    Greedy(Vec<u32>),
    TopK(Vec<Vec<(u32, f32)>>),
}

/// One sequence's coordinates in a [`Qwen35::run_decode_batch`] call: where its
/// KV history lives, where this step's token goes, and its own Gated-DeltaNet
/// state. The batched counterpart of the per-sequence half of
/// [`DecodeCaches`]: the GQA POOL is shared by the whole batch
/// ([`BatchDecodeCaches`]), this is everything that is not.
pub(crate) struct BatchSeq {
    /// The physical block backing this sequence's whole KV history, so its rows
    /// are `phys*gqa_cap .. +gqa_cap` of every layer's pool.
    pub phys: u32,
    /// Absolute decode position of this step's token for this sequence. The
    /// rows of one batch are unrelated positions.
    pub pos: u32,
}

/// One sequence's per-layer recurrent state / conv history, indexed by ABSOLUTE
/// layer index with a dummy at full-attention indices - [`DecodeCaches`]'s own
/// convention, and the same buffers.
pub(crate) struct GdnBufs<'a> {
    pub state: &'a [DeviceBuffer],
    pub hist: &'a [DeviceBuffer],
}

/// Where a batched decode step finds its sequences' Gated-DeltaNet state.
pub(crate) enum GdnStore<'a> {
    /// One buffer set per batch row, in row order: an instance's own decode state.
    PerSeq(&'a [GdnBufs<'a>]),
    /// Per-layer pools holding one row per resident sequence, picked by each
    /// sequence's physical block ([`BatchSeq::phys`]): a whole batch is staged by
    /// one dispatch in and one out per layer, however many sequences it has.
    Pool { state: &'a [DeviceBuffer], hist: &'a [DeviceBuffer] },
}

/// What one [`Qwen35::run_decode_batch`] call reads and updates: the shared
/// per-layer paged KV pool plus one [`BatchSeq`] per sequence.
pub(crate) struct BatchDecodeCaches<'a> {
    /// Per-layer `[num_blocks*gqa_cap, kv_dim]` KV pool for the full-attention
    /// layers (a placeholder at GDN indices), each layer's K and V planes in
    /// the pool's [`KvTier`]. A sequence's own `[gqa_cap, kv_dim]` window is
    /// rows `phys*gqa_cap..`; a caller with one dedicated cache per sequence
    /// is the `num_blocks = 1`, `phys = 0` case.
    pub gqa_kv: &'a [KvLayer],
    /// Pool rows one physical block spans - the per-sequence KV capacity, and
    /// the paged kernels' `block_size`.
    pub gqa_cap: u32,
    /// One entry per batch row, in the same order as the `tokens` argument.
    pub seqs: &'a [BatchSeq],
    pub gdn: GdnStore<'a>,
}

pub(crate) struct DecodeCaches<'a> {
    /// Per-layer `[cap, kv_dim]` KV cache for GQA layers (placeholder at GDN
    /// indices), each layer's K and V planes in the cache's [`KvTier`].
    pub gqa_kv: &'a [KvLayer],
    /// Cache row capacity, shared by every GQA layer's cache in this call
    /// (one per-sequence capacity, not a per-layer one).
    pub gqa_cap: u32,
    /// Where this sequence's `gqa_cap` rows START in the bound buffer. `0` -
    /// the only value every caller but a paged serving engine ever uses --
    /// means the buffers above are this sequence's own dedicated flat cache. A
    /// serving engine that keeps ONE pool per layer instead
    /// (`crate::serve::Engine`) binds the whole pool and sets this to
    /// `physical_block * gqa_cap`; it must be a whole multiple of `gqa_cap`,
    /// since it is also what the paged attention kernels' single-entry block
    /// table is derived from.
    pub gqa_base_row: u32,
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

/// The untied output projection's parameter name.
const UNTIED_HEAD: &str = "lm_head.weight";

impl Qwen35 {
    pub fn new(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(Gpu::new(pipelines()), cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), false, shard)
    }

    /// Build on an existing device handle (test fixtures share one `Gpu` per
    /// binary - see `gpu_core::testgpu`).
    pub fn new_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), false, shard)
    }

    /// [`Self::new`] with the int8 (DP4A) inference tier: the attention/GDN
    /// mixer projections and every routed expert's gate/up/down are
    /// quantized (`crate::q8::Qwen35Q8::is_i8_linear`); the router, shared
    /// expert, embeddings and norms stay fp32. See `crate::q8`'s module doc
    /// for the full rationale. Capability-gated, not assumed: on a device
    /// whose caps report no `int8_dot` (e.g. the CPU JIT), this silently
    /// falls back to the fp32 path instead (a message is printed, never a
    /// silent wrong-result dispatch) - see [`Self::moe_int8_active`].
    /// Inference-only, same as the fp32 path
    /// (`Qwen35::backward` panics regardless).
    pub fn new_i8(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(Gpu::new(pipelines()), cfg, b, t, init, &TierPolicy::uniform(Dtype::I8), false, shard)
    }

    /// [`Self::new_i8`] on an existing device handle - see [`Self::new_on`].
    pub fn new_on_i8(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::I8), false, shard)
    }

    /// [`Self::new_on_i8`] streaming straight from a [`checkpoint::TensorSource`]
    /// (a GGUF through `crate::gguf_load::source`): no host-side fp32 copy of
    /// the model ever exists, one tensor is decoded at a time.
    pub fn new_on_i8_src(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, src, &TierPolicy::uniform(Dtype::I8), false, shard)
    }

    /// [`Self::new_on_i8_src`] at fp32: every weight a float buffer, streamed
    /// from the source one tensor at a time. The reference tier a real-weight
    /// comparison is made against (only a truncated model fits a card in fp32).
    pub fn new_on_src(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, src, &TierPolicy::uniform(Dtype::F32), false, shard)
    }

    /// [`Self::new_on_i8_src`] at a per-leaf [`TierPolicy`]: `F32` and `I8` are
    /// the tiers this model implements (any other is refused by name). The
    /// leaves are matched by substring, so `"self_attn"` / `"out_proj"` /
    /// `"mlp.experts"` select an attention block, one projection or the routed
    /// experts - e.g. `uniform(I8).with(&["linear_attn", "self_attn"], F32)` is
    /// int8 experts under fp32 mixers.
    pub fn new_on_tier_src(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, tier: &TierPolicy) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, src, tier, false, shard)
    }

    /// Build a TRAINABLE model: every weight `Role::Trainable` (full-parameter
    /// backward - no LoRA-specific plumbing here, per this task's scope note),
    /// `forward()` additionally saves the activation cache `backward()` reads.
    /// int8 and training are mutually exclusive (mirrors `qwen3::Qwen`'s own
    /// `assert!(!(i8 && train))`).
    pub fn new_train(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(Gpu::new(pipelines()), cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), true, shard)
    }

    /// [`Self::new_train`] on an existing device handle - see [`Self::new_on`].
    pub fn new_train_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), true, shard)
    }

    /// Build a single pipeline **stage**: only the layers (and endpoint
    /// weights) in `shard` are allocated on this device, as a TRAINABLE
    /// build (`Role::Trainable` full-parameter, or - when `cfg.lora` is
    /// `Some` - frozen base + trainable LoRA adapters). `shard.gpu_index`
    /// names the canonical physical card (device registry); `Shard::ANY_GPU`
    /// keeps the ambient selection. Mirrors `qwen3::Qwen::new_shard` exactly
    /// (see `crate::shard`'s [`model::Shardable`] impl, the only caller this
    /// is meant for outside tests).
    pub fn new_shard(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> Qwen35 {
        let gpu = if shard.gpu_index == Shard::ANY_GPU {
            Gpu::new(pipelines())
        } else {
            Gpu::new_on_index(shard.gpu_index as u32, pipelines()).unwrap_or_else(|e| panic!("qwen35 shard placement: {e}"))
        };
        Qwen35::new_impl_on(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), true, shard)
    }

    fn new_impl_on(
        gpu: Gpu,
        cfg: Qwen35Config,
        b: u32,
        t: u32,
        src: &dyn checkpoint::TensorSource,
        tier: &TierPolicy,
        train: bool,
        shard: Shard,
    ) -> Qwen35 {
        let i8 = tier.quantizes_anything();
        assert!(!(i8 && train), "qwen35: int8 path is inference-only (Qwen35::new_train is fp32-only)");
        // Int8 weights are capability-driven, never assumed: the request only
        // takes effect where the packed-dot GEMM executes (the `Op::
        // MoeExpertLinear`/`Op::MatMul` selector's `PackedInt8` gate).
        // Elsewhere - the CPU JIT - fp32 weights stay, and the fallback is
        // said out loud rather than silently absorbed. Mirrors `qwen3::
        // serve::Engine::from_map_with_gpu`'s `weights_int8`/`w8_on` pattern
        // exactly; gates the `q8` (MoE-expert) build, the mixer-linear
        // upload closure below, and the `ParamStore` role-exclusion filter
        // alike, so all three agree on whether the int8 tier is actually
        // reachable on this device.
        let i8_on = i8 && gpu.caps().numeric.int8_dot;
        if i8 && !i8_on {
            eprintln!("qwen35moe: int8 weights requested but this device has no packed-int8 path; using fp32 weights");
        }
        // Which quantizable linear the per-leaf policy puts at int8 on this
        // device. F32 and I8 are the tiers this model implements; anything
        // else is refused by name rather than silently run at another tier.
        let quant = |name: &str| {
            i8_on
                && (Qwen35Q8::is_i8_linear(name) || name == UNTIED_HEAD)
                && match tier.want(name) {
                    Dtype::I8 => true,
                    Dtype::F32 => false,
                    other => panic!("qwen35moe: tier {other:?} is not implemented for {name} (F32 and I8 only)"),
                }
        };
        // The routed experts are one tier for the whole model (the policy is
        // asked about a representative expert).
        let experts_i8 = quant("blocks.0.mlp.experts.0.gate.weight");
        // The shared expert rides in the banks (as block `n_experts`) when it has
        // the routed experts' shape; the router and its shared gate then live in
        // `q8`'s `router_ext`, so none of the four needs an fp32 copy here.
        let shared_in_bank = experts_i8 && Qwen35Q8::shared_fits_bank(&cfg);
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
            .filter(|(n, _)| {
                !quant(n)
                    && !(shared_in_bank
                        && (Qwen35Q8::is_shared_expert_linear(n) || n.ends_with("mlp.router.weight") || n.ends_with("mlp.shared_expert_gate.weight")))
            })
            .map(|(n, c)| {
                let role = if !train {
                    Role::Frozen
                } else if cfg.lora.is_some() {
                    if model::adapter::device::is_adapter_param(&n) { Role::Trainable } else { Role::Frozen }
                } else {
                    Role::Trainable
                };
                (n, c, role)
            })
            .collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        // See the field's own doc: computed once here so the per-expert
        // forward/backward dispatch loops index into it instead of
        // `format!`ing the same names every pass.
        let moe_expert_names: Vec<Vec<(String, String, String)>> = (0..cfg.n_layers as usize)
            .map(|l| {
                (0..cfg.n_experts as usize)
                    .map(|ei| {
                        (
                            format!("blocks.{l}.mlp.experts.{ei}.gate.weight"),
                            format!("blocks.{l}.mlp.experts.{ei}.up.weight"),
                            format!("blocks.{l}.mlp.experts.{ei}.down.weight"),
                        )
                    })
                    .collect()
            })
            .collect();

        // Quantize+upload the int8 MoE-expert linears from the SAME source,
        // streaming one tensor at a time (see `Qwen35Q8::build`'s own doc -
        // MoE experts only; the mixer linears build `weights` below instead).
        let q8 = if experts_i8 { Some(Qwen35Q8::build(&gpu, src, &cfg, b * t, MAX_ABS_ROW, QUANT_PACK)) } else { None };

        // Prefill/inference grouped-GEMM MoE infra (M5.12) -- built for
        // exactly the instances `moe_sublayer`'s grouped branch runs on: not
        // training (that tape needs `MoeActs`-shaped per-expert saves for
        // `moe_sublayer_bwd`, which the grouped/permuted layout does not
        // produce -- `expert_fwd_grouped` has no backward yet, see its own
        // module doc) and not int8 (the packed-dot path has its own combine,
        // untouched by this change). See `MoeGrouped`'s own doc for why this
        // is a host round-trip over already-uploaded per-expert weights
        // rather than a change to this crate's (fan-out) import layout.
        let moe_grouped = if !train && q8.is_none() {
            let (d, moe_ff, e) = (cfg.d_model, cfg.moe_intermediate_size, cfg.n_experts);
            let per_expert = (moe_ff * d) as usize;
            let banks: Vec<Option<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>> = (0..cfg.n_layers as usize)
                .map(|l| {
                    if !shard.owns(l) {
                        return None;
                    }
                    let mut gate_bank = Vec::with_capacity(per_expert * e as usize);
                    let mut up_bank = Vec::with_capacity(per_expert * e as usize);
                    let mut down_bank = Vec::with_capacity(per_expert * e as usize);
                    for (gn, un, dn) in &moe_expert_names[l] {
                        gate_bank.extend(gpu.read(ps.w(gn), per_expert));
                        up_bank.extend(gpu.read(ps.w(un), per_expert));
                        down_bank.extend(gpu.read(ps.w(dn), per_expert));
                    }
                    let gate_buf = gpu.storage_init(&format!("blocks.{l}.mlp.experts.bank.gate"), &gate_bank);
                    let up_buf = gpu.storage_init(&format!("blocks.{l}.mlp.experts.bank.up"), &up_bank);
                    let down_buf = gpu.storage_init(&format!("blocks.{l}.mlp.experts.bank.down"), &down_bank);
                    Some((gate_buf, up_buf, down_buf))
                })
                .collect();
            let shape = MoeShape { rows: b * t, d_model: d, moe_ff, n_experts: e, top_k: cfg.top_k };
            Some(MoeGrouped { banks, scratch: GroupedExpertScratch::new(&gpu, &shape) })
        } else {
            None
        };

        // Per-layer GDN/GQA mixer linears: every layer this shard owns
        // gets its own leaves (GDN: in_proj_{qkv,z,b,a}/out_proj; GQA:
        // {q,k,v,o}_proj) as a `model::ops::Weight`, built ONCE here. `i8`
        // asks `Weight::upload` for `Dtype::I8`, streaming straight from
        // `src` (these names are excluded from `ps` above, exactly like the
        // MoE-expert linears); the `else` (fp32) arm wraps a `.clone()` of
        // the buffer `ps` already holds (a cheap `Arc` bump), so the common
        // non-i8 case costs no extra VRAM or re-upload. Mirrors `qwen3::
        // model::Qwen::new_impl`'s own B7 `weights` construction exactly.
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("qwen35moe: Ops::new: {e}"));
        let (d_u, conv_dim_u, vdim_u, nvh_u, hqp_u, hkv_u, hq_u) = (
            cfg.d_model as usize,
            cfg.linear_conv_dim() as usize,
            cfg.linear_value_dim() as usize,
            cfg.linear_num_value_heads as usize,
            cfg.q_proj_dim() as usize,
            cfg.kv_dim() as usize,
            cfg.q_dim() as usize,
        );
        let mut weights: HashMap<String, Weight> = HashMap::new();
        let mut upload = |name: String, wn: usize, wk: usize| {
            let w = if quant(&name) {
                let mut built: Option<Weight> = None;
                let found = src.with_tensor(&name, &mut |raw| {
                    built = Some(Weight::upload(&ops, raw, wn, wk, Dtype::I8));
                });
                if !found {
                    panic!("qwen35moe: missing init weight {name}");
                }
                built.unwrap()
            } else {
                Weight::F32 { w: ps.w(&name).clone(), n: wn as u32, k: wk as u32 }
            };
            weights.insert(name, w);
        };
        for (l, ty) in cfg.layer_types().iter().enumerate() {
            if !shard.owns(l) {
                continue;
            }
            match ty {
                LayerType::Linear => {
                    let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");
                    upload(p("in_proj_qkv.weight"), conv_dim_u, d_u);
                    upload(p("in_proj_z.weight"), vdim_u, d_u);
                    upload(p("in_proj_b.weight"), nvh_u, d_u);
                    upload(p("in_proj_a.weight"), nvh_u, d_u);
                    upload(p("out_proj.weight"), d_u, vdim_u);
                }
                LayerType::Full => {
                    let p = |s: &str| format!("blocks.{l}.self_attn.{s}");
                    upload(p("q_proj.weight"), hqp_u, d_u);
                    upload(p("k_proj.weight"), hkv_u, d_u);
                    upload(p("v_proj.weight"), hkv_u, d_u);
                    upload(p("o_proj.weight"), d_u, hq_u);
                }
            }
        }

        // The untied head: int8 under an int8 policy (the GGUF's own `Q8_0`
        // `output.weight`), read once per decode pass instead of 2 GB of fp32.
        if shard.head && cfg.head_weight() == UNTIED_HEAD && quant(UNTIED_HEAD) {
            upload(UNTIED_HEAD.to_string(), cfg.vocab as usize, cfg.d_model as usize);
        }

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
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
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
        let kv_dim = cfg.kv_dim() as u64;
        let mut gqa_kv = Vec::with_capacity(cfg.n_layers as usize);
        let mut gdn_state = Vec::with_capacity(cfg.n_layers as usize);
        let mut gdn_hist = Vec::with_capacity(cfg.n_layers as usize);
        let gdn_bh = cfg.linear_num_value_heads as u64;
        let gdn_state_len = gdn_bh * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
        let gdn_hist_len = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
        for ty in cfg.layer_types() {
            match ty {
                LayerType::Full => {
                    gqa_kv.push(KvLayer::new(&gpu, KvTier::F32, t as u64, kv_dim, cfg.head_dim as u64));
                    gdn_state.push(gpu.storage(1));
                    gdn_hist.push(gpu.storage(1));
                }
                LayerType::Linear => {
                    gqa_kv.push(KvLayer::placeholder(&gpu));
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
            moe_expert_names,
            moe_grouped,
            ops,
            weights,
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
            gqa_kv,
            chunk_arena_min_rows: Cell::new(CHUNK_ARENA_MIN_ROWS),
            moe_grouped_min_rows: Cell::new(MOE_GROUPED_MIN_ROWS),
            decode_fusion: Cell::new(true),
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
    /// dispatches must be skipped - only the input-gradient (dX) path runs to
    /// keep backprop flowing to lower-layer adapters. Mirrors
    /// `qwen3::model.rs`'s own `trainable` helper exactly.
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    /// The gradient buffer for a trainable weight - only valid on a
    /// [`Self::new_train`] instance (every weight is `Role::Trainable` there,
    /// see [`Self::new_impl_on`]'s role filter).
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }

    /// True if a LoRA adapter is configured for the given projection leaf
    /// (one of the 9 targetable leaf names - never an MoE expert leaf).
    /// Mirrors `qwen3::model.rs`'s own `lora_for` exactly.
    fn lora_for(&self, leaf: &str) -> Option<(u32, f32)> {
        self.cfg.lora.as_ref().filter(|lc| lc.targets_leaf(leaf)).map(|lc| (lc.rank, lc.alpha / lc.rank as f32))
    }

    /// Forward LoRA delta for a targeted linear: `y += (alpha/r)·(x·Aᵀ)·Bᵀ`.
    /// No-op for an untargeted leaf. `m`×`k` is the input, `nout` the output -
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

    /// Dispatch one of the 9 GDN/GQA mixer linears via `self.ops`/
    /// `self.weights`. `self.weights` holds whatever dtype `Weight::upload`
    /// picked for this model at construction (uniformly `F32` unless this
    /// model was built int8 AND the device's capability allowed it, in
    /// which case every one of the 9 mixer linears is `I8`) - the forward
    /// never branches on a separate int8-on/off flag itself, only on what
    /// `self.weights` actually holds. Returns whether the dispatch was
    /// `F32` (LoRA only ever targets an unquantized base weight, so a
    /// caller only runs `lora_fwd` when this is `true`). Mirrors
    /// `qwen3::model::Qwen::ops_linear` exactly.
    fn ops_linear(&self, s: &mut Vec<Step>, act: &Act, wname: &str, out: &DeviceBuffer) -> bool {
        let w = self.weights.get(wname).unwrap_or_else(|| panic!("qwen35moe: no Ops weight for {wname}"));
        self.ops.matmul(s, w, act, out, 0);
        matches!(w, Weight::F32 { .. })
    }

    /// [`Self::ops_linear`] for several linears that read the SAME activation,
    /// each into its own buffer: a GDN layer's qkv/b/a/z, a GQA layer's q/k/v.
    /// [`Ops::matmul_group`] makes the group one launch where the device offers it
    /// and an int8 GEMV serves every member, and a matmul per member otherwise
    /// (always, with [`Self::set_decode_fusion`] off). Returns, per member, whether
    /// it was an fp32 dispatch - the one case LoRA applies to, as `ops_linear`
    /// reports it.
    fn ops_linear_group(&self, s: &mut Vec<Step>, act: &Act, group: &[(&str, &DeviceBuffer)]) -> Vec<bool> {
        let members: Vec<(&Weight, &DeviceBuffer)> =
            group.iter().map(|(name, out)| (self.weights.get(*name).unwrap_or_else(|| panic!("qwen35moe: no Ops weight for {name}")), *out)).collect();
        if self.decode_fusion.get() {
            self.ops.matmul_group(s, &members, act);
        } else {
            for (w, out) in &members {
                self.ops.matmul(s, w, act, out, 0);
            }
        }
        members.iter().map(|(w, _)| matches!(w, Weight::F32 { .. })).collect()
    }

    /// Whether a decode step of `rows` rows may use the fused front end
    /// ([`Self::rms_quant_front`]): the device is offered `add_rms_quant`, the
    /// experts' int8 path (which reads the quantised activation) is the model's,
    /// the shape is one the kernel serves, and the unfused chain would have
    /// normalised with the cooperative RMSNorm - the only order the kernel
    /// reproduces. The last clause is what makes the fusion invisible rather than
    /// merely close.
    fn fused_front_ok(&self, rows: u32) -> bool {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let coop = block::rms_variant(g, RMSNORM, Some(RMSNORM_ROWS), rows, d).0 == RMSNORM_ROWS;
        self.decode_fusion.get()
            && coop
            && self.q8.is_some()
            && g.has_fused(gpu_core::Fused::AddRmsQuant)
            && gpu_core::Fused::AddRmsQuant.serves(&[d, rows, f(self.cfg.rms_eps), 1])
    }

    /// Whether a decode step of `rows` rows may use [`Self::quant_epilogue`] at
    /// every width the layers hand it.
    fn fused_epilogue_ok(&self, rows: u32) -> bool {
        let g = &self.gpu;
        let c = &self.cfg;
        let widths = [c.linear_value_dim(), c.q_dim()];
        self.decode_fusion.get()
            && self.q8.is_some()
            && g.has_fused(gpu_core::Fused::QuantEpilogue)
            && widths.iter().all(|&k| gpu_core::Fused::QuantEpilogue.serves(&[k, rows, 0, 0]))
    }

    /// One launch that produces the activation an int8 linear reads AND quantises
    /// it: `mode` 0 quantises `a` as it is, 1 produces `silu(a) * b`, 2 produces
    /// `a * sigmoid(b)`. Returns `(product, act)`; `product` is `None` for mode 0,
    /// where the activation is `a` itself. Only call after
    /// [`Self::fused_epilogue_ok`].
    fn quant_epilogue(&self, mode: u32, a: &DeviceBuffer, b: Option<&DeviceBuffer>, rows: u32, k: u32) -> (Option<DeviceBuffer>, Act) {
        let g = &self.gpu;
        let n = (rows * k) as u64;
        let y = (mode != 0).then(|| g.storage(n));
        let xq = g.storage(n / 4);
        let sx = g.storage(rows as u64);
        // Mode 0 reads no `b` and writes no `y`; the slots are still bound, and
        // alias what is already there.
        let step = g
            .fused_step(gpu_core::Fused::QuantEpilogue, &[a, b.unwrap_or(a), y.as_ref().unwrap_or(&xq), &xq, &sx], &[k, rows, mode, 0])
            .expect("fused_epilogue_ok said the device serves this shape");
        g.submit(&[], &[step]);
        let act = self.ops.act_prequantized(y.as_ref().unwrap_or(a), &sx, &xq, rows, k);
        (y, act)
    }

    /// One launch of the fused front end of an int8 linear: `sum = a + b` (when
    /// `b` is given), `xn = rmsnorm(sum) * w`, and the per-row int8 packing of
    /// `xn` as a ready [`Act`]. Returns `(sum, front)`; `sum` is `None` when
    /// there was no `b`, and the caller keeps `a` as the residual. Only call after
    /// [`Self::fused_front_ok`].
    fn rms_quant_front(&self, a: &DeviceBuffer, b: Option<&DeviceBuffer>, w: &DeviceBuffer, rows: u32) -> (Option<DeviceBuffer>, Front) {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let n = (rows * d) as u64;
        let sum = b.map(|_| g.storage(n));
        let xn = g.storage(n);
        let xq = g.storage(n / 4);
        let sx = g.storage(rows as u64);
        // Without an addend the kernel reads neither `b` nor writes `sum`; the
        // slots still have to be bound, so they alias what is already there.
        let step = g
            .fused_step(
                gpu_core::Fused::AddRmsQuant,
                &[a, b.unwrap_or(a), w, sum.as_ref().unwrap_or(&xn), &xn, &xq, &sx],
                &[d, rows, f(self.cfg.rms_eps), b.is_some() as u32],
            )
            .expect("fused_front_ok said the device serves this shape");
        g.submit(&[], &[step]);
        (sum, Front { xn, xq, sx })
    }

    /// The [`Act`] of a [`Front`]'s packed rows, for the linears that read them.
    fn front_act(&self, front: &Front, rows: u32) -> Act {
        self.ops.act_prequantized(&front.xn, &front.sx, &front.xq, rows, self.cfg.d_model)
    }

    /// Backward for a (possibly-LoRA) linear `y = x·Wᵀ`. Accumulates the input
    /// gradient into `dx` (flag `acc`). For a full weight: `dW += d_outᵀ·x`
    /// (skipped when `wname` is Frozen - a LoRA-mode base, or an untargeted
    /// weight under a LoRA build, e.g. `mlp.router.weight`), `dx = d_out·W`.
    /// For a LoRA-targeted leaf: the base weight is always frozen (dX only, no
    /// dW), and the adapter grads `gA`/`gB` are produced (scale folded into
    /// the private `lora_a`/`lora_da` scratch) - naive `matmul_dx`/`matmul_dw`
    /// only, no tiled-GEMM selection, matching a correctness-first tiny
    /// gradcheck config. Mirrors
    /// `qwen3::model.rs`'s own `proj_bwd` exactly.
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, steps: &mut Vec<Step>, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        let g = &self.gpu;
        match self.lora_for(leaf) {
            Some((r, scale)) => {
                // base: dx += d_out·W (frozen weight - no dW).
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
    /// only when the gain is trainable (frozen under a LoRA build - no norm
    /// gain is ever a LoRA target, so this mirrors `qwen3::Qwen`'s own
    /// LoRA-base-frozen branch, applied here to every norm rather than to a
    /// projection weight).
    fn rmsnorm_bwd_step(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        let inv = self.gpu.storage(rows as u64);
        let gw = self.trainable(wname).then(|| self.g(wname));
        steps.extend(rmsnorm_bwd(&self.gpu, &kernel_ids(), x, self.w(wname), dy, dx, &inv, gw, dim, rows, self.cfg.rms_eps));
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
    /// with the image tokens written via [`Self::write_img_embeds`], and - on
    /// a `new_train` build - `backward` routes their gradient to
    /// [`Self::read_d_img_embeds`] (zeroing them in the residual grad first so
    /// `EMB_BWD` never trains the image-placeholder token id). Unlike
    /// `qwen3::Qwen::enable_mm_splice`, this needs no fwd/bwd step-list
    /// rebuild: `run_forward`/`backward` already build their step lists fresh
    /// on every call (see this module's top-of-file doc), so enabling the
    /// splice is pure buffer allocation + a flag - call once after
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

    /// Read the gradient of the spliced image embeddings after `backward()` -
    /// feeds the vision tower/connector backward. Requires a `new_train` build
    /// (see [`Self::backward`]'s splice-gradient step).
    pub fn read_d_img_embeds(&self) -> Vec<f32> {
        self.gpu.read(&self.d_img_embeds, self.img_numel())
    }

    /// The splice INPUT buffer itself, for a vision tower sharing THIS
    /// decoder's [`Gpu`] - write into it with a `Step` and the embedding never
    /// leaves the device. [`Self::write_img_embeds`] is the cross-device path
    /// (`crate::vl::Qwen35Vl` runs its tower on a second, possibly CPU-backed
    /// device and must round trip through the host); this accessor is purely
    /// additive and changes nothing about that. Valid only after
    /// [`Self::enable_mm_splice`] - before that this is the 1-float
    /// placeholder the constructor allocates.
    pub fn img_embeds_buf(&self) -> &DeviceBuffer {
        &self.img_embeds
    }

    /// The splice GRADIENT buffer itself - the device-side counterpart of
    /// [`Self::read_d_img_embeds`], so a same-device vision tower's backward
    /// can consume it as an input buffer instead of re-uploading a host `Vec`.
    /// Same validity rule as [`Self::img_embeds_buf`].
    pub fn d_img_embeds_buf(&self) -> &DeviceBuffer {
        &self.d_img_embeds
    }

    /// Overwrite the M-RoPE `cos`/`sin` tables (`[b·t, rotary_dim/2]` row-major -
    /// see the `cos`/`sin` fields' own doc, and
    /// `qwen3vl::mrope::{get_rope_index, mrope_tables}` for how to build them
    /// from real 2-D image-grid positions) for the next `forward()`. RoPE here
    /// is unconditionally table-driven already (no `enable_mrope` gating
    /// needed, unlike `qwen3::Qwen`) - this simply replaces the plain-
    /// sequential-position table built at construction.
    pub fn write_mrope_tables(&self, cos: &[f32], sin: &[f32]) {
        self.gpu.write_f32(&self.cos, cos);
        self.gpu.write_f32(&self.sin, sin);
    }

    // ---- one Gated DeltaNet (Linear) layer --------------------------------

    fn layer_gdn_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32, call: GdnCall) -> (DeviceBuffer, Option<GdnLayerActs>) {
        self.layer_gdn_fwd_pre(l, xn1, None, false, n, call)
    }

    /// [`Self::layer_gdn_fwd`] given `xn1`'s activation already quantised (`pre`),
    /// by a fused kernel that produced `xn1` and its int8 packing in one launch.
    /// `None` quantises here, as every other caller does. `epilogue` fuses the
    /// quantisation of what the layer hands `out_proj` into one launch
    /// ([`Self::quant_epilogue`]); only a decode step, after
    /// [`Self::fused_epilogue_ok`], sets it.
    fn layer_gdn_fwd_pre(&self, l: usize, xn1: &DeviceBuffer, pre: Option<Act>, epilogue: bool, n: u32, call: GdnCall) -> (DeviceBuffer, Option<GdnLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let conv_dim = c.linear_conv_dim();
        let value_dim = c.linear_value_dim();
        let nvh = c.linear_num_value_heads;
        let khd = c.linear_key_head_dim;
        let vhd = c.linear_value_head_dim;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // in_proj_qkv/b/a/z all read one activation (`xn1` quantized once): one
        // group, which is one launch where the device allows it. LoRA/int8 dispatch
        // stays local - see `model::gdn_mixer`'s own module doc.
        let mixed_qkv = g.storage((n * conv_dim) as u64);
        let bproj = g.storage((n * nvh) as u64);
        let aproj = g.storage((n * nvh) as u64);
        let z = g.storage((n * value_dim) as u64);
        let mut s1 = Vec::new();
        let act1 = match pre {
            Some(a) => a,
            None => self.ops.act(&mut s1, xn1, 0, n, d),
        };
        let (qkv_w, b_w, a_w, z_w) = (p("in_proj_qkv.weight"), p("in_proj_b.weight"), p("in_proj_a.weight"), p("in_proj_z.weight"));
        let f32s = self.ops_linear_group(&mut s1, &act1, &[(&qkv_w, &mixed_qkv), (&b_w, &bproj), (&a_w, &aproj), (&z_w, &z)]);
        if f32s[0] {
            self.lora_fwd(&mut s1, "in_proj_qkv", xn1, &qkv_w, &mixed_qkv, n, d, conv_dim);
        }
        if f32s[1] {
            self.lora_fwd(&mut s1, "in_proj_b", xn1, &b_w, &bproj, n, d, nvh);
        }
        if f32s[2] {
            self.lora_fwd(&mut s1, "in_proj_a", xn1, &a_w, &aproj, n, d, nvh);
        }
        if f32s[3] {
            self.lora_fwd(&mut s1, "in_proj_z", xn1, &z_w, &z, n, d, value_dim);
        }
        g.submit(&[], &s1);

        // conv+split+l2norm+decay-gate+recurrence+gated-norm - LoRA/dtype-
        // agnostic, shared with `crates/qwen35` (`model::gdn_mixer`). A
        // batched DECODE step is `n` DIFFERENT sequences' one token each, each
        // continuing its own recurrent state and conv window; the
        // whole-sequence forward keeps this instance's shape and starts from
        // zero state.
        let gdn = match &call {
            GdnCall::Whole => GdnShape { b: self.b, h: nvh, t: self.t, dk: khd, dv: vhd, chunk: self.chunk },
            // A chunked round is ONE sequence's `n` rows, with its own chunk
            // size (`n` is a round's length, unrelated to this instance's
            // construction-time `t`).
            GdnCall::Chunk(_) => GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: gdn_chunk_size(n) },
            GdnCall::Decode(_) => GdnShape { b: n, h: nvh, t: 1, dk: khd, dv: vhd, chunk: 1 },
        };
        let shape = model::gdn_mixer::GdnMixerShape { gdn, nkh: c.linear_num_key_heads, conv_kernel: c.linear_conv_kernel_dim, rms_eps: c.rms_eps };
        let weights = model::gdn_mixer::GdnMixerWeights {
            conv1d_weight: self.w(&p("conv1d.weight")),
            a_log: self.w(&p("A_log")),
            dt_bias: self.w(&p("dt_bias")),
            norm_weight: self.w(&p("norm.weight")),
            ones_khd: &self.ones_khd,
        };
        let (gated, internals) = match call {
            GdnCall::Decode(state) => (
                model::gdn_mixer::gdn_mixer_decode_state_fwd(g, &gdn_mixer_ids(), &gdn_mixer_decode_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, &state),
                None,
            ),
            GdnCall::Whole => model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, n, self.is_train, None),
            GdnCall::Chunk(cont) => model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, n, self.is_train, Some(cont)),
        };

        // out_proj (LoRA/int8 dispatch stays local). Fresh `Ops::act` call:
        // `gated` is a different activation from `xn1` above.
        let out = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            let act3 = if epilogue { self.quant_epilogue(0, &gated, None, n, value_dim).1 } else { self.ops.act(&mut s, &gated, 0, n, value_dim) };
            if self.ops_linear(&mut s, &act3, &p("out_proj.weight"), &out) {
                self.lora_fwd(&mut s, "out_proj", &gated, &p("out_proj.weight"), &out, n, value_dim, d);
            }
            g.submit(&[], &s);
        }

        let acts = internals.map(|internals| GdnLayerActs { internals, gated });
        (out, acts)
    }

    // ---- one GQA (Full) layer ----------------------------------------------

    /// The attention kernels for `layer`'s KV tier, resolved by name on this
    /// device. A failure means the tier's kernels are not in [`pipelines`],
    /// which is a build defect, not a runtime condition.
    fn kv_kernels(&self, layer: &KvLayer) -> KvKernels {
        KvKernels::resolve(&self.gpu, layer.k.tier()).unwrap_or_else(|e| panic!("qwen35moe: {e}"))
    }

    fn layer_gqa_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32, cached: Option<GqaCached>) -> (DeviceBuffer, Option<GqaLayerActs>) {
        self.layer_gqa_fwd_pre(l, xn1, None, false, n, cached)
    }

    /// [`Self::layer_gqa_fwd`] with `xn1`'s activation already quantised (`pre`) and,
    /// with `epilogue`, the sigmoid output gate and the quantisation of `o_proj`'s
    /// input in one launch - see [`Self::layer_gdn_fwd_pre`].
    fn layer_gqa_fwd_pre(&self, l: usize, xn1: &DeviceBuffer, pre: Option<Act>, epilogue: bool, n: u32, cached: Option<GqaCached>) -> (DeviceBuffer, Option<GqaLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, kvd) = (c.q_proj_dim(), c.kv_dim());
        let p = |s: &str| format!("blocks.{l}.self_attn.{s}");

        // q/k/v proj (LoRA/int8 dispatch stays local - see `model::gqa_mixer`'s own
        // module doc). xn1 quantized once, shared by q/k/v: one group, which is one
        // launch where the device allows it.
        let q_full = g.storage((n * qpd) as u64);
        let k = g.storage((n * kvd) as u64);
        let v = g.storage((n * kvd) as u64);
        let mut s1 = Vec::new();
        let act1 = match pre {
            Some(a) => a,
            None => self.ops.act(&mut s1, xn1, 0, n, d),
        };
        let (q_w, k_w, v_w) = (p("q_proj.weight"), p("k_proj.weight"), p("v_proj.weight"));
        let f32s = self.ops_linear_group(&mut s1, &act1, &[(&q_w, &q_full), (&k_w, &k), (&v_w, &v)]);
        if f32s[0] {
            self.lora_fwd(&mut s1, "q_proj", xn1, &q_w, &q_full, n, d, qpd);
        }
        if f32s[1] {
            self.lora_fwd(&mut s1, "k_proj", xn1, &k_w, &k, n, d, kvd);
        }
        if f32s[2] {
            self.lora_fwd(&mut s1, "v_proj", xn1, &v_w, &v, n, d, kvd);
        }
        g.submit(&[], &s1);

        // split+qknorm+rope+attention+gating - LoRA/dtype-agnostic, shared
        // with `crates/qwen35` (`model::gqa_mixer`). A batched DECODE step
        // supplies its OWN M-RoPE table (one row per sequence, at that
        // sequence's own absolute position) and attends the shared paged KV
        // pool instead of an isolated `[T,T]` causal block - see
        // `GqaDecodeCtx`.
        let shape = model::gqa_mixer::GqaMixerShape { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: c.rotary_dim() / 2, rms_eps: c.rms_eps };
        let (cos, sin) = match &cached {
            None => (&self.cos, &self.sin),
            Some(GqaCached::Chunk(ch)) => (ch.cos, ch.sin),
            Some(GqaCached::Decode(dc)) => (dc.cos, dc.sin),
        };
        let weights = model::gqa_mixer::GqaMixerWeights { q_norm: self.w(&p("q_norm.weight")), k_norm: self.w(&p("k_norm.weight")), cos, sin };
        let mut o_proj_act: Option<Act> = None;
        let (ctx_gated, internals) = match cached {
            None => model::gqa_mixer::gqa_mixer_fwd(g, &gqa_mixer_ids(), &shape, &weights, &q_full, &k, &v, n, self.is_train),
            Some(GqaCached::Chunk(ch)) => (
                model::gqa_mixer::gqa_mixer_chunk_kv_fwd(
                    g,
                    &gqa_mixer_ids(),
                    &self.kv_kernels(ch.layer),
                    DECODE_SOFTMAX_BATCHED,
                    &shape,
                    &weights,
                    &q_full,
                    &k,
                    &v,
                    n,
                    ch.base_row,
                    ch.start,
                    ch.cap,
                    ch.layer,
                    &ch.block_ids,
                    &ch.offsets,
                    &ch.seq_lens,
                ),
                None,
            ),
            Some(GqaCached::Decode(dc)) => {
                let kv = self.kv_kernels(dc.layer);
                // The front half of the attention (split, QK norm, rotation, KV
                // append) is one native launch for a single sequence on an f32 KV
                // tier where the device is offered it, and the chain of eight
                // otherwise; the sigmoid output gate follows either.
                let fused = self
                    .decode_fusion
                    .get()
                    .then(|| model::gqa_mixer::gqa_mixer_decode_fused_kv_attend(g, &kv, DECODE_SOFTMAX_BATCHED, &shape, &weights, &q_full, &k, &v, dc.layer, n, dc.paged))
                    .flatten();
                let (ctx, q_gate) = fused.unwrap_or_else(|| {
                    model::gqa_mixer::gqa_mixer_decode_batched_kv_attend(g, &gqa_mixer_ids(), &kv, DECODE_SOFTMAX_BATCHED, &shape, &weights, &q_full, &k, &v, dc.layer, n, dc.paged)
                });
                if epilogue {
                    // The gate rides on the quantisation `o_proj` needs anyway.
                    let (gated, act) = self.quant_epilogue(2, &ctx, Some(&q_gate), n, shape.qd());
                    o_proj_act = Some(act);
                    (gated.expect("a gated epilogue hands back its product"), None)
                } else {
                    (model::gqa_mixer::gate_ctx(g, &gqa_mixer_ids(), &ctx, &q_gate, n, shape.qd()).1, None)
                }
            }
        };

        // o_proj (LoRA/int8 dispatch stays local). Fresh `Ops::act` call:
        // `ctx_gated` is a different activation from `xn1` above.
        let out = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            let act2 = match o_proj_act {
                Some(a) => a,
                None => self.ops.act(&mut s, &ctx_gated, 0, n, shape.qd()),
            };
            if self.ops_linear(&mut s, &act2, &p("o_proj.weight"), &out) {
                self.lora_fwd(&mut s, "o_proj", &ctx_gated, &p("o_proj.weight"), &out, n, shape.qd(), d);
            }
            g.submit(&[], &s);
        }

        let acts = internals.map(|internals| GqaLayerActs { internals, ctx_gated });
        (out, acts)
    }

    // ---- MoE sublayer, universal for every layer ---------------------------

    fn moe_sublayer(&self, l: usize, xmid: &DeviceBuffer, n: u32) -> (DeviceBuffer, Option<MoeLayerActs>) {
        self.moe_sublayer_pre(l, xmid, None, n)
    }

    /// [`Self::moe_sublayer`] given the normalised, quantised `xmid` by a fused
    /// front (`front`): the int8 sublayer then starts at the router. Only the int8
    /// tier reads a front.
    fn moe_sublayer_pre(&self, l: usize, xmid: &DeviceBuffer, front: Option<&Front>, n: u32) -> (DeviceBuffer, Option<MoeLayerActs>) {
        // The int8 tier has its own sublayer: device-side routing and one gather
        // GEMV per projection over the fused expert banks (no host readback).
        if let Some(q8) = &self.q8 {
            return (self.moe_sublayer_i8(l, xmid, front, n, q8), None);
        }
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let e = c.n_experts;
        let moe_ff = c.moe_intermediate_size;
        let p = |s: &str| format!("blocks.{l}.{s}");

        let xn2 = g.storage((n * d) as u64);
        let router_logits = g.storage((n * e) as u64);
        let mut steps = vec![
            rmsnorm_fwd(g, &kernel_ids(), xmid, self.w(&p("ln2.weight")), &xn2, d, n, self.cfg.rms_eps),
            g.step(MATMUL, &[&xn2, self.w(&p("mlp.router.weight")), &router_logits], &[n, d, e], n * e),
        ];

        let shape = MoeShape { rows: n, d_model: d, moe_ff, n_experts: e, top_k: c.top_k };
        let gate = g.storage((n * e) as u64);
        // aux_coef/z_coef only affect router_bwd (never reached here -- see
        // model::moe::router_fwd_kind's forward-only call path), so 0.0 is a
        // pure "unused" value, not a behaviour change to the forward gate math.
        steps.push(router_fwd_kind(g, &moe_ids(), RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: true, routed_scaling: 1.0 }, &shape, &router_logits, None, &gate, None));

        let moe_acc = g.storage((n * d) as u64);
        // Router and gate above are fp32 (see `crate::q8`'s module
        // doc for why); only the routed experts' gate/up/down switch tier.
        // Training builds additionally need EVERY expert's OWN gate_pre/up/h/
        // expert_out (not a shared scratch reused across experts -- see
        // `model::moe::MoeActs`'s own doc for why forward's per-call-reused
        // `ExpertScratch` cannot serve backward), so `moe_acts` is `Some` only
        // for a training, non-int8 build (asserted mutually exclusive at
        // construction).
        let moe_acts: Option<MoeActs> = if self.is_train {
            let acts = MoeActs::new(g, &shape);
            for ei in 0..e {
                let (gn, un, dn) = &self.moe_expert_names[l][ei as usize];
                steps.extend(expert_fwd(
                    g,
                    &moe_ids(),
                    &shape,
                    &xn2,
                    &gate,
                    self.w(gn),
                    self.w(un),
                    self.w(dn),
                    &acts.at(ei as usize),
                    &moe_acc,
                    ei,
                    // Each expert owns its own weight tensors here (this
                    // crate's import fans the GGUF stack out per expert), so
                    // there is no bank offset to apply.
                    0,
                    ei != 0,
                ));
            }
            Some(acts)
        } else if n == 1 {
            // Decode's sparse dispatch -- see `Self::moe_sublayer_decode_sparse`'s
            // own doc (cheaper than the grouped path below at a single row:
            // a direct host readback of the exact few experts that fired
            // beats the grouped path's histogram/scan/permute machinery).
            self.moe_sublayer_decode_sparse(l, &mut steps, &shape, &xn2, &gate, &moe_acc);
            None
        } else {
            // Batched/prefill (`n > 1`) inference: device-side grouped-GEMM
            // dispatch (M5.12), replacing the `n_experts`-long dense
            // per-expert loop this branch used before -- see `MoeGrouped`'s
            // own doc and `model::moe::expert_fwd_grouped`'s module doc for
            // the full pipeline. `moe_grouped` is `Some` here by
            // construction: this arm is reached only when `!is_train &&
            // self.q8.is_none()`, exactly `Self::new_impl_on`'s own build
            // condition for it.
            let mg = self
                .moe_grouped
                .as_ref()
                .expect("qwen35: moe_grouped must be built for every non-training, non-int8 instance");
            let (gate_bank, up_bank, down_bank) = mg.banks[l]
                .as_ref()
                .unwrap_or_else(|| panic!("qwen35: moe_grouped bank missing for owned layer {l}"));
            steps.extend(expert_fwd_grouped(
                g,
                &grouped_expert_ids(),
                &shape,
                &xn2,
                &gate,
                gate_bank,
                up_bank,
                down_bank,
                &mg.scratch,
                &moe_acc,
            ));
            None
        };

        let moe_out = g.storage((n * d) as u64);
        let sh = self.shared_expert_steps(l, &xn2, n, &moe_acc, &moe_out, &mut steps);

        g.submit(&[], &steps);

        let acts = moe_acts.map(|acts| MoeLayerActs {
            xn2,
            router_logits,
            gate,
            fe: g.storage(e as u64),
            acts,
            sh_gate_pre: sh.gate_pre,
            sh_up: sh.up,
            sh_h: sh.h,
            sh_mlp_out: sh.mlp_out,
            sh_gate_logits: sh.gate_logits,
            sh_gate_scalar: sh.gate_scalar,
        });
        (moe_out, acts)
    }

    /// The shared expert at fp32 - `moe_out = moe_acc + sigmoid(gate . x) *
    /// SwiGLU(x)` - appended to `steps`, with its scratch handed back for the
    /// training tape. The one definition of the shared expert's fp32 path,
    /// reached from the dense/grouped/decode-sparse forwards and from the int8
    /// forward whenever the shared expert does not fit the expert banks.
    fn shared_expert_steps(&self, l: usize, xn2: &DeviceBuffer, n: u32, moe_acc: &DeviceBuffer, moe_out: &DeviceBuffer, steps: &mut Vec<Step>) -> SharedBufs {
        let g = &self.gpu;
        let (d, shared_ff) = (self.cfg.d_model, self.cfg.shared_expert_intermediate_size);
        let p = |s: &str| format!("blocks.{l}.{s}");
        let bufs = SharedBufs {
            gate_pre: g.storage((n * shared_ff) as u64),
            up: g.storage((n * shared_ff) as u64),
            h: g.storage((n * shared_ff) as u64),
            mlp_out: g.storage((n * d) as u64),
            gate_logits: g.storage(n as u64),
            gate_scalar: g.storage(n as u64),
            scaled: g.storage((n * d) as u64),
        };
        let scratch = SharedExpertScratch {
            gate_pre: &bufs.gate_pre,
            up: &bufs.up,
            h: &bufs.h,
            mlp_out: &bufs.mlp_out,
            gate_logits: &bufs.gate_logits,
            gate_scalar: &bufs.gate_scalar,
            scaled: &bufs.scaled,
        };
        steps.extend(shared_expert_fwd(
            g,
            &shared_expert_ids(),
            n,
            d,
            shared_ff,
            xn2,
            self.w(&p("mlp.shared_expert.gate.weight")),
            self.w(&p("mlp.shared_expert.up.weight")),
            self.w(&p("mlp.shared_expert.down.weight")),
            Some(self.w(&p("mlp.shared_expert_gate.weight"))),
            &scratch,
            moe_acc,
            moe_out,
        ));
        bufs
    }

    /// The int8 MoE sublayer: router, routed experts and (when it fits the
    /// banks) the shared expert, with NO host synchronisation and no
    /// per-expert dispatch.
    ///
    /// ```text
    /// xn2 --quant--> xq, sx
    /// router matmul [n, d] x [E(+1), d] --> logits         (fp32: a routing decision is a hard top-k)
    /// moe_router_topk --> ids[n, S], weight[n, S]          (S = top_k, +1 for the shared expert)
    /// gather GEMV gate, up over the fused banks --> [n*S, ff]
    /// moe_swiglu_quant --> hq, sh                          (SiLU(gate) * up, requantised per slot)
    /// gather GEMV down --> y[n*S, d]
    /// moe_slot_combine: out[row] = sum_s weight * y        (shared expert = slot top_k)
    /// ```
    ///
    /// Every selected expert of every row is one `slot`; its expert is read out
    /// of the bank by id. At decode that is `top_k + 1` slots per token and the
    /// layer streams exactly the ~25 MB of weights those experts hold, in a
    /// handful of dispatches instead of the 256-expert, ~1280-dispatch loop
    /// this replaced.
    fn moe_sublayer_i8(&self, l: usize, xmid: &DeviceBuffer, front: Option<&Front>, n: u32, q8: &Qwen35Q8) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let (d, e, ff, top_k) = (c.d_model, c.n_experts, c.moe_intermediate_size, c.top_k);
        let ml = &q8.moe[l];
        let shared = u32::from(ml.shared_in_bank());
        let (slots_per_row, width) = (top_k + shared, e + shared);
        let slots = n * slots_per_row;
        let p = |s: &str| format!("blocks.{l}.{s}");

        // `xn2` and its int8 packing: from the fused front when there is one, else
        // made here (the normalisation, then the model's shared packing scratch).
        let own_xn2;
        let (xn2, xq, sx): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer) = match front {
            Some(f) => (&f.xn, &f.xq, &f.sx),
            None => {
                own_xn2 = g.storage((n * d) as u64);
                (&own_xn2, &q8.xq, &q8.sx)
            }
        };
        let logits = g.storage((n * width) as u64);
        let (ids, weight) = (g.storage(slots as u64), g.storage(slots as u64));
        let (gate_out, up_out) = (g.storage((slots * ff) as u64), g.storage((slots * ff) as u64));
        let (hq, sh) = (g.storage((slots * ff / 4) as u64), g.storage(slots as u64));
        let y = g.storage((slots * d) as u64);
        let own_router;
        let router_w: &Weight = match &ml.router_ext {
            Some(w) => w,
            None => {
                own_router = Weight::F32 { w: self.w(&p("mlp.router.weight")).clone(), n: e, k: d };
                &own_router
            }
        };
        // At prefill row counts every expert has many slots: order them by expert
        // (three tiny dispatches) so each weight row is read once per tile of
        // slots instead of once per slot.
        let ne = e + shared;
        let grouped = n >= self.moe_grouped_min_rows.get();
        let (route_tab, route_perm) = if grouped { (g.storage(2 * (ne as u64 + 1)), g.storage(slots as u64)) } else { (g.storage(1), g.storage(1)) };
        let gather = |bank: &Bank8, xq: &DeviceBuffer, sx: &DeviceBuffer, out: &DeviceBuffer, xdiv: u32| {
            if grouped {
                // Worst-case tile count (an expert's last tile may be partial); a
                // workgroup past the real count finds the table's end and exits.
                let tiles = ne + slots.div_ceil(MOE_GROUPED_MR);
                g.dispatch(
                    MOE_I8_GROUPED,
                    &[xq, sx, &route_tab, &route_perm, &bank.packed, &bank.scale, out],
                    &[tiles, bank.k / 4, bank.n, xdiv, ne],
                    Dispatch::Workgroups(tiles * bank.n.div_ceil(MOE_GROUPED_COLS)),
                )
            } else {
                g.dispatch(
                    MOE_I8_GEMV_GATHER,
                    &[xq, sx, &ids, &bank.packed, &bank.scale, out],
                    &[slots, bank.k / 4, bank.n, xdiv],
                    Dispatch::Workgroups(slots * bank.n.div_ceil(MOE_GATHER_COLS)),
                )
            }
        };

        let mut steps = Vec::new();
        if front.is_none() {
            steps.push(rmsnorm_fwd(g, &kernel_ids(), xmid, self.w(&p("ln2.weight")), xn2, d, n, c.rms_eps));
            q8.quant(g, &mut steps, xn2, d, n);
        }
        // Through the `Ops` façade, not the naive one-thread-per-output `matmul`: at
        // decode's one row that kernel walks the whole `k = d_model` serially per
        // output and was a third of a decode step's device time.
        self.ops.matmul(&mut steps, router_w, &self.ops.act_f32(xn2, 0, n, d), &logits, 0);
        steps.push(g.dispatch(MOE_ROUTER_TOPK, &[&logits, &ids, &weight], &[n, e, top_k, shared], Dispatch::Workgroups(n)));
        if grouped {
            let counts = g.storage(ne as u64);
            steps.push(g.dispatch(MOE_ROUTE_COUNT, &[&ids, &counts], &[slots, ne], Dispatch::Workgroups(ne)));
            steps.push(g.dispatch(MOE_ROUTE_SCAN, &[&counts, &route_tab], &[ne, MOE_GROUPED_MR], Dispatch::Workgroups(1)));
            steps.push(g.dispatch(MOE_ROUTE_EMIT, &[&ids, &route_tab, &route_perm], &[slots, ne], Dispatch::Workgroups(ne)));
        }
        steps.push(gather(&ml.gate, xq, sx, &gate_out, slots_per_row));
        steps.push(gather(&ml.up, xq, sx, &up_out, slots_per_row));
        steps.push(g.dispatch(MOE_SWIGLU_QUANT, &[&gate_out, &up_out, &hq, &sh], &[slots, ff], Dispatch::Workgroups(slots)));
        steps.push(gather(&ml.down, &hq, &sh, &y, 1));

        let moe_out = g.storage((n * d) as u64);
        if shared == 1 {
            steps.push(g.step(MOE_SLOT_COMBINE, &[&y, &weight, &moe_out], &[n, d, slots_per_row], n * d));
        } else {
            let routed = g.storage((n * d) as u64);
            steps.push(g.step(MOE_SLOT_COMBINE, &[&y, &weight, &routed], &[n, d, slots_per_row], n * d));
            self.shared_expert_steps(l, xn2, n, &routed, &moe_out, &mut steps);
        }
        g.submit(&[], &steps);
        moe_out
    }

    /// Decode's sparse expert dispatch -- the `n==1` sibling of
    /// [`Self::moe_sublayer`]'s dense per-expert loop, called from inside it
    /// (see the call site's own comment) rather than replacing it, so the
    /// batched/prefill path is untouched.
    ///
    /// The dense loop calls `model::moe::expert_fwd` once per expert, for all
    /// `n_experts` (256) experts, unconditionally -- `moe_linear_gated.wgsl`
    /// already early-exits per-row for a non-selected expert (so no wasted
    /// FLOPs), but the DISPATCH itself still happens 256 times, 5 GPU submits
    /// each. At decode's `rows=1` that fixed per-dispatch overhead, not FLOPs,
    /// dominates: only `top_k` (8) of those 256 experts are ever selected for
    /// the single row. This function finds out WHICH `top_k` experts those
    /// are and calls `expert_fwd` only for those, cheaply:
    ///
    /// 1. `steps` (this layer's ln2 rmsnorm, router matmul, and
    ///    `router_fwd_kind` -- queued but not yet executed by the caller) is
    ///    submitted now and cleared: the compaction readback below needs
    ///    `gate`'s real device contents, so this is a genuine synchronisation
    ///    point, the same shape of trade `model::moe::expert_fwd_compact`
    ///    already makes (see its own doc) -- unavoidable, since only the HOST
    ///    can decide how many/which GEMM dispatches to issue next, and no
    ///    indirect-dispatch primitive exists anywhere in this engine.
    /// 2. `router_topk_compact.wgsl` (new kernel, see its own header) turns
    ///    the dense `[1, n_experts]` gate row into a compact `[top_k]` list of
    ///    the selected expert ids (padded with the sentinel `n_experts` for
    ///    any shortfall -- shouldn't happen, defensive only).
    /// 3. That `[top_k]` u32 buffer (8 words at the real 256-expert/top-8
    ///    scale, vs 256 f32 words for the dense gate `expert_fwd_compact`
    ///    would need) is read back and deduplicated on the host -- a no-op
    ///    given `router_gate.wgsl`'s own construction (its selection is
    ///    already distinct), done anyway because it's nearly free at 8
    ///    elements and turns "trust the invariant" into "checked".
    /// 4. `expert_fwd` (the SAME step-builder the dense loop uses, unmodified)
    ///    is called once per distinct real expert id, first with
    ///    `accumulate=false` (mirrors the dense loop's own `ei != 0` set-vs-
    ///    add contract for `moe_acc`) and the rest `accumulate=true`.
    ///
    /// Correctness: `moe_linear_gated.wgsl` writes exactly 0 for any row
    /// whose gate weight for the dispatched expert is 0 (`model::moe`'s own
    /// module doc), so the dense loop's non-selected experts are already a
    /// provable no-op on `moe_acc` -- excluding them here must not change
    /// `moe_acc`'s final contents, only how many dispatches produce them. See
    /// this crate's `moe_sublayer_decode_sparse_matches_dense_loop_bit_identical`
    /// test (both backends) for the check.
    fn moe_sublayer_decode_sparse(
        &self,
        l: usize,
        steps: &mut Vec<Step>,
        shape: &MoeShape,
        xn2: &DeviceBuffer,
        gate: &DeviceBuffer,
        moe_acc: &DeviceBuffer,
    ) {
        let g = &self.gpu;
        let (d, moe_ff, e, top_k, rows) = (shape.d_model, shape.moe_ff, shape.n_experts, shape.top_k, shape.rows);
        debug_assert_eq!(rows, 1, "moe_sublayer_decode_sparse: decode-only, expected rows==1, got {rows}");

        // Flush the router steps queued so far -- the readback below needs
        // `gate`'s real device contents, not just its recorded dispatch.
        g.submit(&[], steps);
        steps.clear();

        let top_ids = g.storage((rows * top_k) as u64);
        g.submit(&[], &[g.step(ROUTER_TOPK_COMPACT, &[gate, &top_ids], &[rows, e, top_k], rows)]);
        let host_ids: Vec<u32> = g.read(&top_ids, (rows * top_k) as usize).iter().map(|v| v.to_bits()).collect();

        // Dedup + drop the `e` sentinel (no real expert has that id). Defensive
        // no-op given router_gate.wgsl's own top_k-distinct construction.
        let mut real_ids: Vec<u32> = Vec::with_capacity(top_k as usize);
        for id in host_ids {
            if id < e && !real_ids.contains(&id) {
                real_ids.push(id);
            }
        }

        if real_ids.is_empty() {
            // Defensive only (router_gate.wgsl always selects >=1 expert for
            // top_k>=1): still honour the set-vs-add contract so `moe_acc`
            // is never left uninitialised for the shared-expert combine below.
            g.submit(&[moe_acc], &[]);
            return;
        }

        let scratch = ExpertScratch {
            gate_pre: &g.storage(moe_ff as u64),
            up: &g.storage(moe_ff as u64),
            h: &g.storage(moe_ff as u64),
            expert_out: &g.storage(d as u64),
        };
        for (i, &ei) in real_ids.iter().enumerate() {
            let (gn, un, dn) = &self.moe_expert_names[l][ei as usize];
            steps.extend(expert_fwd(
                g,
                &moe_ids(),
                shape,
                xn2,
                gate,
                self.w(gn),
                self.w(un),
                self.w(dn),
                &scratch,
                moe_acc,
                ei,
                0,
                i != 0,
            ));
        }
    }

    // ---- full stack ----------------------------------------------------------

    /// Run this stage's forward graph over its own layer range
    /// (`self.shard.start..self.shard.end`, ABSOLUTE layer indices - `res`
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
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), xres, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n, self.cfg.rms_eps)]);

            let (attn_out, mixer_acts) = match ty {
                LayerType::Linear => {
                    let (o, a) = self.layer_gdn_fwd(l, &xn1, n, GdnCall::Whole);
                    (o, a.map(|a| MixerActs::Gdn(Box::new(a))))
                }
                LayerType::Full => {
                    let (o, a) = self.layer_gqa_fwd(l, &xn1, n, None);
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
        // `false` in `Self::forward`'s CE step too - see `Qwen35::forward`'s
        // own doc), so a size-1 dummy stands in, matching this file's
        // "size-1 dummy where a value doesn't apply" convention used
        // elsewhere (`gqa_kcache`/`gdn_state`).
        let xn_final = if self.shard.head {
            let xn_final = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &self.res[self.cfg.n_layers as usize], self.w("norm.weight"), &xn_final, d, n, self.cfg.rms_eps)]);
            let mut head_steps = Vec::new();
            self.head_matmul(&mut head_steps, &xn_final, None, n, &self.logits);
            g.submit(&[], &head_steps);
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
        let value_dim = c.linear_value_dim();
        let nvh = c.linear_num_value_heads;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // out_proj backward (LoRA/int8 dispatch stays local).
        let d_gated = g.storage((n * value_dim) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "out_proj", d_out, &la.gated, &p("out_proj.weight"), &d_gated, n, value_dim, d, 0);
            g.submit(&[], &s);
        }

        // Reverse of the hoisted `model::gdn_mixer::gdn_mixer_fwd` internals.
        let shape = model::gdn_mixer::GdnMixerShape {
            gdn: la.internals.shape,
            nkh: c.linear_num_key_heads,
            conv_kernel: c.linear_conv_kernel_dim,
            rms_eps: c.rms_eps,
        };
        let weights = model::gdn_mixer::GdnMixerWeights {
            conv1d_weight: self.w(&p("conv1d.weight")),
            a_log: self.w(&p("A_log")),
            dt_bias: self.w(&p("dt_bias")),
            norm_weight: self.w(&p("norm.weight")),
            ones_khd: &self.ones_khd,
        };
        let grads = model::gdn_mixer::GdnMixerGrads {
            conv1d_weight: self.trainable(&p("conv1d.weight")).then(|| self.g(&p("conv1d.weight"))),
            a_log: self.trainable(&p("A_log")).then(|| self.g(&p("A_log"))),
            dt_bias: self.trainable(&p("dt_bias")).then(|| self.g(&p("dt_bias"))),
            norm_weight: self.trainable(&p("norm.weight")).then(|| self.g(&p("norm.weight"))),
        };
        let (d_mixed_qkv, d_bproj, d_aproj, d_z) = model::gdn_mixer::gdn_mixer_bwd(g, &gdn_mixer_ids(), &shape, &weights, &grads, &la.internals, &d_gated, n);

        // in_proj_b/a/z backward (LoRA/int8 dispatch stays local). FIRST
        // touch to d_xn1 in this function (acc=0) -- in_proj_qkv (below)
        // accumulates last of all.
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "in_proj_b", &d_bproj, xn1, &p("in_proj_b.weight"), d_xn1, n, d, nvh, 0);
            self.proj_bwd(&mut s, "in_proj_a", &d_aproj, xn1, &p("in_proj_a.weight"), d_xn1, n, d, nvh, 1);
            self.proj_bwd(&mut s, "in_proj_z", &d_z, xn1, &p("in_proj_z.weight"), d_xn1, n, d, value_dim, 1);
            g.submit(&[], &s);
        }

        // in_proj_qkv backward (last accumulate into d_xn1; LoRA/int8
        // dispatch stays local).
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

        // o_proj backward (LoRA/int8 dispatch stays local).
        let d_ctx_gated = g.storage((n * qd) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "o_proj", d_out, &la.ctx_gated, &p("o_proj.weight"), &d_ctx_gated, n, qd, d, 0);
            g.submit(&[], &s);
        }

        // Reverse of the hoisted `model::gqa_mixer::gqa_mixer_fwd` internals.
        let shape = model::gqa_mixer::GqaMixerShape { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: c.rotary_dim() / 2, rms_eps: c.rms_eps };
        let weights = model::gqa_mixer::GqaMixerWeights { q_norm: self.w(&p("q_norm.weight")), k_norm: self.w(&p("k_norm.weight")), cos: &self.cos, sin: &self.sin };
        let grads = model::gqa_mixer::GqaMixerGrads {
            q_norm: self.trainable(&p("q_norm.weight")).then(|| self.g(&p("q_norm.weight"))),
            k_norm: self.trainable(&p("k_norm.weight")).then(|| self.g(&p("k_norm.weight"))),
        };
        let (d_q_full, d_k, d_v) = model::gqa_mixer::gqa_mixer_bwd(g, &gqa_mixer_ids(), &shape, &weights, &grads, &la.internals, &d_ctx_gated, n);

        // q/k/v proj backward (LoRA/int8 dispatch stays local).
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
        let expert_weights: Vec<ExpertWeights> = (0..e)
            .map(|ei| {
                let (gn, un, dn) = &self.moe_expert_names[l][ei as usize];
                // `w_off = 0`: one tensor per expert, not a fused bank.
                ExpertWeights { gate_w: self.w(gn).clone(), up_w: self.w(un).clone(), down_w: self.w(dn).clone(), w_off: 0 }
            })
            .collect();
        // Never a LoRA target (per the standing LoRA task's own scope note:
        // the 256-expert MoE linears are out of scope) -- Frozen under a LoRA
        // build, so each field is `None` there (`ExpertGrads`' own contract).
        let expert_grads: Vec<ExpertGrads> = (0..e)
            .map(|ei| {
                let (gn, un, dn) = &self.moe_expert_names[l][ei as usize];
                ExpertGrads {
                    gate_w: self.trainable(gn).then(|| self.g(gn)),
                    up_w: self.trainable(un).then(|| self.g(un)),
                    down_w: self.trainable(dn).then(|| self.g(dn)),
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
            RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: true, routed_scaling: 1.0 },
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

    /// Full backward pass - mirrors [`Self::run_forward`]'s layer loop in
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
    /// read - mirrors `qwen3::Qwen::build_backward_steps`'s own shard gating,
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

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, adam, clip, extra_scale);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    /// Run the forward graph and return the scalar loss. Only meaningful on
    /// a whole (single-device) instance or a pipeline stage that owns the
    /// head (`self.shard.head`) - `self.logits`/`self.ce_buf` are only
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
    // that is `model::serve::PagedDecoder`, built on top of this).
    //
    // `run_decode_batch` is the whole of the decode tape - one token for each
    // of `bsz` INDEPENDENT sequences, in one dispatch set per layer - and
    // `step`/`run_decode_step` are its one-row case, not a separate tape.
    // There is no decode-specific per-layer code: a decode step is
    // `layer_gdn_fwd`/`layer_gqa_fwd`/`moe_sublayer` at `n = bsz` rows, with
    // the two stateful mixers in their DECODE arm (`model::gdn_mixer::
    // gdn_mixer_decode_fwd` over the recurrent state and conv window,
    // `model::gqa_mixer::gqa_mixer_decode_batched_fwd` over the shared paged
    // KV pool) - the same pair `qwen35::model::Qwen35::run_decode_batch`
    // drives, so "a batched decode step" means the same thing for both models.
    // =========================================================================

    /// Reset decode state for a fresh sequence: the position counter and
    /// every GDN layer's persistent recurrent `state`/conv `hist` (both must
    /// start at zero for a fresh sequence -- see `model::gdn::gdn_recurrent_step`
    /// and `gdn_causal_conv1d_step`'s own docs). GQA layers' KV caches are
    /// deliberately left untouched: a decode step's attention only ever reads
    /// cache rows `0..=pos` (its `seq_lens` bound), so
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
            gqa_kv: &self.gqa_kv,
            gqa_cap: self.dec_cap,
            // This instance's own caches are dedicated per-sequence buffers,
            // not a pool window -- see `DecodeCaches::gqa_base_row`.
            gqa_base_row: 0,
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
        assert!(
            caches.gqa_cap > 0 && caches.gqa_base_row.is_multiple_of(caches.gqa_cap),
            "qwen35moe::run_decode_step: gqa_base_row {} is not a whole number of {}-row blocks",
            caches.gqa_base_row,
            caches.gqa_cap
        );
        let seqs = [BatchSeq { phys: caches.gqa_base_row / caches.gqa_cap, pos }];
        let gdn = [GdnBufs { state: caches.gdn_state, hist: caches.gdn_hist }];
        let batch = BatchDecodeCaches { gqa_kv: caches.gqa_kv, gqa_cap: caches.gqa_cap, seqs: &seqs, gdn: GdnStore::PerSeq(&gdn) };
        self.run_decode_batch(&[token_id], &batch).xn
    }

    /// [`Self::run_decode_step`] for a BATCH of independent sequences -- one
    /// decode token each, all in one set of dispatches per layer.
    ///
    /// This is the function [`Self::run_decode_step`] is the `bsz = 1` case of,
    /// and the reason a serving engine can raise aggregate throughput without
    /// raising per-sequence latency: a decode step's cost is dominated by
    /// reading every weight the step touches, and those reads are what the
    /// batch shares. Per layer the projections become `[bsz, d] x [d, *]` GEMMs
    /// instead of `bsz` separate `m = 1` GEMVs against the same weights, and
    /// the two stateful mixers each take the whole batch in one dispatch set --
    /// full attention through one shared paged KV pool
    /// ([`model::block::gqa_decode_batched_step`]), Gated-DeltaNet through the
    /// `b*h` axis its kernels already have
    /// ([`model::gdn_mixer::gdn_mixer_decode_fwd`]). Both are the SAME shared
    /// primitives `qwen35::model::Qwen35::run_decode_batch` drives, which is
    /// what makes a batched decode step mean the same thing for both models.
    ///
    /// `tokens` is one token id per sequence, in `caches.seqs` order; the
    /// return value is the `[bsz, d_model]` final-norm hidden block.
    pub(crate) fn run_decode_batch(&self, tokens: &[u32], caches: &BatchDecodeCaches) -> Hidden {
        let bsz = tokens.len() as u32;
        assert!(bsz > 0, "qwen35moe::run_decode_batch: empty batch");
        assert_eq!(tokens.len(), caches.seqs.len(), "qwen35moe::run_decode_batch: {} tokens for {} sequences", tokens.len(), caches.seqs.len());
        for s in caches.seqs {
            assert!(s.pos < caches.gqa_cap, "qwen35moe::run_decode_batch: decode position {} exceeds the per-sequence capacity {}", s.pos, caches.gqa_cap);
        }
        let meta = self.alloc_decode_meta(bsz);
        self.write_decode_meta(&meta, tokens, caches.seqs);
        // The scores/probs stride of the unfused attention path, sized to the
        // batch's longest LIVE sequence rather than the engine's configured
        // ceiling: it is pure addressing, not a compute bound (see
        // `gqa_decode_batched_step`).
        let cap = caches.seqs.iter().map(|s| s.pos + 1).max().unwrap_or(1);
        self.decode_layers(&meta, bsz, caches, cap)
    }

    /// The device buffers a decode step READS for everything that changes from
    /// one token to the next: the token ids, the batch's M-RoPE rows (each
    /// sequence at its OWN position) and the four paged-KV index buffers. A
    /// step's dispatches bind these buffers and nothing else varies between
    /// tokens, which is what lets a step be recorded once ([`Self::record_decode`])
    /// and replayed with [`Self::write_decode_meta`] updating them.
    fn alloc_decode_meta(&self, bsz: u32) -> DecodeMeta {
        let g = &self.gpu;
        let rows = bsz as u64 * (self.cfg.rotary_dim() / 2) as u64;
        DecodeMeta {
            tokens: g.storage(bsz as u64),
            cos: g.storage(rows),
            sin: g.storage(rows),
            blocks: g.storage(bsz as u64),
            offsets: g.storage(bsz as u64),
            block_tables: g.storage(bsz as u64),
            seq_lens: g.storage(bsz as u64),
        }
    }

    /// Fill a [`DecodeMeta`] for this step's tokens and sequences.
    fn write_decode_meta(&self, m: &DecodeMeta, tokens: &[u32], seqs: &[BatchSeq]) {
        let (g, c) = (&self.gpu, &self.cfg);
        g.write(&m.tokens, tokens);
        let positions: Vec<[u32; 3]> = seqs.iter().map(|s| [s.pos, s.pos, s.pos]).collect();
        let (cos_rows, sin_rows) = qwen3vl::mrope::mrope_tables(&positions, c.mrope_section, c.rotary_dim(), c.rope_theta);
        g.write_f32(&m.cos, &cos_rows);
        g.write_f32(&m.sin, &sin_rows);
        let phys: Vec<u32> = seqs.iter().map(|s| s.phys).collect();
        g.write(&m.blocks, &phys);
        g.write(&m.offsets, &seqs.iter().map(|s| s.pos).collect::<Vec<u32>>());
        // `max_bt = 1`: this engine gives a sequence ONE physical block for its
        // whole KV history, so its block table is a single entry.
        g.write(&m.block_tables, &phys);
        g.write(&m.seq_lens, &seqs.iter().map(|s| s.pos + 1).collect::<Vec<u32>>());
    }

    /// The layer stack of one decode step over `meta`'s buffers: embedding,
    /// every layer, the final norm. `cap` is the unfused attention's scores
    /// stride and the fused attention's split count, so a step that will be
    /// replayed passes the pool's whole capacity (the kernels mask by each
    /// sequence's length) and one that will not passes the live maximum.
    fn decode_layers(&self, meta: &DecodeMeta, bsz: u32, caches: &BatchDecodeCaches, cap: u32) -> Hidden {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let mut res = g.storage((bsz * d) as u64);
        g.submit(&[], &[g.step(EMBED, &[&meta.tokens, self.w("tok.weight"), &res], &[d, bsz], bsz * d)]);

        let paged = model::gqa_mixer::PagedDecodeBatch {
            blocks: &meta.blocks,
            offsets: &meta.offsets,
            block_tables: &meta.block_tables,
            seq_lens: &meta.seq_lens,
            block_size: caches.gqa_cap,
            max_bt: 1,
            cap,
        };

        // Where the residual add, the norm and the activation quantisation that sit
        // between two linears are ONE launch instead of four (`add2`,
        // `rmsnorm_rows`, `max_abs_rows`, `quant_pack`) - bit-identical to those
        // four, which is why the gate is "the device offers the kernel and the
        // unfused chain would have used the cooperative RMSNorm". Off, the loop is
        // the unfused chain.
        let fused = self.fused_front_ok(bsz);
        let epilogue = self.fused_epilogue_ok(bsz);
        let n_layers = c.n_layers as usize;
        // The next layer's front, when the previous step produced it.
        let mut front = fused.then(|| self.rms_quant_front(&res, None, self.w("blocks.0.ln1.weight"), bsz).1);
        let mut hidden = None;

        for (l, ty) in c.layer_types().iter().enumerate() {
            let (xn1, pre1) = match front.take() {
                Some(f) => {
                    let act = self.front_act(&f, bsz);
                    (f.xn, Some(act))
                }
                None => {
                    let xn1 = g.storage((bsz * d) as u64);
                    g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, bsz, self.cfg.rms_eps)]);
                    (xn1, None)
                }
            };

            let attn_out = match ty {
                LayerType::Linear => {
                    let streams: Vec<model::gdn_mixer::GdnStream>;
                    let state = match &caches.gdn {
                        GdnStore::PerSeq(bufs) => {
                            streams = bufs.iter().map(|b| model::gdn_mixer::GdnStream { state: &b.state[l], hist: &b.hist[l] }).collect();
                            model::gdn_mixer::GdnDecodeState::Streams(&streams)
                        }
                        GdnStore::Pool { state, hist } => model::gdn_mixer::GdnDecodeState::Pool(model::gdn_mixer::GdnPoolRows {
                            state: &state[l],
                            hist: &hist[l],
                            rows: &meta.blocks,
                            gather: POOL_ROWS_GATHER2,
                            scatter: POOL_ROWS_SCATTER2,
                            fuse: self.decode_fusion.get(),
                        }),
                    };
                    self.layer_gdn_fwd_pre(l, &xn1, pre1, epilogue, bsz, GdnCall::Decode(state)).0
                }
                LayerType::Full => {
                    let dctx = GqaDecodeCtx { paged: &paged, layer: &caches.gqa_kv[l], cos: &meta.cos, sin: &meta.sin };
                    self.layer_gqa_fwd_pre(l, &xn1, pre1, epilogue, bsz, Some(GqaCached::Decode(&dctx))).0
                }
            };

            // The residual add and the MoE sublayer's norm and packing: one launch
            // when fused.
            let ln2 = format!("blocks.{l}.ln2.weight");
            let (xmid, front2) = if fused {
                let (sum, f) = self.rms_quant_front(&res, Some(&attn_out), self.w(&ln2), bsz);
                (sum.expect("an add front hands back its sum"), Some(f))
            } else {
                let xmid = g.storage((bsz * d) as u64);
                g.submit(&[], &[g.step(ADD2, &[&res, &attn_out, &xmid], &[bsz * d], bsz * d)]);
                (xmid, None)
            };

            // Same `moe_sublayer` this file's batched path uses -- only the
            // row count differs, so no decode-specific MoE function is needed
            // at all.
            let (moe_out, _) = self.moe_sublayer_pre(l, &xmid, front2.as_ref(), bsz);
            if fused {
                // The next layer's first norm - or, after the last, the final one -
                // rides on this residual add.
                let next = if l + 1 < n_layers { format!("blocks.{}.ln1.weight", l + 1) } else { "norm.weight".to_string() };
                let (sum, f) = self.rms_quant_front(&xmid, Some(&moe_out), self.w(&next), bsz);
                res = sum.expect("an add front hands back its sum");
                if l + 1 < n_layers {
                    front = Some(f);
                } else {
                    let act = self.front_act(&f, bsz);
                    hidden = Some(Hidden { xn: f.xn, act: Some(act) });
                }
            } else {
                let res_next = g.storage((bsz * d) as u64);
                g.submit(&[], &[g.step(ADD2, &[&xmid, &moe_out, &res_next], &[bsz * d], bsz * d)]);
                res = res_next;
            }

            // Hand this layer to the device NOW and keep building the next one:
            // `Gpu::submit` appends to a pending list rather than submitting, so
            // without this the whole step is recorded on the host before the
            // card starts any of it. At decode row counts the host side of that
            // is not small next to the device side.
            g.flush();
        }

        hidden.unwrap_or_else(|| {
            let xn_final = g.storage((bsz * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res, self.w("norm.weight"), &xn_final, d, bsz, self.cfg.rms_eps)]);
            Hidden::plain(xn_final)
        })
    }

    /// **Record one whole decode step** - layers AND head - as a [`DecodeTape`]:
    /// the same dispatches [`Self::run_decode_batch`] builds, captured instead of
    /// launched, over buffers the tape keeps alive. [`Self::replay_decode`] then
    /// runs a step for the price of one submission plus a handful of small
    /// writes, where building it costs tens of milliseconds of host time - more
    /// than the card spends executing it.
    ///
    /// The tape is bound to this batch's SHAPE and sequences: its row count, the
    /// per-sequence recurrent-state buffers in `caches.seqs`, the KV pool. What
    /// differs token to token (ids, positions, KV lengths) goes through the
    /// tape's [`DecodeMeta`], rewritten by each replay. Needs the int8 tier: the
    /// fp32 decode's sparse expert dispatch reads the router's choice back to the
    /// host mid-step, which a recording cannot do.
    pub(crate) fn record_decode(&self, tokens: &[u32], caches: &BatchDecodeCaches, head: DecodeHead) -> DecodeTape {
        let bsz = tokens.len() as u32;
        assert!(self.q8.is_some(), "qwen35moe::record_decode needs the int8 expert tier: the fp32 decode reads routing back to the host mid-step");
        assert!(bsz > 0 && tokens.len() == caches.seqs.len(), "qwen35moe::record_decode: {} tokens for {} sequences", tokens.len(), caches.seqs.len());
        let meta = self.alloc_decode_meta(bsz);
        self.write_decode_meta(&meta, tokens, caches.seqs);
        let g = &self.gpu;
        g.begin_tape();
        let hidden = self.decode_layers(&meta, bsz, caches, caches.gqa_cap);
        let out = match head {
            DecodeHead::Greedy => TapeOut::Greedy(self.head_argmax_rows_record(&hidden, bsz)),
            DecodeHead::TopK(cap) => {
                let (vals, idx) = self.head_topk_rows_record(&hidden, bsz, cap);
                TapeOut::TopK { vals, idx, cap }
            }
        };
        DecodeTape { tape: g.end_tape(), meta, bsz, out }
    }

    /// Run a recorded decode step for `tokens` at the positions in `seqs` (the
    /// SAME sequences the tape was recorded for, in the same order), and read
    /// back what the recorded head produced. See [`Self::record_decode`].
    pub(crate) fn replay_decode(&self, tape: &DecodeTape, tokens: &[u32], seqs: &[BatchSeq]) -> DecodeOut {
        assert_eq!(tokens.len(), tape.bsz as usize, "qwen35moe::replay_decode: a tape recorded for {} rows replayed with {}", tape.bsz, tokens.len());
        self.write_decode_meta(&tape.meta, tokens, seqs);
        self.gpu.replay_tape(&tape.tape);
        match &tape.out {
            TapeOut::Greedy(out) => DecodeOut::Greedy(self.gpu.read(out, tape.bsz as usize).into_iter().map(|x| x as u32).collect()),
            TapeOut::TopK { vals, idx, cap } => DecodeOut::TopK(self.read_topk(vals, idx, tape.bsz, *cap)),
        }
    }

    /// Rows at or above which a chunk round pools its per-layer scratch (see
    /// [`CHUNK_ARENA_MIN_ROWS`]). A test hook: the tiny configs this crate's
    /// tests run at have a `block_size` far below any useful threshold, so
    /// without it only the unpooled side would ever be exercised.
    /// Rows from which [`Self::moe_sublayer_i8`] groups each expert's slots
    /// (default [`MOE_GROUPED_MIN_ROWS`]); `u32::MAX` never does, `1` always does.
    /// Both compute every slot identically, so this is a speed choice - and a test
    /// hook, since the tiny configs the tests run are far below any useful
    /// threshold.
    pub fn set_moe_grouped_min_rows(&self, rows: u32) {
        self.moe_grouped_min_rows.set(rows.max(1));
    }

    /// Whether a decode step takes the fused native kernels where the device is
    /// offered them (default on). Every one of them is gated byte-for-byte against
    /// the chain of kernels it replaces, so off is the reference a test compares to
    /// and never a different answer.
    pub fn set_decode_fusion(&self, on: bool) {
        self.decode_fusion.set(on);
    }

    pub fn set_chunk_arena_min_rows(&self, rows: u32) {
        assert!(rows > 0, "qwen35moe::set_chunk_arena_min_rows: 0 would open a scope for an empty round; 1 is 'always pool'");
        self.chunk_arena_min_rows.set(rows);
    }

    /// **One ROUND of a chunked prefill**: `tokens.len()` consecutive prompt
    /// tokens starting at absolute position `pos_start`, pushed through the
    /// whole layer stack with ONE dispatch shape per layer instead of
    /// [`Self::run_decode_step`]'s one per token. Returns the round's LAST
    /// token's final-norm hidden state (`[d_model]`, unread) - the only row a
    /// prefill's caller wants, and the row a following round or decode step
    /// continues from.
    ///
    /// **State contract.** `caches` is left in EXACTLY the state a
    /// token-by-token replay of the same tokens would have left it in: rows
    /// `pos_start..pos_start+n` of every GQA layer's K/V planes hold the round's
    /// QK-normed, RoPE'd keys and values (the round's queries attended rows
    /// `0..=pos_start+i`), and every Gated-DeltaNet layer's state/conv window
    /// continues from the previous round, so rounds and single-token steps
    /// mix freely on one sequence. The `qwen35` twin's contract, with this
    /// model's MoE sublayer in place of the dense MLP.
    pub(crate) fn run_prefill_chunk(&self, tokens: &[u32], pos_start: u32, caches: &DecodeCaches) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let n = tokens.len() as u32;
        assert!(n > 0, "qwen35moe::run_prefill_chunk: empty chunk (no token to produce a hidden state from)");
        assert!(
            self.shard.embed && self.shard.head,
            "qwen35moe::run_prefill_chunk is whole-model only (this shard has embed={}, head={})",
            self.shard.embed,
            self.shard.head
        );
        assert!(
            pos_start + n <= caches.gqa_cap,
            "qwen35moe::run_prefill_chunk: chunk ends at position {} but the KV cache holds {} rows",
            pos_start + n,
            caches.gqa_cap
        );
        assert!(
            caches.gqa_cap > 0 && caches.gqa_base_row.is_multiple_of(caches.gqa_cap),
            "qwen35moe::run_prefill_chunk: gqa_base_row {} is not a whole number of {}-row blocks",
            caches.gqa_base_row,
            caches.gqa_cap
        );

        let tok_buf = g.storage(n as u64);
        g.write(&tok_buf, tokens);
        let mut res = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(EMBED, &[&tok_buf, self.w("tok.weight"), &res], &[d, n], n * d)]);

        // Built ONCE per round, shared by every GQA layer in it: the round's
        // own M-RoPE table (absolute positions), its causal `seq_lens`, and the
        // single-block table a flat per-sequence KV cache degenerates to.
        let positions: Vec<[u32; 3]> = (0..n).map(|i| [pos_start + i, pos_start + i, pos_start + i]).collect();
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, c.mrope_section, c.rotary_dim(), c.rope_theta);
        let cos = g.storage_init("qwen35moe.prefill_chunk.cos", &cos);
        let sin = g.storage_init("qwen35moe.prefill_chunk.sin", &sin);
        let block_ids = g.storage(n as u64);
        let offsets = g.storage(n as u64);
        let seq_lens = g.storage(n as u64);
        g.write(&block_ids, &vec![caches.gqa_base_row / caches.gqa_cap; n as usize]);
        g.write(&offsets, &(0..n).map(|i| pos_start + i).collect::<Vec<u32>>());
        g.write(&seq_lens, &(0..n).map(|i| pos_start + i + 1).collect::<Vec<u32>>());

        let pooled = n >= self.chunk_arena_min_rows.get();
        // What an UNPOOLED round owes instead of the arena's drain: nothing is
        // recycled there, but every layer's temporaries are still DROPPED as
        // the next layer's are taken, and the backend refuses an allocation once
        // more than its reclaim ceiling sits dropped-and-unreclaimed. So an
        // unpooled round drains on BYTES, not on a layer count.
        let reclaim_budget = g.reclaim_ceiling_bytes() / 2;
        let types = c.layer_types();
        #[allow(clippy::needless_range_loop)]
        for l in self.shard.start..self.shard.end {
            // A replay arena, not a fixed drain schedule: every layer's outer
            // temporaries (`xn1`, `xmid`, `res_next`, ...) are the same size
            // requested in the same call order whatever the layer's own type,
            // each dead the moment its one consumer's dispatch is recorded.
            let _scope = pooled.then(|| g.scratch_scope());

            let xn1 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n, c.rms_eps)]);

            let mixer_out = match types[l] {
                LayerType::Linear => {
                    let cont = model::gdn_mixer::GdnStream { state: &caches.gdn_state[l], hist: &caches.gdn_hist[l] };
                    self.layer_gdn_fwd(l, &xn1, n, GdnCall::Chunk(cont)).0
                }
                LayerType::Full => {
                    let ctx = GqaChunkCtx {
                        base_row: caches.gqa_base_row,
                        start: pos_start,
                        cap: caches.gqa_cap,
                        layer: &caches.gqa_kv[l],
                        block_ids: block_ids.clone(),
                        offsets: offsets.clone(),
                        seq_lens: seq_lens.clone(),
                        cos: &cos,
                        sin: &sin,
                    };
                    self.layer_gqa_fwd(l, &xn1, n, Some(GqaCached::Chunk(&ctx))).0
                }
            };

            let xmid = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[&res, &mixer_out, &xmid], &[n * d], n * d)]);

            let (moe_out, _) = self.moe_sublayer(l, &xmid, n);
            let res_next = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &moe_out, &res_next], &[n * d], n * d)]);
            res = res_next;

            // The liveness boundary: building the whole layer first and THEN
            // draining lets the host overlap recording with the previous
            // layer's device work, while the `poll_wait` proves no recycled slot
            // is still being read before `flush` lets this layer's writes
            // reach the device.
            if pooled || g.pending_reclaim_bytes() > reclaim_budget {
                g.poll_wait();
            }
            g.flush();
        }
        if pooled {
            g.scratch_release();
        }

        let xn_final = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res, self.w("norm.weight"), &xn_final, d, n, c.rms_eps)]);
        // Only the LAST row is ever wanted (the round's next-token prediction,
        // or the seam into the next round).
        let last = g.storage(d as u64);
        g.submit(&[], &[g.step(CONCAT_SPLIT, &[&xn_final, &last], &[1, n * d, d, (n - 1) * d, 1, 1], d)]);
        last
    }

    /// **Chunked prefill** of a whole prompt against THIS instance's own
    /// per-sequence decode state - the multi-token-per-dispatch sibling of
    /// calling [`Self::step`] once per prompt token. Consumes `tokens` in rounds
    /// of at most `max_chunk`, each continuing from the state the previous one
    /// left, advances `decode_pos` by the whole prompt and returns the LAST
    /// token's final-norm hidden state. A following [`Self::step`] continues as
    /// if the prompt had been replayed one token at a time.
    pub fn prefill_chunked(&self, tokens: &[u32], max_chunk: u32) -> Vec<f32> {
        assert_eq!(self.b, 1, "qwen35moe::prefill_chunked requires b==1 (single sequence)");
        assert!(!tokens.is_empty(), "qwen35moe::prefill_chunked: empty prompt");
        assert!(max_chunk > 0, "qwen35moe::prefill_chunked: max_chunk must be > 0");
        if let Some(&bad) = tokens.iter().find(|&&t| t >= self.cfg.vocab) {
            panic!("qwen35moe::prefill_chunked: token {bad} exceeds vocab {}", self.cfg.vocab);
        }
        let mut pos = self.dec_pos.get();
        assert!(
            pos + tokens.len() as u32 <= self.dec_cap,
            "qwen35moe::prefill_chunked: prompt ends at position {} but this instance's decode capacity is {}",
            pos + tokens.len() as u32,
            self.dec_cap
        );
        let caches = DecodeCaches { gqa_kv: &self.gqa_kv, gqa_cap: self.dec_cap, gqa_base_row: 0, gdn_state: &self.gdn_state, gdn_hist: &self.gdn_hist };
        let mut hidden = None;
        for round in tokens.chunks(max_chunk as usize) {
            hidden = Some(self.run_prefill_chunk(round, pos, &caches));
            pos += round.len() as u32;
        }
        self.dec_pos.set(pos);
        let hidden = hidden.expect("prefill_chunked: prompt is non-empty (asserted above)");
        self.gpu.read(&hidden, self.cfg.d_model as usize)
    }

    /// Device-side head epilogue over a `[rows, d_model]` block of final-normed
    /// hidden states: `logits[rows, vocab]` in one GEMM against the resident
    /// head weight.
    pub(crate) fn head_logits_rows_dev(&self, hidden: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        self.head_logits_rows_hidden(&Hidden::plain(hidden.clone()), rows)
    }

    /// [`Self::head_logits_rows_dev`] over a [`Hidden`] block, reading the int8
    /// activation a fused front already packed instead of packing it again.
    fn head_logits_rows_hidden(&self, hidden: &Hidden, rows: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let logits = g.storage(rows as u64 * v as u64);
        let mut steps = Vec::new();
        self.head_matmul(&mut steps, &hidden.xn, hidden.act.as_ref(), rows, &logits);
        g.submit(&[], &steps);
        logits
    }

    /// `logits = hidden @ head^T` through the façade, at whatever tier the head
    /// was loaded: the int8 weight in `self.weights` (activation quantised
    /// here) or the fp32 parameter.
    fn head_matmul(&self, steps: &mut Vec<Step>, hidden: &DeviceBuffer, pre: Option<&Act>, rows: u32, logits: &DeviceBuffer) {
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        let name = self.cfg.head_weight();
        match self.weights.get(name) {
            Some(head) => match pre {
                Some(act) => self.ops.matmul(steps, head, act, logits, 0),
                None => {
                    let act = self.ops.act(steps, hidden, 0, rows, d);
                    self.ops.matmul(steps, head, &act, logits, 0);
                }
            },
            None => {
                let head = Weight::F32 { w: self.w(name).clone(), n: v, k: d };
                self.ops.matmul(steps, &head, &self.ops.act_f32(hidden, 0, rows, d), logits, 0);
            }
        }
    }

    /// [`Self::head_logits_rows_dev`] reduced to `rows` greedy picks entirely on
    /// the device (`argmax_part` + `argmax_final`): only the winning indices are
    /// read back, never the `[rows, vocab]` logits.
    pub(crate) fn head_argmax_rows_dev(&self, hidden: &Hidden, rows: u32) -> Vec<u32> {
        let out = self.head_argmax_rows_record(hidden, rows);
        self.gpu.read(&out, rows as usize).into_iter().map(|x| x as u32).collect()
    }

    /// The dispatches of [`Self::head_argmax_rows_dev`] without the readback:
    /// returns the `[rows]` buffer of winning indices, written once the
    /// dispatches have run - so a recording can include the head.
    fn head_argmax_rows_record(&self, hidden: &Hidden, rows: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let logits = self.head_logits_rows_hidden(hidden, rows);
        let chunk = v.div_ceil(HEAD_ARGMAX_CHUNKS);
        let part = g.storage(rows as u64 * HEAD_ARGMAX_CHUNKS as u64 * 2);
        let out = g.storage(rows as u64);
        g.submit(
            &[],
            &[
                g.step(ARGMAX_PART, &[&logits, &part], &[rows, v, HEAD_ARGMAX_CHUNKS, chunk], rows * HEAD_ARGMAX_CHUNKS),
                g.step(ARGMAX_FINAL, &[&part, &out], &[rows, HEAD_ARGMAX_CHUNKS], rows),
            ],
        );
        out
    }

    /// [`Self::head_logits_rows_dev`] reduced to every row's top-`cap` (token id,
    /// logit) candidates, best first, entirely on the device: `cap` rounds of
    /// (`argmax_part`+`argmax_final`, `topk_extract_step`), each masking the
    /// winner out of `logits` before the next - only `rows * cap` pairs are read
    /// back.
    pub(crate) fn head_topk_rows_dev(&self, hidden: &Hidden, rows: u32, cap: u32) -> Vec<Vec<(u32, f32)>> {
        let (vals, idx) = self.head_topk_rows_record(hidden, rows, cap);
        self.read_topk(&vals, &idx, rows, cap)
    }

    /// The dispatches of [`Self::head_topk_rows_dev`] without the readback:
    /// returns the `[rows, cap]` value and index buffers.
    fn head_topk_rows_record(&self, hidden: &Hidden, rows: u32, cap: u32) -> (DeviceBuffer, DeviceBuffer) {
        assert!(cap > 0, "head_topk_rows_dev: cap must be > 0");
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let logits = self.head_logits_rows_hidden(hidden, rows);
        let chunk = v.div_ceil(HEAD_ARGMAX_CHUNKS);
        let part = g.storage(rows as u64 * HEAD_ARGMAX_CHUNKS as u64 * 2);
        let arg = g.storage(rows as u64);
        let vals = g.storage((rows * cap) as u64);
        let idx = g.storage((rows * cap) as u64);
        let mut steps: Vec<Step> = Vec::new();
        for col in 0..cap {
            steps.push(g.step(ARGMAX_PART, &[&logits, &part], &[rows, v, HEAD_ARGMAX_CHUNKS, chunk], rows * HEAD_ARGMAX_CHUNKS));
            steps.push(g.step(ARGMAX_FINAL, &[&part, &arg], &[rows, HEAD_ARGMAX_CHUNKS], rows));
            steps.push(g.step(TOPK_EXTRACT_STEP, &[&arg, &logits, &vals, &idx], &[rows, v, cap, col], rows));
        }
        g.submit(&[], &steps);
        (vals, idx)
    }

    fn read_topk(&self, vals: &DeviceBuffer, idx: &DeviceBuffer, rows: u32, cap: u32) -> Vec<Vec<(u32, f32)>> {
        let vals = self.gpu.read(vals, (rows * cap) as usize);
        let idx = self.gpu.read(idx, (rows * cap) as usize);
        (0..rows as usize)
            .map(|r| {
                let s = r * cap as usize;
                idx[s..s + cap as usize].iter().map(|&x| x as u32).zip(vals[s..s + cap as usize].iter().copied()).collect()
            })
            .collect()
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
    /// (for the previous stage's [`Self::write_out_dres`]) - populated by the
    /// preceding `backward()` call (see [`Self::backward`]'s own doc for why
    /// this is a stashed buffer rather than a `self.dres[..]` array index).
    pub fn read_in_dres(&self) -> Vec<f32> {
        self.gpu.read(&self.dres_boundary_out.borrow(), self.res_numel())
    }
    /// Write this stage's OUTPUT-boundary residual gradient `dres[shard.end]`
    /// (from the next stage's [`Self::read_in_dres`]) - consumed by the next
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

    /// Whether the int8 MoE-expert dispatch (`crate::q8::Qwen35Q8`,
    /// `model::moe::expert_fwd_i8`/`moe_linear_gated_i8.wgsl`) is actually
    /// reachable on this instance: `true` only for an int8-requested build
    /// (`new_i8`/`new_on_i8`) on a device whose caps report
    /// `numeric.int8_dot` (see [`Self::new_impl_on`]'s `i8_on` gate). `false`
    /// for a plain fp32 build AND for an int8-requested build that fell back
    /// to fp32 because the device lacks the packed-dot path - lets a test
    /// observe the gate without reaching into the private `q8` field.
    pub fn moe_int8_active(&self) -> bool {
        self.q8.is_some()
    }

    /// Whether the shared expert rides in the int8 expert banks (as block
    /// `n_experts`) rather than on the fp32 path - see `Qwen35Q8::shared_fits_bank`.
    pub fn moe_shared_in_bank(&self) -> bool {
        self.q8.as_ref().is_some_and(|q| q.moe.iter().all(|m| m.shared_in_bank()))
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
        // "brain/qwen35moe"/"qwen35moe": must NOT collide with `brain/qwen35`
        // family (the dense sibling, `crates/qwen35`) - `model_dir.rs`'s
        // `resident_for` dispatches on this exact family string.
        checkpoint::save_carded(path, config, &tensors, &checkpoint::st::ModelCard::new("brain/qwen35moe", "qwen35moe"));
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
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        Qwen35::adamw_step(self, t, lr, wd, adam, clip, extra_scale)
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

/// Bit-identical gate for [`Qwen35::moe_sublayer_decode_sparse`] -- the
/// decode-only (`n==1`) sparse MoE dispatch this task added. Decode is
/// inference-only (`Qwen35::step` has no backward), so there is no gradcheck
/// entry point here; a direct bit-identical comparison against the dense
/// per-expert loop it replaces is both stronger and cheaper.
#[cfg(test)]
mod decode_sparse_moe_tests {
    use super::*;
    use crate::config::Qwen35Config;

    /// The exact dense per-expert loop `moe_sublayer`'s `else` arm used
    /// (unconditionally, for every `n`) before this task's `n==1` sparse
    /// branch - reimplemented standalone here as the independent baseline,
    /// since `moe_sublayer` itself now always takes the sparse path at
    /// `n==1`. Any future edit to the real dense loop must be mirrored here
    /// or this test stops being an honest baseline.
    fn moe_out_dense_reference(m: &Qwen35, l: usize, xmid: &DeviceBuffer) -> Vec<f32> {
        let g = &m.gpu;
        let c = &m.cfg;
        let n = 1u32;
        let d = c.d_model;
        let e = c.n_experts;
        let moe_ff = c.moe_intermediate_size;
        let shared_ff = c.shared_expert_intermediate_size;
        let p = |s: &str| format!("blocks.{l}.{s}");

        let xn2 = g.storage((n * d) as u64);
        let router_logits = g.storage((n * e) as u64);
        let mut steps = vec![
            rmsnorm_fwd(g, &kernel_ids(), xmid, m.w(&p("ln2.weight")), &xn2, d, n, m.cfg.rms_eps),
            g.step(MATMUL, &[&xn2, m.w(&p("mlp.router.weight")), &router_logits], &[n, d, e], n * e),
        ];
        let shape = MoeShape { rows: n, d_model: d, moe_ff, n_experts: e, top_k: c.top_k };
        let gate = g.storage((n * e) as u64);
        steps.push(router_fwd_kind(
            g,
            &moe_ids(),
            RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: true, routed_scaling: 1.0 },
            &shape,
            &router_logits,
            None,
            &gate,
            None,
        ));

        let moe_acc = g.storage((n * d) as u64);
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
                m.w(&ep("gate.weight")),
                m.w(&ep("up.weight")),
                m.w(&ep("down.weight")),
                &scratch,
                &moe_acc,
                ei,
                0,
                ei != 0,
            ));
        }

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
            m.w(&p("mlp.shared_expert.gate.weight")),
            m.w(&p("mlp.shared_expert.up.weight")),
            m.w(&p("mlp.shared_expert.down.weight")),
            Some(m.w(&p("mlp.shared_expert_gate.weight"))),
            &sh_scratch,
            &moe_acc,
            &moe_out,
        ));

        g.submit(&[], &steps);
        g.read(&moe_out, d as usize)
    }

    fn run(gpu: Gpu) {
        let cfg = Qwen35Config::tiny();
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 11);
        let m = Qwen35::new_on(gpu, cfg.clone(), 1, cfg.block_size, &init);

        // Arbitrary but deterministic single-row hidden state fed straight
        // into `moe_sublayer` -- this isolates the MoE dispatch itself from
        // the rest of the layer stack (attention/GDN), which is not this
        // task's concern.
        let xvals: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.017 - 3.0).sin()).collect();
        let xmid = m.gpu.storage_init("decode_sparse_test_xmid", &xvals);

        for l in 0..cfg.n_layers as usize {
            let dense = moe_out_dense_reference(&m, l, &xmid);
            let (sparse_buf, _) = m.moe_sublayer(l, &xmid, 1);
            let sparse = m.gpu.read(&sparse_buf, d);
            assert_eq!(
                dense.len(),
                sparse.len(),
                "layer {l}: dense/sparse moe_out length mismatch"
            );
            assert_eq!(
                dense, sparse,
                "layer {l}: sparse decode-path MoE dispatch diverged from the dense \
                 per-expert loop it replaces -- must be bit-identical, not just close"
            );
        }
    }

    /// Pin the CPU JIT explicitly regardless of `BRAIN_DEVICE` (a
    /// barrier-crossing kernel can silently misbehave on exactly one
    /// backend), mirroring `tests/decode_step.rs`'s own convention.
    #[test]
    fn moe_sublayer_decode_sparse_matches_dense_loop_bit_identical_cpu() {
        run(Gpu::new_cpu(pipelines()));
    }

    /// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
    /// backend otherwise -- run this under both `BRAIN_DEVICE=cpu` and unset.
    #[test]
    fn moe_sublayer_decode_sparse_matches_dense_loop_bit_identical_default_backend() {
        run(Gpu::new(pipelines()));
    }

    /// The actual claim behind this task ("fewer GPU dispatches per decode
    /// step"), measured via `Gpu::stats()`, not asserted (closing the loop
    /// rather than declaring victory unmeasured), at the REAL 256-expert/top-8 shape
    /// (`Qwen35Config::qwen35_35b_a3b`'s own `n_experts`/`top_k` - the
    /// dispatch COUNT this measures depends only on those two numbers, not on
    /// `d_model`/`vocab`/`n_layers`, so the rest of the config stays
    /// `tiny()`-cheap to build and run in milliseconds on the CPU backend).
    ///
    /// `dispatches` (individual `pass.dispatch_workgroups` calls -- pipeline
    /// bind + launch, the unit "per-dispatch overhead" means here) is the honest
    /// metric here, not `submits`: this engine lazily coalesces every queued
    /// step into ONE real hardware submission at the next readback
    /// (`backend_wgpu::Backend::submit`'s own doc - "a whole forward's
    /// dispatches coalesce into a single queue.submit"), and the ORIGINAL
    /// dense per-expert loop already queued its whole 1280-dispatch layer
    /// through exactly one host-side `Gpu::submit` call, same as the sparse
    /// path below -- `submits` was already 1-vs-1 before this task and stays
    /// that way; it is `dispatches` that this task's whole point is to cut.
    /// See this crate's final report for the exact numbers.
    #[test]
    fn moe_sublayer_decode_sparse_cuts_gpu_dispatches_at_real_expert_scale() {
        let mut cfg = Qwen35Config::tiny();
        cfg.n_experts = 256;
        cfg.top_k = 8;
        cfg.n_layers = 1;
        let gpu = Gpu::new_cpu(pipelines());
        let init = crate::init::init_weights(&cfg, 3);
        let m = Qwen35::new_on(gpu, cfg.clone(), 1, cfg.block_size, &init);
        let d = cfg.d_model as usize;
        let xvals: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.031 + 1.0).cos()).collect();
        let xmid = m.gpu.storage_init("dispatch_count_test_xmid", &xvals);

        let before_dense = m.gpu.stats().expect("cpu backend reports device stats");
        let _dense = moe_out_dense_reference(&m, 0, &xmid);
        let after_dense = m.gpu.stats().unwrap();

        let before_sparse = m.gpu.stats().unwrap();
        let (_sparse_buf, _) = m.moe_sublayer(0, &xmid, 1);
        let after_sparse = m.gpu.stats().unwrap();

        let dense_dispatches = after_dense.dispatches - before_dense.dispatches;
        let sparse_dispatches = after_sparse.dispatches - before_sparse.dispatches;
        println!(
            "moe_sublayer decode dispatch @ n_experts={} top_k={}: dense={dense_dispatches} \
             dispatches, sparse={sparse_dispatches} dispatches ({:.1}x fewer)",
            cfg.n_experts,
            cfg.top_k,
            dense_dispatches as f64 / sparse_dispatches.max(1) as f64
        );
        assert!(
            sparse_dispatches < dense_dispatches,
            "sparse decode dispatch ({sparse_dispatches} dispatches) did not beat the dense \
             per-expert loop ({dense_dispatches} dispatches)"
        );
    }

    /// Real-hardware wall-clock, nice-to-have evidence alongside the
    /// dispatch-count test above: one `moe_sublayer` call, dense vs sparse,
    /// on the real wgpu backend (this box's Tesla P40s) at the real MoE
    /// layer shape (`n_experts=256`, `top_k=8`, `d_model=2048`,
    /// `moe_intermediate_size=512` -- `Qwen35Config::qwen35_35b_a3b`'s own
    /// numbers). `vocab`/`block_size`/`n_layers` stay `tiny()`-small (this
    /// benchmark calls `moe_sublayer` directly, never touching the embed/lm
    /// -head/attention tensors those gate) so the weight upload itself stays
    /// small -- this is real GEMM/dispatch cost at real MoE dimensions, NOT
    /// a full 140 GB fp32 checkpoint load (out of scope for this pass; see
    /// this crate's final report for why). `#[ignore]`d: needs a real GPU
    /// and is a benchmark, not a correctness gate --
    /// `cargo test -p brain-qwen35moe --lib -- --ignored --nocapture
    /// moe_sublayer_decode_sparse_wallclock_at_real_scale_gpu`.
    #[test]
    #[ignore]
    fn moe_sublayer_decode_sparse_wallclock_at_real_scale_gpu() {
        let mut cfg = Qwen35Config::tiny();
        cfg.d_model = 2048;
        cfg.n_experts = 256;
        cfg.top_k = 8;
        cfg.moe_intermediate_size = 512;
        cfg.shared_expert_intermediate_size = 512;
        cfg.n_layers = 1;
        let gpu = Gpu::new(pipelines()); // real default backend (wgpu on this box)
        println!("backend: {}", gpu.kind());
        let init = crate::init::init_weights(&cfg, 5);
        let m = Qwen35::new_on(gpu, cfg.clone(), 1, cfg.block_size, &init);
        let d = cfg.d_model as usize;
        let xvals: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.031 + 1.0).cos()).collect();
        let xmid = m.gpu.storage_init("wallclock_test_xmid", &xvals);

        // Warm up (pipeline compilation, allocator caches) before timing.
        let _ = moe_out_dense_reference(&m, 0, &xmid);
        let (_wsparse, _) = m.moe_sublayer(0, &xmid, 1);
        m.gpu.poll_wait();

        const ITERS: u32 = 5;
        let mut dense_best = f64::INFINITY;
        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            let _ = moe_out_dense_reference(&m, 0, &xmid);
            m.gpu.poll_wait();
            dense_best = dense_best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        let mut sparse_best = f64::INFINITY;
        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            let (out, _) = m.moe_sublayer(0, &xmid, 1);
            m.gpu.poll_wait();
            sparse_best = sparse_best.min(t0.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&out);
        }
        println!(
            "moe_sublayer wall-clock @ n_experts={} top_k={} d_model={} moe_ff={} (best of {ITERS}): \
             dense={dense_best:.2} ms, sparse={sparse_best:.2} ms ({:.1}x)",
            cfg.n_experts,
            cfg.top_k,
            cfg.d_model,
            cfg.moe_intermediate_size,
            dense_best / sparse_best
        );
    }
}

/// Prefill/inference (`n > 1`) grouped-GEMM MoE dispatch (M5.12a) - the
/// batched sibling of `decode_sparse_moe_tests` above, gating
/// `Self::moe_sublayer`'s grouped-forward branch (`MoeGrouped` /
/// `model::moe::expert_fwd_grouped`) against the exact dense per-expert loop
/// it replaced. Unlike the decode test's bit-identical bar, this compares
/// two genuinely different kernels (`moe_linear_gated.wgsl`'s per-row-gated
/// naive accumulation vs `matmul_reg3_grouped.wgsl`'s tiled register GEMM) -
/// the SAME "mathematically equivalent, not textually identical" tolerance
/// category `crates/model/tests/moe_grouped_parity.rs` already establishes
/// for `expert_fwd_grouped` itself (see [`TOLERANCE`]'s own doc for the
/// measured numbers this crate's wiring adds on top).
#[cfg(test)]
mod grouped_moe_tests {
    use super::*;
    use crate::config::Qwen35Config;

    /// The exact dense per-expert loop `moe_sublayer`'s prefill (`n>1`)
    /// `else` arm used before this task's grouped-GEMM branch replaced it -
    /// reimplemented standalone as the independent baseline (mirrors
    /// `decode_sparse_moe_tests::moe_out_dense_reference`'s own doc: any
    /// future edit to the real dense loop must be mirrored here or this test
    /// stops being an honest baseline). Generalized over `n` (this module
    /// only ever calls it with `n>1`; `decode_sparse_moe_tests`'s own copy
    /// stays pinned at the `n==1` decode shape it tests).
    fn moe_out_dense_reference_batched(m: &Qwen35, l: usize, xmid: &DeviceBuffer, n: u32) -> Vec<f32> {
        let g = &m.gpu;
        let c = &m.cfg;
        let d = c.d_model;
        let e = c.n_experts;
        let moe_ff = c.moe_intermediate_size;
        let shared_ff = c.shared_expert_intermediate_size;
        let p = |s: &str| format!("blocks.{l}.{s}");

        let xn2 = g.storage((n * d) as u64);
        let router_logits = g.storage((n * e) as u64);
        let mut steps = vec![
            rmsnorm_fwd(g, &kernel_ids(), xmid, m.w(&p("ln2.weight")), &xn2, d, n, m.cfg.rms_eps),
            g.step(MATMUL, &[&xn2, m.w(&p("mlp.router.weight")), &router_logits], &[n, d, e], n * e),
        ];
        let shape = MoeShape { rows: n, d_model: d, moe_ff, n_experts: e, top_k: c.top_k };
        let gate = g.storage((n * e) as u64);
        steps.push(router_fwd_kind(
            g,
            &moe_ids(),
            RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: true, routed_scaling: 1.0 },
            &shape,
            &router_logits,
            None,
            &gate,
            None,
        ));

        let moe_acc = g.storage((n * d) as u64);
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
                m.w(&ep("gate.weight")),
                m.w(&ep("up.weight")),
                m.w(&ep("down.weight")),
                &scratch,
                &moe_acc,
                ei,
                0,
                ei != 0,
            ));
        }

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
            m.w(&p("mlp.shared_expert.gate.weight")),
            m.w(&p("mlp.shared_expert.up.weight")),
            m.w(&p("mlp.shared_expert.down.weight")),
            Some(m.w(&p("mlp.shared_expert_gate.weight"))),
            &sh_scratch,
            &moe_acc,
            &moe_out,
        ));

        g.submit(&[], &steps);
        g.read(&moe_out, (n * d) as usize)
    }

    /// Relative L2 (`||grouped - dense|| / ||dense||`), NOT an absolute
    /// tolerance: this model's `tiny()` MoE output lands at magnitude
    /// ~1e-9-1e-8 (small `N(0, 0.02)` weights through several unselected-
    /// expert-masked SwiGLU stages), so an absolute bound loose enough to
    /// give real headroom over the two kernels' legitimate rounding
    /// difference would ALSO trivially pass a genuinely wrong wiring, whose
    /// output at this scale is just as small.
    ///
    /// Both numbers below are MEASURED, not assumed - mutation-verified
    /// against swapping the `gate_bank`/`up_bank` arguments into
    /// `expert_fwd_grouped` (a real wiring bug: gate/up feed different
    /// projections of the SwiGLU, `h = silu(gate) * up`).
    ///
    /// Genuine agreement (unmutated): `rel_l2` 1.7e-8 - 4.1e-8 at every layer
    /// (`matmul_reg3_grouped.wgsl`'s tiled reduction order vs
    /// `moe_linear_gated.wgsl`'s per-row-gated naive one - the SAME
    /// "different kernel, same math" tolerance category `crates/model/
    /// tests/moe_grouped_parity.rs`'s own `TOLERANCE` documents).
    ///
    /// The swapped-bank mutation: `rel_l2` 3.6e-4 - 7.3e-4 at every layer -
    /// four orders of magnitude above genuine agreement, but NOT ~1.0: at
    /// `tiny()`'s small `N(0, 0.02)` weight scale `silu(x) ~= x/2` for both
    /// operands, so `silu(gate)*up` and `silu(up)*gate` agree to FIRST order
    /// and only diverge in the second-order term - a smaller effect than a
    /// naive "wrong direction entirely" bug would produce, but still 10000x
    /// the genuine noise floor.
    ///
    /// `1e-5` sits two orders of magnitude above the measured genuine
    /// agreement and one order below the measured mutation - real headroom
    /// on both sides, not a number backed into passing.
    const TOLERANCE: f32 = 1e-5;

    fn run(gpu: Gpu) {
        let cfg = Qwen35Config::tiny();
        let b = 1u32;
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 13);
        let m = Qwen35::new_on(gpu, cfg.clone(), b, t, &init);

        // Arbitrary but deterministic multi-row hidden state (`t=24` rows)
        // fed straight into `moe_sublayer`, isolating the MoE dispatch from
        // the rest of the layer stack exactly like the decode test above.
        let n = (b * t) as usize;
        let xvals: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.013 - 5.0).sin()).collect();
        let xmid = m.gpu.storage_init("grouped_prefill_test_xmid", &xvals);

        for l in 0..cfg.n_layers as usize {
            let dense = moe_out_dense_reference_batched(&m, l, &xmid, b * t);
            let (grouped_buf, _) = m.moe_sublayer(l, &xmid, b * t);
            let grouped = m.gpu.read(&grouped_buf, n * d);
            assert_eq!(dense.len(), grouped.len(), "layer {l}: dense/grouped moe_out length mismatch");

            let mut sq_diff = 0.0f64;
            let mut sq_dense = 0.0f64;
            for (a, bv) in grouped.iter().zip(dense.iter()) {
                sq_diff += ((*a - *bv) as f64).powi(2);
                sq_dense += (*bv as f64).powi(2);
            }
            assert!(sq_dense.sqrt() > 1e-12, "layer {l}: oracle output is all-zero - the test shape routes nothing");
            let rel_l2 = (sq_diff.sqrt() / sq_dense.sqrt()) as f32;
            assert!(
                rel_l2 < TOLERANCE,
                "layer {l}: grouped-GEMM prefill MoE diverged from the dense per-expert \
                 loop it replaces: rel_l2={rel_l2} (tolerance={TOLERANCE}) \
                 grouped[..4]={:?} dense[..4]={:?}",
                &grouped[..4.min(grouped.len())],
                &dense[..4.min(dense.len())],
            );
        }
    }

    /// Pin the CPU JIT explicitly regardless of `BRAIN_DEVICE`, mirroring
    /// `decode_sparse_moe_tests`'s own convention.
    #[test]
    fn moe_sublayer_grouped_prefill_matches_dense_loop_cpu() {
        run(Gpu::new_cpu(pipelines()));
    }

    /// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
    /// backend otherwise - run this under both `BRAIN_DEVICE=cpu` and unset.
    #[test]
    fn moe_sublayer_grouped_prefill_matches_dense_loop_default_backend() {
        run(Gpu::new(pipelines()));
    }

    /// The actual claim behind this task ("fewer GPU dispatches per prefill
    /// layer, at real expert scale"), measured via `Gpu::stats()`, mirroring
    /// `decode_sparse_moe_tests::moe_sublayer_decode_sparse_cuts_gpu_dispatches_at_real_expert_scale`'s
    /// own rationale exactly (`dispatches`, not `submits` - both paths queue
    /// their whole layer through exactly one `Gpu::submit`).
    #[test]
    fn moe_sublayer_grouped_prefill_cuts_gpu_dispatches_at_real_expert_scale() {
        let mut cfg = Qwen35Config::tiny();
        cfg.n_experts = 256;
        cfg.top_k = 8;
        cfg.n_layers = 1;
        let b = 1u32;
        let t = cfg.block_size;
        let gpu = Gpu::new_cpu(pipelines());
        let init = crate::init::init_weights(&cfg, 9);
        let m = Qwen35::new_on(gpu, cfg.clone(), b, t, &init);
        let d = cfg.d_model as usize;
        let n = (b * t) as usize;
        let xvals: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.029 + 2.0).cos()).collect();
        let xmid = m.gpu.storage_init("grouped_dispatch_count_test_xmid", &xvals);

        let before_dense = m.gpu.stats().expect("cpu backend reports device stats");
        let _dense = moe_out_dense_reference_batched(&m, 0, &xmid, b * t);
        let after_dense = m.gpu.stats().unwrap();

        let before_grouped = m.gpu.stats().unwrap();
        let (_grouped_buf, _) = m.moe_sublayer(0, &xmid, b * t);
        let after_grouped = m.gpu.stats().unwrap();

        let dense_dispatches = after_dense.dispatches - before_dense.dispatches;
        let grouped_dispatches = after_grouped.dispatches - before_grouped.dispatches;
        println!(
            "moe_sublayer prefill dispatch @ rows={n} n_experts={} top_k={}: dense={dense_dispatches} \
             dispatches, grouped={grouped_dispatches} dispatches ({:.1}x fewer)",
            cfg.n_experts,
            cfg.top_k,
            dense_dispatches as f64 / grouped_dispatches.max(1) as f64
        );
        assert!(
            grouped_dispatches < dense_dispatches,
            "grouped prefill dispatch ({grouped_dispatches} dispatches) did not beat the dense \
             per-expert loop ({dense_dispatches} dispatches)"
        );
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

    /// The slot really names the coalesced kernel. A registration this model
    /// gets wrong by one index does not fail - it silently dispatches a
    /// DIFFERENT kernel through the RMSNorm bindings.
    #[test]
    fn the_registered_slot_names_the_coalesced_kernel() {
        assert_eq!(STATIC_PIPELINES[kernel_ids().rmsnorm_rows].0, "rmsnorm_rows");
    }

    #[test]
    fn the_decode_tape_norms_match_the_host_reference() {
        let c = crate::config::Qwen35Config::qwen35_35b_a3b();
        // Every `(rows, dim)` a decode step dispatches an RMSNorm at, read off
        // the config rather than written down. Row counts are per SEQUENCE: a
        // batched step multiplies each of them by the batch size, which
        // `block::rms_variant`'s own selector already handles as just a wider
        // dispatch.
        let shapes = [
            (1, c.d_model, "ln1/ln2/final norm: one residual row"),
            (c.n_heads, c.head_dim, "GQA q_norm"),
            (c.n_kv_heads, c.head_dim, "GQA k_norm"),
            (c.linear_num_value_heads, c.linear_value_head_dim, "GDN gated norm"),
        ];
        let gpu = gpu_core::testgpu::dev(pipelines());
        block::assert_rmsnorm_variant_agrees(&gpu, &kernel_ids(), c.rms_eps, &shapes);
    }
}

/// The backward-half twin of the gate above: the coalesced `rmsnorm_dx_rows`
/// this model now selects (via the SAME `block::rms_variant` policy, inside
/// `block::rmsnorm_bwd`) folds the row's two reductions as 64 partials in a
/// different order than `rmsnorm_dx`'s single-threaded double walk, so it
/// agrees to fp32 rounding, not to the bit. Every gradient this model produces
/// flows through it, so the swap is gated numerically against a HOST
/// reference at the shapes this model's own backward tape dispatches.
#[cfg(test)]
mod rmsnorm_dx_variant_agreement {
    use super::*;

    /// The slot really names the coalesced kernel. A registration this model
    /// gets wrong by one index does not fail - it silently dispatches a
    /// DIFFERENT kernel through the RMSNorm-backward bindings.
    #[test]
    fn the_registered_slot_names_the_coalesced_kernel() {
        assert_eq!(STATIC_PIPELINES[kernel_ids().rmsnorm_dx_rows].0, "rmsnorm_dx_rows");
    }

    #[test]
    fn the_backward_tape_norms_match_the_host_reference() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let c = crate::config::Qwen35Config::qwen35_35b_a3b();
        let tiny = crate::config::Qwen35Config::tiny();
        // Every `(rows, dim)` `build_backward_steps` dispatches an RMSNorm
        // backward at, read off the config rather than written down, at a
        // real training microbatch (b=2, t=128) plus the gradcheck fixture's
        // own tiny shape (dims far below the 64-thread workgroup, so the
        // cooperative kernel's idle-lane tail is gated too).
        let n = 2 * 128u32;
        let shapes = [
            (n, c.d_model, "ln1/ln2/final norm at training width"),
            (n * c.n_heads, c.head_dim, "GQA q_norm"),
            (n * c.n_kv_heads, c.head_dim, "GQA k_norm"),
            (n * c.linear_num_value_heads, c.linear_value_head_dim, "GDN gated norm"),
            (12, tiny.d_model, "the gradcheck fixture's block norms"),
            (12 * tiny.n_heads, tiny.head_dim, "the gradcheck fixture's QK-norms"),
        ];
        let gpu = gpu_core::testgpu::dev(pipelines());
        block::assert_rmsnorm_dx_variant_agrees(&gpu, &kernel_ids(), c.rms_eps, &shapes);
    }
}
