// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.8-27B dense hybrid decoder - forward AND backward, text-only.
//! See `crate::config` for the architecture summary.
//!
//! **Scope**: a single whole-sequence prefill forward/backward (`t` must
//! already be a multiple of the derived GDN chunk size - asserted loudly in
//! [`Qwen35::new_impl_on`], see [`gdn_chunk_size`]). Two base construction
//! paths, mirroring `qwen35moe::model` exactly: [`Qwen35::new_on`] builds a
//! frozen (`Role::Frozen`), forward-only instance (`backward`/`zero_grads`/
//! `adamw_step` all assert and panic on such an instance); [`Qwen35::new_train_on`]
//! builds a fully trainable (`Role::Trainable` everywhere, or - when
//! `cfg.lora` is `Some` - frozen base + trainable LoRA adapters) instance
//! whose `forward()` additionally saves the activation cache `backward()`
//! reads (the `train_acts` field's own doc has the exact "one forward, one
//! backward, then the cache is gone" contract). [`Qwen35::new_i8`]/
//! [`Qwen35::new_on_i8`] build the same forward-only shape as `new_on`, but
//! with the 12 per-layer mixer/MLP linears [`is_quantizable_linear`] names
//! quantized to int8 (DP4A) wherever the device's capabilities support it - see
//! [`Qwen35::ops_linear`]'s own doc. [`Qwen35::new_shard`] builds a single
//! pipeline stage (`crate::shard`'s `model::Shardable` impl). Also present:
//! an MTP head (`cfg.mtp`, one extra full-attention decoder layer sharing the
//! main `lm_head`), LoRA/full finetune (`cfg.lora`), a vision-language
//! embedding splice seam (`crate::vl::Qwen35Vl`), single-sequence incremental
//! decode (`Self::step`), and pipeline-parallel sharding (`crate::shard`) -
//! each mirrors the matching piece of `qwen35moe::model`.
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

use gpu_core::select::Dtype;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use paramstore::{ParamStore, Role};

use audio::conv::ConvKernels;
use model::block::{self, rmsnorm_bwd, rmsnorm_fwd, swiglu_bwd, KernelIds};
use model::gdn::{GdnBwdIds, GdnConvIds, GdnIds, GdnShape};
pub use model::gdn::gdn_chunk_size;
use model::ops::{Act, Ops, TierPolicy, Weight};
use model::Shard;
use optim::Optim;

use crate::config::{LayerType, Qwen35Config};

// ---- kernel pipeline (order fixes the indices below) -----------------------
// Forward + backward subset of qwen35moe::model::STATIC_PIPELINES (no MoE -
// this model has none).

const STATIC_PIPELINES: &[(&str, &str)] = &[
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
    // -- LoRA tier -- see `Qwen35::lora_fwd`/`Qwen35::proj_bwd`'s LoRA branch.
    ("axpy", kernels::AXPY), // 73
    // -- vision-language splice tier -- see `Qwen35::enable_mm_splice`.
    ("splice", kernels::SPLICE), // 74
    ("splice_bwd", kernels::SPLICE_BWD), // 75
    // -- single-sequence incremental decode tier -- see `Qwen35::step`.
    ("causal_conv1d_step", kernels::CAUSAL_CONV1D_STEP), // 76
    ("kv_append", kernels::KV_APPEND), // 77
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES), // 78
    ("decode_softmax", kernels::DECODE_SOFTMAX), // 79
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY), // 80
    // -- int8/q4 inference tiers -- see `crate::model::is_quantizable_linear`.
    ("max_abs_row", kernels::MAX_ABS_ROW), // 81
    ("quant_pack", kernels::QUANT_PACK), // 82
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN), // 83
    // -- coalesced RMSNorm -- see `rms_step`.
    ("rmsnorm_rows", kernels::RMSNORM_ROWS), // 84
    // -- device head (greedy argmax / top-k), see `Qwen35::head_argmax_dev`/
    // `Qwen35::head_topk_dev`. Registered LAST so every hand-numbered const
    // above keeps its position.
    ("argmax_part", kernels::ARGMAX_PART), // 85
    ("argmax_final", kernels::ARGMAX_FINAL), // 86
    ("topk_extract_step", kernels::TOPK_EXTRACT_STEP), // 87
    // -- chunked (multi-token-per-dispatch) prefill tier -- see
    // `Qwen35::run_prefill_chunk`. The two `paged_*_batched` kernels were
    // already registered further down in `pipelines()` purely to satisfy
    // `Ops::REQUIRED_KERNELS` ("compiled, never dispatched"); they are named
    // here instead because this crate now genuinely dispatches them - a
    // chunk's queries attending a flat per-sequence KV cache is exactly their
    // degenerate one-block case (see `model::block::gqa_chunk_step`).
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED), // 88
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED), // 89
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED), // 90
    // M2.6: real, dispatched - `gqa_chunk_ids`'s `fused_prefill_hd256` slot,
    // `model::block::gqa_chunk_step`'s single-dispatch alternative to the
    // three kernels just above, at this model's real `head_dim=256`.
    ("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256), // 91
    // The batched KV append `model::block::gqa_decode_batched_step` needs -
    // hand-numbered (rather than appended by name in `pipelines()` below, where
    // it used to sit purely to satisfy `Ops::REQUIRED_KERNELS`) because the
    // batched decode path dispatches it directly, off a `const` index like
    // every other kernel it uses.
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED), // 92
];

/// This model's FULL kernel set: `STATIC_PIPELINES` (every hand-numbered
/// const above indexes into this - unchanged positions 0..83) followed by
/// the `model::ops::Ops` façade's own required kernels, appended with NO
/// named consts of their own - resolved by `Ops::new` purely BY NAME
/// (`Gpu::kernel_index`), never by position. Mirrors `qwen35moe::model::
/// pipelines`'s own recipe (including the `matmul_reg2` -> `matmul_reg3`
/// bit-identical-faster-twin registration) exactly.
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
        // This model has no MoE tier at all (see the module doc), so unlike
        // `qwen35moe::model::STATIC_PIPELINES` - which already carries
        // `moe_linear_gated` for its own routed experts - the plain (f32)
        // `moe_linear_gated` name is never registered above; `Ops::
        // REQUIRED_KERNELS` still demands it (see the bf16/f16 loop below,
        // which only adds the DTYPE VARIANTS, not this base name).
        v.push(("moe_linear_gated", kernels::MOE_LINEAR_GATED));
        // `Ops::REQUIRED_KERNELS` also demands the bf16/f16 storage-tier
        // variants even though this crate never builds a `Weight::BF16`/
        // `Weight::F16` and never dispatches the generic `paged_*_batched`
        // family - see `Ops::new`'s own doc comment ("every model that
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
        // The f32 `paged_decode_{scores,apply}_batched` themselves are in
        // `STATIC_PIPELINES` (this crate dispatches them from its chunked
        // prefill); only their bf16 storage-tier variants, which nothing here
        // ever builds a `Weight` for, are appended by name only.
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
        v
    })
}

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
const AXPY: usize = 73;
const SPLICE: usize = 74;
const SPLICE_BWD: usize = 75;
const CAUSAL_CONV1D_STEP: usize = 76;
const RMSNORM_ROWS: usize = 84;
const ARGMAX_PART: usize = 85;
const ARGMAX_FINAL: usize = 86;
const TOPK_EXTRACT_STEP: usize = 87;
const DECODE_SOFTMAX_BATCHED: usize = 88;
const PAGED_DECODE_SCORES_BATCHED: usize = 89;
const PAGED_DECODE_APPLY_BATCHED: usize = 90;
const PAGED_FLASH_PREFILL_HD256: usize = 91;
const PAGED_KV_APPEND_BATCHED: usize = 92;

/// Two-stage argmax reduction width for [`Qwen35::head_argmax_dev`]/
/// [`Qwen35::head_topk_dev`] - matches `qwen3::serve::Engine`'s own
/// `ARGMAX_CHUNKS`. `Op::ArgMaxRow`'s `SplitReduction` shape is
/// capability-free (no `caps` gate anywhere in that arm - see
/// `backend_api::select`'s own doc), so dispatching it unconditionally here
/// is safe even though this crate carries no `KernelSelector` of its own.
const HEAD_ARGMAX_CHUNKS: u32 = 256;

// M6.10 (kernel-performance ledger) replaced the fixed-schedule constant
// that used to live here (`DRAIN_EVERY_N_LAYERS`, a periodic BLOCKING drain
// every 4 layers regardless of actual buffer liveness) with real liveness
// tracking in `Qwen35::run_prefill_chunk_stage`'s layer loop: a
// `gpu_core::scratch::Arena` scope per layer, with the device drain moved to
// exactly the point that scope's buffer reuse requires one, rather than a
// count of layers. See that function's own comments for the mechanism and
// the ledger entry for the measured before/after.

/// Rows at or above which a chunk round draws its per-layer temporaries from
/// [`gpu_core::scratch::Arena`] - **and therefore pays that arena's drain**, a
/// blocking `poll_wait` at every layer boundary.
///
/// The arena trades allocations for a fence. A round's per-layer temporaries
/// are `[n, d_model]`, so what it saves scales with `n` while what it costs -
/// one device fence per layer, which stops the host's recording of layer `l+1`
/// from overlapping the card's execution of layer `l` - does not scale with `n`
/// at all. That makes the trade a function of the row count and nothing else,
/// which is why this is a threshold rather than a policy.
///
/// Below this, no scope is opened, so `Gpu::storage` allocates exactly as it
/// did before any caller opted in - the arena's own documented behaviour
/// outside a scope - which means no buffer is ever recycled and hence the drain
/// the arena's contract asks for is not owed. The small-round path is then
/// structurally the decode tape's: record a layer, `flush`, keep going. That is
/// not a relaxed correctness argument, it is the absence of the condition the
/// drain exists for. (An unpooled round still owes RECLAIM, which is a
/// different obligation and is met on bytes - see the layer loop.)
///
/// **Measured**, real 27B across two P40s, layer-stack ms per round, pooled
/// divided by unpooled - above 1 the pool is losing. The qwen35 ledger's M29
/// records how these were taken, and
/// `tests/gguf_resident_verify_cost_real::the_verify_round_cost_ladder`
/// re-derives the whole row on demand:
///
/// | rows |   1  |   3  |   7  |   8  |  16  |  32  |  64  |  128 |  256 |
/// |------|------|------|------|------|------|------|------|------|------|
/// | pooled/unpooled | 1.68 | 1.66 | 1.44 | 1.36 | 1.10 | 1.10 | 1.04 | 0.96 | 0.92 |
///
/// The fence costs 1.4-1.7x at exactly the row counts a speculative verify
/// round uses (`k + 1`, so 2-8), and the curve crosses over between 64 and 128.
/// Anywhere in 16..=64 is within measurement noise of the same choice and no
/// production caller runs a round in that range anyway - a verify round is
/// `k + 1`, a prompt round is `MAX_PREFILL_TOKENS`, and only a ragged final
/// prefill chunk lands between. 16 is the conservative end of that window: it
/// keeps MORE rounds on the pool, and the pool is what bounds allocation churn
/// (M6.10's actual purpose, whose stability claim is not re-measured here).
const CHUNK_ARENA_MIN_ROWS: u32 = 16;

/// This model's RMSNorm epsilon. Exactly what `rmsnorm.wgsl` hardcodes, but it
/// has to be passed explicitly here - see [`rms_step`].
const RMS_EPS: f32 = 1e-6;

/// One RMSNorm through `block::rms_variant`: the cooperative workgroup-per-row
/// kernel (`rmsnorm_rows`) where the device can run a workgroup reduction, the
/// per-element `rmsnorm` otherwise.
///
/// Used by the DECODE tape ([`Qwen35::run_decode_step`] and the two mixer step
/// functions), which is also this model's per-token PREFILL tape - the resident
/// replays a prompt one token at a time through the same primitive, so this one
/// seam covers every RMSNorm a served request pays for.
///
/// Why it was worth a seam: `rmsnorm.wgsl` gives thread `t` row `t`, so at the
/// `rows = 1` of a decode step it runs a 5120-element reduction on ONE thread
/// of a 3840-core card, with every 32-byte sector fetched serving a single
/// useful float. Measured on the real two-card Qwen3.8-27B resident it was the
/// TOP row of the decode profile at 48% of all device time - more than the
/// entire 27 GB int8 weight stream underneath it - at 210 calls and 117 ms per
/// token. `rmsnorm_rows` walks the row with 64 threads instead.
///
/// The epsilon is passed explicitly even though 1e-6 is what `rmsnorm.wgsl`
/// hardcodes, because the two kernels must share one `Params` layout and the
/// cooperative one reads a third `eps` field; a two-field list would hand it
/// whatever the uniform happened to hold. (Same reasoning, same shape, as
/// `minimaxmusic3::depth_decoder`'s own `rms_step`.)
///
/// This is NOT a bit-identical swap - the 64 partial sums fold in a different
/// order, agreeing to ~3e-6 max_abs - which is exactly why it lives at the call
/// site behind `rms_variant` rather than in `gpu_core::upgrade`, whose bar is
/// bit-identity.
pub fn rms_step(g: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32) -> Step {
    let (kind, threads) = block::rms_variant(g, RMSNORM, Some(RMSNORM_ROWS), rows, dim);
    g.dispatch(kind, &[x, w, out], &[dim, rows, f(RMS_EPS)], threads)
}

pub(crate) fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
        rmsnorm_dx_rows: block::UNREGISTERED,
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
        rmsnorm_rows: block::UNREGISTERED,
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

/// [`model::gdn::gdn_causal_conv1d_step`]'s kernel id - the streaming
/// causal-conv decode step, dispatched by
/// [`model::gdn_mixer::gdn_mixer_decode_fwd`] in place of a whole-sequence
/// `conv1d_fwd`.
fn gdn_conv_ids() -> GdnConvIds {
    GdnConvIds { causal_conv1d_step: CAUSAL_CONV1D_STEP }
}

/// [`model::block::gqa_decode_batched_step`]'s kernel ids - the
/// CROSS-SEQUENCE sibling of [`gqa_chunk_ids`], dispatched by
/// [`Qwen35::run_decode_batch`]: every resident sequence's one new token
/// appended to, and attended against, ONE shared paged KV pool per layer in a
/// single dispatch set.
fn gqa_decode_batched_ids() -> model::block::GqaDecodeBatchedIds {
    model::block::GqaDecodeBatchedIds {
        kv_append_batched: PAGED_KV_APPEND_BATCHED,
        scores_batched: PAGED_DECODE_SCORES_BATCHED,
        softmax_batched: DECODE_SOFTMAX_BATCHED,
        apply_batched: PAGED_DECODE_APPLY_BATCHED,
    }
}

/// [`model::gdn_mixer::gdn_mixer_decode_fwd`]'s decode-only kernel ids.
fn gdn_mixer_decode_ids() -> model::gdn_mixer::GdnMixerDecodeIds {
    model::gdn_mixer::GdnMixerDecodeIds { conv: gdn_conv_ids(), splice: SPLICE }
}

/// [`model::block::gqa_chunk_step`]'s kernel ids - the CHUNKED sibling of
/// [`gqa_decode_ids`], dispatched by [`Qwen35::layer_gqa_prefill_chunk`]:
/// many query rows appended to and attended against the same flat per-layer
/// KV cache in one dispatch triad.
fn gqa_chunk_ids() -> model::block::GqaChunkIds {
    model::block::GqaChunkIds {
        splice: SPLICE,
        scores_batched: PAGED_DECODE_SCORES_BATCHED,
        softmax_batched: DECODE_SOFTMAX_BATCHED,
        apply_batched: PAGED_DECODE_APPLY_BATCHED,
        // M2.6: this model's real head_dim (256) has a real fused kernel -
        // `gqa_chunk_step` dispatches it instead of the triad above whenever
        // `Op::PagedAttentionFused`'s selector agrees (`caps.
        // workgroup_reductions`; always true on a real GPU, never on the CPU
        // JIT, which keeps the triad exactly as before this existed).
        fused_prefill_hd256: Some(PAGED_FLASH_PREFILL_HD256),
    }
}

/// Everything [`Qwen35::layer_gdn_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GDN mixer arm. `internals` is the LoRA-agnostic
/// mixer math's own saved state (`model::gdn_mixer`, shared with
/// `crates/qwen35moe`); `gated` is this crate's own `out_proj` input (the
/// hoisted forward's return value), kept here since only the LOCAL
/// `out_proj` backward reads it - see `model::gdn_mixer`'s own module doc.
struct GdnLayerActs {
    internals: model::gdn_mixer::GdnMixerActs,
    gated: DeviceBuffer,
}

/// What one CHUNKED-prefill round hands [`Qwen35::layer_gqa_fwd`] so its GQA
/// layers attend the sequence's persistent KV cache instead of an isolated
/// `[T,T]` causal block over the round alone. Every field is built ONCE per
/// round by [`Qwen35::run_prefill_chunk`] and shared unchanged by every GQA
/// layer in it (`kcache`/`vcache` excepted - those are per layer).
struct GqaChunkCtx<'a> {
    /// Where this sequence's KV rows START in the bound buffer: `0` for a flat
    /// dedicated cache, `phys * cap` for a window of a shared pool
    /// (`DecodeCaches::gqa_base_row`).
    base_row: u32,
    /// Absolute position of this round's FIRST token (`0` on round 1).
    start: u32,
    /// The per-sequence KV cache row capacity (`DecodeCaches::gqa_cap`).
    cap: u32,
    kcache: &'a DeviceBuffer,
    vcache: &'a DeviceBuffer,
    /// `[n]` u32, every entry `base_row / cap` - the single-block table this
    /// sequence's KV window is, see `model::block::gqa_chunk_step`.
    block_ids: DeviceBuffer,
    /// `[n]` u32 with `seq_lens[i] == start+i+1`: this round's causal mask.
    seq_lens: DeviceBuffer,
    /// This round's own `[n, rotary_dim/2]` M-RoPE tables, for absolute
    /// positions `start..start+n`.
    cos: &'a DeviceBuffer,
    sin: &'a DeviceBuffer,
}

/// What one BATCHED DECODE step hands [`Qwen35::layer_gqa_fwd`] - the
/// cross-sequence counterpart of [`GqaChunkCtx`]. `paged`/`cos`/`sin` are built
/// ONCE per step and shared unchanged by every GQA layer; the two pools are per
/// layer, so they are rebound each time round the layer loop.
struct GqaDecodeCtx<'a> {
    paged: &'a model::gqa_mixer::PagedDecodeBatch<'a>,
    pool_k: &'a DeviceBuffer,
    pool_v: &'a DeviceBuffer,
    /// `[bsz, rotary_dim/2]` M-RoPE tables - row `b` is sequence `b`'s OWN
    /// decode position, and the rows of one batch are unrelated positions.
    cos: &'a DeviceBuffer,
    sin: &'a DeviceBuffer,
}

/// Which recurrent-state shape a [`Qwen35::layer_gdn_fwd`] call is running in:
/// the Gated-DeltaNet counterpart of [`GqaCached`], and the same three-way
/// split - a whole-sequence forward that starts from zero state, one sequence's
/// prefill round continuing its own state, or one decode token for each of
/// several independent sequences.
enum GdnCall<'a> {
    Whole,
    Chunk(model::gdn_mixer::GdnStream<'a>),
    Decode(&'a [model::gdn_mixer::GdnStream<'a>]),
}

/// Which cached-KV shape a [`Qwen35::layer_gqa_fwd`] call is running in.
/// `None` is the isolated `[T,T]` causal block a training/whole-sequence
/// forward wants; the two `Some` arms are the two ways a SERVING pass attends
/// a persistent cache - one sequence's many tokens ([`GqaChunkCtx`]) or many
/// sequences' one token each ([`GqaDecodeCtx`]).
enum GqaCached<'a> {
    Chunk(&'a GqaChunkCtx<'a>),
    Decode(&'a GqaDecodeCtx<'a>),
}

/// Everything [`Qwen35::layer_gqa_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GQA mixer arm. `internals`/`ctx_gated` split the
/// same way as [`GdnLayerActs`] - see that struct's own doc.
struct GqaLayerActs {
    internals: model::gqa_mixer::GqaMixerActs,
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

/// Everything the MTP head's forward saves for its own backward -
/// DeepSeek-V3-style: normalize the next token's own embedding and the main
/// stack's final hidden state independently, fuse them with a `fc_e`/`fc_h`
/// projection pair, run through ONE standard full-attention decoder layer
/// (`mtp.layers.0.*` - same shape as any other `Full` block, reusing
/// [`Qwen35::layer_gqa_fwd`]/[`Qwen35::mlp_fwd`] via the `"mtp.layers.0.*"`
/// prefix), then `mtp.norm` and the SHARED `lm_head`. No reference oracle
/// exists for this head on this box (`transformers`' own loader discards
/// `mtp.*` on load) - structural only, gradchecked and overfit-tested, never
/// parity-claimed.
struct MtpActs {
    /// Raw next-token embedding (pre `pre_fc_norm_embedding`).
    e: DeviceBuffer,
    /// `norm(e, "mtp.pre_fc_norm_embedding.weight")` - the `fc_e` matmul's input.
    en: DeviceBuffer,
    /// `norm(res[last], "mtp.pre_fc_norm_hidden.weight")` - the `fc_h` matmul's input.
    hn: DeviceBuffer,
    /// `fc_e(en) + fc_h(hn)` - the one extra layer's residual-stream INPUT
    /// (`mlp_fwd`/`layer_gqa_fwd`'s doc calls this role `xres`).
    ehp: DeviceBuffer,
    /// The one extra full-attention decoder layer's own saved activations -
    /// same shape [`LayerTrainActs`] gives every other layer.
    layer: LayerTrainActs,
    /// The layer's output (post both residual adds), pre `mtp.norm`.
    block_out: DeviceBuffer,
    /// `norm(block_out, "mtp.norm.weight")` - the shared `lm_head` matmul's input.
    final_h: DeviceBuffer,
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
    /// `new_on` path) keeps inference-only behaviour byte-for-byte with the
    /// model's original, training-free forward implementation.
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

    // ---- multi-token prediction head (`cfg.mtp`) ---------------------------
    // Size-1 dummies when `cfg.mtp` is false, matching this file's own
    // "size-1 dummy where a value doesn't apply" convention.
    /// Next-token input for the MTP embedding gather (`x` shifted by 1),
    /// written by [`Self::set_batch`].
    mtp_input: DeviceBuffer,
    /// MTP's own target (`x` shifted by 2 - predicts token `t+2`), written
    /// by [`Self::set_batch`].
    mtp_target: DeviceBuffer,
    mtp_logits: DeviceBuffer,
    mtp_ce_buf: DeviceBuffer,
    /// MTP's own activation cache - see [`MtpActs`]'s own doc. `Some` only
    /// right after a `forward()` call on a `cfg.mtp` training build, same
    /// "forward reallocates fresh, backward takes it" contract as
    /// `train_acts`.
    mtp_acts: RefCell<Option<MtpActs>>,

    /// The `model::ops` façade driving every per-layer mixer/MLP linear's
    /// dispatch - see [`Self::ops_linear`]'s own doc.
    ops: Ops,
    /// Every per-layer mixer/MLP linear [`is_quantizable_linear`] names, as a
    /// `model::ops::Weight` - `F32` unless a `TierPolicy` asked for a
    /// narrower tier AND the device caps support it, in which case
    /// `Weight::upload` promotes it to that tier (see [`Self::ops_linear`]).
    weights: HashMap<String, Weight>,

    // ---- LoRA scratch (persistent, reused across every targeted linear) ----
    // Sized once at construction for `cfg.lora`'s rank and the widest output
    // dimension across the 12 targetable leaves (`crate::config::
    // lora_targets`). See [`Self::lora_fwd`]/[`Self::proj_bwd`]'s LoRA
    // branch. Size-1 dummies when `cfg.lora` is `None` (rank forced to 1 in
    // `new_impl_on`, never read) - mirrors `qwen35moe::model::Qwen35`'s own
    // `lora_a`/`lora_da`/`lora_out` fields exactly.
    /// `[n*r]` : `a = x @ Aᵀ`.
    lora_a: DeviceBuffer,
    /// `[n*r]` : grad wrt `a`.
    lora_da: DeviceBuffer,
    /// `[n*max_out]` : `delta = a @ Bᵀ`.
    lora_out: DeviceBuffer,

    // ---- vision-language embedding splice seam (see `crate::vl::Qwen35Vl`) ------
    /// `Some((row0, n_rows))` once [`Self::enable_mm_splice`] has run: the
    /// image-placeholder rows `run_forward` overwrites and `backward` routes
    /// to [`Self::read_d_img_embeds`]. `None` (the default) makes both a
    /// pure no-op, matching `qwen35moe::model::Qwen35`'s own splice seam.
    mm_splice: Cell<Option<(u32, u32)>>,
    /// `[n_rows*d_model]` : the projected image tokens to splice in on the
    /// next `forward()`. 1-element dummy until `enable_mm_splice` resizes it.
    img_embeds: DeviceBuffer,
    /// `[n_rows*d_model]` : gradient of the spliced image tokens after
    /// `backward()`. 1-element dummy until `enable_mm_splice` resizes it.
    d_img_embeds: DeviceBuffer,

    // ---- single-sequence incremental decode (see `Self::step`) ------------
    // Mirrors `qwen35moe::model::Qwen35`'s own decode-cache fields exactly
    // (structure, not code - this crate's own orchestration is prefix-
    // parameterized instead of layer-index-parameterized, matching the MTP
    // head's `layer_gqa_fwd`/`mlp_fwd` convention, and reuses the SAME already-
    // shared `model::block::gqa_decode_step`/`model::gdn::{gdn_recurrent_step,
    // gdn_causal_conv1d_step}` primitives qwen35moe's own decode step calls -
    // only the per-model orchestration around them (weight lookups, the
    // layer loop) is duplicated, the low-level cache-append/attend and
    // recurrent-state kernels are not).
    /// The absolute position the next [`Self::step`] will decode.
    dec_pos: Cell<u32>,
    /// Decode capacity - this instance's own fixed prefill length `t` (no
    /// independent "max decode length" constructor parameter, same
    /// simplification qwen35moe's own `dec_cap` doc explains).
    dec_cap: u32,
    /// Per-layer plain (non-paged) KV cache for GQA layers, `[dec_cap,
    /// kv_dim]`; a size-1 dummy at GDN layer indices.
    gqa_kcache: Vec<DeviceBuffer>,
    gqa_vcache: Vec<DeviceBuffer>,
    /// Per-layer persistent Gated DeltaNet recurrent state, `[bh, dk, dv]`,
    /// for GDN layers; a size-1 dummy at GQA layer indices. Threaded across
    /// `step` calls; zeroed by [`Self::reset_decode_cache`].
    gdn_state: Vec<DeviceBuffer>,
    /// Per-layer persistent causal-conv history ring buffer, `[1, conv_dim,
    /// K-1]`, for GDN layers; a size-1 dummy at GQA layer indices.
    gdn_hist: Vec<DeviceBuffer>,

    // ---- pipeline-parallel cross-stage seam (`model::Shardable`) ----------
    /// This stage's upstream gradient at `res[shard.end]`, written externally
    /// by [`Self::write_out_dres`] before a non-head stage's `backward()`.
    dres_boundary_in: DeviceBuffer,
    /// This stage's gradient at `res[shard.start]`, refreshed by every
    /// `backward()` call, read externally by [`Self::read_in_dres`].
    dres_boundary_out: RefCell<DeviceBuffer>,
    /// Which layers/endpoints this instance owns - `Shard::whole` for every
    /// constructor except [`Self::new_shard`].
    pub shard: Shard,

    // ---- hidden-state tap (see `Self::set_hidden_taps`) --------------------
    /// ABSOLUTE decoder-layer indices whose post-MLP residual a chunk pass
    /// should copy back to the host. Empty (the default) makes the tap a
    /// compiled-in `is_empty()` check per layer and nothing else.
    taps: RefCell<Vec<usize>>,
    /// What the last chunk pass tapped, `(layer, [n, d_model])` in layer
    /// order. Drained by [`Self::take_hidden_taps`].
    tapped: RefCell<Vec<(usize, Vec<f32>)>>,

    /// This instance's [`CHUNK_ARENA_MIN_ROWS`] - see
    /// [`Self::set_chunk_arena_min_rows`].
    chunk_arena_min_rows: Cell<u32>,
}

/// Which per-sequence GQA cache / GDN recurrent state one [`Qwen35::
/// run_decode_step`] call reads and updates - this instance's own persistent
/// decode state (`self.gqa_kcache`/`self.gqa_vcache`/`self.gdn_state`/
/// `self.gdn_hist`), threaded across `step` calls exactly as before this
/// struct existed. `pub(crate)` since a future paged serving engine (see
/// `qwen35moe::serve::Engine`'s own precedent) would own a SEPARATE cache/
/// slot per admitted request and need to say "run one decode step, but
/// against THIS request's own state" - the same reason `qwen35moe::model::
/// Qwen35::run_decode_step` takes this as an explicit parameter rather than
/// always reading `self.*` directly.
///
/// Every field is indexed by absolute layer index `l` (length `cfg.n_layers`),
/// with a size-1 dummy buffer at the layer indices that don't apply to that
/// field - the SAME "every layer index has a plain buffer, dummy where
/// irrelevant" convention this struct's own `gqa_kcache`/`gdn_state` fields
/// already use.
pub(crate) struct DecodeCaches<'a> {
    /// Per-layer `[cap, kv_dim]` KV cache for GQA layers (dummy at GDN
    /// indices) - `model::block::gqa_decode_step`'s own `kcache`/`vcache`.
    pub gqa_kcache: &'a [DeviceBuffer],
    pub gqa_vcache: &'a [DeviceBuffer],
    /// Cache row capacity, shared by every GQA layer's cache in this call
    /// (one per-sequence capacity, not a per-layer one).
    pub gqa_cap: u32,
    /// Where this sequence's `gqa_cap` rows START in the bound buffer. `0` -
    /// the only value every caller but a paged serving engine ever uses - means
    /// the buffers above are this sequence's own dedicated flat cache. A
    /// serving engine that keeps ONE pool per layer instead
    /// (`crate::serve::Engine`) binds the whole pool and sets this to
    /// `physical_block * gqa_cap`; it must be a whole multiple of `gqa_cap`,
    /// since it is also what the paged attention kernels' single-entry block
    /// table is derived from.
    pub gqa_base_row: u32,
    /// Per-layer Gated-DeltaNet recurrent `state`/conv `hist` for GDN layers
    /// (dummy at GQA indices) - `gdn_recurrent_step`'s `state`,
    /// `gdn_causal_conv1d_step`'s `hist`.
    pub gdn_state: &'a [DeviceBuffer],
    pub gdn_hist: &'a [DeviceBuffer],
}

/// One sequence's coordinates in a [`Qwen35::run_decode_batch`] call: where
/// its KV history lives, where this step's token goes, and its own
/// Gated-DeltaNet state. The batched counterpart of the per-sequence half of
/// [`DecodeCaches`]: the GQA POOL is shared by the whole batch
/// ([`BatchDecodeCaches`]), this is everything that is not.
pub(crate) struct BatchSeq<'a> {
    /// The physical block backing this sequence's whole KV history, so its
    /// rows are `phys*gqa_cap .. +gqa_cap` of every layer's pool.
    pub phys: u32,
    /// Absolute decode position of this step's token for this sequence. The
    /// rows of one batch are unrelated positions.
    pub pos: u32,
    /// Per-layer recurrent state / conv history, indexed by ABSOLUTE layer
    /// index with a dummy at full-attention indices - [`DecodeCaches`]'s own
    /// convention, and the same buffers.
    pub gdn_state: &'a [DeviceBuffer],
    pub gdn_hist: &'a [DeviceBuffer],
}

/// What one [`Qwen35::run_decode_batch`] call reads and updates: the shared
/// per-layer paged KV pool plus one [`BatchSeq`] per sequence.
pub(crate) struct BatchDecodeCaches<'a> {
    /// Per-layer `[num_blocks*gqa_cap, kv_dim]` KV pool for the full-attention
    /// layers (a dummy at GDN indices). A sequence's own `[gqa_cap, kv_dim]`
    /// window is rows `phys*gqa_cap..`; a caller with one dedicated cache per
    /// sequence is the `num_blocks = 1`, `phys = 0` case.
    pub gqa_kpool: &'a [DeviceBuffer],
    pub gqa_vpool: &'a [DeviceBuffer],
    /// Pool rows one physical block spans - the per-sequence KV capacity, and
    /// the paged kernels' `block_size`.
    pub gqa_cap: u32,
    /// One entry per batch row, in the same order as the `tokens` argument.
    pub seqs: &'a [BatchSeq<'a>],
}

/// Cold copies of every Gated-DeltaNet layer's recurrent state and conv
/// history, taken before a speculative verify pass so a rejected tail can be
/// undone - see [`Qwen35::spec_decode_greedy`] and the section comment above
/// it for why the recurrent layers need this and the GQA layers do not.
/// Indexed by ABSOLUTE layer index, with 1-word dummies at GQA indices, the
/// same convention as every other per-layer cache here.
pub struct GdnSnapshot {
    pub(crate) state: Vec<DeviceBuffer>,
    pub(crate) hist: Vec<DeviceBuffer>,
}

/// What a speculative run actually cost, which is the only way to know
/// whether the drafter was worth running. `accepted / rounds` is the number
/// that decides it: the expected tokens per round is that plus one (the
/// target's own correction or bonus token always lands), against the one
/// token per forward plain decoding gets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecDecodeStats {
    /// Verify chunks dispatched, plus the re-commit chunks a partial
    /// rejection forces. This is the cost, in units of "one plain decode
    /// step", because a memory-bound decode's chunk of `k+1` rows streams the
    /// same weights a single row does.
    pub target_forwards: usize,
    /// Draft tokens offered across all rounds.
    pub proposed: usize,
    /// Draft tokens the target agreed with, i.e. the ones that were NOT
    /// re-derived from a target forward of their own.
    pub accepted: usize,
    /// Verify rounds run.
    pub rounds: usize,
}

impl SpecDecodeStats {
    /// Mean draft tokens accepted per round - the model/workload-dependent
    /// number that determines whether speculative decoding pays here at all.
    pub fn accepted_per_round(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.accepted as f64 / self.rounds as f64
    }
}

/// The device a shard runs on: `shard.gpu_index`'s canonical physical card,
/// or the ambient selection for [`Shard::ANY_GPU`]. Written once so every
/// shard constructor places the same way (they used to carry a byte-identical
/// copy each).
fn shard_gpu(shard: &Shard) -> Gpu {
    if shard.gpu_index == Shard::ANY_GPU {
        Gpu::new(pipelines())
    } else {
        Gpu::new_on_index(shard.gpu_index as u32, pipelines()).unwrap_or_else(|e| panic!("qwen35 shard placement: {e}"))
    }
}

/// The parameter subset a shard holds. A whole shard returns `cfg.param_list()`
/// verbatim (so the single-device store is byte-identical). A partial shard
/// keeps only its layers' weights, plus `tok.weight` when it embeds and/or
/// carries the tied head, and `norm.weight`+head when it is the head stage.
/// Mirrors `qwen35moe::model::shard_param_list` exactly, adapted for this
/// config's `"blocks.{l}."`-prefixed naming.
///
/// `pub(crate)` because it is also THE definition of "which tensors must a
/// loader supply for this stage": `crate::int8_gguf_resident` builds its
/// `checkpoint::remap::RemapSource` fetch plan from exactly this list (and
/// validates the plan against its numels), so the loader and
/// [`Qwen35::new_i8_shard`] can never disagree about a stage's tensor set.
pub(crate) fn shard_param_list(cfg: &Qwen35Config, shard: &Shard) -> Vec<(String, usize)> {
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
            if name.starts_with("mtp.") {
                return shard.head; // MTP is forced whole-shard-only, see new_impl_on's assert
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

/// Names one of this model's 12 per-layer quantizable linears (5 GDN +
/// 4 GQA + 3 MLP leaves) regardless of prefix - matches both the main
/// decoder stack (`blocks.{l}.{linear_attn,self_attn,mlp}.*`) and the MTP
/// head's own single extra layer (`mtp.layers.0.{self_attn,mlp}.*`, see
/// `Self::run_mtp_forward` - it reuses `layer_gqa_fwd`/`mlp_fwd` unchanged,
/// so it must be quantized identically or `ops_linear` would panic looking
/// up a name `Self::new_impl_on` never uploaded). Embeddings, norms, the LM
/// head, and (on a LoRA build) `.lora_a`/`.lora_b` adapters always stay fp32
/// - none of their leaf names appear here.
pub(crate) fn is_quantizable_linear(name: &str) -> bool {
    const LEAVES: &[&str] = &[
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "in_proj_b.weight",
        "in_proj_a.weight",
        "out_proj.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate.weight",
        "up.weight",
        "down.weight",
    ];
    LEAVES.iter().any(|leaf| name.ends_with(leaf))
}

impl Qwen35 {
    pub fn new_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), false, shard)
    }

    /// [`Self::new_on`] with the int8 (DP4A) inference tier: the 12
    /// per-layer mixer/MLP linears [`is_quantizable_linear`] names are
    /// quantized (see that function's own doc); embeddings, norms and the LM
    /// head stay fp32. Inference-only, same as the fp32 path
    /// (`Qwen35::backward` panics regardless). A thin alias over
    /// [`Self::new_shard_dt`] - see that function's own doc for the general
    /// per-leaf case this specializes.
    pub fn new_i8(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_shard_dt(cfg, b, t, init, shard, &TierPolicy::uniform(Dtype::I8))
    }

    /// [`Self::new_i8`] on an existing device handle - see [`Self::new_on`].
    /// A thin alias over [`Self::new_on_dt`].
    pub fn new_on_i8(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_on_dt(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::I8))
    }

    /// [`Self::new_on`] at a per-leaf [`TierPolicy`] on an EXISTING device
    /// handle - [`Self::new_shard_dt`]'s direct-`Gpu` sibling. Exists because
    /// `new_shard_dt` always derives its own device from the shard
    /// (`shard_gpu`'s ambient selection), which cannot express "force this
    /// backend" the way a test comparing CPU-JIT vs GPU behaviour at the
    /// same tier needs (`Qwen35::new_on`/`new_on_i8` already had this
    /// property; a bare `Dtype`-parameterised tier deserves the same one).
    pub fn new_on_dt(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, tier: &TierPolicy) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, tier, false, shard)
    }

    /// Build a fully trainable instance (every weight `Role::Trainable`,
    /// full-parameter, or - when `cfg.lora` is `Some` - frozen base +
    /// trainable LoRA adapters). See the struct's own `is_train` doc.
    pub fn new_train_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), true, shard)
    }

    /// Build a single pipeline **stage**: only the layers (and endpoint
    /// weights) in `shard` are allocated on this device, as a TRAINABLE
    /// build (`Role::Trainable` full-parameter, or - when `cfg.lora` is
    /// `Some` - frozen base + trainable LoRA adapters). `shard.gpu_index`
    /// names the canonical physical card; `Shard::ANY_GPU` keeps the ambient
    /// selection. Mirrors `qwen35moe::model::Qwen35::new_shard` exactly (see
    /// `crate::shard`'s `model::Shardable` impl, the only caller this is
    /// meant for outside tests). `cfg.mtp` requires a whole shard - see this
    /// function's own assert.
    pub fn new_shard(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> Qwen35 {
        Qwen35::new_impl_on(shard_gpu(&shard), cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), true, shard)
    }

    /// Build a single pipeline **stage** at a per-leaf [`TierPolicy`] rather
    /// than one uniform [`Dtype`] - the general case [`Self::new_i8_shard`]
    /// and [`Self::new_fp32_shard_src`] both specialize (M24: a `TierPolicy`
    /// generalizes the `i8: bool` this crate used to take, so "Q4 on the MLP,
    /// F32 on the GDN decay/beta gates" is expressible without a third
    /// constructor per combination). Inference-only when `tier.
    /// quantizes_anything()` - see [`Self::new_impl_on`]'s own assert.
    pub fn new_shard_dt(
        cfg: Qwen35Config,
        b: u32,
        t: u32,
        src: &dyn checkpoint::TensorSource,
        shard: Shard,
        tier: &TierPolicy,
    ) -> Qwen35 {
        Qwen35::new_impl_on(shard_gpu(&shard), cfg, b, t, src, tier, false, shard)
    }

    /// [`Self::new_shard`]'s int8-INFERENCE sibling (`i8 = true`, `train =
    /// false`): one pipeline stage of a quantized model, placed on
    /// `shard.gpu_index`'s physical card exactly as `new_shard` does for
    /// training. This is what a multi-GPU resident serving path builds, one
    /// instance per card, driving each stage's decode with
    /// [`Self::run_decode_step`]'s `input_override` seam.
    ///
    /// `src` is any [`checkpoint::TensorSource`] rather than the
    /// `&HashMap<String, Vec<f32>>` the training constructors take, so a real
    /// checkpoint loads straight from its own mmap with no whole-model fp32
    /// intermediate - `checkpoint::gguf::MmapGguf` implements the trait and
    /// dequantizes Q8_0 to f32 on read. A `HashMap` still works (it
    /// implements the trait too), which is what tests pass.
    ///
    /// `cfg.mtp` requires a whole shard, asserted in `new_impl_on` - shared
    /// with every other constructor, so it covers this one unchanged.
    pub fn new_i8_shard(cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, shard: Shard) -> Qwen35 {
        Qwen35::new_shard_dt(cfg, b, t, src, shard, &TierPolicy::uniform(Dtype::I8))
    }

    /// [`Self::new_i8_shard`]'s **fp32** sibling: the same stage, the same
    /// [`checkpoint::TensorSource`] streaming load, weights kept at full
    /// precision.
    ///
    /// Exists so a quantized stage can be compared against an
    /// otherwise-identical unquantized one built from the SAME real
    /// checkpoint bytes - the only reference available for "does int8 still
    /// track fp32 on REAL weights, through a real stack of layers", which
    /// per-leaf cosine (`tests/int8_real_weight_sanity.rs`) and tiny-config
    /// parity (`tests/model_i8_smoke.rs`) both leave open. A truncated
    /// `cfg.n_layers` makes that comparison fit one card at 27B dims; see
    /// `tests/gguf_i8_vs_fp32_real.rs`.
    ///
    /// It is not a serving path: at the real depth the fp32 weights are
    /// ~108 GB.
    pub fn new_fp32_shard_src(cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, shard: Shard) -> Qwen35 {
        Qwen35::new_shard_dt(cfg, b, t, src, shard, &TierPolicy::uniform(Dtype::F32))
    }

    /// [`Self::new_fp32_shard_src`]'s TRAIN-mode sibling: identical weights,
    /// shape and real-checkpoint source, but `is_train: true` so
    /// `run_forward` actually saves `GdnMixerActs`/`GqaMixerActs` per layer
    /// instead of discarding them. Never used to actually train (nothing
    /// here calls `backward()` or touches the Adam buffers this allocates) -
    /// it exists purely so [`Self::debug_gdn_trace`] can read back a GDN
    /// layer's own internal per-step activations on a real checkpoint, the
    /// same "expose it for parity debugging" reason [`Self::debug_res`]
    /// exists for the residual stream.
    pub fn new_fp32_shard_src_train(cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, shard: Shard) -> Qwen35 {
        Qwen35::new_impl_on(shard_gpu(&shard), cfg, b, t, src, &TierPolicy::uniform(Dtype::F32), true, shard)
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
        assert!(
            !(tier.quantizes_anything() && train),
            "qwen35: a quantized tier is inference-only (Qwen35::new_train_on is fp32-only)"
        );
        let chunk = gdn_chunk_size(t);
        assert_eq!(
            t % chunk,
            0,
            "qwen35: t={t} is not a multiple of the derived GDN chunk size {chunk} -- \
             model::gdn is prefill-only (no T-padding support, see its module doc); \
             gdn_chunk_size always returns a value that divides t by construction, so \
             this assert failing would mean a logic error in gdn_chunk_size itself"
        );
        assert!(
            !cfg.mtp || shard.is_whole(cfg.n_layers as usize),
            "qwen35: MTP + pipeline sharding is not yet supported - the MTP head needs \
             res[n_layers] and the shared lm_head, both only valid on a whole shard"
        );

        // Role assignment:
        //  - inference (`!train`): every weight Role::Frozen (no grad/Adam
        //    buffers allocated at all).
        //  - LoRA training (`train && cfg.lora.is_some()`): only the
        //    `.lora_a`/`.lora_b` adapter tensors `Qwen35Config::param_list`
        //    added for each targeted leaf are Trainable; every other weight
        //    (including a LoRA-targeted leaf's own frozen base) is Frozen -
        //    mirrors `qwen35moe::model::Qwen35::new_impl_on`'s own role
        //    filter exactly.
        //  - full training (`train && cfg.lora.is_none()`): every weight
        //    Role::Trainable (full-parameter backward).
        // A leaf `tier.want`s at any non-F32 dtype lives in `weights`
        // (below), NOT the fp32 store - filter it out here so no redundant
        // fp32 copy is ever uploaded. A quantized tier and LoRA/training are
        // mutually exclusive (the `assert!` above), so this filter and
        // `cfg.lora.is_some()` never both fire here.
        let roles: Vec<(String, usize, Role)> = shard_param_list(&cfg, &shard)
            .into_iter()
            .filter(|(n, _)| !(is_quantizable_linear(n) && tier.want(n) != Dtype::F32))
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

        // Per-layer mixer/MLP linears: every layer this shard owns gets its
        // own leaves (GDN: in_proj_{qkv,z,b,a}/out_proj; GQA: {q,k,v,o}_proj;
        // MLP: gate/up/down - `Qwen35Config::layer_leaves`, the SAME table
        // `layer_weight_bytes` folds for its byte-cost estimate, so the two
        // cannot drift the way a hand-transcribed formula once did - lesson
        // #68) as a `model::ops::Weight`, built ONCE here. `tier.want(name)`
        // asks `Weight::upload` for that leaf's own dtype, streaming straight
        // from `src` for anything non-F32 (these names are excluded from
        // `ps` above); the F32 arm wraps a `.clone()` of the buffer `ps`
        // already holds (a cheap `Arc` bump), so an all-F32 policy costs no
        // extra VRAM or re-upload - unchanged from the old `i8: bool`'s
        // `false` arm. Mirrors `qwen35moe::model::Qwen35::new_impl_on`'s own
        // `weights` construction exactly (that crate has no per-leaf policy
        // yet - a separate milestone, see `TierPolicy`'s own doc).
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("qwen35: Ops::new: {e}"));
        let mut weights: HashMap<String, Weight> = HashMap::new();
        let mut upload = |name: String, wn: u32, wk: u32| {
            let want = tier.want(&name);
            let w = if want == Dtype::F32 {
                Weight::F32 { w: ps.w(&name).clone(), n: wn, k: wk }
            } else {
                let mut built: Option<Weight> = None;
                let found = src.with_tensor(&name, &mut |raw| {
                    built = Some(Weight::upload(&ops, raw, wn as usize, wk as usize, want));
                });
                if !found {
                    panic!("qwen35: missing init weight {name}");
                }
                built.unwrap()
            };
            weights.insert(name, w);
        };
        for (l, ty) in cfg.layer_types().iter().enumerate() {
            if !shard.owns(l) {
                continue;
            }
            for (suffix, n, k) in cfg.layer_leaves(*ty) {
                upload(format!("blocks.{l}.{suffix}"), n, k);
            }
        }
        if cfg.mtp && shard.head {
            // MTP's single extra layer is architecturally a GQA layer + MLP
            // (`Self::run_mtp_forward` reuses `layer_gqa_fwd`/`mlp_fwd`
            // unchanged) - `layer_leaves(LayerType::Full)` names the exact
            // same 7 leaves, just under `mtp.layers.0.` instead of
            // `blocks.{l}.`.
            for (suffix, n, k) in cfg.layer_leaves(LayerType::Full) {
                upload(format!("mtp.layers.0.{suffix}"), n, k);
            }
        }

        // LoRA scratch (rank `r`; max projection output across all 12
        // targetable leaves - `crate::config::lora_targets` - mirrors
        // `qwen35moe::model::Qwen35::new_impl_on`'s own sizing, plus
        // `intermediate_size` for this model's dense-MLP `gate`/`up` leaves,
        // which qwen35moe never targets). `.max(1)` so a `cfg.lora: None`
        // build still gets a valid (unused) 1-element rank.
        let lora_r = cfg.lora.as_ref().map(|l| l.rank as u64).unwrap_or(0).max(1);
        let lora_max_out = cfg
            .linear_conv_dim()
            .max(cfg.linear_value_dim())
            .max(cfg.linear_num_value_heads)
            .max(cfg.d_model)
            .max(cfg.q_proj_dim())
            .max(cfg.kv_dim())
            .max(cfg.intermediate_size) as u64;
        let lora_a = gpu.storage((b * t) as u64 * lora_r);
        let lora_da = gpu.storage((b * t) as u64 * lora_r);
        let lora_out = gpu.storage((b * t) as u64 * lora_max_out);

        let ones_khd = gpu.storage_init("qwen35.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);

        // Text-only: every axis of the M-RoPE table carries the same plain
        // sequential position, reset per sequence (row = batch*t + pos).
        let positions: Vec<[u32; 3]> = (0..b).flat_map(|_| (0..t).map(|ti| [ti, ti, ti])).collect();
        // `yarn_scaling()` is `None` unless `cfg.rope_scaling` is set, which
        // makes this call bit-for-bit today's plain `mrope_tables` output -
        // see `Qwen35Config::yarn_scaling`'s own doc.
        let yarn = cfg.yarn_scaling();
        let (cos, sin) = qwen3vl::mrope::mrope_tables_scaled(
            &positions,
            cfg.mrope_section,
            cfg.rotary_dim(),
            cfg.rope_theta,
            yarn.as_ref().map(|(f, a)| (f.as_slice(), *a)),
        );
        let cos = gpu.storage_init("qwen35.rope_cos", &cos);
        let sin = gpu.storage_init("qwen35.rope_sin", &sin);

        let n = (b * t) as u64;
        let tokens = gpu.storage(n);
        let targets = gpu.storage(n);
        // A non-head shard never reads `logits` (`run_forward` only writes it
        // `if self.shard.head`) - on an inference build this is `n * vocab`
        // (248320) of dead VRAM per non-head card.
        let logits = gpu.storage(if shard.head { n * cfg.vocab as u64 } else { 1 });
        let ce_buf = gpu.storage(n);
        let ce_grad_uni = gpu.uniform_dynamic(4);
        let d = cfg.d_model as u64;
        let res = RefCell::new((0..=cfg.n_layers).map(|_| gpu.storage(n * d)).collect());

        let mtp_input = gpu.storage(if cfg.mtp { n } else { 1 });
        let mtp_target = gpu.storage(if cfg.mtp { n } else { 1 });
        let mtp_logits = gpu.storage(if cfg.mtp { n * cfg.vocab as u64 } else { 1 });
        let mtp_ce_buf = gpu.storage(if cfg.mtp { n } else { 1 });

        let img_embeds = gpu.storage(1);
        let d_img_embeds = gpu.storage(1);

        // Single-sequence incremental decode state - see `Qwen35::step`'s doc
        // and the struct fields' own docs for what each buffer holds.
        // `dec_cap = t`: this pass's decode capacity is this instance's own
        // fixed prefill length (see `dec_cap`'s own doc).
        let kv_dim = cfg.kv_dim() as u64;
        let mut gqa_kcache = Vec::with_capacity(cfg.n_layers as usize);
        let mut gqa_vcache = Vec::with_capacity(cfg.n_layers as usize);
        let mut gdn_state = Vec::with_capacity(cfg.n_layers as usize);
        let mut gdn_hist = Vec::with_capacity(cfg.n_layers as usize);
        let (gdn_state_len, gdn_hist_len) = (cfg.gdn_state_len(), cfg.gdn_hist_len());
        // A layer this shard does not own never runs its decode step
        // (`run_decode_step` loops only `shard.start..shard.end`), so its
        // cache/state/history buffers are dead weight - a full-size GQA KV
        // cache or GDN state per un-owned layer, on every non-owning shard.
        for (l, ty) in cfg.layer_types().into_iter().enumerate() {
            if !shard.owns(l) {
                gqa_kcache.push(gpu.storage(1));
                gqa_vcache.push(gpu.storage(1));
                gdn_state.push(gpu.storage(1));
                gdn_hist.push(gpu.storage(1));
                continue;
            }
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

        // Backward-pass boundary scratch for gradient checkpointing across
        // shard seams - only meaningful when this instance actually trains.
        let dres_boundary_in = gpu.storage(if train { n * d } else { 1 });
        let dres_boundary_out = RefCell::new(gpu.storage(if train { n * d } else { 1 }));

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
            mtp_input,
            mtp_target,
            mtp_logits,
            mtp_ce_buf,
            mtp_acts: RefCell::new(None),
            ops,
            weights,
            lora_a,
            lora_da,
            lora_out,
            mm_splice: Cell::new(None),
            img_embeds,
            d_img_embeds,
            dec_pos: Cell::new(0),
            dec_cap: t,
            gqa_kcache,
            gqa_vcache,
            gdn_state,
            gdn_hist,
            dres_boundary_in,
            dres_boundary_out,
            shard,
            taps: RefCell::new(Vec::new()),
            tapped: RefCell::new(Vec::new()),
            chunk_arena_min_rows: Cell::new(CHUNK_ARENA_MIN_ROWS),
        }
    }

    /// **Move this instance's scratch-arena threshold**, the row count at or
    /// above which [`Self::run_prefill_chunk_stage`] pools its per-layer
    /// temporaries and pays the per-layer device drain that pooling requires.
    /// Defaults to [`CHUNK_ARENA_MIN_ROWS`].
    ///
    /// Two behaviours exist behind one function, and both are production
    /// paths, so both have to be reachable by a test. The equivalence gates
    /// that own this function's correctness (`tests/chunked_prefill.rs`,
    /// `two_shard_chunked_prefill_matches_token_by_token_replay`) run at
    /// `tiny()` dims, whose whole `block_size` is far below any useful
    /// threshold - so without this they would only ever exercise the unpooled
    /// side, and the pooled side would be gated by nothing. Setting this to 1
    /// is what lets them assert the same claim on both.
    ///
    /// It is not a knob for a serving caller to tune: the default is measured
    /// (see [`CHUNK_ARENA_MIN_ROWS`]) and a caller that moves it is choosing
    /// against a measurement, not for one.
    pub fn set_chunk_arena_min_rows(&self, rows: u32) {
        assert!(rows > 0, "qwen35::set_chunk_arena_min_rows: 0 would open a scope for an empty round; 1 is 'always pool'");
        self.chunk_arena_min_rows.set(rows);
    }

    /// **Ask a chunk pass to hand back the residual leaving specific layers.**
    ///
    /// `layers` are ABSOLUTE decoder-layer indices; ones outside this stage's
    /// own `shard` range are simply never reached, so a caller driving a
    /// pipeline hands the same list to every stage and each answers for its
    /// own range. The tapped value is the post-MLP residual - the standard
    /// "hidden state after decoder layer `l`", and exactly what HF's
    /// `output_hidden_states=True` returns at tuple index `l + 1`.
    ///
    /// This exists for cross-model consumers of the target's INTERNALS, of
    /// which the DFlash2 drafter (`crate::dflash2`) is the first: it
    /// cross-attends to the concatenated residuals of five specific layers,
    /// which no amount of reading this model's OUTPUT can reconstruct. It is
    /// deliberately not an env-var dump like `QWEN35_DEBUG_LAYER_RMS` beside
    /// it - a drafter needs the values in-process, every round, not printed.
    ///
    /// Costs one `[n, d_model]` readback per tapped layer per chunk when the
    /// list is non-empty, and one `Vec::is_empty` per layer when it is not.
    pub fn set_hidden_taps(&self, layers: &[usize]) {
        *self.taps.borrow_mut() = layers.to_vec();
        self.tapped.borrow_mut().clear();
    }

    /// Take what the last chunk pass tapped, `(absolute layer, `[n, d_model]`)`
    /// in the order the layers ran. Draining rather than borrowing, so a
    /// caller cannot accidentally read a previous round's rows.
    pub fn take_hidden_taps(&self) -> Vec<(usize, Vec<f32>)> {
        std::mem::take(&mut *self.tapped.borrow_mut())
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    /// True if `name` has a gradient buffer (i.e. is optimised). Frozen
    /// parameters have none, so their weight-gradient dispatches must be
    /// skipped - on a LoRA build this is exactly the targeted leaves' own
    /// frozen base weights (their `.lora_a`/`.lora_b` adapters are the only
    /// trainable tensors; see [`Self::new_impl_on`]'s role filter). Mirrors
    /// `qwen35moe::model::Qwen35::trainable` exactly.
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    /// `Some((rank, alpha/rank))` if `leaf` (one of `crate::config::
    /// lora_targets`'s 12 names) has a LoRA adapter configured; `None`
    /// otherwise (either `cfg.lora` is unset, or this leaf isn't targeted -
    /// e.g. `"fc_e"`/`"fc_h"`/`"lm_head"`, which are never LoRA-targetable).
    /// Mirrors `qwen35moe::model::Qwen35::lora_for` exactly.
    fn lora_for(&self, leaf: &str) -> Option<(u32, f32)> {
        self.cfg.lora.as_ref().filter(|lc| lc.targets_leaf(leaf)).map(|lc| (lc.rank, lc.alpha / lc.rank as f32))
    }

    /// Forward LoRA delta for a targeted linear: `y += (alpha/r)*(x*A^t)*B^t`.
    /// No-op for an untargeted leaf. `m`x`k` is the input, `nout` the output -
    /// mirrors `qwen35moe::model::Qwen35::lora_fwd` exactly (same two-matmul +
    /// `AXPY` fusion, using this file's own persistent `lora_a`/`lora_out`
    /// scratch).
    #[allow(clippy::too_many_arguments)]
    fn lora_fwd(&self, s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, wname: &str, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let Some((r, scale)) = self.lora_for(leaf) else { return };
        let g = &self.gpu;
        let a = format!("{wname}.lora_a");
        let bnm = format!("{wname}.lora_b");
        s.push(g.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
        s.push(g.step(MATMUL, &[&self.lora_a, self.w(&bnm), &self.lora_out], &[m, r, nout], m * nout));
        s.push(g.step(AXPY, &[y, &self.lora_out], &[m * nout, f(scale)], m * nout));
    }

    /// Dispatch one of the 12 per-layer mixer/MLP linears via `self.ops`/
    /// `self.weights`. Returns whether the dispatch was `F32` (LoRA only
    /// ever targets an unquantized base weight, so a caller only runs
    /// `lora_fwd` when this is `true`). Mirrors `qwen35moe::model::Qwen35::
    /// ops_linear` exactly.
    fn ops_linear(&self, s: &mut Vec<Step>, act: &Act, wname: &str, out: &DeviceBuffer) -> bool {
        let w = self.weights.get(wname).unwrap_or_else(|| panic!("qwen35: no Ops weight for {wname}"));
        self.ops.matmul(s, w, act, out, 0);
        matches!(w, Weight::F32 { .. })
    }

    /// The dtype leaf `name` actually landed at after
    /// `want.promote(caps)` - what a test asserts to confirm a per-leaf
    /// [`TierPolicy`] was really PLACED, not silently collapsed to uniform
    /// (`Weight::dtype()`'s own doc). `None` for a name this instance never
    /// uploaded as an `Ops` weight (norms, embeddings, the LM head, and any
    /// leaf `tier` left at F32 - see [`Self::w`] for those).
    pub fn linear_dtype(&self, name: &str) -> Option<Dtype> {
        self.weights.get(name).map(Weight::dtype)
    }

    /// The activation an [`Self::ops_linear`] dispatch reads, packed for int8
    /// only where an int8 weight will actually read it. Mirrors
    /// `qwen3::model::Qwen::ops_act` exactly, and for the same reason:
    /// `Ops::act`'s packing is two dispatches plus an `I8Scratch` allocation
    /// per activation, all of it dead on an fp32 build, and a DECODE tape
    /// pays that per layer per token (prefill amortizes it over `t` rows,
    /// which is why the prefill helpers still call `Ops::act` directly). The
    /// tier is read off the weights this instance actually holds, not off a
    /// remembered "int8 was requested" flag - `Weight::upload` demotes to
    /// fp32 on a device without the DP4A path, and such a build should get
    /// the cheap activation.
    fn ops_act(&self, s: &mut Vec<Step>, x: &DeviceBuffer, rows: u32, k: u32) -> Act {
        if self.weights.values().all(|w| matches!(w, Weight::F32 { .. })) {
            return self.ops.act_f32(x, 0, rows, k);
        }
        self.ops.act(s, x, 0, rows, k)
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

        if self.cfg.mtp {
            // MTP predicts token t+2 from hidden_t + embed(x[t+1]). Per
            // sequence (b seqs of `self.t`): input = x shifted +1, target =
            // x shifted +2 - same construction as `glmdsa::model`'s own MTP
            // input/target shift.
            let seqlen = self.t as usize;
            let bsz = self.b as usize;
            let mut inp = vec![0u32; tokens.len()];
            let mut tgt = vec![model::IGNORE; tokens.len()];
            for s in 0..bsz {
                for ti in 0..seqlen {
                    let i = s * seqlen + ti;
                    inp[i] = if ti + 1 < seqlen { tokens[s * seqlen + ti + 1] } else { 0 };
                    tgt[i] = if ti + 2 < seqlen { tokens[s * seqlen + ti + 2] } else { model::IGNORE };
                }
            }
            self.gpu.write(&self.mtp_input, &inp);
            self.gpu.write(&self.mtp_target, &tgt);
        }
    }

    /// Backward for a (possibly-LoRA) linear `y = x*Wt`. Accumulates the input
    /// gradient into `dx` (flag `acc`). For a full weight: `dW += d_out^t*x`
    /// (skipped when `wname` is Frozen - a LoRA-mode base, or an untargeted
    /// weight under a LoRA build), `dx = d_out*W`. For a LoRA-targeted leaf:
    /// the base weight is always frozen (dX only, no dW), and the adapter
    /// grads `gA`/`gB` are produced (scale folded into the private
    /// `lora_a`/`lora_da` scratch). `leaf` is the bare leaf name (e.g.
    /// `"q_proj"`, `"gate"`) `crate::config::lora_targets` matches against -
    /// mirrors `qwen35moe::model::Qwen35::proj_bwd` exactly.
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, steps: &mut Vec<Step>, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        let g = &self.gpu;
        match self.lora_for(leaf) {
            Some((r, scale)) => {
                // base: dx += d_out*W (frozen weight - no dW).
                steps.push(g.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
                let a = format!("{wname}.lora_a");
                let bnm = format!("{wname}.lora_b");
                // a = (alpha/r)*(x*A^t)  -> gB += d_out^t*a
                steps.push(g.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
                steps.push(g.step(GRAD_SCALE, &[&self.lora_a], &[m * r, f(scale)], m * r));
                steps.push(g.step(MATMUL_DW, &[d_out, &self.lora_a, self.g(&bnm)], &[m, r, nout], nout * r));
                // da = (alpha/r)*(d_out*B) -> gA += da^t*x ; dx += da*A
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

    // ---- vision-language embedding splice seam (see `crate::vl::Qwen35Vl`) ----

    /// Enable the VLM embedding splice at residual rows `[row0, row0+n_rows)`:
    /// after the token-embedding gather, `run_forward` overwrites those rows
    /// with the image tokens written via [`Self::write_img_embeds`], and - on
    /// a `new_train_on` build - `backward` routes their gradient to
    /// [`Self::read_d_img_embeds`] (zeroing them in the residual grad first so
    /// `EMB_BWD` never trains the image-placeholder token id). `run_forward`/
    /// `backward` already build their step lists fresh on every call, so
    /// enabling the splice is pure buffer allocation + a flag - call once
    /// after construction, before the first `forward()`. Mirrors
    /// `qwen35moe::model::Qwen35::enable_mm_splice` exactly.
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

    /// Number of spliced image-embedding elements (`n_rows*d_model`); 0 if off.
    fn img_numel(&self) -> usize {
        self.mm_splice.get().map_or(0, |(_, n)| (n * self.cfg.d_model) as usize)
    }

    /// Read the gradient of the spliced image embeddings after `backward()` -
    /// feeds the vision tower/connector backward. Requires a `new_train_on`
    /// build (see [`Self::backward`]'s splice-gradient step).
    pub fn read_d_img_embeds(&self) -> Vec<f32> {
        self.gpu.read(&self.d_img_embeds, self.img_numel())
    }

    /// Overwrite the M-RoPE `cos`/`sin` tables (`[b*t, rotary_dim/2]`
    /// row-major - see `qwen3vl::mrope::{get_rope_index, mrope_tables}` for
    /// how to build them from real 2-D image-grid positions) for the next
    /// `forward()`. RoPE here is unconditionally table-driven already - this
    /// simply replaces the plain-sequential-position table built at
    /// construction.
    pub fn write_mrope_tables(&self, cos: &[f32], sin: &[f32]) {
        self.gpu.write_f32(&self.cos, cos);
        self.gpu.write_f32(&self.sin, sin);
    }

    // ---- one Gated DeltaNet (Linear) layer --------------------------------

    /// One Gated DeltaNet layer over `n` rows. `cont` is `None` for the
    /// whole-sequence (training / `logits_all`) forward, where the sequence
    /// starts fresh and `self.b`/`self.t`/`self.chunk` describe it; `Some` for
    /// one round of a CHUNKED prefill ([`Self::run_prefill_chunk`]), where the
    /// rows are one sequence's `n` next tokens continuing from a persistent
    /// recurrent state and conv history - see `model::gdn_mixer::GdnStream`.
    fn layer_gdn_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32, call: GdnCall) -> (DeviceBuffer, Option<GdnLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let conv_dim = c.linear_conv_dim();
        let value_dim = c.linear_value_dim();
        let nvh = c.linear_num_value_heads;
        let khd = c.linear_key_head_dim;
        let vhd = c.linear_value_head_dim;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // in_proj_qkv (LoRA/int8 dispatch stays local - see `model::gdn_mixer`'s
        // own module doc). `xn1` quantized once here (`Ops::act`), reused
        // unchanged by in_proj_b/a/z below (no further `act` call on `xn1`
        // happens in between).
        let mixed_qkv = g.storage((n * conv_dim) as u64);
        let mut s1 = Vec::new();
        let act1 = self.ops_act(&mut s1, xn1, n, d);
        if self.ops_linear(&mut s1, &act1, &p("in_proj_qkv.weight"), &mixed_qkv) {
            self.lora_fwd(&mut s1, "in_proj_qkv", xn1, &p("in_proj_qkv.weight"), &mixed_qkv, n, d, conv_dim);
        }
        g.submit(&[], &s1);

        // in_proj_b/a/z (LoRA/int8 dispatch stays local). Reuses `act1`
        // (xn1 quantized once, shared) - no fresh `Ops::act` call needed.
        let bproj = g.storage((n * nvh) as u64);
        let aproj = g.storage((n * nvh) as u64);
        let z = g.storage((n * value_dim) as u64);
        {
            let mut s = Vec::new();
            if self.ops_linear(&mut s, &act1, &p("in_proj_b.weight"), &bproj) {
                self.lora_fwd(&mut s, "in_proj_b", xn1, &p("in_proj_b.weight"), &bproj, n, d, nvh);
            }
            if self.ops_linear(&mut s, &act1, &p("in_proj_a.weight"), &aproj) {
                self.lora_fwd(&mut s, "in_proj_a", xn1, &p("in_proj_a.weight"), &aproj, n, d, nvh);
            }
            if self.ops_linear(&mut s, &act1, &p("in_proj_z.weight"), &z) {
                self.lora_fwd(&mut s, "in_proj_z", xn1, &p("in_proj_z.weight"), &z, n, d, value_dim);
            }
            g.submit(&[], &s);
        }

        // conv+split+l2norm+decay-gate+recurrence+gated-norm - LoRA/dtype-
        // agnostic, shared with `crates/qwen35moe` (`model::gdn_mixer`).
        // A chunked round is ONE sequence's `n` rows, with its own chunk size
        // (`n` is a round's length, unrelated to this instance's construction
        // -time `t`); a batched decode step is `n` DIFFERENT sequences' one
        // token each; the whole-sequence forward keeps this instance's shape.
        let gdn_shape = match &call {
            GdnCall::Whole => GdnShape { b: self.b, h: nvh, t: self.t, dk: khd, dv: vhd, chunk: self.chunk },
            GdnCall::Chunk(_) => GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: gdn_chunk_size(n) },
            GdnCall::Decode(_) => GdnShape { b: n, h: nvh, t: 1, dk: khd, dv: vhd, chunk: 1 },
        };
        let shape = model::gdn_mixer::GdnMixerShape { gdn: gdn_shape, nkh: c.linear_num_key_heads, conv_kernel: c.linear_conv_kernel_dim };
        let weights = model::gdn_mixer::GdnMixerWeights {
            conv1d_weight: self.w(&p("conv1d.weight")),
            a_log: self.w(&p("A_log")),
            dt_bias: self.w(&p("dt_bias")),
            norm_weight: self.w(&p("norm.weight")),
            ones_khd: &self.ones_khd,
        };
        let (gated, internals) = match call {
            GdnCall::Decode(streams) => (
                model::gdn_mixer::gdn_mixer_decode_fwd(g, &gdn_mixer_ids(), &gdn_mixer_decode_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, streams),
                None,
            ),
            GdnCall::Whole => model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, n, self.is_train, None),
            GdnCall::Chunk(cont) => {
                model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, n, self.is_train, Some(cont))
            }
        };

        // out_proj (LoRA/int8 dispatch stays local). Fresh `Ops::act` call:
        // `gated` is a different activation from `xn1` above.
        let out = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            let act3 = self.ops_act(&mut s, &gated, n, value_dim);
            if self.ops_linear(&mut s, &act3, &p("out_proj.weight"), &out) {
                self.lora_fwd(&mut s, "out_proj", &gated, &p("out_proj.weight"), &out, n, value_dim, d);
            }
            g.submit(&[], &s);
        }

        let acts = internals.map(|internals| GdnLayerActs { internals, gated });
        (out, acts)
    }

    // ---- one GQA (Full) layer ----------------------------------------------

    /// `prefix` is the weight-name prefix up to (not including) the leaf
    /// name - `"blocks.{l}.self_attn"` for a normal layer, `"mtp.layers.0.
    /// self_attn"` for the MTP head's own full-attention sublayer -
    /// reusing this function unchanged, since the mechanism is identical.
    fn layer_gqa_fwd(&self, prefix: &str, xn1: &DeviceBuffer, n: u32, cached: Option<GqaCached>) -> (DeviceBuffer, Option<GqaLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, kvd) = (c.q_proj_dim(), c.kv_dim());
        let p = |s: &str| format!("{prefix}.{s}");

        // q/k/v proj (LoRA/int8 dispatch stays local - see `model::gqa_mixer`'s
        // own module doc). xn1 quantized once (`Ops::act`), shared by q/k/v.
        let q_full = g.storage((n * qpd) as u64);
        let k = g.storage((n * kvd) as u64);
        let v = g.storage((n * kvd) as u64);
        let mut s1 = Vec::new();
        let act1 = self.ops_act(&mut s1, xn1, n, d);
        if self.ops_linear(&mut s1, &act1, &p("q_proj.weight"), &q_full) {
            self.lora_fwd(&mut s1, "q_proj", xn1, &p("q_proj.weight"), &q_full, n, d, qpd);
        }
        if self.ops_linear(&mut s1, &act1, &p("k_proj.weight"), &k) {
            self.lora_fwd(&mut s1, "k_proj", xn1, &p("k_proj.weight"), &k, n, d, kvd);
        }
        if self.ops_linear(&mut s1, &act1, &p("v_proj.weight"), &v) {
            self.lora_fwd(&mut s1, "v_proj", xn1, &p("v_proj.weight"), &v, n, d, kvd);
        }
        g.submit(&[], &s1);

        // split+qknorm+rope+attention+gating - LoRA/dtype-agnostic, shared
        // with `crates/qwen35moe` (`model::gqa_mixer`). Both SERVING arms
        // supply their OWN M-RoPE table (real absolute positions, not `0..t`)
        // and attend the persistent KV cache instead of an isolated `[T,T]`
        // causal block - see `GqaCached`.
        let shape = model::gqa_mixer::GqaMixerShape { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: c.rotary_dim() / 2 };
        let (cos, sin) = match &cached {
            None => (&self.cos, &self.sin),
            Some(GqaCached::Chunk(ch)) => (ch.cos, ch.sin),
            Some(GqaCached::Decode(dc)) => (dc.cos, dc.sin),
        };
        let weights = model::gqa_mixer::GqaMixerWeights { q_norm: self.w(&p("q_norm.weight")), k_norm: self.w(&p("k_norm.weight")), cos, sin };
        let (ctx_gated, internals) = match cached {
            None => model::gqa_mixer::gqa_mixer_fwd(g, &gqa_mixer_ids(), &shape, &weights, &q_full, &k, &v, n, self.is_train),
            Some(GqaCached::Chunk(ch)) => (
                model::gqa_mixer::gqa_mixer_chunk_fwd(
                    g,
                    &gqa_mixer_ids(),
                    &gqa_chunk_ids(),
                    &shape,
                    &weights,
                    &q_full,
                    &k,
                    &v,
                    n,
                    ch.base_row,
                    ch.start,
                    ch.cap,
                    ch.kcache,
                    ch.vcache,
                    &ch.block_ids,
                    &ch.seq_lens,
                ),
                None,
            ),
            Some(GqaCached::Decode(dc)) => (
                model::gqa_mixer::gqa_mixer_decode_batched_fwd(
                    g,
                    &gqa_mixer_ids(),
                    &gqa_decode_batched_ids(),
                    &shape,
                    &weights,
                    &q_full,
                    &k,
                    &v,
                    dc.pool_k,
                    dc.pool_v,
                    n,
                    dc.paged,
                ),
                None,
            ),
        };

        // o_proj (LoRA/int8 dispatch stays local). Fresh `Ops::act` call:
        // `ctx_gated` is a different activation from `xn1` above.
        let out = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            let act2 = self.ops_act(&mut s, &ctx_gated, n, shape.qd());
            if self.ops_linear(&mut s, &act2, &p("o_proj.weight"), &out) {
                self.lora_fwd(&mut s, "o_proj", &ctx_gated, &p("o_proj.weight"), &out, n, shape.qd(), d);
            }
            g.submit(&[], &s);
        }

        let acts = internals.map(|internals| GqaLayerActs { internals, ctx_gated });
        (out, acts)
    }

    // ---- dense SwiGLU MLP, universal for every layer -----------------------

    /// `prefix` is the weight-name prefix - `"blocks.{l}.mlp"` for a normal
    /// layer, `"mtp.layers.0.mlp"` for the MTP head. `gate`/`up` share
    /// ONE `Ops::act` call on `xn2` (both consume the same activation);
    /// `down` gets its own fresh `Ops::act` call on `h` (a different
    /// activation) - same "quantize once, share across sibling linears"
    /// discipline `layer_gdn_fwd`/`layer_gqa_fwd` use.
    fn mlp_fwd(&self, prefix: &str, xn2: &DeviceBuffer, n: u32) -> (DeviceBuffer, Option<MlpLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.intermediate_size;
        let p = |s: &str| format!("{prefix}.{s}");

        let gate_pre = g.storage((n * ff) as u64);
        let up = g.storage((n * ff) as u64);
        {
            let mut s = Vec::new();
            let act1 = self.ops_act(&mut s, xn2, n, d);
            if self.ops_linear(&mut s, &act1, &p("gate.weight"), &gate_pre) {
                self.lora_fwd(&mut s, "gate", xn2, &p("gate.weight"), &gate_pre, n, d, ff);
            }
            if self.ops_linear(&mut s, &act1, &p("up.weight"), &up) {
                self.lora_fwd(&mut s, "up", xn2, &p("up.weight"), &up, n, d, ff);
            }
            g.submit(&[], &s);
        }
        let h = g.storage((n * ff) as u64);
        g.submit(&[], &[g.step(SILU_MUL, &[&gate_pre, &up, &h], &[n * ff], n * ff)]);
        let down = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            let act2 = self.ops_act(&mut s, &h, n, ff);
            if self.ops_linear(&mut s, &act2, &p("down.weight"), &down) {
                self.lora_fwd(&mut s, "down", &h, &p("down.weight"), &down, n, ff, d);
            }
            g.submit(&[], &s);
        }

        let acts = self.is_train.then(|| MlpLayerActs { xn2: xn2.clone(), gate_pre, up, h });
        (down, acts)
    }

    // ---- multi-token-prediction head (`cfg.mtp`) ---------------------------

    /// Forward pass for the MTP head - see [`MtpActs`]'s own doc for the
    /// exact chain: `embed(mtp_input)` -> two independent pre-norms (one over
    /// the embedding, one over `res_last`) -> `fc_e`/`fc_h` fused into one
    /// residual-stream input -> ONE full Gated-Attention decoder layer
    /// (`"mtp.layers.0.*"`, reusing [`Self::layer_gqa_fwd`]/[`Self::mlp_fwd`]
    /// unchanged - the exact per-layer shape [`Self::run_forward`]'s own loop
    /// body uses) -> `mtp.norm` -> the SHARED `lm_head`/`tok.weight` matmul
    /// into `self.mtp_logits` -> CE against `self.mtp_target` into
    /// `self.mtp_ce_buf`. Always writes `mtp_logits`/`mtp_ce_buf` (mirrors the
    /// main head always having somewhere to write); returns the saved
    /// activations for [`Self::backward`] only when `is_train`.
    fn run_mtp_forward(&self, res_last: &DeviceBuffer, n: u32) -> Option<MtpActs> {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let v = c.vocab;

        let e = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(EMBED, &[&self.mtp_input, self.w("tok.weight"), &e], &[d, n], n * d)]);

        let en = g.storage((n * d) as u64);
        let hn = g.storage((n * d) as u64);
        g.submit(
            &[],
            &[
                rmsnorm_fwd(g, &kernel_ids(), &e, self.w("mtp.pre_fc_norm_embedding.weight"), &en, d, n),
                rmsnorm_fwd(g, &kernel_ids(), res_last, self.w("mtp.pre_fc_norm_hidden.weight"), &hn, d, n),
            ],
        );

        let ehp_e = g.storage((n * d) as u64);
        let ehp_h = g.storage((n * d) as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[&en, self.w("mtp.fc_e.weight"), &ehp_e], &[n, d, d], n * d),
                g.step(MATMUL, &[&hn, self.w("mtp.fc_h.weight"), &ehp_h], &[n, d, d], n * d),
            ],
        );
        let ehp = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(ADD2, &[&ehp_e, &ehp_h, &ehp], &[n * d], n * d)]);

        let xn1 = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &ehp, self.w("mtp.layers.0.ln1.weight"), &xn1, d, n)]);
        let (mixer_out, mixer_acts) = self.layer_gqa_fwd("mtp.layers.0.self_attn", &xn1, n, None);

        let xmid = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(ADD2, &[&ehp, &mixer_out, &xmid], &[n * d], n * d)]);

        let xn2 = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &xmid, self.w("mtp.layers.0.ln2.weight"), &xn2, d, n)]);
        let (mlp_out, mlp_acts) = self.mlp_fwd("mtp.layers.0.mlp", &xn2, n);

        let block_out = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &block_out], &[n * d], n * d)]);

        let final_h = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &block_out, self.w("mtp.norm.weight"), &final_h, d, n)]);
        g.submit(&[], &[g.step(MATMUL, &[&final_h, self.w(c.head_weight()), &self.mtp_logits], &[n, d, v], n * v)]);
        g.submit(&[], &[g.step(CE_VALUE, &[&self.mtp_logits, &self.mtp_target, &self.mtp_ce_buf], &[n, v, model::IGNORE], n)]);

        self.is_train.then(|| MtpActs {
            e,
            en,
            hn,
            ehp,
            layer: LayerTrainActs {
                xn1,
                mixer: MixerActs::Gqa(mixer_acts.expect("qwen35: is_train but layer_gqa_fwd returned no acts for the MTP layer")),
                xmid,
                mlp: mlp_acts.expect("qwen35: is_train but mlp_fwd returned no acts for the MTP layer"),
            },
            block_out,
            final_h,
        })
    }

    /// Reverse of [`Self::run_mtp_forward`]. Must run before
    /// [`Self::backward`]'s main reverse layer loop consumes `d_res_next` -
    /// returns the updated gradient (the `pre_fc_norm_hidden` branch's
    /// contribution folded in), mirroring `glmdsa::model::Qwen35::
    /// build_backward`'s own MTP-before-layer-loop ordering
    /// (`crates/glmdsa/src/model.rs:930-960`), adapted for a full
    /// self-attention sublayer instead of glmdsa's position-wise-only block.
    fn mtp_backward(&self, ma: &MtpActs, res_last: &DeviceBuffer, d_res_next: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let v = c.vocab;
        let head = c.head_weight();

        g.write(&self.ce_grad_uni, &[n, v, model::IGNORE, f(self.count.get())]);
        let d_mtp_logits = g.storage((n * v) as u64);
        g.submit(&[], &[g.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.mtp_logits, &self.mtp_target, &d_mtp_logits], n * v)]);

        let d_final_h = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "lm_head", &d_mtp_logits, &ma.final_h, head, &d_final_h, n, d, v, 0);
            g.submit(&[], &s);
        }

        let d_block_out = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.rmsnorm_bwd_step(&mut s, &ma.block_out, "mtp.norm.weight", &d_final_h, &d_block_out, d, n);
            g.submit(&[], &s);
        }

        // ---- reverse of the one decoder layer (mirrors `backward`'s own per-layer body) ----
        let d_xn2 = self.mlp_bwd("mtp.layers.0.mlp", &ma.layer.mlp, &d_block_out, n);
        let d_ln2_dx = g.storage((n * d) as u64);
        let d_xmid = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.rmsnorm_bwd_step(&mut s, &ma.layer.xmid, "mtp.layers.0.ln2.weight", &d_xn2, &d_ln2_dx, d, n);
            s.push(g.step(ADD2, &[&d_block_out, &d_ln2_dx, &d_xmid], &[n * d], n * d));
            g.submit(&[], &s);
        }

        let d_xn1 = g.storage((n * d) as u64);
        match &ma.layer.mixer {
            MixerActs::Gqa(acts) => self.gqa_mixer_bwd("mtp.layers.0.self_attn", &ma.layer.xn1, acts, &d_xmid, &d_xn1, n),
            MixerActs::Gdn(_) => unreachable!("qwen35: the MTP layer is always Full-attention, never GDN"),
        }

        let d_ln1_dx = g.storage((n * d) as u64);
        let d_ehp = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.rmsnorm_bwd_step(&mut s, &ma.ehp, "mtp.layers.0.ln1.weight", &d_xn1, &d_ln1_dx, d, n);
            s.push(g.step(ADD2, &[&d_xmid, &d_ln1_dx, &d_ehp], &[n * d], n * d));
            g.submit(&[], &s);
        }

        // ---- ehp = fc_e(en) + fc_h(hn) backward ----
        let d_en = g.storage((n * d) as u64);
        let d_hn = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "fc_e", &d_ehp, &ma.en, "mtp.fc_e.weight", &d_en, n, d, d, 0);
            self.proj_bwd(&mut s, "fc_h", &d_ehp, &ma.hn, "mtp.fc_h.weight", &d_hn, n, d, d, 0);
            g.submit(&[], &s);
        }

        // ---- pre-fc norms backward ----
        let d_e = g.storage((n * d) as u64);
        let d_res_from_hn = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            self.rmsnorm_bwd_step(&mut s, &ma.e, "mtp.pre_fc_norm_embedding.weight", &d_en, &d_e, d, n);
            self.rmsnorm_bwd_step(&mut s, res_last, "mtp.pre_fc_norm_hidden.weight", &d_hn, &d_res_from_hn, d, n);
            g.submit(&[], &s);
        }

        if self.trainable("tok.weight") {
            g.submit(&[], &[g.step(EMB_BWD, &[&self.mtp_input, &d_e, self.g("tok.weight")], &[n, d, v], v * d)]);
        }

        let d_res_next2 = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(ADD2, &[d_res_next, &d_res_from_hn, &d_res_next2], &[n * d], n * d)]);
        d_res_next2
    }

    pub(crate) fn run_forward(&self) {
        let g = &self.gpu;
        let n = self.b * self.t;
        let d = self.cfg.d_model;
        let res = self.res.borrow();
        let mut layer_acts: Vec<LayerTrainActs> = Vec::new();

        if self.shard.embed {
            g.submit(&[], &[g.step(EMBED, &[&self.tokens, self.w("tok.weight"), &res[0]], &[d, n], n * d)]);

            // Vision-language splice: overwrite the image-placeholder rows of
            // the freshly-gathered residual stream with the projected image
            // tokens (see `Self::enable_mm_splice`'s doc). No-op unless
            // enabled. Only meaningful on the embed stage (operates on
            // `res[0]`, right after the gather above).
            if let Some((row0, n_rows)) = self.mm_splice.get() {
                g.submit(&[], &[model::vlm::splice_fwd(g, SPLICE, &self.img_embeds, &res[0], row0 * d, n_rows * d)]);
            }
        }

        let types = self.cfg.layer_types();
        // `l` is the ABSOLUTE layer index (into `types`/`res`/the
        // `blocks.{l}.*` weight names below), not just a loop counter.
        #[allow(clippy::needless_range_loop)]
        for l in self.shard.start..self.shard.end {
            let ty = types[l];
            let xres = &res[l];
            let xn1 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), xres, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n)]);

            let (mixer_out, mixer_acts) = match ty {
                LayerType::Linear => {
                    let (o, a) = self.layer_gdn_fwd(l, &xn1, n, GdnCall::Whole);
                    (o, a.map(|a| MixerActs::Gdn(Box::new(a))))
                }
                LayerType::Full => {
                    let (o, a) = self.layer_gqa_fwd(&format!("blocks.{l}.self_attn"), &xn1, n, None);
                    (o, a.map(MixerActs::Gqa))
                }
            };

            let xmid = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

            let xn2 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &xmid, self.w(&format!("blocks.{l}.ln2.weight")), &xn2, d, n)]);

            let (mlp_out, mlp_acts) = self.mlp_fwd(&format!("blocks.{l}.mlp"), &xn2, n);
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

        // Head epilogue (final norm + lm_head/logits) and MTP: head stage
        // only. `cfg.mtp` requires a whole shard (asserted in `new_impl_on`),
        // so it is always true here whenever `self.cfg.mtp` is set. On a
        // non-head stage `xn_final` is never read (a size-1 dummy stands in,
        // matching this file's own "size-1 dummy where a value doesn't
        // apply" convention used elsewhere).
        let xn_final = if self.shard.head {
            let xn_final = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res[self.cfg.n_layers as usize], self.w("norm.weight"), &xn_final, d, n)]);
            let v = self.cfg.vocab;
            g.submit(&[], &[g.step(MATMUL, &[&xn_final, self.w(self.cfg.head_weight()), &self.logits], &[n, d, v], n * v)]);

            if self.cfg.mtp {
                let mtp_acts = self.run_mtp_forward(&res[self.cfg.n_layers as usize], n);
                *self.mtp_acts.borrow_mut() = mtp_acts;
            }
            xn_final
        } else {
            g.storage(1)
        };

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
        let value_dim = c.linear_value_dim();
        let nvh = c.linear_num_value_heads;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // out_proj backward (LoRA dispatch stays local).
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

        // in_proj_b/a/z backward (LoRA dispatch stays local). FIRST touch to
        // d_xn1 in this function (acc=0) -- in_proj_qkv (below) accumulates
        // last of all.
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "in_proj_b", &d_bproj, xn1, &p("in_proj_b.weight"), d_xn1, n, d, nvh, 0);
            self.proj_bwd(&mut s, "in_proj_a", &d_aproj, xn1, &p("in_proj_a.weight"), d_xn1, n, d, nvh, 1);
            self.proj_bwd(&mut s, "in_proj_z", &d_z, xn1, &p("in_proj_z.weight"), d_xn1, n, d, value_dim, 1);
            g.submit(&[], &s);
        }

        // in_proj_qkv backward (last accumulate into d_xn1; LoRA dispatch
        // stays local).
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "in_proj_qkv", &d_mixed_qkv, xn1, &p("in_proj_qkv.weight"), d_xn1, n, d, conv_dim, 1);
            g.submit(&[], &s);
        }
    }

    /// Reverse of [`Self::layer_gqa_fwd`]'s 7 steps. Mirrors
    /// `qwen35moe::model::Qwen35::gqa_mixer_bwd` exactly. `prefix` matches
    /// the forward call's own (see [`Self::layer_gqa_fwd`]'s doc).
    fn gqa_mixer_bwd(&self, prefix: &str, xn1: &DeviceBuffer, la: &GqaLayerActs, d_out: &DeviceBuffer, d_xn1: &DeviceBuffer, n: u32) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, qd, kvd) = (c.q_proj_dim(), c.q_dim(), c.kv_dim());
        let p = |s: &str| format!("{prefix}.{s}");

        // o_proj backward (LoRA dispatch stays local).
        let d_ctx_gated = g.storage((n * qd) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "o_proj", d_out, &la.ctx_gated, &p("o_proj.weight"), &d_ctx_gated, n, qd, d, 0);
            g.submit(&[], &s);
        }

        // Reverse of the hoisted `model::gqa_mixer::gqa_mixer_fwd` internals.
        let shape = model::gqa_mixer::GqaMixerShape { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: c.rotary_dim() / 2 };
        let weights = model::gqa_mixer::GqaMixerWeights { q_norm: self.w(&p("q_norm.weight")), k_norm: self.w(&p("k_norm.weight")), cos: &self.cos, sin: &self.sin };
        let grads = model::gqa_mixer::GqaMixerGrads {
            q_norm: self.trainable(&p("q_norm.weight")).then(|| self.g(&p("q_norm.weight"))),
            k_norm: self.trainable(&p("k_norm.weight")).then(|| self.g(&p("k_norm.weight"))),
        };
        let (d_q_full, d_k, d_v) = model::gqa_mixer::gqa_mixer_bwd(g, &gqa_mixer_ids(), &shape, &weights, &grads, &la.internals, &d_ctx_gated, n);

        // q/k/v proj backward (LoRA dispatch stays local).
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "q_proj", &d_q_full, xn1, &p("q_proj.weight"), d_xn1, n, d, qpd, 0);
            self.proj_bwd(&mut s, "k_proj", &d_k, xn1, &p("k_proj.weight"), d_xn1, n, d, kvd, 1);
            self.proj_bwd(&mut s, "v_proj", &d_v, xn1, &p("v_proj.weight"), d_xn1, n, d, kvd, 1);
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
    fn mlp_bwd(&self, prefix: &str, la: &MlpLayerActs, d_mlp_out: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.intermediate_size;
        let p = |s: &str| format!("{prefix}.{s}");

        let d_h = g.storage((n * ff) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "down", d_mlp_out, &la.h, &p("down.weight"), &d_h, n, ff, d, 0);
            g.submit(&[], &s);
        }

        let d_gate_pre = g.storage((n * ff) as u64);
        let d_up = g.storage((n * ff) as u64);
        g.submit(&[], &swiglu_bwd(g, &kernel_ids(), &la.gate_pre, &la.up, &d_h, &d_gate_pre, &d_up, n * ff));

        let d_xn2 = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            // FIRST touch to d_xn2 (acc=0); gate accumulates on top.
            self.proj_bwd(&mut s, "up", &d_up, &la.xn2, &p("up.weight"), &d_xn2, n, d, ff, 0);
            self.proj_bwd(&mut s, "gate", &d_gate_pre, &la.xn2, &p("gate.weight"), &d_xn2, n, d, ff, 1);
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

        // ---- head epilogue backward (head stage only): CE-grad, lm_head,
        // final norm -- a non-head stage starts instead from the externally
        // supplied gradient at `res[shard.end]` (`Self::write_out_dres`).
        let mut d_res_next = if self.shard.head {
            g.write(&self.ce_grad_uni, &[n, v, model::IGNORE, f(self.count.get())]);
            let d_logits = g.storage((n * v) as u64);
            g.submit(&[], &[g.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &d_logits], n * v)]);

            let d_xn_final = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.proj_bwd(&mut s, "lm_head", &d_logits, &ta.xn_final, self.cfg.head_weight(), &d_xn_final, n, d, v, 0);
                g.submit(&[], &s);
            }

            let d_res_next = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.rmsnorm_bwd_step(&mut s, &self.res.borrow()[self.cfg.n_layers as usize], "norm.weight", &d_xn_final, &d_res_next, d, n);
                g.submit(&[], &s);
            }
            d_res_next
        } else {
            self.dres_boundary_in.clone()
        };

        if self.cfg.mtp {
            let ma = self.mtp_acts.borrow_mut().take().expect(
                "qwen35: backward() called with cfg.mtp but no MTP activations cached -- \
                 run_forward must populate mtp_acts whenever is_train && cfg.mtp, same \
                 \"forward reallocates fresh, backward takes it\" contract as train_acts",
            );
            let res_borrow = self.res.borrow();
            let res_last = &res_borrow[self.cfg.n_layers as usize];
            d_res_next = self.mtp_backward(&ma, res_last, &d_res_next, n);
            drop(res_borrow);
        }

        let res = self.res.borrow();
        for l in (self.shard.start..self.shard.end).rev() {
            let la = &ta.layers[l - self.shard.start];

            // ---- second residual add backward: res[l+1] = xmid + mlp_out ----
            let d_mlp_out = &d_res_next;
            let d_xn2 = self.mlp_bwd(&format!("blocks.{l}.mlp"), &la.mlp, d_mlp_out, n);

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
                MixerActs::Gqa(acts) => self.gqa_mixer_bwd(&format!("blocks.{l}.self_attn"), &la.xn1, acts, &d_xmid, &d_xn1, n),
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

        // This stage's INPUT-boundary gradient (`dres[shard.start]`), for the
        // previous stage to read via `Self::read_in_dres`. Stashed
        // unconditionally (cheap: one buffer handle) -- a whole/embed-stage
        // build simply never has this read.
        *self.dres_boundary_out.borrow_mut() = d_res_next.clone();

        if self.shard.embed {
            // ---- vision-language splice backward: route the image rows'
            // grad to `d_img_embeds` and ZERO them in `d_res_next` BEFORE
            // `EMB_BWD`, so the image-placeholder token id never accumulates
            // a spurious `tok.weight` gradient from those rows. No-op unless
            // `enable_mm_splice` was called. Only meaningful on the embed
            // stage (operates on `res[0]`/`dres[0]`).
            if let Some((row0, n_rows)) = self.mm_splice.get() {
                g.submit(&[], &[model::vlm::splice_bwd(g, SPLICE_BWD, &d_res_next, &self.d_img_embeds, row0 * d, n_rows * d)]);
            }

            // ---- embedding backward (tok.weight) ----
            if self.trainable("tok.weight") {
                g.submit(&[], &[g.step(EMB_BWD, &[&self.tokens, &d_res_next, self.g("tok.weight")], &[n, d, v], v * d)]);
            }
        }
    }

    // =========================================================================
    // Incremental decode. `run_decode_batch` is the whole of it - one token
    // for each of `bsz` INDEPENDENT sequences, in one dispatch set per layer -
    // and `step`/`run_decode_step` are its one-row case, not a separate tape.
    //
    // There is no decode-specific per-layer code here: a decode step is
    // `layer_gdn_fwd`/`layer_gqa_fwd`/`mlp_fwd` at `n = bsz` rows, with the
    // two stateful mixers in their DECODE arm (`model::gdn_mixer::
    // gdn_mixer_decode_fwd` over the recurrent state and conv window,
    // `model::gqa_mixer::gqa_mixer_decode_batched_fwd` over the shared paged
    // KV pool). Those two are shared with `crates/qwen35moe`, which drives the
    // same pair from its own `run_decode_batch`, so "a batched decode step"
    // means the same thing for both models.
    // =========================================================================

    /// Reset decode state for a fresh sequence: the position counter and
    /// every GDN layer's persistent recurrent `state`/conv `hist` (both must
    /// start at zero for a fresh sequence). GQA layers' KV caches are
    /// deliberately left untouched: a decode step's attention only ever reads
    /// cache rows `0..=pos` (its `seq_lens` bound), so stale rows beyond the
    /// new sequence's own length are never read.
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
    /// token - apply this instance's head (`Self::cfg.head_weight()`) to it
    /// on the host to get logits, exactly as `logits_all`'s own caller would
    /// from a row of its output.
    pub fn step(&self, token_id: u32) -> Vec<f32> {
        self.step_with_input(token_id, None)
    }

    /// [`Self::step`] for a stage that does NOT own the token embedding:
    /// `input` is this stage's `[d_model]` INPUT residual, exactly what
    /// [`Self::run_decode_step`]'s `input_override` seam takes (the previous
    /// stage's output, or - on the first stage of a pipeline whose embedding
    /// lives off-device - the token's embedding row).
    ///
    /// Uses THIS instance's own per-sequence decode state, so its capacity is
    /// the `t` it was built with. A serving path that needs a capacity
    /// independent of `t`, or several concurrent sequences, supplies its own
    /// `DecodeCaches` instead (`crate::serve::Engine`,
    /// `crate::int8_gguf_resident`); this is the single-sequence convenience
    /// both of those wrap.
    ///
    /// `None` is exactly [`Self::step`] and requires `shard.embed`.
    pub fn step_with_input(&self, token_id: u32, input: Option<&[f32]>) -> Vec<f32> {
        assert_eq!(self.b, 1, "qwen35::step requires b==1 (single sequence)");
        assert!(
            (token_id as usize) < self.cfg.vocab as usize,
            "decode token id {token_id} exceeds vocab {} (checkpoint/tokenizer mismatch?)",
            self.cfg.vocab
        );
        let pos = self.dec_pos.get();
        assert!(pos < self.dec_cap, "qwen35::step: decode position {pos} exceeds capacity {}", self.dec_cap);
        let caches = DecodeCaches {
            gqa_kcache: &self.gqa_kcache,
            gqa_vcache: &self.gqa_vcache,
            gqa_cap: self.dec_cap,
            // This instance's own caches are dedicated per-sequence buffers,
            // not a pool window - see `DecodeCaches::gqa_base_row`.
            gqa_base_row: 0,
            gdn_state: &self.gdn_state,
            gdn_hist: &self.gdn_hist,
        };
        let hidden = self.run_decode_step(token_id, pos, &caches, input);
        self.dec_pos.set(pos + 1);
        self.gpu.read(&hidden, self.cfg.d_model as usize)
    }

    /// One incremental decode step's full layer stack - the decode-shaped
    /// (`n=1`) sibling of [`Self::run_forward`], **shard-aware in exactly the
    /// same way**: only `self.shard`'s own layers run, the token embedding
    /// happens on the embed stage only, and the final norm on the head stage
    /// only. `caches` selects WHICH sequence's per-layer GQA cache / GDN
    /// state this call reads and updates - see [`DecodeCaches`]'s own doc
    /// (its fields are indexed by ABSOLUTE layer index, so a partial shard
    /// indexes into them with its own absolute `l` and needs no remapping).
    ///
    /// `input_override` carries the cross-stage seam, the decode-shaped
    /// counterpart of the training pipeline's `read_out_res`/`write_in_res`
    /// (which are `b*t`-shaped and so cannot serve an `n=1` decode step):
    /// * embed stage - must be `None`; `token_id` is embedded as usual.
    /// * non-embed stage - must be `Some(x)` with `x.len() == d_model`: the
    ///   PREVIOUS stage's returned residual, staged through the host. Passing
    ///   both, or neither, is a caller wiring error and panics rather than
    ///   silently picking one. `token_id` is unused on such a stage.
    ///
    /// Returns the final-norm hidden state buffer (unread) on the head stage;
    /// on a non-head stage it returns that stage's LAST-LAYER RESIDUAL,
    /// **unnormed** - the next stage's `input_override`, not a finished
    /// hidden state.
    pub(crate) fn run_decode_step(&self, token_id: u32, pos: u32, caches: &DecodeCaches, input_override: Option<&[f32]>) -> DeviceBuffer {
        assert!(
            caches.gqa_cap > 0 && caches.gqa_base_row.is_multiple_of(caches.gqa_cap),
            "qwen35::run_decode_step: gqa_base_row {} is not a whole number of {}-row blocks",
            caches.gqa_base_row,
            caches.gqa_cap
        );
        let seqs = [BatchSeq {
            phys: caches.gqa_base_row / caches.gqa_cap,
            pos,
            gdn_state: caches.gdn_state,
            gdn_hist: caches.gdn_hist,
        }];
        let batch = BatchDecodeCaches { gqa_kpool: caches.gqa_kcache, gqa_vpool: caches.gqa_vcache, gqa_cap: caches.gqa_cap, seqs: &seqs };
        self.run_decode_batch(&[token_id], &batch, input_override)
    }

    /// [`Self::run_decode_step`] for a BATCH of independent sequences - one
    /// decode token each, all in one set of dispatches per layer.
    ///
    /// This is the function [`Self::run_decode_step`] is the `bsz = 1` case of,
    /// and the reason a serving engine can raise aggregate throughput without
    /// raising per-sequence latency: a decode step's cost is dominated by
    /// reading every weight in the model, and those reads are what the batch
    /// shares. Per-layer the projections become `[bsz, d] x [d, *]` GEMMs
    /// instead of `bsz` separate `m = 1` GEMVs against the same weights, and
    /// the two stateful mixers each take the whole batch in one dispatch set -
    /// full attention through one shared paged KV pool
    /// ([`model::block::gqa_decode_batched_step`]), Gated-DeltaNet through the
    /// `b*h` axis its kernels already have
    /// ([`model::gdn_mixer::gdn_mixer_decode_fwd`]).
    ///
    /// `tokens` is one token id per sequence, in `caches.seqs` order.
    /// `input_override`, when present, is the whole `[bsz, d_model]` boundary
    /// block, the same cross-stage seam [`Self::run_decode_step`] documents at
    /// one row.
    pub(crate) fn run_decode_batch(&self, tokens: &[u32], caches: &BatchDecodeCaches, input_override: Option<&[f32]>) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let bsz = tokens.len() as u32;
        assert!(bsz > 0, "qwen35::run_decode_batch: empty batch");
        assert_eq!(tokens.len(), caches.seqs.len(), "qwen35::run_decode_batch: {} tokens for {} sequences", tokens.len(), caches.seqs.len());
        for s in caches.seqs {
            assert!(s.pos < caches.gqa_cap, "qwen35::run_decode_batch: decode position {} exceeds the per-sequence capacity {}", s.pos, caches.gqa_cap);
        }

        let mut res = if self.shard.embed {
            assert!(input_override.is_none(), "qwen35: run_decode_batch got an input_override on the EMBED stage - it embeds `tokens` itself, so a caller supplying both is a wiring error");
            let tok_buf = g.storage(bsz as u64);
            g.write(&tok_buf, tokens);
            let res = g.storage((bsz * d) as u64);
            g.submit(&[], &[g.step(EMBED, &[&tok_buf, self.w("tok.weight"), &res], &[d, bsz], bsz * d)]);
            res
        } else {
            let x = input_override.expect("qwen35: run_decode_batch on a NON-EMBED stage needs the previous stage's residual as `input_override` (this stage holds no `tok.weight`)");
            assert_eq!(x.len(), (bsz * d) as usize, "qwen35: run_decode_batch input_override must be the batch's whole [bsz, d_model] block");
            g.storage_init("qwen35.decode.res_in", x)
        };

        // Built ONCE per step, shared by every layer in it (the same discipline
        // `run_prefill_chunk_stage` uses for a round): the batch's M-RoPE
        // table at each sequence's OWN position, and the four paged-KV index
        // buffers. The rows of one batch are unrelated positions and unrelated
        // physical blocks, which is the whole difference from a chunk.
        let yarn = c.yarn_scaling();
        let positions: Vec<[u32; 3]> = caches.seqs.iter().map(|s| [s.pos, s.pos, s.pos]).collect();
        let (cos_rows, sin_rows) =
            qwen3vl::mrope::mrope_tables_scaled(&positions, c.mrope_section, c.rotary_dim(), c.rope_theta, yarn.as_ref().map(|(f, a)| (f.as_slice(), *a)));
        let cos = g.storage_init("qwen35.decode_batch.cos", &cos_rows);
        let sin = g.storage_init("qwen35.decode_batch.sin", &sin_rows);
        let blocks = g.storage(bsz as u64);
        g.write(&blocks, &caches.seqs.iter().map(|s| s.phys).collect::<Vec<u32>>());
        let offsets = g.storage(bsz as u64);
        g.write(&offsets, &caches.seqs.iter().map(|s| s.pos).collect::<Vec<u32>>());
        // `max_bt = 1`: this engine gives a sequence ONE physical block for its
        // whole KV history, so its block table is a single entry.
        let block_tables = g.storage(bsz as u64);
        g.write(&block_tables, &caches.seqs.iter().map(|s| s.phys).collect::<Vec<u32>>());
        let seq_lens = g.storage(bsz as u64);
        g.write(&seq_lens, &caches.seqs.iter().map(|s| s.pos + 1).collect::<Vec<u32>>());
        let paged = model::gqa_mixer::PagedDecodeBatch {
            blocks: &blocks,
            offsets: &offsets,
            block_tables: &block_tables,
            seq_lens: &seq_lens,
            block_size: caches.gqa_cap,
            max_bt: 1,
            // The scores/probs stride. `max(pos)+1` rather than the whole
            // per-sequence capacity: it is pure addressing (see
            // `gqa_decode_batched_step`'s own doc), so sizing it to the batch's
            // longest LIVE sequence keeps the scratch proportional to real
            // context rather than to the engine's configured ceiling.
            cap: caches.seqs.iter().map(|s| s.pos + 1).max().unwrap_or(1),
        };

        let types = c.layer_types();
        // `l` is the ABSOLUTE layer index (into `types`, `caches.*` and the
        // `blocks.{l}.*` weight names below), not just a loop counter -
        // exactly as in `Self::run_forward`.
        #[allow(clippy::needless_range_loop)]
        for l in self.shard.start..self.shard.end {
            let ty = types[l];
            let xn1 = g.storage((bsz * d) as u64);
            g.submit(&[], &[rms_step(g, &res, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, bsz)]);

            let mixer_out = match ty {
                LayerType::Linear => {
                    let streams: Vec<model::gdn_mixer::GdnStream> =
                        caches.seqs.iter().map(|s| model::gdn_mixer::GdnStream { state: &s.gdn_state[l], hist: &s.gdn_hist[l] }).collect();
                    self.layer_gdn_fwd(l, &xn1, bsz, GdnCall::Decode(&streams)).0
                }
                LayerType::Full => {
                    let dctx = GqaDecodeCtx { paged: &paged, pool_k: &caches.gqa_kpool[l], pool_v: &caches.gqa_vpool[l], cos: &cos, sin: &sin };
                    self.layer_gqa_fwd(&format!("blocks.{l}.self_attn"), &xn1, bsz, Some(GqaCached::Decode(&dctx))).0
                }
            };

            let xmid = g.storage((bsz * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[&res, &mixer_out, &xmid], &[bsz * d], bsz * d)]);

            let xn2 = g.storage((bsz * d) as u64);
            g.submit(&[], &[rms_step(g, &xmid, self.w(&format!("blocks.{l}.ln2.weight")), &xn2, d, bsz)]);

            let (mlp_out, _) = self.mlp_fwd(&format!("blocks.{l}.mlp"), &xn2, bsz);
            let res_next = g.storage((bsz * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &res_next], &[bsz * d], bsz * d)]);
            res = res_next;

            // Hand this layer to the device NOW and keep building the next one.
            //
            // `Gpu::submit` on this backend does not submit - it appends to a
            // pending list that is flushed at the terminal readback - so
            // WITHOUT this a decode step records every one of its ~1250
            // dispatches (bind groups, uniforms and all) on the host before the
            // card starts any of them, and then waits. Zero overlap, and at
            // `n = 1` the host side of that is not small next to the device
            // side.
            //
            // Measured, and it is not a small effect: this model used to get
            // the overlap BY ACCIDENT, because the GQA decode path uploaded an
            // M-RoPE row per GQA layer and `Gpu::write*` flushes the queue
            // first. Deduplicating those uploads to one per step - a strictly
            // smaller amount of work - cost 1.45x on the whole pass
            // (7.19 -> 4.96 tok/s) purely by removing the flushes that had been
            // pipelining the pass. Flushing on purpose, per layer, is what that
            // accident was worth and is why the dedup is now free.
            g.flush();
        }

        // Head epilogue (final norm): head stage only - `run_forward`'s own
        // `shard.head` branch, at `bsz` rows. A non-head stage returns its raw
        // residual for the next stage to pick up as `input_override`.
        if self.shard.head {
            let xn_final = g.storage((bsz * d) as u64);
            g.submit(&[], &[rms_step(g, &res, self.w("norm.weight"), &xn_final, d, bsz)]);
            xn_final
        } else {
            res
        }
    }

    /// **One ROUND of a chunked prefill**: `tokens.len()` consecutive prompt
    /// tokens starting at absolute position `pos_start`, pushed through the
    /// whole layer stack with ONE dispatch shape per layer instead of
    /// [`Self::run_decode_step`]'s one-per-token. Returns the round's LAST
    /// token's final-norm hidden state (`[d_model]`, unread) - the only row a
    /// prefill's caller wants, and the row a following round or
    /// [`Self::run_decode_step`] call continues from.
    ///
    /// **State contract - this is the whole point of the function.** `caches`
    /// is left in EXACTLY the state a token-by-token replay of the same
    /// `tokens` would have left it in:
    /// * GQA layers: rows `pos_start..pos_start+n` of `gqa_kcache`/
    ///   `gqa_vcache` hold this round's QK-normed, RoPE'd K/V, written by
    ///   `model::block::gqa_chunk_step`'s bulk fill, and the round's queries
    ///   attended rows `0..=pos_start+i` (everything earlier rounds cached,
    ///   plus this round's own keys up to each query - the same causal set the
    ///   per-token path sees).
    /// * GDN layers: `gdn_state` holds `gdn_chunk_fwd`'s `final_state` for the
    ///   round (seeded with the state the PREVIOUS round left, not zero), and
    ///   `gdn_hist` the last `conv_kernel-1` rows of the round's own
    ///   `in_proj_qkv` output - `gdn_causal_conv1d_step`'s own window, so the
    ///   next single-token decode step continues seamlessly.
    ///
    /// so a caller may freely mix rounds and single-token steps on one
    /// sequence. Gated by `tests/chunked_prefill.rs`.
    ///
    /// Whole-model only (`shard.embed && shard.head`), because that is the only
    /// shape whose "the round's last row" return value means anything. A
    /// PIPELINE-PARALLEL caller wants the same round through one stage, with
    /// the `[n, d_model]` boundary block on both ends: that is
    /// [`Self::prefill_chunk_stage`], and both funnel into the same
    /// [`Self::run_prefill_chunk_stage`] body.
    pub(crate) fn run_prefill_chunk(&self, tokens: &[u32], pos_start: u32, caches: &DecodeCaches) -> DeviceBuffer {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let n = tokens.len() as u32;
        assert!(
            self.shard.embed && self.shard.head,
            "qwen35::run_prefill_chunk is whole-model only (this shard has embed={}, head={}) - see this function's own doc",
            self.shard.embed,
            self.shard.head
        );
        let xn_final = self.run_prefill_chunk_stage(tokens, pos_start, caches, None);
        // Only the LAST row is ever wanted (this round's next-token
        // prediction, or the seam into the next round).
        let last = g.storage(d as u64);
        g.submit(&[], &[g.step(CONCAT_SPLIT, &[&xn_final, &last], &[1, n * d, d, (n - 1) * d, 1, 1], d)]);
        last
    }

    /// [`Self::run_prefill_chunk`] for ONE PIPELINE STAGE - the chunked
    /// sibling of [`Self::run_decode_step`], standing to it exactly as
    /// `run_prefill_chunk` stands to a whole-model decode step. Same per-layer
    /// GDN/GQA chunk math, same [`DecodeCaches`] state contract (see
    /// `run_prefill_chunk`'s doc, which owns that contract for both), and the
    /// same shard-awareness `run_decode_step` already has - only `self.shard`'s
    /// layers run, the embedding gather happens on the embed stage only, the
    /// final norm on the head stage only.
    ///
    /// What is new here is the SEAM WIDTH. `run_decode_step`'s cross-stage
    /// `input_override` is `[d_model]`, one row, which cannot express a round's
    /// boundary; this one is a whole `[n, d_model]` block on both ends:
    /// * embed stage - `input_override` must be `None`; `tokens` are gathered
    ///   through `tok.weight` as usual.
    /// * non-embed stage - `input_override` must be `Some(x)` with
    ///   `x.len() == n * d_model`, the previous stage's returned block in token
    ///   order. `tokens` is then used only for its LENGTH and for the round's
    ///   absolute positions; the ids themselves are unread.
    ///
    /// Returns `[n, d_model]`: the head stage's final-normed hidden states, or
    /// a non-head stage's raw last-layer residual block (the next stage's
    /// `input_override`, not a finished hidden state) - the same distinction
    /// `run_decode_step` makes at `n = 1`.
    pub(crate) fn run_prefill_chunk_stage(
        &self,
        tokens: &[u32],
        pos_start: u32,
        caches: &DecodeCaches,
        input_override: Option<&[f32]>,
    ) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let n = tokens.len() as u32;
        assert!(n > 0, "qwen35::run_prefill_chunk_stage: empty chunk (no token to produce a hidden state from)");
        assert!(
            pos_start + n <= caches.gqa_cap,
            "qwen35::run_prefill_chunk_stage: chunk ends at position {} but the KV cache holds {} rows",
            pos_start + n,
            caches.gqa_cap
        );
        // `lora_fwd`'s scratch (`self.lora_a`/`lora_out`) is sized `b*t` rows
        // at construction; a chunk wider than that would write past it.
        assert!(
            self.cfg.lora.is_none() || n <= self.b * self.t,
            "qwen35::run_prefill_chunk_stage: a LoRA build's adapter scratch holds {} rows, chunk is {n}",
            self.b * self.t
        );

        let mut res = if self.shard.embed {
            assert!(
                input_override.is_none(),
                "qwen35: run_prefill_chunk_stage got an input_override on the EMBED stage - it gathers `tokens` itself, so a caller supplying both is a wiring error"
            );
            let tok_buf = g.storage(n as u64);
            g.write(&tok_buf, tokens);
            let res = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(EMBED, &[&tok_buf, self.w("tok.weight"), &res], &[d, n], n * d)]);
            res
        } else {
            let x = input_override
                .expect("qwen35: run_prefill_chunk_stage on a NON-EMBED stage needs the previous stage's residual block as `input_override` (this stage holds no `tok.weight`)");
            assert_eq!(x.len(), (n * d) as usize, "qwen35: run_prefill_chunk_stage input_override must be the round's whole [n, d_model] block");
            g.storage_init("qwen35.prefill.res_in", x)
        };

        // Built ONCE per round, shared by every GQA layer in it: the round's
        // own M-RoPE table (absolute positions), its causal `seq_lens`, and
        // the single-block table a flat per-sequence KV cache degenerates to.
        let positions: Vec<[u32; 3]> = (0..n).map(|i| [pos_start + i, pos_start + i, pos_start + i]).collect();
        // `mrope_tables_scaled`, not plain `mrope_tables`: this is the THIRD
        // M-RoPE call site (`run_forward` and `run_decode_batch` are the other
        // two), and a round that rotates its keys unscaled while the decode
        // step that later reads them rotates its queries with YaRN applied
        // writes a cache the rest of the sequence silently misreads. See
        // `tests/yarn_rope_scaling.rs`'s chunked-replay gate.
        let yarn = c.yarn_scaling();
        let (cos, sin) =
            qwen3vl::mrope::mrope_tables_scaled(&positions, c.mrope_section, c.rotary_dim(), c.rope_theta, yarn.as_ref().map(|(f, a)| (f.as_slice(), *a)));
        let cos = g.storage_init("qwen35.prefill_chunk.cos", &cos);
        let sin = g.storage_init("qwen35.prefill_chunk.sin", &sin);
        let block_ids = g.storage(n as u64);
        let seq_lens = g.storage(n as u64);
        // The single-block table this sequence's KV window is: every row holds
        // the same physical block, `0` when the caller bound a dedicated flat
        // cache - see `DecodeCaches::gqa_base_row`.
        assert!(
            caches.gqa_base_row.is_multiple_of(caches.gqa_cap),
            "qwen35::run_prefill_chunk_stage: gqa_base_row {} is not a whole number of {}-row blocks",
            caches.gqa_base_row,
            caches.gqa_cap
        );
        g.write(&block_ids, &vec![caches.gqa_base_row / caches.gqa_cap; n as usize]);
        g.write(&seq_lens, &(0..n).map(|i| pos_start + i + 1).collect::<Vec<u32>>());

        let pooled = n >= self.chunk_arena_min_rows.get();
        // What an UNPOOLED round owes instead of the arena's drain. Nothing is
        // recycled there, so no dispatch can be overwritten while it still
        // reads - but every layer's temporaries are still DROPPED as the next
        // layer's are taken, and this backend refuses the next allocation once
        // more than its own reclaim ceiling is sitting dropped-and-unreclaimed
        // (`backend_wgpu::WgpuBackend::track`, which panics rather than
        // letting it become a generic OOM hours into a run). Measured: a
        // 128-row unpooled round crosses a P40's 4.29 GB ceiling partway
        // through its 34 layers.
        //
        // So the unpooled path drains on BYTES, not on a layer count. A 1-8
        // row verify round's whole scratch is a few hundred MB and never comes
        // near this, so it pays nothing; a round large enough to matter drains
        // a handful of times instead of once per layer. Half the ceiling, so
        // one more layer's allocations after the check cannot cross it.
        let reclaim_budget = g.reclaim_ceiling_bytes() / 2;
        let types = self.cfg.layer_types();
        #[allow(clippy::needless_range_loop)]
        for l in self.shard.start..self.shard.end {
            let ty = types[l];

            // M6.10: a replay arena, not a fixed drain schedule. Every layer's
            // outer temporaries (`xn1`, `xmid`, `xn2`, `res_next`) are the
            // same `[n, d_model]` size requested in the same call order
            // whatever the layer's own TYPE, and each is read by exactly one
            // downstream consumer within this same iteration - dead the
            // moment that consumer's dispatch is recorded. That is exactly
            // the repeated-shape pattern `gpu_core::scratch::Arena` exists
            // for (see its module doc for the aliasing argument this scope
            // relies on); `res` itself is the one value this loop carries
            // ACROSS scopes (this iteration's `res` is still held by this
            // binding when the next iteration's `res_next` asks for the same
            // slot), which the arena's own `is_unique` check already handles
            // for free - that slot gets a fresh allocation instead of
            // aliasing the value this loop is still reading, with no special
            // case needed here.
            //
            // M29: only above `CHUNK_ARENA_MIN_ROWS`. Below it no scope is
            // opened at all, `Gpu::storage` allocates as it does everywhere
            // else, and the `poll_wait` this scope's reuse requires is not
            // owed - see that constant's own doc for why the trade flips with
            // the row count.
            let _scope = pooled.then(|| g.scratch_scope());

            let xn1 = g.storage((n * d) as u64);
            g.submit(&[], &[rms_step(g, &res, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n)]);

            let mixer_out = match ty {
                LayerType::Linear => {
                    let cont = model::gdn_mixer::GdnStream { state: &caches.gdn_state[l], hist: &caches.gdn_hist[l] };
                    self.layer_gdn_fwd(l, &xn1, n, GdnCall::Chunk(cont)).0
                }
                LayerType::Full => {
                    let ctx = GqaChunkCtx {
                        base_row: caches.gqa_base_row,
                        start: pos_start,
                        cap: caches.gqa_cap,
                        kcache: &caches.gqa_kcache[l],
                        vcache: &caches.gqa_vcache[l],
                        block_ids: block_ids.clone(),
                        seq_lens: seq_lens.clone(),
                        cos: &cos,
                        sin: &sin,
                    };
                    self.layer_gqa_fwd(&format!("blocks.{l}.self_attn"), &xn1, n, Some(GqaCached::Chunk(&ctx))).0
                }
            };

            let xmid = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[&res, &mixer_out, &xmid], &[n * d], n * d)]);

            let xn2 = g.storage((n * d) as u64);
            g.submit(&[], &[rms_step(g, &xmid, self.w(&format!("blocks.{l}.ln2.weight")), &xn2, d, n)]);

            let (mlp_out, _) = self.mlp_fwd(&format!("blocks.{l}.mlp"), &xn2, n);
            let res_next = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &res_next], &[n * d], n * d)]);
            res = res_next;

            // The liveness boundary itself: recording (the `g.submit` calls
            // above) is not dispatching - a bind group names a buffer, it
            // does not write it - so building this WHOLE layer's step
            // sequence first, THEN draining, is what lets the host overlap
            // this layer's CPU-side recording with the PREVIOUS layer's
            // device work. What is not safe is letting this layer's writes
            // reach the device (`flush`) before the dispatch that read the
            // slot they may have just been handed back by the arena is
            // confirmed complete - `poll_wait` is what proves that, and it
            // has to run before `flush`, not after, or it drains nothing
            // this layer just recorded. On the FIRST layer of this call
            // there is nothing queued yet to drain (`poll_wait` on an idle
            // queue returns immediately), so within the pooled regime the
            // ordering is unconditional rather than gated on a layer count.
            //
            // The wait is a real cost, and how real depends entirely on `n`:
            // a 256-row round has enough per-layer device work to hide it
            // behind, a 1-row one has none and spends 60% of the round
            // blocked. That is why it is now gated on `pooled` (M29) rather
            // than paid by every round - it belongs to the POOL, not to the
            // tape. See M6.10's kernel-performance ledger entry for the pool
            // itself and the qwen35 ledger's M29 for the profile that priced
            // its drain.
            //
            // The second disjunct is a different obligation with the same
            // remedy - see `reclaim_budget` above.
            if pooled || g.pending_reclaim_bytes() > reclaim_budget {
                g.poll_wait();
            }
            g.flush();
            // The tap ([`Self::set_hidden_taps`]). Here, not after the loop,
            // because `res` is rebound every iteration and only this point
            // holds layer `l`'s own output. A tapped read is its own drain
            // (`Gpu::read` flushes and blocks), so it is correct on both sides
            // of the `pooled` branch above.
            if !self.taps.borrow().is_empty() && self.taps.borrow().contains(&l) {
                self.tapped.borrow_mut().push((l, g.read(&res, (n * d) as usize)));
            }
            if std::env::var_os("QWEN35_DEBUG_LAYER_RMS").is_some() {
                let last_row = g.read(&res, (n * d) as usize)[((n - 1) * d) as usize..].to_vec();
                let rms = (last_row.iter().map(|v| v * v).sum::<f32>() / last_row.len() as f32).sqrt();
                eprintln!("qwen35::run_prefill_chunk_stage: pos_start={pos_start} n={n} layer {l} ({ty:?}) last-row rms = {rms}");
            }
        }
        if std::env::var("QWEN35_DEBUG_SCRATCH_HELD").is_ok() {
            let (n_slots, words) = g.scratch_held();
            eprintln!("qwen35::run_prefill_chunk_stage: arena holds {n_slots} slots, {words} words ({:.2} MiB) at round end (n_layers this shard={}, n={})", words as f64 * 4.0 / (1024.0 * 1024.0), self.shard.end - self.shard.start, n);
        }
        // The per-layer scratch pool this loop built is worth exactly as
        // much as the layer count that reused it; a caller entering the next
        // round at a different `n` (a ragged final chunk) would only see it
        // mismatch on size and fall back to fresh allocation anyway, so
        // there is nothing to gain by keeping it - and holding it would mean
        // every decode step for the rest of this instance's life pays a
        // permanently resident `~n_layers`-th of a round's scratch for
        // buffers a decode step never touches. Release it here, the same
        // point a `for` loop's locals would drop on their own if the arena
        // did not deliberately outlive them. Skipped when no scope was opened
        // (`n < CHUNK_ARENA_MIN_ROWS`): there is nothing this round put in the
        // arena to give back, and a round that ran unpooled has no business
        // reaching into the allocator state at all.
        if pooled {
            g.scratch_release();
        }

        // Head epilogue (final norm): head stage only - `run_decode_step`'s own
        // `shard.head` branch, at `n` rows. A non-head stage hands back its raw
        // residual block for the next stage to pick up as `input_override`.
        if self.shard.head {
            let xn_final = g.storage((n * d) as u64);
            g.submit(&[], &[rms_step(g, &res, self.w("norm.weight"), &xn_final, d, n)]);
            xn_final
        } else {
            res
        }
    }

    /// Whether pushing a prompt through this build in multi-row ROUNDS
    /// ([`Self::run_prefill_chunk_stage`]) actually beats replaying it one
    /// token at a time - a property of the WEIGHT TIER, not of the prompt.
    ///
    /// A round's whole premise is that one `[n, d] x [d, *]` GEMM beats `n`
    /// GEMVs. That holds when the tier's `m > 1` kernel is a real tiled GEMM,
    /// which `matmul` / `matmul_reg2` / `matmul_i8_dyn` all are - and, as of
    /// `model::ops::Ops::bind` wiring in `matmul_q4_dyn_reg` (kernel-
    /// performance ledger M5.5), `matmul_q4_dyn` now resolves to a real
    /// 128x128 register-tiled GEMM too, not the one-thread-per-output naive
    /// kernel this function used to exclude Q4 for. Measured on the real
    /// 27B, uniform Q4 on one P40, the real 1555-token prompt this crate's
    /// own long-context gate uses: chunked prefill through the now-tiled
    /// kernel is 22.9 s (68.0 tok/s), against 152.1 s (10.2 tok/s) per
    /// token - a ~6.6x win, not a regression, and slightly FASTER than the
    /// INT8 two-card stack's own chunked result (64.6-65.4 tok/s). See
    /// `crate::int8_gguf_resident::MAX_PREFILL_TOKENS`'s own doc for the
    /// full before/after.
    pub(crate) fn chunked_prefill_is_profitable(&self) -> bool {
        true
    }

    /// [`Self::run_prefill_chunk_stage`] staged to the HOST - the one call a
    /// multi-stage CHUNKED prefill driver (`crate::int8_gguf_resident`) needs
    /// per stage per round, and the exact counterpart of what
    /// [`Self::decode_step_stage`] is for a single-token pass.
    ///
    /// Returns `[n * d_model]` in token order: this stage's boundary residual
    /// block, ready to be handed to the next stage as its `input_override` (or,
    /// on a head stage, already through the final `norm.weight`). The same
    /// "no `lm_head` here, ever" rule [`Self::decode_step_stage`]'s doc
    /// explains applies - at the real shape an fp32 head cannot be a device
    /// buffer, so every sharded caller projects with
    /// `crate::stream::head_logits_on` instead.
    ///
    /// One host round trip per stage per ROUND, where the per-token path pays
    /// one per stage per TOKEN: that ratio is the point of the function.
    pub(crate) fn prefill_chunk_stage(&self, tokens: &[u32], pos_start: u32, caches: &DecodeCaches, input_override: Option<&[f32]>) -> Vec<f32> {
        let out = self.run_prefill_chunk_stage(tokens, pos_start, caches, input_override);
        self.gpu.read(&out, tokens.len() * self.cfg.d_model as usize)
    }

    /// **Chunked prefill** of a whole prompt against THIS instance's own
    /// per-sequence decode state - the multi-token-per-dispatch sibling of
    /// calling [`Self::step`] once per prompt token, and the exact analogue of
    /// how [`Self::step`] wraps [`Self::run_decode_step`].
    ///
    /// Consumes `tokens` in rounds of at most `max_chunk`, each round
    /// continuing from the state the previous one left
    /// ([`Self::run_prefill_chunk`]'s own contract), advances `decode_pos` by
    /// the whole prompt length, and returns the LAST token's final-norm hidden
    /// state. A following [`Self::step`] continues exactly as if the prompt
    /// had been replayed one token at a time.
    ///
    /// `max_chunk` bounds the round length because attention scratch grows as
    /// `chunk * n_heads * (pos + chunk)` - the one cost that does not shrink
    /// with the dispatch count, and the reason this is CHUNKED rather than one
    /// whole-prompt forward.
    pub fn prefill_chunked(&self, tokens: &[u32], max_chunk: u32) -> Vec<f32> {
        assert_eq!(self.b, 1, "qwen35::prefill_chunked requires b==1 (single sequence)");
        assert!(!tokens.is_empty(), "qwen35::prefill_chunked: empty prompt");
        assert!(max_chunk > 0, "qwen35::prefill_chunked: max_chunk must be > 0");
        if let Some(&bad) = tokens.iter().find(|&&t| t >= self.cfg.vocab) {
            panic!("qwen35::prefill_chunked: token {bad} exceeds vocab {}", self.cfg.vocab);
        }
        let mut pos = self.dec_pos.get();
        assert!(
            pos + tokens.len() as u32 <= self.dec_cap,
            "qwen35::prefill_chunked: prompt ends at position {} but this instance's decode capacity is {}",
            pos + tokens.len() as u32,
            self.dec_cap
        );
        let caches = DecodeCaches {
            gqa_kcache: &self.gqa_kcache,
            gqa_vcache: &self.gqa_vcache,
            gqa_cap: self.dec_cap,
            // This instance's own caches are dedicated per-sequence buffers,
            // not a pool window - see `DecodeCaches::gqa_base_row`.
            gqa_base_row: 0,
            gdn_state: &self.gdn_state,
            gdn_hist: &self.gdn_hist,
        };
        let mut hidden = None;
        for round in tokens.chunks(max_chunk as usize) {
            hidden = Some(self.run_prefill_chunk(round, pos, &caches));
            pos += round.len() as u32;
        }
        self.dec_pos.set(pos);
        let hidden = hidden.expect("prefill_chunked: prompt is non-empty (asserted above)");
        self.gpu.read(&hidden, self.cfg.d_model as usize)
    }

    /// One Gated-DeltaNet layer's recurrent state, `[bh, dk, dv]` flattened -
    /// parity-debugging introspection only, and the buffer
    /// `tests/spec_decode.rs` compares directly rather than through its
    /// effect on a sampled token. Panics at a GQA layer index, where the
    /// buffer is a 1-word dummy and reading it would silently compare
    /// nothing.
    pub fn debug_gdn_state(&self, l: usize) -> Vec<f32> {
        assert_eq!(self.cfg.layer_types()[l], LayerType::Linear, "layer {l} is GQA and has no recurrent state");
        self.gpu.read(&self.gdn_state[l], self.cfg.gdn_state_len() as usize)
    }

    // =========================================================================
    // Speculative decoding.
    //
    // The verify pass is a CHUNK, not a decode step: `k+1` rows of ONE
    // sequence at consecutive positions, which is exactly
    // `run_prefill_chunk_stage`'s shape and exactly NOT `run_decode_batch`'s
    // (that one is `k` rows of `k` different sequences). Reusing the chunk
    // path is what makes a verify cost roughly one decode step - the weight
    // stream dominates, and scoring a few more positions against it is close
    // to free on a memory-bound decode.
    //
    // The part that is not textbook is the rollback. This decoder is three
    // quarters recurrent (`full_attention_interval = 4`), and the two kinds
    // of layer fail differently when a speculative tail is rejected:
    //
    //   * GQA layers need NO rollback. The verify chunk writes KV rows
    //     `pos_start..pos_start+n`, but a later step's attention is bounded
    //     by its own `seq_lens`, so rows past the committed position are
    //     never read - the same argument `reset_decode_cache` already makes
    //     for leaving a stale KV pool alone.
    //   * GDN layers need a real one. The recurrent state and the conv
    //     history are single buffers that the verify pass has already
    //     advanced through every row it was given, rejected ones included.
    //     There is no "row past the position" to leave behind, so the only
    //     honest options are to snapshot before and restore after, or to
    //     teach the chunk kernel to emit per-row intermediate states.
    //
    // This takes the first: snapshot (a device-side `REGION_COPY`, no host
    // round trip), and on a partial rejection restore and re-run the accepted
    // prefix as its own small chunk. The cost is a second target forward in
    // any round that rejects, which is why the break-even is two accepted
    // draft tokens per round rather than one. A round that accepts EVERY
    // proposal skips the restore entirely - its state is already correct for
    // all `k+1` rows - so a drafter that is usually right pays nothing.
    //
    // The alternative (per-row intermediate states out of `gdn_chunk_fwd`)
    // would remove that second forward, at the price of a new kernel output
    // and `k+1` times the state memory per GDN layer. Left undone
    // deliberately: it is an optimization to the accept path, not a
    // correctness question, and it should be measured against a real
    // acceptance rate before it is paid for.
    // =========================================================================

    /// Scratch mirroring every recurrent layer's `state`/`hist`, allocated
    /// once per speculative run and reused every round. Indexed by ABSOLUTE
    /// layer index like every other per-layer cache in this file; GQA layer
    /// slots hold 1-word dummies and are never touched.
    pub(crate) fn new_gdn_snapshot(&self) -> GdnSnapshot {
        let (state_len, hist_len) = (self.cfg.gdn_state_len(), self.cfg.gdn_hist_len());
        let mut snap = GdnSnapshot { state: Vec::new(), hist: Vec::new() };
        for ty in self.cfg.layer_types() {
            let linear = ty == LayerType::Linear;
            snap.state.push(self.gpu.storage(if linear { state_len } else { 1 }));
            snap.hist.push(self.gpu.storage(if linear { hist_len } else { 1 }));
        }
        snap
    }

    /// Copy every recurrent layer's live state into (`save`) or out of
    /// (`!save`) `snap`, device-side. `REGION_COPY` at `off = 0`,
    /// `width = row_stride = len` is a whole-buffer `dst[i] = src[i]`.
    pub(crate) fn gdn_snapshot_xfer(&self, caches: &DecodeCaches, snap: &GdnSnapshot, save: bool) {
        let (state_len, hist_len) = (self.cfg.gdn_state_len(), self.cfg.gdn_hist_len());
        let mut steps = Vec::new();
        for (l, ty) in self.cfg.layer_types().into_iter().enumerate() {
            if ty != LayerType::Linear || !self.shard.owns(l) {
                continue;
            }
            for (live, cold, len) in [(&caches.gdn_state[l], &snap.state[l], state_len), (&caches.gdn_hist[l], &snap.hist[l], hist_len)] {
                let (src, dst) = if save { (live, cold) } else { (cold, live) };
                let n = len as u32;
                steps.push(self.gpu.step(REGION_COPY, &[src, dst], &[1, n, n, 0], n));
            }
        }
        self.gpu.submit(&[], &steps);
    }

    /// Take a copy of every recurrent layer's state and conv history, as they
    /// stand right now, against THIS instance's own decode caches - the
    /// operation a speculative round performs before a verify pass it may
    /// have to undo. Device-side; nothing crosses the host.
    pub fn gdn_snapshot(&self) -> GdnSnapshot {
        let snap = self.new_gdn_snapshot();
        self.gdn_snapshot_xfer(&self.own_caches(), &snap, true);
        snap
    }

    /// Put a [`Self::gdn_snapshot`] back, discarding whatever the recurrent
    /// state has absorbed since it was taken.
    pub fn gdn_restore(&self, snap: &GdnSnapshot) {
        self.gdn_snapshot_xfer(&self.own_caches(), snap, false);
    }

    /// This instance's own single-sequence caches, as `run_*` wants them -
    /// the same literal `step`/`prefill_chunked` build, factored out because
    /// the speculative loop needs it in three places.
    fn own_caches(&self) -> DecodeCaches<'_> {
        DecodeCaches {
            gqa_kcache: &self.gqa_kcache,
            gqa_vcache: &self.gqa_vcache,
            gqa_cap: self.dec_cap,
            gqa_base_row: 0,
            gdn_state: &self.gdn_state,
            gdn_hist: &self.gdn_hist,
        }
    }

    /// **Speculative decoding**, greedy, against this instance's own
    /// single-sequence decode state. `draft(ctx, want) -> tokens` proposes up
    /// to `want` continuations of `ctx`; `argmax_of` projects one `[d_model]`
    /// final-norm hidden state to its greedy token id (this crate keeps the
    /// vocab head off-device, so the caller owns that matvec - see
    /// [`Self::decode_step_stage`]'s doc for why).
    ///
    /// The output is IDENTICAL to plain greedy decoding through
    /// [`Self::step`], whatever the drafter proposes - a rejected proposal
    /// costs time, never correctness. `tests/spec_decode.rs` gates that on
    /// both a perfect and an adversarial drafter, and gates the recurrent
    /// rollback directly against the state buffers.
    pub fn spec_decode_greedy(
        &self,
        prompt: &[u32],
        max_new: usize,
        k: u32,
        argmax_of: &dyn Fn(&[f32]) -> u32,
        draft: &mut dyn FnMut(&[u32], u32) -> Vec<u32>,
    ) -> (Vec<u32>, SpecDecodeStats) {
        let (toks, _, stats) = self.spec_decode_inner(prompt, max_new, k, argmax_of, draft, false);
        (toks, stats)
    }

    /// [`Self::spec_decode_greedy`], additionally returning the `[d_model]`
    /// final-norm hidden state each emitted token was argmaxed from. Token
    /// equality is a step function over these, so a gate that wants to see a
    /// small state error rather than only a changed id compares them instead.
    pub fn spec_decode_greedy_hidden(
        &self,
        prompt: &[u32],
        max_new: usize,
        k: u32,
        argmax_of: &dyn Fn(&[f32]) -> u32,
        draft: &mut dyn FnMut(&[u32], u32) -> Vec<u32>,
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        let (toks, hidden, _) = self.spec_decode_inner(prompt, max_new, k, argmax_of, draft, true);
        (toks, hidden)
    }

    fn spec_decode_inner(
        &self,
        prompt: &[u32],
        max_new: usize,
        k: u32,
        argmax_of: &dyn Fn(&[f32]) -> u32,
        draft: &mut dyn FnMut(&[u32], u32) -> Vec<u32>,
        want_hidden: bool,
    ) -> (Vec<u32>, Vec<Vec<f32>>, SpecDecodeStats) {
        assert_eq!(self.b, 1, "qwen35::spec_decode_greedy requires b==1 (single sequence)");
        assert!(!prompt.is_empty(), "qwen35::spec_decode_greedy: empty prompt");
        assert!(k >= 1, "qwen35::spec_decode_greedy: k must be at least 1");
        let d = self.cfg.d_model as usize;
        let caches = self.own_caches();
        let snap = self.new_gdn_snapshot();

        // Replay all but the last prompt token; the last is the first row of
        // the first verify chunk, exactly as it is the first `pending` of the
        // textbook loop. A one-token prompt replays nothing and starts at 0.
        self.reset_decode_cache();
        if prompt.len() > 1 {
            self.prefill_chunked(&prompt[..prompt.len() - 1], self.dec_cap.min(256));
        }
        let mut pending = *prompt.last().expect("non-empty (asserted above)");
        let mut ctx: Vec<u32> = prompt.to_vec();
        let (mut out, mut out_hidden) = (Vec::new(), Vec::new());
        let mut stats = SpecDecodeStats::default();

        while out.len() < max_new {
            let want = ((max_new - out.len()) as u32).min(k);
            let mut props = draft(&ctx, want);
            props.truncate(want as usize);
            let kk = props.len();
            assert!(props.iter().all(|&t| t < self.cfg.vocab), "drafter proposed a token outside the vocab");
            stats.rounds += 1;
            stats.proposed += kk;

            let base = self.dec_pos.get();
            let rows: Vec<u32> = std::iter::once(pending).chain(props.iter().copied()).collect();
            assert!(base + rows.len() as u32 <= self.dec_cap, "qwen35::spec_decode_greedy: verify chunk ends at {} past capacity {}", base + rows.len() as u32, self.dec_cap);

            // Only a round that can REJECT can need the state back.
            if kk > 0 {
                self.gdn_snapshot_xfer(&caches, &snap, true);
            }
            let hidden = self.prefill_chunk_stage(&rows, base, &caches, None);
            stats.target_forwards += 1;

            // Row `j` holds the distribution that should have produced
            // `props[j]`. Accept while it agrees; the first row that does not
            // supplies the target's OWN token instead, which is what makes
            // the result identical to plain greedy rather than merely close.
            let mut accepted = 0usize;
            while accepted < kk && argmax_of(&hidden[accepted * d..(accepted + 1) * d]) == props[accepted] {
                accepted += 1;
            }
            let correction = argmax_of(&hidden[accepted * d..(accepted + 1) * d]);

            if accepted == kk {
                // Every proposal survived, so the chunk's own end state is
                // the correct state for all `kk+1` committed rows: no
                // restore, no second forward.
                self.dec_pos.set(base + kk as u32 + 1);
            } else {
                // Rows `0..=accepted` of that chunk were computed from
                // genuinely accepted context and are already correct; it is
                // only the recurrent state, which absorbed the rejected tail
                // too, that has to be rewound and re-advanced.
                self.gdn_snapshot_xfer(&caches, &snap, false);
                self.prefill_chunk_stage(&rows[..=accepted], base, &caches, None);
                stats.target_forwards += 1;
                self.dec_pos.set(base + accepted as u32 + 1);
            }

            stats.accepted += accepted;
            for j in 0..accepted {
                out.push(props[j]);
                ctx.push(props[j]);
                if want_hidden {
                    out_hidden.push(hidden[j * d..(j + 1) * d].to_vec());
                }
            }
            out.push(correction);
            ctx.push(correction);
            if want_hidden {
                out_hidden.push(hidden[accepted * d..(accepted + 1) * d].to_vec());
            }
            pending = correction;
        }
        out.truncate(max_new);
        out_hidden.truncate(max_new);
        (out, out_hidden, stats)
    }

    /// [`Self::run_decode_step`] staged to the HOST - the one call a
    /// multi-stage decode driver (`crate::int8_gguf_resident`) needs per stage
    /// per token. Returns `[d_model]`: this stage's last-layer residual, ready
    /// to be handed to the next stage as its `input_override` (or, on a head
    /// stage, already through the final `norm.weight`).
    ///
    /// Deliberately does NOT apply an `lm_head` projection, even on a head
    /// stage. At the real Qwen3.8-27B shape a `[248320, 5120]` fp32 head is
    /// 5.09 GB - past `max_buffer_size` on a 24 GB P40, so it cannot be a
    /// device buffer at all and a shard that owned it would fail to load.
    /// Both real-weight callers therefore keep the head OUTSIDE the shard, as
    /// an int8 `model::ops::Weight` (1.27 GB packed, inside the binding
    /// limit), and project with `crate::stream::head_logits_on`.
    pub(crate) fn decode_step_stage(&self, token_id: u32, pos: u32, caches: &DecodeCaches, input_override: Option<&[f32]>) -> Vec<f32> {
        let hidden = self.run_decode_step(token_id, pos, caches, input_override);
        self.gpu.read(&hidden, self.cfg.d_model as usize)
    }

    /// [`Self::decode_step_stage`] for a BATCH: one decode token for each of
    /// `tokens.len()` independent sequences through THIS stage, staged to the
    /// host as the whole `[bsz, d_model]` boundary block.
    ///
    /// One host round trip per stage per STEP, where the per-sequence form
    /// pays one per stage per sequence per step - which is the same ratio, on
    /// the pipeline seam, that [`Self::run_decode_batch`] buys inside a stage.
    pub(crate) fn decode_batch_stage(&self, tokens: &[u32], caches: &BatchDecodeCaches, input_override: Option<&[f32]>) -> Vec<f32> {
        let hidden = self.run_decode_batch(tokens, caches, input_override);
        self.gpu.read(&hidden, tokens.len() * self.cfg.d_model as usize)
    }

    /// Device-side head epilogue: `logits[vocab] = hidden[d_model] @
    /// head[vocab, d_model]^T` - the SAME `MATMUL` dispatch [`Self::
    /// run_forward`]'s head epilogue and [`Self::logits_all`] already issue,
    /// at `n = 1`. `hidden` must already be through the final `norm.weight`
    /// RMSNorm - exactly what [`Self::run_decode_step`]'s head-stage branch
    /// returns. Unlike [`Self::decode_step_stage`]'s doc (which explains why
    /// a STREAMED/SHARDED head must stay int8 and off-shard), a whole
    /// unsharded `Qwen35` - this method's only caller
    /// (`crate::serve::Engine`) - already holds the fp32 head resident via
    /// `Self::w` regardless, so projecting through it costs no extra device
    /// memory over what the model already carries.
    pub(crate) fn head_logits_dev(&self, hidden: &DeviceBuffer) -> DeviceBuffer {
        self.head_logits_rows_dev(hidden, 1)
    }

    /// [`Self::head_logits_dev`] over a `[rows, d_model]` block, returning
    /// `[rows, vocab]` - one GEMM against the resident head weight instead of
    /// `rows` GEMVs, the head-side counterpart of what
    /// [`Self::run_decode_batch`] does for the body.
    pub(crate) fn head_logits_rows_dev(&self, hidden: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let v = self.cfg.vocab;
        let logits = g.storage(rows as u64 * v as u64);
        g.submit(&[], &[g.step(MATMUL, &[hidden, self.w(self.cfg.head_weight()), &logits], &[rows, d, v], rows * v)]);
        logits
    }

    /// [`Self::head_logits_dev`] reduced to ONE greedy index, entirely on the
    /// device (`argmax_part` + `argmax_final` - `Op::ArgMaxRow`'s
    /// `SplitReduction` shape, `backend_api::select`) - only the winning
    /// index is read back, never the `[vocab]` logits block. The serving
    /// path's device head: see `crate::serve::Engine::forward_batched_greedy`/
    /// `admit_greedy`.
    pub(crate) fn head_argmax_dev(&self, hidden: &DeviceBuffer) -> u32 {
        self.head_argmax_rows_dev(hidden, 1)[0]
    }

    /// [`Self::head_argmax_dev`] over a `[rows, d_model]` block: `rows` greedy
    /// picks, one dispatch per stage for the whole block, and only the `rows`
    /// winning indices read back.
    pub(crate) fn head_argmax_rows_dev(&self, hidden: &DeviceBuffer, rows: u32) -> Vec<u32> {
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let logits = self.head_logits_rows_dev(hidden, rows);
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
        g.read(&out, rows as usize).into_iter().map(|x| x as u32).collect()
    }

    /// [`Self::head_logits_dev`] reduced to the top-`cap` (token id, logit)
    /// candidates, best first, entirely on the device: `cap` iterations of
    /// (`argmax_part`+`argmax_final`, `topk_extract_step`), each masking the
    /// current winner out of `logits` before the next iteration finds the
    /// row's next-best value - only `cap` pairs are read back, never the
    /// `[vocab]` logits block. Mirrors `qwen3::serve::Engine::
    /// submit_topk_head`/`topk_from_hidden` exactly, at `bsz = 1` and with a
    /// freshly-sized scratch per call rather than a persistent
    /// `[max_batch, TOPK_CAPACITY]` buffer - this engine's own "one truly
    /// active sequence at a time" scope (see `crate::serve`'s module doc)
    /// never needs the batched form.
    pub(crate) fn head_topk_dev(&self, hidden: &DeviceBuffer, cap: u32) -> Vec<(u32, f32)> {
        self.head_topk_rows_dev(hidden, 1, cap).pop().expect("head_topk_rows_dev always returns one entry per row")
    }

    /// [`Self::head_topk_dev`] over a `[rows, d_model]` block: every row's own
    /// top-`cap`, with all `rows` masked and re-reduced together each
    /// iteration, so the whole batch costs `cap` dispatch triples rather than
    /// `rows * cap`.
    pub(crate) fn head_topk_rows_dev(&self, hidden: &DeviceBuffer, rows: u32, cap: u32) -> Vec<Vec<(u32, f32)>> {
        assert!(cap > 0, "head_topk_rows_dev: cap must be > 0");
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let logits = self.head_logits_rows_dev(hidden, rows);
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
        let vals = g.read(&vals, (rows * cap) as usize);
        let idx = g.read(&idx, (rows * cap) as usize);
        (0..rows as usize)
            .map(|r| {
                let s = r * cap as usize;
                idx[s..s + cap as usize].iter().map(|&x| x as u32).zip(vals[s..s + cap as usize].iter().copied()).collect()
            })
            .collect()
    }

    // ---- pipeline-parallel cross-stage seam (`model::Shardable`) ----------

    /// Element count of one residual-stream boundary slab (`b*t*d_model`).
    fn res_numel(&self) -> usize {
        (self.b * self.t * self.cfg.d_model) as usize
    }

    /// This stage's OUTPUT residual `res[shard.end]` (input to the next stage).
    pub fn read_out_res(&self) -> Vec<f32> {
        self.gpu.read(&self.res.borrow()[self.shard.end], self.res_numel())
    }

    /// This stage's INPUT residual `res[shard.start]` (from the previous stage).
    pub fn write_in_res(&self, data: &[f32]) {
        self.gpu.write_f32(&self.res.borrow()[self.shard.start], data);
    }

    /// This stage's INPUT-side residual grad `dres[shard.start]` (to the
    /// previous stage), refreshed by every `backward()` call.
    pub fn read_in_dres(&self) -> Vec<f32> {
        self.gpu.read(&self.dres_boundary_out.borrow(), self.res_numel())
    }

    /// This stage's OUTPUT-side residual grad `dres[shard.end]` (from the
    /// next stage), written externally before a non-head stage's `backward()`.
    pub fn write_out_dres(&self, data: &[f32]) {
        self.gpu.write_f32(&self.dres_boundary_in, data);
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

    /// Every fp32-store name for an inference or full-training build
    /// (`self.ps.params`). A LoRA training build (`self.is_train &&
    /// cfg.lora.is_some()`) instead lists only the trainable `.lora_a`/
    /// `.lora_b` adapter tensors (`self.ps.trainable`) - the frozen base has
    /// no gradient buffer (see [`Self::trainable`]), so listing it here would
    /// make any `read_grad` caller (gradcheck's `directional_check`) panic.
    /// Mirrors `qwen35moe::model::Qwen35::param_names` exactly.
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

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    /// Run the forward graph and return the scalar loss.
    pub fn forward(&self) -> f32 {
        self.run_forward();
        let n = self.b * self.t;
        self.gpu.submit(&[], &[self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, self.cfg.vocab, model::IGNORE], n)]);
        let vals = self.gpu.read(&self.ce_buf, n as usize);
        let mut total = vals.iter().sum::<f32>();
        if self.cfg.mtp {
            // Unweighted sum, matching `glmdsa::model::Qwen35::forward`'s own
            // MTP loss addition - `run_forward` already dispatched this
            // buffer's CE_VALUE inside `run_mtp_forward`.
            total += self.gpu.read(&self.mtp_ce_buf, n as usize).iter().sum::<f32>();
        }
        total / self.count.get()
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

    /// Runs the forward graph (same graph [`Self::forward`] runs, minus the
    /// loss) and returns GDN layer `l`'s own 11-step math's intermediates,
    /// named, in the order the layer itself computes them - a sharper tool
    /// than [`Self::debug_res`] for localizing exactly where
    /// `tools/goldens/qwen35_gguf_reference_forward.py`'s independent
    /// CPython re-implementation diverges from this one on real weights
    /// (see `crates/qwen35/tests/gguf_reference_parity_real.rs`'s doc
    /// comment for why that comparison exists and what it has already
    /// ruled out).
    ///
    /// Every entry is `[n, width]` row-major token-major EXCEPT `ncl_in`/
    /// `ncl_out` (`[C, T]` channel-major - `conv1d_fwd`'s own NCL layout,
    /// `b` folded away since every real caller here has `b == 1`) and
    /// `value_cm`/`beta_cm` (`model::gdn_mixer::gdn_mixer_stream_fwd`'s own
    /// chunk-major permute, `[nvh, n_chunks, chunk, dim]`). `ncl_out` is
    /// PRE-SiLU (apply `silu` before comparing to the reference script's
    /// own post-conv `act`); `z` is likewise pre-SiLU (`z_silu` is the
    /// post-SiLU one the reference script's own `z` variable already is).
    ///
    /// Requires a [`Self::new_fp32_shard_src_train`] instance (`is_train`,
    /// so `run_forward` actually populates `train_acts`) with
    /// `res[shard.start]` already fed via [`Self::write_in_res`] - mirrors
    /// [`Self::forward`]'s own contract, minus the loss/logits it also
    /// computes, which a truncated non-head shard cannot provide anyway.
    pub fn debug_gdn_trace(&self, l: usize) -> Vec<(&'static str, Vec<f32>)> {
        self.run_forward();
        let n = (self.b * self.t) as usize;
        let c = &self.cfg;
        let d = c.d_model as usize;
        let conv_dim = c.linear_conv_dim() as usize;
        let key_dim = (c.linear_num_key_heads * c.linear_key_head_dim) as usize;
        let value_dim = c.linear_value_dim() as usize;
        let nvh = c.linear_num_value_heads as usize;
        let vhd = c.linear_value_head_dim as usize;
        let g = &self.gpu;
        let ta = self.train_acts.borrow();
        let ta = ta.as_ref().expect("qwen35: debug_gdn_trace requires an is_train instance (Self::new_fp32_shard_src_train)");
        let la = &ta.layers[l - self.shard.start];
        let MixerActs::Gdn(gdn) = &la.mixer else { panic!("qwen35: debug_gdn_trace: layer {l} is not a GDN (Linear) layer") };
        let ia = &gdn.internals;
        vec![
            ("xn1", g.read(&la.xn1, n * d)),
            ("ncl_in", g.read(&ia.ncl_in, n * conv_dim)),
            ("ncl_out", g.read(&ia.ncl_out, n * conv_dim)),
            ("query_pre_l2norm", g.read(&ia.query, n * key_dim)),
            ("key_pre_l2norm", g.read(&ia.key, n * key_dim)),
            ("value", g.read(&ia.value, n * value_dim)),
            ("bproj", g.read(&ia.bproj, n * nvh)),
            ("aproj", g.read(&ia.aproj, n * nvh)),
            ("g_decay", g.read(&ia.g_decay, n * nvh)),
            ("value_cm", g.read(&ia.value_cm, n * nvh * vhd)),
            ("beta_cm", g.read(&ia.beta_cm, n * nvh)),
            ("out_tok", g.read(&ia.out_tok, n * value_dim)),
            ("normed", g.read(&ia.normed, n * value_dim)),
            ("z", g.read(&ia.z, n * value_dim)),
            ("z_silu", g.read(&ia.z_silu, n * value_dim)),
            ("gated", g.read(&gdn.gated, n * value_dim)),
        ]
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
        Qwen35::new_on(Gpu::new(pipelines()), cfg, b, t, init)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One decode step of ONE pipeline stage, driven by hand: the same
    /// `DecodeCaches` [`Qwen35::step`] builds from this instance's own
    /// per-sequence state, but with an explicit `pos` and the cross-stage
    /// `input_override` seam exposed (`step` itself is the whole-shard
    /// convenience wrapper and always passes `None`).
    fn decode_stage(m: &Qwen35, token_id: u32, pos: u32, input: Option<&[f32]>) -> Vec<f32> {
        let caches = DecodeCaches {
            gqa_kcache: &m.gqa_kcache,
            gqa_vcache: &m.gqa_vcache,
            gqa_cap: m.dec_cap,
            gqa_base_row: 0,
            gdn_state: &m.gdn_state,
            gdn_hist: &m.gdn_hist,
        };
        let out = m.run_decode_step(token_id, pos, &caches, input);
        m.gpu.read(&out, m.cfg.d_model as usize)
    }

    /// The two-GPU int8 resident-serving seam, end to end at tiny scale: the
    /// same weights loaded as TWO [`Qwen35::new_i8_shard`] stages must decode
    /// exactly what ONE whole-shard [`Qwen35::new_i8`] instance decodes, for
    /// several consecutive positions.
    ///
    /// This is the one thing "the existing whole-shard decode tests still
    /// pass" cannot show, because every one of them constructs a
    /// `Shard::whole` model where the new code paths are unreachable. Here
    /// all three compose at once: the layer-range partition
    /// (`shard.start..shard.end` over ABSOLUTE indices, so each stage hits
    /// its own `blocks.{l}.*` weights and its own `caches.*[l]` slot), the
    /// embed/head endpoint skips (stage 0 embeds and does NOT final-norm,
    /// stage 1 final-norms and does NOT embed), and the host round-trip of
    /// the boundary residual.
    ///
    /// The cut at layer 2 puts a GDN layer on each side and the single GQA
    /// layer (3, `full_attention_interval = 4`) on the downstream stage, so
    /// BOTH kinds of per-layer decode state - the GDN recurrent state/conv
    /// history and the GQA KV cache - are threaded across steps on a
    /// PARTIAL shard, which is where an absolute-vs-relative layer indexing
    /// mistake would surface. Several steps, not one: a stage-local cache
    /// indexed wrongly can still agree at `pos = 0`.
    ///
    /// Equality is exact. The two paths run the identical kernels over
    /// identically quantized weights in the identical order; the only
    /// difference is that the boundary residual makes a lossless fp32 round
    /// trip through host memory. Anything but bit-equality here is a real
    /// difference in what was computed, so there is no tolerance to pick.
    /// Re-run this test binary as a child process, executing only the named
    /// `#[ignore]`d helper, with a generous ceiling published on both memory
    /// classes so [`gpu_core::Gpu::charged_bytes`] is populated regardless of
    /// which backend the ambient device selection resolves to here. Mirrors
    /// `crates/qwen3/tests/mem_budget_inference.rs`'s own helper exactly.
    fn measure_charged_bytes(helper: &str) -> u64 {
        let exe = std::env::current_exe().expect("current_exe");
        let mut cmd = std::process::Command::new(exe);
        cmd.args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"]);
        cmd.env("BRAIN_LIMIT_VRAM_TOTAL", "64G");
        cmd.env("BRAIN_LIMIT_RAM_TOTAL", "64G");
        let out = cmd.output().expect("spawn subprocess");
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "child {helper} exited {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", out.status.code());
        stdout
            .lines()
            .find_map(|l| l.split_once("CHARGED=").map(|(_, rest)| rest.trim().to_string()))
            .unwrap_or_else(|| panic!("child never printed CHARGED; stdout:\n{stdout}"))
            .parse()
            .expect("CHARGED must be a byte count")
    }

    /// A shard that owns only ONE layer out of many must not pay for every
    /// OTHER layer's decode state as if it owned those too.
    ///
    /// Before `run_decode_step` was made shard-aware, `new_impl_on` built a
    /// full-size GQA KV cache or GDN state/history buffer for every layer
    /// position regardless of `shard.owns(l)`, a full-size `logits` buffer
    /// regardless of `shard.head`, and full-size gradient-checkpoint boundary
    /// scratch regardless of `train` - the same defect class already fixed
    /// for `qwen3` (`crates/qwen3/src/model.rs`'s `train`-gated buffers). On
    /// the real 64-layer/GQA-every-4th-layer checkpoint this means a
    /// single-layer resident stage would pay for all 64 layers' worth of
    /// cache/state, not its own one.
    ///
    /// This config makes every layer `Full` (`full_attention_interval = 1`,
    /// the type with the expensive `t`-sized KV cache) so a shard owning only
    /// ONE of many identically-expensive layers gives the clearest possible
    /// signal: correctly gated, its charged bytes are dominated by the fixed
    /// per-model residual-stream buffers (`res`, sized by `cfg.n_layers`
    /// regardless of shard size) plus its own one layer's cache; regressed,
    /// every one of the other unowned layers leaks a full KV cache on top.
    #[test]
    fn a_shard_owning_one_layer_does_not_pay_for_every_other_layers_decode_state() {
        let charged = measure_charged_bytes("model::tests::child_measure_narrow_shard_charged_bytes");
        const MIB: u64 = 1 << 20;
        // Correctly gated: ~res_fixed (33 * 96 * 4 * 2048 = ~25.7 MiB) + this
        // shard's one owned layer's KV cache (~2 MiB) + small weights/rope/
        // lora scratch. Regressed (ownership check removed): the other 31
        // unowned `Full` layers each leak a full KV cache on top, an extra
        // ~62 MiB - comfortably clear of this bound either way.
        assert!(
            charged < 40 * MIB,
            "a shard owning 1 of 32 layers requested {charged} bytes ({:.1} MiB) - \
             expected well under 40 MiB given only one layer's decode state should be \
             live; this is the per-layer cache/state-not-gated-on-ownership regression",
            charged as f64 / MIB as f64
        );
    }

    #[test]
    #[ignore = "child process helper, driven by a_shard_owning_one_layer_does_not_pay_for_every_other_layers_decode_state"]
    fn child_measure_narrow_shard_charged_bytes() {
        let cfg = Qwen35Config { n_layers: 32, full_attention_interval: 1, block_size: 2048, ..Qwen35Config::tiny_i8() };
        let init = crate::init::init_weights(&cfg, 7);
        let shard = Shard { start: 16, end: 17, embed: false, head: false, gpu_index: Shard::ANY_GPU };
        let m = Qwen35::new_i8_shard(cfg.clone(), 1, cfg.block_size, &init, shard);
        println!("CHARGED={}", m.gpu.charged_bytes());
    }

    /// One PREFILL ROUND of ONE pipeline stage, driven by hand - the chunked
    /// sibling of [`decode_stage`], with the widened `[n, d_model]` seam
    /// exposed. Returns the whole `[n, d_model]` boundary block.
    fn prefill_stage(m: &Qwen35, tokens: &[u32], pos: u32, input: Option<&[f32]>) -> Vec<f32> {
        let caches = DecodeCaches {
            gqa_kcache: &m.gqa_kcache,
            gqa_vcache: &m.gqa_vcache,
            gqa_cap: m.dec_cap,
            gqa_base_row: 0,
            gdn_state: &m.gdn_state,
            gdn_hist: &m.gdn_hist,
        };
        m.prefill_chunk_stage(tokens, pos, &caches, input)
    }

    /// Retune the Gated-DeltaNet decay gate so the recurrent state actually
    /// SURVIVES from token to token - `tests/chunked_prefill.rs`'s own
    /// `slow_decay`, and see that file's doc for the measurement behind the
    /// numbers: at `init_weights`' fresh-model values the per-token decay is
    /// `~exp(-10)` and "did round 2 continue from round 1's state?" has no
    /// observable answer.
    fn slow_decay(cfg: &Qwen35Config, mut w: HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
        for (name, numel) in cfg.param_list() {
            if name.ends_with(".A_log") {
                w.insert(name, vec![0.05f32.ln(); numel]);
            } else if name.ends_with(".dt_bias") {
                w.insert(name, vec![-1.0f32; numel]);
            }
        }
        w
    }

    /// **The pipeline-parallel chunked prefill gate.** Two stages, a prompt
    /// consumed in bounded ROUNDS through [`Qwen35::prefill_chunk_stage`],
    /// must leave both stages in EXACTLY the decode state the one-token-at-a-
    /// time replay of the same prompt through [`Qwen35::run_decode_step`]
    /// leaves - proven the only way it can be, by CONTINUITY: several further
    /// single-token steps after the prompt must produce the same hidden
    /// states either way.
    ///
    /// This is the sharded counterpart of `tests/chunked_prefill.rs` (which
    /// gates the same claim for a WHOLE-model `Qwen35`), and it exists
    /// because the whole-model gate cannot reach any of what is new here: the
    /// `[n, d_model]` cross-stage seam (a stage that must take its input from
    /// `input_override` instead of embedding, and hand back its whole
    /// last-layer residual block instead of one row or a final-normed one),
    /// and a round whose GDN/GQA state must continue across BOTH a round
    /// boundary and a card boundary. `crate::int8_gguf_resident` drives
    /// exactly this seam on the real 27B checkpoint.
    ///
    /// The cut at layer 5 puts a GQA layer on BOTH sides (layers 3 and 7 at
    /// `full_attention_interval = 4`) as well as GDN layers on both, so
    /// neither stage is a degenerate single-layer-type case, and stage 1
    /// carries the round's interior causal masking into a cache that every
    /// later token reads.
    ///
    /// Tolerance: `1e-5`, for `tests/chunked_prefill.rs`' reasons exactly -
    /// same kernels and weights, different dispatch shapes (an `n`-row chunk
    /// selects different matmul/RMSNorm variants than the `n = 1` decode tape,
    /// and the GDN recurrence runs `gdn_chunk_fwd`'s chunked-parallel form
    /// rather than `gdn_recurrent_step`'s sequential one). Measured, not
    /// guessed: 3.7e-9 for the correct implementation, and a stage-1 seam
    /// deliberately broken to seed itself with zeros instead of
    /// `input_override` moves the prompt's last hidden state by 1.86 - eight
    /// orders of magnitude clear of the bound on both sides.
    #[test]
    fn two_shard_chunked_prefill_matches_token_by_token_replay() {
        two_shard_chunked_prefill_replay(false);
    }

    /// [`two_shard_chunked_prefill_matches_token_by_token_replay`] with the
    /// scratch arena OPEN - the regime a production-sized round takes. See
    /// `tests/chunked_prefill.rs`'s `pooled_chunked_prefill_matches_token_by_
    /// token_replay_default_backend` for why the two are separate tests and
    /// not one.
    #[test]
    fn two_shard_pooled_chunked_prefill_matches_token_by_token_replay() {
        two_shard_chunked_prefill_replay(true);
    }

    fn two_shard_chunked_prefill_replay(pooled: bool) {
        let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny() };
        let n_layers = cfg.n_layers as usize;
        let cut = 5usize;
        let types = cfg.layer_types();
        assert_eq!(types[3], LayerType::Full, "the cut must leave a GQA layer UPSTREAM");
        assert_eq!(types[7], LayerType::Full, "and another one DOWNSTREAM");
        assert_eq!(types[cut - 1], LayerType::Linear, "and a GDN layer immediately upstream of the seam");

        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = slow_decay(&cfg, crate::init::init_weights(&cfg, 7));
        // `new_fp32_shard_src`, not `new_shard`: the latter builds a TRAINABLE
        // stage, and a streaming/chunked GDN forward is inference-only
        // (`model::gdn_mixer::gdn_mixer_stream_fwd` asserts it saves no
        // backward history). Every real caller of this seam is inference too.
        let stage0 =
            Qwen35::new_fp32_shard_src(cfg.clone(), 1, t, &init, Shard { start: 0, end: cut, embed: true, head: false, gpu_index: Shard::ANY_GPU });
        let stage1 = Qwen35::new_fp32_shard_src(
            cfg.clone(),
            1,
            t,
            &init,
            Shard { start: cut, end: n_layers, embed: false, head: true, gpu_index: Shard::ANY_GPU },
        );
        for s in [&stage0, &stage1] {
            s.set_chunk_arena_min_rows(if pooled { 1 } else { u32::MAX });
        }

        // 14 prompt tokens at chunk 4 is 4+4+4+2 - several rounds with a
        // ragged last one, so every round after the first must continue from
        // the state the previous one left rather than from zero.
        let prompt: Vec<u32> = (0..14).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let tail: Vec<u32> = (0..3).map(|i| (i * 7 + 1) % cfg.vocab).collect();
        let chunk = 4usize;
        assert!(prompt.len() as u32 + tail.len() as u32 <= t, "the whole run must fit one instance's decode capacity");

        // Reference: the existing one-token-per-pass, stage-by-stage replay -
        // exactly what `int8_gguf_resident::stack_step` drives.
        let step_both = |tok: u32, pos: u32| {
            let boundary = decode_stage(&stage0, tok, pos, None);
            assert_eq!(boundary.len(), d);
            // `token_id` is unused on a non-embed stage. Feeding stage 1 a
            // DELIBERATELY WRONG token makes that a checked fact.
            decode_stage(&stage1, (tok + 1) % cfg.vocab, pos, Some(&boundary))
        };
        stage0.reset_decode_cache();
        stage1.reset_decode_cache();
        let mut want_last = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            want_last = step_both(tok, i as u32);
        }
        let want_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| step_both(tok, (prompt.len() + i) as u32)).collect();

        // Under test: the same prompt in ROUNDS, then the SAME per-token
        // continuation.
        stage0.reset_decode_cache();
        stage1.reset_decode_cache();
        let mut got_last = Vec::new();
        let mut pos = 0u32;
        for round in prompt.chunks(chunk) {
            let n = round.len();
            let boundary = prefill_stage(&stage0, round, pos, None);
            assert_eq!(boundary.len(), n * d, "a non-head stage must hand back its whole [n, d_model] residual block");
            assert!(boundary.iter().all(|x| x.is_finite()), "stage 0 emitted a non-finite boundary residual");
            assert!(boundary.iter().any(|x| x.abs() > 1e-6), "stage 0's boundary residual is all ~0 - the seam would carry no information");
            let out = prefill_stage(&stage1, round, pos, Some(&boundary));
            assert_eq!(out.len(), n * d);
            got_last = out[(n - 1) * d..].to_vec();
            pos += n as u32;
        }
        let got_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| step_both(tok, (prompt.len() + i) as u32)).collect();

        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let mut worst = maxabs(&got_last, &want_last);
        assert!(worst < 1e-5, "sharded chunked prefill: prompt's last hidden state maxabs={worst}");
        for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
            let err = maxabs(got, want);
            worst = worst.max(err);
            assert!(err < 1e-5, "continuation token {i} maxabs={err} (the chunked prefill left the two stages' decode state wrong)");
        }
        println!("two_shard_chunked_prefill(pooled={pooled}): worst maxabs over prompt-last + {} continuation steps = {worst:e}", tail.len());
    }

    #[test]
    fn two_shard_int8_decode_matches_the_whole_shard_model() {
        let cfg = Qwen35Config::tiny_i8();
        let n_layers = cfg.n_layers as usize;
        let cut = 2usize;
        assert_eq!(cfg.layer_types()[cut - 1], LayerType::Linear, "cut must leave a GDN layer upstream");
        assert_eq!(cfg.layer_types()[n_layers - 1], LayerType::Full, "the GQA layer must sit downstream of the cut");

        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let stage0 = Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: 0, end: cut, embed: true, head: false, gpu_index: Shard::ANY_GPU });
        let stage1 = Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: cut, end: n_layers, embed: false, head: true, gpu_index: Shard::ANY_GPU });

        whole.reset_decode_cache();
        stage0.reset_decode_cache();
        stage1.reset_decode_cache();

        let tokens: Vec<u32> = (0..6).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = i as u32;
            let want = whole.step(tok);

            let boundary = decode_stage(&stage0, tok, pos, None);
            assert_eq!(boundary.len(), d);
            assert!(boundary.iter().all(|x| x.is_finite()), "pos {pos}: stage 0 emitted a non-finite boundary residual");
            assert!(boundary.iter().any(|x| x.abs() > 1e-6), "pos {pos}: stage 0's boundary residual is all ~0 - the seam would carry no information and this test would be vacuous");

            // `token_id` is unused on a non-embed stage, which reads its input
            // from the seam instead. Feeding stage 1 a DELIBERATELY WRONG
            // token makes that a checked fact: if it ever embedded the token
            // rather than using `input_override`, this step would diverge.
            let got = decode_stage(&stage1, (tok + 1) % cfg.vocab, pos, Some(&boundary));
            assert_eq!(got, want, "pos {pos}: two-shard decode diverged from the whole-shard model");
        }
    }

    /// A chunked-prefill round at the INT8 tier, wide enough to cross the
    /// register-tiled `matmul_i8_dyn` GEMM's own 128-row tile boundary, must
    /// match a token-by-token replay of the same prompt - the INT8 twin of
    /// [`two_shard_chunked_prefill_matches_token_by_token_replay`], which
    /// only ever ran at FP32 (`Qwen35::new_fp32_shard_src`) with chunks of 4
    /// rows, far short of `matmul_i8_dyn`/`matmul_q4_dyn_reg`'s 128-row tile,
    /// so a defect specific to the multi-tile dispatch path (real
    /// production rounds are 256 rows, `int8_gguf_resident::
    /// MAX_PREFILL_TOKENS`) had no gate at any tier that actually
    /// dispatches that kernel family.
    #[test]
    fn chunked_prefill_at_int8_matches_token_by_token_past_one_gemm_tile() {
        let cfg = Qwen35Config { block_size: 320, max_position_embeddings: 320, ..Qwen35Config::tiny_i8() };
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let whole_chunked = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        whole.reset_decode_cache();
        whole_chunked.reset_decode_cache();

        // 260 tokens: past the 128-row tile once (128), then a second full
        // tile (256), then a ragged remainder (4) - all three tile shapes
        // `matmul_i8_dyn`'s dispatch grid must cover in one round.
        let prompt: Vec<u32> = (0..260).map(|i| (i * 5 + 3) % cfg.vocab).collect();

        let mut want_last = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            want_last = decode_stage(&whole, tok, i as u32, None);
        }

        let got = prefill_stage(&whole_chunked, &prompt, 0, None);
        assert_eq!(got.len(), prompt.len() * d, "a whole-model chunk must hand back its whole [n, d_model] block");
        let got_last = got[(prompt.len() - 1) * d..].to_vec();

        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let worst = maxabs(&got_last, &want_last);
        assert!(worst < 1e-4, "chunked int8 prefill past one GEMM tile diverged from token-by-token replay: maxabs={worst}, want={want_last:?}, got={got_last:?}");
    }

    /// SEVERAL rounds of exactly [`crate::int8_gguf_resident::
    /// MAX_PREFILL_TOKENS`] (256), the real production round size, then a
    /// few token-by-token DECODE steps continuing from that state - the
    /// exact shape `int8_gguf_resident::Qwen35GgufInstance::replay_prompt`
    /// drives on a real long prompt (`gguf_resident_real.rs`'s
    /// `prefill_throughput_at_a_real_long_context`/
    /// `decode_throughput_at_a_real_long_context`, both of which produce
    /// REPEATING GARBAGE on the real 27B checkpoint at ~1500-1700 real
    /// tokens - several 256-token rounds - despite the single-round test
    /// above passing clean). If a defect only shows up ACROSS a round
    /// boundary (GDN state or GQA cache carried wrong from one round to the
    /// next), one round can never see it; this is the first gate that
    /// replays MULTIPLE real-sized rounds and then decodes from the result.
    #[test]
    fn chunked_prefill_at_int8_matches_token_by_token_across_several_max_size_rounds() {
        let cfg = Qwen35Config { block_size: 896, max_position_embeddings: 896, ..Qwen35Config::tiny_i8() };
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let whole_chunked = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        whole.reset_decode_cache();
        whole_chunked.reset_decode_cache();

        // 3 rounds of the real MAX_PREFILL_TOKENS size (256, 256, 256), then
        // a ragged one (68) - mirrors a real ~836-token prompt.
        let prompt: Vec<u32> = (0..836).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let tail: Vec<u32> = (0..6).map(|i| (i * 7 + 1) % cfg.vocab).collect();

        let mut want_last = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            want_last = decode_stage(&whole, tok, i as u32, None);
        }
        let want_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole, tok, (prompt.len() + i) as u32, None)).collect();

        let mut pos = 0u32;
        let mut got_last = Vec::new();
        for round in prompt.chunks(256) {
            let out = prefill_stage(&whole_chunked, round, pos, None);
            got_last = out[(round.len() - 1) * d..].to_vec();
            pos += round.len() as u32;
        }
        let got_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole_chunked, tok, (prompt.len() + i) as u32, None)).collect();

        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let mut worst = maxabs(&got_last, &want_last);
        assert!(worst < 1e-4, "multi-round chunked int8 prefill diverged from token-by-token replay at the prompt's last token: maxabs={worst}");
        for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
            let err = maxabs(got, want);
            worst = worst.max(err);
            assert!(err < 1e-4, "decode step {i} after multi-round chunked prefill diverged: maxabs={err} (a round boundary left GDN/GQA state wrong)");
        }
        println!("multi-round chunked int8 prefill: worst maxabs = {worst:e}");
    }

    /// A ragged LAST round whose length is not a multiple of anything
    /// `model::gdn::gdn_chunk_size` tries but 1 (a prime `n`) - the case a
    /// round of 256 or 68 (both multiples of 4, this file's other fixtures)
    /// never exercises. Real long prompts hit exactly this: 1731 tokens
    /// (`prefill_throughput_at_a_real_long_context`) leaves a 195-token
    /// last round, and 1555 tokens (`decode_throughput_at_a_real_long_context`)
    /// leaves 19 - `gdn_chunk_size(195)` and `gdn_chunk_size(19)` both fall
    /// through every candidate in `[64,32,16,8,4,2,1]` to plain 1, since
    /// neither is even. Both of those real runs produce repeating garbage;
    /// every OTHER gate in this file (including the one directly above)
    /// happens to only ever use ragged remainders divisible by 4.
    #[test]
    fn chunked_prefill_at_int8_matches_token_by_token_with_a_prime_length_ragged_round() {
        let cfg = Qwen35Config { block_size: 560, max_position_embeddings: 560, ..Qwen35Config::tiny_i8() };
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let whole_chunked = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        whole.reset_decode_cache();
        whole_chunked.reset_decode_cache();

        // 256 + 256 + 13 = 525: the last round (13) is prime, so
        // `gdn_chunk_size(13)` collapses to 1.
        let prompt: Vec<u32> = (0..525).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let tail: Vec<u32> = (0..6).map(|i| (i * 7 + 1) % cfg.vocab).collect();

        let mut want_last = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            want_last = decode_stage(&whole, tok, i as u32, None);
        }
        let want_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole, tok, (prompt.len() + i) as u32, None)).collect();

        let mut pos = 0u32;
        let mut got_last = Vec::new();
        for round in prompt.chunks(256) {
            let out = prefill_stage(&whole_chunked, round, pos, None);
            got_last = out[(round.len() - 1) * d..].to_vec();
            pos += round.len() as u32;
        }
        let got_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole_chunked, tok, (prompt.len() + i) as u32, None)).collect();

        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let mut worst = maxabs(&got_last, &want_last);
        assert!(worst < 1e-4, "chunked int8 prefill with a prime-length ragged round diverged at the prompt's last token: maxabs={worst}");
        for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
            let err = maxabs(got, want);
            worst = worst.max(err);
            assert!(err < 1e-4, "decode step {i} after a prime-length ragged round diverged: maxabs={err}");
        }
        println!("prime-ragged-round chunked int8 prefill: worst maxabs = {worst:e}");
    }

    /// The same multi-round chunked-prefill-vs-token-by-token comparison as
    /// the two tests above, but at REAL `qwen38_27b` magnitude for every
    /// dimension that could plausibly gate a kernel's correctness on its own
    /// size (`d_model=5120`, GQA `head_dim=256`/`n_heads=24`/`n_kv_heads=4`,
    /// GDN `linear_key_head_dim=128`/`linear_value_head_dim=128`/
    /// `linear_num_key_heads=16`/`linear_num_value_heads=48`,
    /// `intermediate_size=17408`) - only `vocab`/`n_layers` shrink, purely to
    /// keep a RANDOM-weight run affordable (real weight VALUES are not the
    /// suspect; `tiny_i8`-scale versions of every test above already passed
    /// clean, so if this one diverges too the defect is tied to real
    /// MAGNITUDE, not to the real trained checkpoint's own numbers).
    #[test]
    fn chunked_prefill_at_int8_matches_token_by_token_at_real_qwen38_27b_dims() {
        let cfg = Qwen35Config { vocab: 320, n_layers: 4, block_size: 560, max_position_embeddings: 560, ..Qwen35Config::qwen38_27b() };
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let whole_chunked = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        whole.reset_decode_cache();
        whole_chunked.reset_decode_cache();

        // Same shape as the prime-ragged-round test: 256 + 256 + 13.
        let prompt: Vec<u32> = (0..525).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let tail: Vec<u32> = (0..6).map(|i| (i * 7 + 1) % cfg.vocab).collect();

        let mut want_last = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            want_last = decode_stage(&whole, tok, i as u32, None);
        }
        let want_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole, tok, (prompt.len() + i) as u32, None)).collect();

        let mut pos = 0u32;
        let mut got_last = Vec::new();
        for round in prompt.chunks(256) {
            let out = prefill_stage(&whole_chunked, round, pos, None);
            got_last = out[(round.len() - 1) * d..].to_vec();
            pos += round.len() as u32;
        }
        let got_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole_chunked, tok, (prompt.len() + i) as u32, None)).collect();

        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let mut worst = maxabs(&got_last, &want_last);
        assert!(worst < 1e-3, "chunked int8 prefill at real qwen38_27b dims diverged at the prompt's last token: maxabs={worst}");
        for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
            let err = maxabs(got, want);
            worst = worst.max(err);
            assert!(err < 1e-3, "decode step {i} after real-dims chunked prefill diverged: maxabs={err}");
        }
        println!("real-dims chunked int8 prefill: worst maxabs = {worst:e}");
    }

    /// EVERY chunked-INT8 test above uses `init::init_weights`'s DEFAULT
    /// decay, which this file's own `slow_decay` helper's doc says is
    /// `~exp(-10)` per step - so close to total forgetting that the GDN
    /// recurrent state is reset to ~0 every token regardless of how it is
    /// replayed, and a bug that only shows up once state genuinely PERSISTS
    /// across many tokens (the real checkpoint's own trained decay, which
    /// must retain long-range memory to be useful at all) would be
    /// completely invisible to it. This test is the same real-dims,
    /// multi-round, INT8 chunked-vs-token-by-token comparison as
    /// [`chunked_prefill_at_int8_matches_token_by_token_at_real_qwen38_27b_dims`],
    /// with ONLY the decay retuned via `slow_decay` (`A_log = ln(0.05)`, the
    /// same retuning the FP32 gate already applies) so the recurrent state
    /// actually accumulates instead of resetting every step - the condition
    /// `tests/gguf_resident_real.rs`'s real-checkpoint failure needs to
    /// reproduce at all.
    #[test]
    fn chunked_prefill_at_int8_matches_token_by_token_with_persistent_gdn_state() {
        let cfg = Qwen35Config { vocab: 320, n_layers: 4, block_size: 560, max_position_embeddings: 560, ..Qwen35Config::qwen38_27b() };
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = slow_decay(&cfg, crate::init::init_weights(&cfg, 7));

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let whole_chunked = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        whole.reset_decode_cache();
        whole_chunked.reset_decode_cache();

        let prompt: Vec<u32> = (0..525).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let tail: Vec<u32> = (0..6).map(|i| (i * 7 + 1) % cfg.vocab).collect();

        let mut want_last = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            want_last = decode_stage(&whole, tok, i as u32, None);
        }
        let want_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole, tok, (prompt.len() + i) as u32, None)).collect();

        let mut pos = 0u32;
        let mut got_last = Vec::new();
        for round in prompt.chunks(256) {
            let out = prefill_stage(&whole_chunked, round, pos, None);
            got_last = out[(round.len() - 1) * d..].to_vec();
            pos += round.len() as u32;
        }
        let got_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole_chunked, tok, (prompt.len() + i) as u32, None)).collect();

        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        let mut worst = maxabs(&got_last, &want_last);
        println!("persistent-state chunked int8 prefill: last-token maxabs = {worst:e}, want_rms={:.4}, got_rms={:.4}", rms(&want_last), rms(&got_last));
        assert!(worst < 1e-2, "chunked int8 prefill with persistent GDN state diverged at the prompt's last token: maxabs={worst}, want_rms={:.4}, got_rms={:.4}", rms(&want_last), rms(&got_last));
        for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
            let err = maxabs(got, want);
            worst = worst.max(err);
            println!("  decode step {i} after chunked prefill: maxabs = {err:e}");
            assert!(err < 1e-2, "decode step {i} after persistent-state chunked prefill diverged: maxabs={err}");
        }
        println!("persistent-state chunked int8 prefill: worst maxabs = {worst:e}");
    }

    /// The combination none of the tests above try together: TWO SHARDS
    /// (the real pipeline-parallel seam `int8_gguf_resident` drives, host
    /// round-tripping the `[n, d_model]` boundary block between them) AND
    /// several real-sized (256-row) chunked-prefill rounds, at the INT8
    /// tier. Every single-shard INT8 chunked test above passed clean; every
    /// two-shard chunked test that exists (`two_shard_chunked_prefill_
    /// matches_token_by_token_replay`) only ever ran at FP32 with 4-row
    /// chunks. This is the first gate combining both seams at once - the
    /// exact shape a real long prompt through the two-card GGUF resident
    /// exercises.
    #[test]
    fn two_shard_chunked_prefill_at_int8_matches_token_by_token_across_several_max_size_rounds() {
        let cfg = Qwen35Config { block_size: 896, max_position_embeddings: 896, ..Qwen35Config::tiny_i8() };
        let n_layers = cfg.n_layers as usize;
        let cut = 2usize;
        assert_eq!(cfg.layer_types()[cut - 1], LayerType::Linear, "cut must leave a GDN layer upstream");
        assert_eq!(cfg.layer_types()[n_layers - 1], LayerType::Full, "the GQA layer must sit downstream of the cut");
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let stage0 = Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: 0, end: cut, embed: true, head: false, gpu_index: Shard::ANY_GPU });
        let stage1 =
            Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: cut, end: n_layers, embed: false, head: true, gpu_index: Shard::ANY_GPU });
        whole.reset_decode_cache();
        stage0.reset_decode_cache();
        stage1.reset_decode_cache();

        let prompt: Vec<u32> = (0..836).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let tail: Vec<u32> = (0..6).map(|i| (i * 7 + 1) % cfg.vocab).collect();

        let mut want_last = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            want_last = decode_stage(&whole, tok, i as u32, None);
        }
        let want_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| decode_stage(&whole, tok, (prompt.len() + i) as u32, None)).collect();

        let step_both = |tok: u32, pos: u32| {
            let boundary = decode_stage(&stage0, tok, pos, None);
            decode_stage(&stage1, (tok + 1) % cfg.vocab, pos, Some(&boundary))
        };

        let mut pos = 0u32;
        let mut got_last = Vec::new();
        for round in prompt.chunks(256) {
            let boundary = prefill_stage(&stage0, round, pos, None);
            let out = prefill_stage(&stage1, round, pos, Some(&boundary));
            got_last = out[(round.len() - 1) * d..].to_vec();
            pos += round.len() as u32;
        }
        let got_tail: Vec<Vec<f32>> = tail.iter().enumerate().map(|(i, &tok)| step_both(tok, (prompt.len() + i) as u32)).collect();

        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let mut worst = maxabs(&got_last, &want_last);
        assert!(worst < 1e-4, "two-shard multi-round chunked int8 prefill diverged at the prompt's last token: maxabs={worst}");
        for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
            let err = maxabs(got, want);
            worst = worst.max(err);
            assert!(err < 1e-4, "decode step {i} after two-shard multi-round chunked prefill diverged: maxabs={err}");
        }
        println!("two-shard multi-round chunked int8 prefill: worst maxabs = {worst:e}");
    }

    /// The Q4 (W4A8) twin of the test above, same reasoning: a shard split
    /// must be bit-exact regardless of tier, and this crosses M24's
    /// `TierPolicy` seam with the shard seam - two places a byte could
    /// silently diverge (a policy applied inconsistently across shard
    /// boundaries, or the seam itself losing precision) collapsed into one
    /// gate.
    #[test]
    fn two_shard_q4_decode_matches_the_whole_shard_model() {
        let cfg = Qwen35Config::tiny_i8();
        let n_layers = cfg.n_layers as usize;
        let cut = 2usize;
        assert_eq!(cfg.layer_types()[cut - 1], LayerType::Linear, "cut must leave a GDN layer upstream");
        assert_eq!(cfg.layer_types()[n_layers - 1], LayerType::Full, "the GQA layer must sit downstream of the cut");

        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);
        let q4 = TierPolicy::uniform(Dtype::Q4);

        let whole = Qwen35::new_shard_dt(cfg.clone(), 1, t, &init, Shard::whole(n_layers), &q4);
        let stage0 =
            Qwen35::new_shard_dt(cfg.clone(), 1, t, &init, Shard { start: 0, end: cut, embed: true, head: false, gpu_index: Shard::ANY_GPU }, &q4);
        let stage1 = Qwen35::new_shard_dt(
            cfg.clone(),
            1,
            t,
            &init,
            Shard { start: cut, end: n_layers, embed: false, head: true, gpu_index: Shard::ANY_GPU },
            &q4,
        );

        whole.reset_decode_cache();
        stage0.reset_decode_cache();
        stage1.reset_decode_cache();

        let tokens: Vec<u32> = (0..6).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = i as u32;
            let want = whole.step(tok);

            let boundary = decode_stage(&stage0, tok, pos, None);
            assert_eq!(boundary.len(), d);
            assert!(boundary.iter().all(|x| x.is_finite()), "pos {pos}: stage 0 emitted a non-finite boundary residual");
            assert!(boundary.iter().any(|x| x.abs() > 1e-6), "pos {pos}: stage 0's boundary residual is all ~0 - the seam would carry no information and this test would be vacuous");

            let got = decode_stage(&stage1, (tok + 1) % cfg.vocab, pos, Some(&boundary));
            assert_eq!(got, want, "pos {pos}: two-shard q4 decode diverged from the whole-shard q4 model");
        }
    }
}
