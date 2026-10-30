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

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;

use gpu_core::select::{Dtype, DECODE_REGIME_MAX_ROWS};
use gpu_core::provider::library;
use gpu_core::tape::Tape;
use gpu_core::{f, DeviceBuffer, Gpu, Step};

use paramstore::{ParamStore, Role};

use audio::conv::ConvKernels;
use model::adapter::exec::AdapterRuntime;
use model::adapter::bank::{AdapterBank, AdapterBankConfig, AdapterError};
use model::adapter::sites::LoraSites;
use model::head_ce::{GradScale, HeadBackward, HeadCe, HeadCeConfig, HeadKernels};
use model::block::{self, rmsnorm_bwd, rmsnorm_fwd, swiglu_bwd, KernelIds};
use model::gdn::{GdnBwdIds, GdnConvIds, GdnIds, GdnShape};
use model::kv_tier::{KvKernels, KvLayer};
pub use model::gdn::gdn_chunk_size;
use model::ops::{Act, Ops, TierPolicy, Weight};
use model::saved::SavedFormat;
use model::Shard;
use optim::Optim;

use crate::config::{LayerType, Qwen35Config};
pub use crate::decode_state::{slot_state_bytes, SlotPool, SlotTiers};
use crate::decode_tape::{DecodeInputs, SlotDecodeTape};
pub(crate) use crate::stage_rows::StageRows;
pub use crate::train_base::TrainOpts;
use crate::decode_state::DecodeState;
#[path = "objectives.rs"]
mod objectives;
#[path = "spec_verify.rs"]
pub(crate) mod verify;
use crate::train_base::{is_table, memory_linear_tier, reads_float_acts, table_tier, write_weight_host};

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
    // -- paged attention over the KV cache, f32 tier -- see
    // `model::kv_tier::KvKernels`, which binds a layer's KV planes to these by
    // NAME (the compact tiers' siblings are appended in `pipelines()`):
    // `Qwen35::run_prefill_chunk`'s chunk and `run_decode_batch`'s batch both
    // attend a flat per-sequence cache as the degenerate one-block case of a
    // paged pool. Only the softmax is dispatched off a `const` index.
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED), // 88
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED), // 89
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED), // 90
    // M2.6: the single-dispatch alternative to the three kernels just above,
    // at this model's real `head_dim=256`.
    ("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256), // 91
    // The batched KV append of every tier's decode step.
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED), // 92
    // Prefill fast path, see `model::gdn::use_fast_kernels`. Registered LAST
    // for the same reason as the head kernels above.
    ("gdn_ut_fwd", kernels::GDN_UT_FWD), // 93
    ("bmm_tiled", kernels::BMM_TILED), // 94
    // The coalesced RMSNorm backward-x twin: the training backward's norms select it
    // through `block::rms_variant` where the device can run a workgroup reduction.
    ("rmsnorm_dx_rows", kernels::RMSNORM_DX_ROWS), // 95
    // Row-indexed staging of a serving engine's pooled Gated-DeltaNet state,
    // see `GdnPoolStore`. Registered LAST so every const above keeps its place.
    ("pool_rows_gather2", kernels::POOL_ROWS_GATHER2), // 96
    ("pool_rows_scatter2", kernels::POOL_ROWS_SCATTER2), // 97
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
        // Cooperative grad-norm: `optim::Optim` resolves the pair by name and runs it on
        // devices whose policy allows (the CPU JIT keeps the serial walk). The reference
        // `gradnorm_sq` gives each tensor ONE thread to sum its elements, which for 118 M
        // adapter parameters is seconds of a single thread's time per step.
        v.extend(model::dispatch::LORA_BANK_PIPELINES);
        // EdgeRock: the anchor memory's kernels and the windowed attention scores, resolved by name.
        v.extend(model::anchor_memory::anchor_memory_pipelines());
        // The Gated-DeltaNet pool staging for a pool stored as bf16: the gather decodes, the scatter rounds.
        let gather_a = kernels::template::dtype_variant("pool_rows_gather2", kernels::POOL_ROWS_GATHER2, "pool_a", Dtype::BF16).unwrap();
        v.push(kernels::template::dtype_variant(gather_a.0, gather_a.1, "pool_b", Dtype::BF16).unwrap());
        v.push(("pool_rows_scatter2_bf16", kernels::POOL_ROWS_SCATTER2_BF16));
        // The exact fp32 product of the memory's scorer at a large decode batch (`memory_project`).
        v.push(("matmul_tile16", kernels::MATMUL_TILE16));
        v.push(("gqa_scores_win", kernels::GQA_SCORES_WIN));
        v.push(("gqa_ring_chunk_attn", kernels::GQA_RING_CHUNK_ATTN));
        // Device AdamW over bf16 or blockwise-int8 moments (`optim::Optim` resolves them by name).
        v.extend(optim::MOMENT_KERNELS);
        v.push(("gradnorm_part", kernels::GRADNORM_PART));
        v.push(("clip_coef_wg", kernels::CLIP_COEF_WG));
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
        // A 248k-row vocabulary's table exceeds one storage binding; `Ops::embed` gathers it in vocab tiles.
        v.push(("embed_tile", kernels::EMBED_TILE));
        // Copies between two ranges of one buffer (a KV prefix or a pooled state row into another slot).
        v.push(("copy_within", kernels::COPY_WITHIN));
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
            v.push(kernels::template::dtype_variant("embed_tile", kernels::EMBED_TILE, "emb", dt).unwrap());
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
        // A frozen bf16/NF4 base trains through the tiled bf16 dX and the
        // float-read NF4 kernels, and the chunked LM head resolves its
        // elementwise passes by name. Appended unless already registered above.
        // Training attention also runs the flash trio at long sequences, so
        // its three kernels are registered too.
        // A quantized layer-boundary residual stream packs and unpacks on the device
        // (`model::saved::kernels`).
        let flash = [
            ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA),
            ("flash_attn_causal_gqa_bwd_dq", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DQ),
            ("flash_attn_causal_gqa_bwd_dkv", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DKV),
        ];
        for &(name, src) in model::ops::frozen_base_kernels().iter().chain(model::head_ce::kernels()).chain(model::saved::kernels()).chain(&flash) {
            if !v.iter().any(|(have, _)| *have == name) {
                v.push((name, src));
            }
        }
        // The compact KV tiers (`model::kv_tier`) are resolved BY NAME when a
        // cache of that tier is used, so their kernels ride along here.
        let kv = model::kv_tier::kernel_list(&v);
        v.extend(kv);
        model::adapter::exec::add_kernels(&mut v);
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
const SPLICE: usize = 74;
const SPLICE_BWD: usize = 75;
const CAUSAL_CONV1D_STEP: usize = 76;
const RMSNORM_ROWS: usize = 84;
const RMSNORM_DX_ROWS: usize = 95;
const ARGMAX_PART: usize = 85;
const ARGMAX_FINAL: usize = 86;
const TOPK_EXTRACT_STEP: usize = 87;
const DECODE_SOFTMAX_BATCHED: usize = 88;
const GDN_UT_FWD: usize = 93;
const BMM_TILED: usize = 94;
/// Rows from which the memory's fp32 scorer takes the tiled product instead of the skinny-M kernel's 32-row groups.
const MEMORY_TILED_MIN_ROWS: u32 = 32;
const POOL_ROWS_GATHER2: usize = 96;
const POOL_ROWS_SCATTER2: usize = 97;
/// The bf16-pool builds of the two staging kernels, resolved by name (`Qwen35::kernel_by_name`).
const POOL_ROWS_GATHER2_BF16: &str = "pool_rows_gather2#pool_a=bf16#pool_b=bf16";
const POOL_ROWS_SCATTER2_BF16: &str = "pool_rows_scatter2_bf16";

/// The readout kernels of this crate's pipeline list (`model::readout`).
const READOUT: model::readout::ReadoutKernels = model::readout::ReadoutKernels { part: ARGMAX_PART, fin: ARGMAX_FINAL, extract: TOPK_EXTRACT_STEP };

/// The greedy pick of every row of a `[rows, vocab]` logits block, reduced on
/// the device: only the `rows` winning indices are read back, never the
/// logits. Shared by the resident-weight head ([`Qwen35::head_argmax_rows_dev`])
/// and the quantized out-of-shard head `crate::int8_gguf_resident` owns.
pub(crate) fn argmax_rows_of(g: &Gpu, logits: &DeviceBuffer, rows: u32, vocab: u32) -> Vec<u32> {
    model::readout::argmax_rows(g, READOUT, logits, rows, vocab)
}

/// Every row's top-`cap` `(token id, logit)` candidates, best first, from a
/// `[rows, vocab]` logits block, extracted on the device
/// (`model::readout::topk_rows`). Destroys `logits`.
pub(crate) fn topk_rows_of(g: &Gpu, logits: &DeviceBuffer, rows: u32, vocab: u32, cap: u32) -> Vec<Vec<(u32, f32)>> {
    model::readout::topk_rows(g, READOUT, logits, rows, vocab, cap)
}

/// [`argmax_rows_of`] without the read: the dispatches are submitted and the indices stay in the returned buffer.
pub(crate) fn argmax_rows_issue_of(g: &Gpu, logits: &DeviceBuffer, rows: u32, vocab: u32) -> DeviceBuffer {
    model::readout::argmax_rows_issue(g, READOUT, logits, rows, vocab)
}

/// [`topk_rows_of`] without the read.
pub(crate) fn topk_rows_issue_of(g: &Gpu, logits: &DeviceBuffer, rows: u32, vocab: u32, cap: u32) -> model::readout::TopkOut {
    model::readout::topk_rows_issue(g, READOUT, logits, rows, vocab, cap)
}

// M6.10 (kernel-performance ledger) replaced the fixed-schedule constant
// that used to live here (`DRAIN_EVERY_N_LAYERS`, a periodic BLOCKING drain
// every 4 layers regardless of actual buffer liveness) with real liveness
// tracking in `Qwen35::run_prefill_chunk_stage`'s layer loop: a
// `gpu_core::scratch::Arena` scope per layer, with the device drain moved to
// exactly the point that scope's buffer reuse requires one, rather than a
// count of layers. See that function's own comments for the mechanism and
// the ledger entry for the measured before/after.

/// The scratch arena a decode step's layer stack replays. Named, because the
/// head that follows the stack runs on the same handle and replays an arena of
/// its own: one shared arena would be evicted by each of them in turn.
pub(crate) const DECODE_ARENA: &str = "qwen35.decode";

/// The scratch arenas of a verify round of `2..=8` rows (one row is a decode
/// step and uses [`DECODE_ARENA`]), one per row count: an arena replays a fixed
/// sequence of requests and every row count asks for a different one.
const ROUND_ARENAS: [&str; 7] = ["qwen35.round.2", "qwen35.round.3", "qwen35.round.4", "qwen35.round.5", "qwen35.round.6", "qwen35.round.7", "qwen35.round.8"];

/// The arena a round of `rows` consecutive tokens of one sequence replays.
fn round_arena(rows: u32) -> &'static str {
    match rows {
        0 | 1 => DECODE_ARENA,
        r => ROUND_ARENAS[r as usize - 2],
    }
}

/// The scratch arena the logits head of a decode step replays.
pub(crate) const HEAD_ARENA: &str = "qwen35.head";

/// Granularity, in key positions, of a decode step's attention scratch (the
/// `scores`/`probs` stride). Rounding the stride up to a multiple of this makes
/// the step identical from token to token within a bucket, which is what lets
/// its buffers be recycled and its dispatches replayed; the cost is at most
/// this many idle threads per attention head in the scores kernel.
const DECODE_CAP_BUCKET: u32 = 128;

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
// perf-number: measured by `gguf_resident_verify_cost_real::the_verify_round_cost_ladder`
/// The fence costs 1.4-1.7x at exactly the row counts a speculative verify
/// round uses (`k + 1`, so 2-8), and the curve crosses over between 64 and 128.
/// Anywhere in 16..=64 is within measurement noise of the same choice and no
/// production caller runs a round in that range anyway - a verify round is
/// `k + 1`, a prompt round is `MAX_PREFILL_TOKENS`, and only a ragged final
/// prefill chunk lands between. 16 is the conservative end of that window: it
/// keeps MORE rounds on the pool, and the pool is what bounds allocation churn
/// (M6.10's actual purpose, whose stability claim is not re-measured here).
const CHUNK_ARENA_MIN_ROWS: u32 = 16;

/// Words of logits the chunked LM head keeps alive at once unless
/// [`TrainOpts::head_budget_words`] says otherwise: 1 GiB of fp32.
const HEAD_BUDGET_WORDS: u64 = 1 << 28;

/// The Gated-DeltaNet chunk a long training sequence is padded to a multiple of.
const GDN_PAD_QUANTUM: u32 = 64;

/// This model's RMSNorm epsilon. Exactly what `rmsnorm.wgsl` hardcodes, but it
/// has to be passed explicitly here - see [`rms_step`].
const RMS_EPS: f32 = 1e-6;

/// The native fused residual-add + RMSNorm launch (`add_rmsnorm_fwd`), `None` off CUDA and on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn native_add_rmsnorm(g: &Gpu, bufs: [&DeviceBuffer; 5], d: u32, rows: u32, eps: f32) -> Option<Step> {
    gpu_core::provider::cuda::add_rmsnorm_fwd_step(g, &bufs, d, rows, eps)
}

#[cfg(target_arch = "wasm32")]
fn native_add_rmsnorm(_g: &Gpu, _bufs: [&DeviceBuffer; 5], _d: u32, _rows: u32, _eps: f32) -> Option<Step> {
    None
}

/// The native RMSNorm backward with the residual gradient added (`rmsnorm_dx_add`), `None` off CUDA and on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn native_rmsnorm_dx_add(g: &Gpu, bufs: [&DeviceBuffer; 5], d: u32, rows: u32, eps: f32) -> Option<Step> {
    gpu_core::provider::cuda::rmsnorm_dx_add_step(g, &bufs, d, rows, eps)
}

#[cfg(target_arch = "wasm32")]
fn native_rmsnorm_dx_add(_g: &Gpu, _bufs: [&DeviceBuffer; 5], _d: u32, _rows: u32, _eps: f32) -> Option<Step> {
    None
}


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
// perf-number: measured by `qwen35_decode_profile` on the real 27B resident
/// TOP row of the decode profile at 48% of all device time - more than the
// perf-number: measured by `qwen35_decode_profile` on the real 27B resident
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
/// [`rms_step`] as the front end kernel (`add_rms_quant` without the add and the packing) where the device offers it and the unfused
/// norm would have been the cooperative RMSNorm - the only order that kernel reproduces, bit for bit: one launch of 4 us in place of
/// the generated norm's 40 us a row. The head of a decode token pays it once, after the last layer.
pub(crate) fn final_norm_step(g: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32) -> Step {
    let coop = block::rms_variant(g, RMSNORM, Some(RMSNORM_ROWS), rows, dim).0 == RMSNORM_ROWS;
    let params = [dim, rows, f(RMS_EPS), 2];
    if coop && g.has_fused(gpu_core::Fused::AddRmsQuant) && gpu_core::Fused::AddRmsQuant.serves(&params) {
        // Without an addend the kernel reads neither `b` nor writes `sum`; the slots still have to be bound, and alias what is there.
        let (xq, sx) = (g.storage(1), g.storage(1));
        if let Some(step) = g.fused_step(gpu_core::Fused::AddRmsQuant, &[x, x, w, out, out, &xq, &sx], &params) {
            return step;
        }
    }
    rms_step(g, x, w, out, dim, rows)
}

pub fn rms_step(g: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32) -> Step {
    let (kind, threads) = block::rms_variant(g, RMSNORM, Some(RMSNORM_ROWS), rows, dim);
    g.dispatch(kind, &[x, w, out], &[dim, rows, f(RMS_EPS)], threads)
}

pub(crate) fn kernel_ids() -> KernelIds {
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
        fast: Some(model::gdn::GdnFastIds { ut_fwd: GDN_UT_FWD, bmm_tiled: BMM_TILED, tensor_core: true }),
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

/// [`model::gdn_mixer::gdn_mixer_decode_fwd`]'s decode-only kernel ids.
fn gdn_mixer_decode_ids() -> model::gdn_mixer::GdnMixerDecodeIds {
    model::gdn_mixer::GdnMixerDecodeIds { conv: gdn_conv_ids(), splice: SPLICE }
}

/// Everything [`Qwen35::layer_gdn_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GDN mixer arm. `internals` is the LoRA-agnostic
/// mixer math's own saved state (`model::gdn_mixer`, shared with
/// `crates/qwen35moe`); `gated` is this crate's own `out_proj` input (the
/// hoisted forward's return value), kept here since only the LOCAL
/// `out_proj` backward reads it - see `model::gdn_mixer`'s own module doc.
/// What a GDN layer's backward starts from: every intermediate of the whole
/// sequence, or - with `TrainOpts::gdn_segment_tokens` - only the four
/// projections and a recurrent-state snapshot per segment, the intermediates
/// being rebuilt one segment at a time (`model::gdn_segment`).
enum GdnSaved {
    Full(Box<model::gdn_mixer::GdnMixerActs>),
    Segmented { bounds: model::gdn_segment::GdnSegmentBounds, mixed_qkv: DeviceBuffer, bproj: DeviceBuffer, aproj: DeviceBuffer, z: DeviceBuffer },
}

struct GdnLayerActs {
    saved: GdnSaved,
    gated: DeviceBuffer,
}

/// The physical KV blocks that back one sequence, in logical order, in a pool
/// whose blocks hold `gqa_cap` rows each. [`Self::Block`] is the layout every
/// engine had before paging: ONE block holds the whole sequence, so its rows are
/// `block * gqa_cap ..` of every layer's pool. [`Self::Table`] is a sequence that
/// spans several blocks (the pool's rows `table[i] * gqa_cap ..` hold its logical
/// rows `i * gqa_cap ..`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum KvWindow<'a> {
    Block(u32),
    Table(&'a [u32]),
    /// A sliding-window layer's ring ([`model::gqa_mixer::ring_row`]): ONE block of exactly the window's rows,
    /// position `p` in row `p % window`, reused for as long as the sequence runs. It holds the last `window`
    /// positions and no more, so it has no capacity to exceed.
    Ring(u32),
}

impl KvWindow<'_> {
    /// Rows the window holds: `block_size` per block; a ring holds the same rows forever, so any position fits.
    fn capacity(&self, block_size: u32) -> u32 {
        match self {
            KvWindow::Ring(_) => u32::MAX,
            _ => block_size * self.blocks().len() as u32,
        }
    }

    /// The blocks, in logical order.
    fn blocks(&self) -> Vec<u32> {
        match self {
            KvWindow::Block(b) | KvWindow::Ring(b) => vec![*b],
            KvWindow::Table(t) => t.to_vec(),
        }
    }

    fn is_ring(&self) -> bool {
        matches!(self, KvWindow::Ring(_))
    }

    /// The per-row addressing of `n` rows at positions `pos_start..`: the block
    /// each row's K/V is appended to, its offset inside the block, and the
    /// sequence's block table repeated for every row, with the table's length.
    fn chunk_rows(&self, block_size: u32, pos_start: u32, n: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32) {
        assert!(!self.is_ring(), "qwen35: a ring window is prefilled through the ring chunk path, not paged rows");
        let table = self.blocks();
        let max_bt = table.len() as u32;
        assert!(max_bt > 0, "qwen35: a sequence's KV window names no block");
        assert!(pos_start + n <= block_size * max_bt, "qwen35: rows {pos_start}..{} past the {} rows its {max_bt} block(s) hold", pos_start + n, block_size * max_bt);
        let blocks = (0..n).map(|i| table[((pos_start + i) / block_size) as usize]).collect();
        let offsets = (0..n).map(|i| (pos_start + i) % block_size).collect();
        let tables = (0..n).flat_map(|_| table.iter().copied()).collect();
        (blocks, offsets, tables, max_bt)
    }
}

/// One named intermediate of a window/memory layer, `(name, values)`: see [`Qwen35::enable_memory_taps`].
pub type MemoryTap = (String, Vec<f32>);

/// What one CHUNKED-prefill round hands [`Qwen35::layer_gqa_fwd`] so its GQA
/// layers attend the sequence's persistent KV cache instead of an isolated
/// `[T,T]` causal block over the round alone. Every field is built ONCE per
/// round by [`Qwen35::run_prefill_chunk`] and shared unchanged by every GQA
/// layer in it (`kcache`/`vcache` excepted - those are per layer).
struct GqaChunkCtx<'a> {
    /// Absolute position of this round's FIRST token (`0` on round 1).
    start: u32,
    /// Rows per physical block (`DecodeCaches::gqa_cap`).
    block_size: u32,
    /// Entries of the sequence's block table.
    max_bt: u32,
    layer: &'a KvLayer,
    /// `[n]` u32: the physical block each row's K/V is appended to.
    blocks: DeviceBuffer,
    /// `[n]` u32: each row's offset inside that block.
    offsets: DeviceBuffer,
    /// `[n, max_bt]` u32: the sequence's block table, for every row.
    tables: DeviceBuffer,
    /// `[n]` u32 with `seq_lens[i] == start+i+1`: this round's causal mask.
    seq_lens: DeviceBuffer,
    /// This round's own `[n, rotary_dim/2]` M-RoPE tables, for absolute
    /// positions `start..start+n`.
    cos: &'a DeviceBuffer,
    sin: &'a DeviceBuffer,
}

/// What a chunk of a sliding-window layer hands [`Qwen35::layer_gqa_fwd`]: where the sequence's ring is and which
/// positions the chunk covers.
struct RingChunkCtx<'a> {
    start: u32,
    window: u32,
    /// First row of the sequence's ring in the layer's pool.
    row0: u32,
    layer: &'a KvLayer,
    cos: &'a DeviceBuffer,
    sin: &'a DeviceBuffer,
}

/// What one BATCHED DECODE step hands [`Qwen35::layer_gqa_fwd`] - the
/// cross-sequence counterpart of [`GqaChunkCtx`]. `paged`/`cos`/`sin` are built
/// ONCE per step and shared unchanged by every GQA layer; the two pools are per
/// layer, so they are rebound each time round the layer loop.
struct GqaDecodeCtx<'a> {
    paged: &'a model::gqa_mixer::PagedDecodeBatch<'a>,
    layer: &'a KvLayer,
    /// `[bsz, rotary_dim/2]` M-RoPE tables - row `b` is sequence `b`'s OWN
    /// decode position, and the rows of one batch are unrelated positions.
    cos: &'a DeviceBuffer,
    sin: &'a DeviceBuffer,
}

/// How a sequence's `rows` prompt rows run through the Gated DeltaNet recurrence: `(row offset, rows, chunk)` pieces in
/// order, each continuing from the state the one before left. The recurrence is sequential over chunks (one launch
/// walks them, about a hundred microseconds a chunk), so the chunk must be as large as the kernels take (64):
/// `model::gdn::gdn_chunk_size`, the largest divisor of the round, would walk a 600-row slice in 75 chunks of 8 and a
/// 195-row one in 195 chunks of 1. The whole multiple of 64 runs as full chunks and the remainder (below 64) as one
/// chunk of its own size.
fn gdn_stream_pieces(rows: u32) -> impl Iterator<Item = (u32, u32, u32)> {
    const CHUNK: u32 = 64;
    let full = rows / CHUNK * CHUNK;
    let head = (full > 0).then_some((0, full, CHUNK));
    let tail = (rows > full).then_some((full, rows - full, rows - full));
    head.into_iter().chain(tail)
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
    /// [`Self::Decode`] with every sequence's state a row of a pool.
    DecodePool(model::gdn_mixer::GdnPoolRows<'a>),
    /// `n` consecutive decode steps of one sequence, in order - a short verify
    /// round, which must reproduce plain decode's arithmetic row by row.
    Rows(model::gdn_mixer::GdnStream<'a>),
    /// A speculative verify step: several consecutive rows per sequence, each
    /// sequence's state a row of a pool, the state between rows snapshotted.
    VerifyPool { pool: model::gdn_mixer::GdnPoolRows<'a>, layout: model::gdn_verify::VerifyLayout<'a>, snaps: &'a model::gdn_verify::GdnRowSnapshots, n_tokens: u32 },
    /// Several sequences' prefill rounds one after another in the rows, each
    /// continuing its own state row of a pool ([`Qwen35::run_prefill_multi_stage`]).
    Segments(&'a MultiRound<'a>),
}

/// Which cached-KV shape a [`Qwen35::layer_gqa_fwd`] call is running in.
/// `None` is the isolated `[T,T]` causal block a training/whole-sequence
/// forward wants; the two `Some` arms are the two ways a SERVING pass attends
/// a persistent cache - one sequence's many tokens ([`GqaChunkCtx`]) or many
/// sequences' one token each ([`GqaDecodeCtx`]).
enum GqaCached<'a> {
    Chunk(&'a GqaChunkCtx<'a>),
    /// One sequence's chunk of a sliding-window layer, against its ring ([`RingChunkCtx`]).
    Ring(&'a RingChunkCtx<'a>),
    Decode(&'a GqaDecodeCtx<'a>),
    /// Several sequences' rounds laid out one after another in the activation
    /// rows ([`Qwen35::run_prefill_multi_stage`]), attending this layer's cache.
    Segments(&'a MultiRound<'a>, &'a KvLayer),
}

/// One sequence's rows within a [`Qwen35::run_prefill_multi_stage`] round.
pub(crate) struct PrefillSeg<'a> {
    pub tokens: &'a [u32],
    /// Absolute position of the first of `tokens`.
    pub pos_start: u32,
    /// The physical blocks that back the sequence's KV in each layer's pool.
    pub gqa_window: KvWindow<'a>,
    /// The sequence's row of the recurrent-state pools.
    pub slot: u32,
}

/// The pooled caches a multi-sequence prefill round reads and extends: every
/// sequence's KV window is a slice of one pool per layer, and its recurrent state
/// a row of one pool per layer. The chunk kernels run on one state at a time, so a
/// sequence's row is gathered into the `work` copy before its rows go through a
/// layer and scattered back after.
pub(crate) struct PooledCaches<'a> {
    pub gqa_kv: &'a [KvLayer],
    pub gqa_cap: u32,
    pub gdn_state: &'a [DeviceBuffer],
    pub gdn_hist: &'a [DeviceBuffer],
    pub work_state: &'a [DeviceBuffer],
    pub work_hist: &'a [DeviceBuffer],
    /// Pipeline indices of `pool_rows_gather2` / `pool_rows_scatter2`: row of the
    /// pool -> work copy, and back.
    pub gather: usize,
    pub scatter: usize,
}

/// Per-row coordinates of a whole round (decode rows, then each sequence's prompt rows): where a row's K/V is
/// appended and the M-RoPE table at its own position.
struct RoundRows {
    /// `[n]` u32: the physical block of each row's K/V append.
    blocks: DeviceBuffer,
    /// `[n]` u32: the row inside that block.
    offsets: DeviceBuffer,
    /// `[n, rotary_dim/2]` M-RoPE tables at each row's position.
    cos: DeviceBuffer,
    sin: DeviceBuffer,
}

/// One sequence of a multi-sequence round, as the layers see it.
struct SegRound {
    rows: u32,
    /// First activation row of the sequence.
    row0: u32,
    /// `[1]` u32: the sequence's row of the state pools.
    slot: DeviceBuffer,
    start: u32,
    max_bt: u32,
    tables: DeviceBuffer,
    seq_lens: DeviceBuffer,
}

/// A multi-sequence round: the caches and every sequence's coordinates.
pub(crate) struct MultiRound<'a> {
    caches: &'a PooledCaches<'a>,
    segs: Vec<SegRound>,
    /// The coordinates of every row of the round, decode rows first, for the attention front half and the K/V
    /// append, which run once over the whole round.
    rows: RoundRows,
    /// The segments' table for the one-pass Gated DeltaNet kernels (`model::gdn_segments`), where the device serves them.
    gdn: Option<model::gdn_segments::GdnSegmentRound>,
    /// Decode rows that lead the round's activation rows (a mixed round, [`Qwen35::run_mixed_stage`]); the
    /// sequences' prompt rows follow them.
    decode: Option<MixedDecode<'a>>,
}

/// The decode rows of a mixed round, as the mixers see them: one token of each of `rows` unrelated sequences.
struct MixedDecode<'a> {
    rows: u32,
    /// `[rows]` u32: each row's recurrent-state pool row.
    slot_rows: &'a DeviceBuffer,
    paged: &'a model::gqa_mixer::PagedDecodeBatch<'a>,
    /// Set when the leading rows are a speculative verify step (several consecutive rows of a sequence) and not one
    /// token each of unrelated sequences: the recurrent mixer then runs the verify kernel, which snapshots the state
    /// between a sequence's rows.
    verify: Option<MixedVerify<'a>>,
}

/// The leading rows of a mixed round as a verify step: its sequences, and where each layer snapshots their states.
struct MixedVerify<'a> {
    batch: &'a verify::VerifyBatch<'a>,
    layout: &'a verify::VerifyLayout,
}

/// Everything [`Qwen35::layer_gqa_fwd`]'s training branch saves for
/// [`Qwen35::backward`]'s GQA mixer arm. `internals`/`ctx_gated` split the
/// same way as [`GdnLayerActs`] - see that struct's own doc.
struct GqaLayerActs {
    internals: model::gqa_mixer::GqaMixerActs,
    ctx_gated: DeviceBuffer,
    /// The layer's anchor memory (EdgeRock), whose output is added to the attention's.
    memory: Option<Box<model::anchor_memory::AnchorMemoryActs>>,
}

/// Who reads the output of a fused front end ([`Qwen35::rms_quant_front`]): which linears take the normalised rows, and so in which
/// form they are written.
#[derive(Clone, Copy, Debug)]
enum FrontUse {
    /// The projections of the token mixer of a layer of this type.
    Mixer(LayerType),
    /// The gate and up projections of the MLP.
    Mlp,
    /// The final norm: the head reads the fp32 rows.
    Head,
}

impl FrontUse {
    /// The output widths of the products that read E4M3 rows of the front's output when the device tunes them so; empty when
    /// the readers are not FP8 projections.
    fn product_widths(self, c: &Qwen35Config) -> Vec<u32> {
        match self {
            FrontUse::Mixer(LayerType::Linear) => vec![c.linear_conv_dim(), c.linear_value_dim()],
            FrontUse::Mixer(LayerType::Full) => vec![c.q_proj_dim(), c.kv_dim()],
            FrontUse::Mlp => vec![c.intermediate_size],
            FrontUse::Head => Vec::new(),
        }
    }

    /// Whether a reader takes the rows as fp32 whatever the projections do: a Gated DeltaNet layer's two gate projections are
    /// dense and read the fp32 rows.
    fn reads_f32(self) -> bool {
        matches!(self, FrontUse::Mixer(LayerType::Linear) | FrontUse::Head)
    }
}

/// Everything [`Qwen35::mlp_fwd`]'s training branch saves - universal, every
/// layer, both mixer types. The dense-MLP analogue of `qwen35moe::model::
/// MoeLayerActs` (this model has no MoE at all, so no router/expert acts).
/// What a layer's MLP backward starts from: its activations, or - when the
/// MLP runs in row chunks (`TrainOpts::mlp_rows`) - only its input, the
/// activations being rebuilt chunk by chunk.
enum MlpSaved {
    Full(MlpLayerActs),
    Rows { xn2: DeviceBuffer },
}

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
    mlp: MlpSaved,
}

/// The full backward activation cache for one `forward()` call on a
/// [`Qwen35::new_train_on`] instance. `Some` only right after a `forward()`
/// call (populated by `run_forward`'s train branch; read and taken by
/// `backward()`) - mirrors the engine-wide "forward reallocates fresh
/// buffers every call" convention, so `backward()` MUST run against the
/// same `forward()` call whose gradient it computes.
struct TrainActs {
    /// One entry per owned layer, or empty when the forward ran with
    /// recompute and the backward rebuilds each layer's from `res`.
    layers: Vec<LayerTrainActs>,
    recomputed: bool,
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
    /// Tokens per sequence the buffers are sized for: `t_user` rounded up to
    /// whole GDN chunks for a long training sequence (see `new_impl_on`).
    t: u32,
    /// Tokens per sequence callers pass and read back.
    t_user: u32,
    /// Tokens per sequence the last [`Self::set_batch`] computes: `t`, or, for a
    /// one-sequence training batch whose supervised positions end early, just
    /// past them (see [`Self::run_len`]). Every forward and backward of that
    /// batch runs this many tokens; the buffers keep their `t`-sized layout.
    t_run: Cell<u32>,
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
    /// The distinct ids of `tokens`, padded to `b * t` entries ([`model::ops::uniq_padded`]): the embedding gradient
    /// scatters over these rows only. Written beside `tokens` by [`Self::write_tokens`]; a training build's embed
    /// stage has one.
    tok_uniq: Option<DeviceBuffer>,
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
    /// True when `cos`/`sin` hold the compact table a trimmed multi-row batch
    /// uploaded (see [`Self::set_batch`]) and a full-length row therefore has
    /// to put the `t`-row table back first. Always false for a one-row build,
    /// which addresses the construction table directly.
    rope_compact: Cell<bool>,

    logits: DeviceBuffer,
    /// The chunked LM head + cross-entropy of a training build without MTP;
    /// `None` elsewhere (inference reads `logits`; MTP shares the head with
    /// its own fp32 path).
    head_ce: Option<HeadCe>,
    ce_buf: DeviceBuffer,
    /// CE-gradient uniform (`[n, vocab, IGNORE, count]`), written once per
    /// `backward()` call (`count` is only known after `set_batch`).
    ce_grad_uni: DeviceBuffer,
    /// Per-token objectives (weighted loss, logit scale) when opted into.
    objectives: objectives::State,

    /// Residual stream, one entry per layer boundary (`res[0]` = embeddings,
    /// `res[n_layers]` = input to the final norm) - the SSA activation-cache
    /// convention `qwen35moe::model` uses, kept even though nothing
    /// backprops through it yet: useful for parity debugging, any layer's
    /// residual output is independently readable via [`Self::debug_res`].
    res: RefCell<Vec<DeviceBuffer>>,

    /// The layer-boundary residuals in a narrow format when [`TrainOpts::saved`]
    /// is [`model::saved::SavedPolicy::Quantized`]; `res` then holds just two fp32 buffers,
    /// the current layer's input (`res[0]`) and output (`res[1]`).
    res_saved: Option<model::saved::BoundaryStore>,

    /// Backward's activation cache - see [`TrainActs`]'s own doc.
    train_acts: RefCell<Option<TrainActs>>,
    /// The flash-attention training kernels' pipeline indices, when they are
    /// registered on this device's kernel list.
    flash: Option<model::block::FlashGqaIds>,
    /// Rows per Gated-DeltaNet segment of a training layer, or `None` to keep
    /// the whole sequence's intermediates (see [`TrainOpts::gdn_segment_tokens`]).
    gdn_segment_tokens: Cell<Option<u32>>,
    /// Rows per chunk the MLP runs in on a recomputing training build, or
    /// `None` for the whole pass (see [`TrainOpts::mlp_rows`]).
    mlp_rows: Cell<Option<u32>>,
    /// Shortest sequence the training forward hands to [`Self::flash`].
    flash_min_t: Cell<u32>,
    /// Whether `forward()` drops each layer's activations after use and
    /// `backward()` recomputes them from the layer-boundary residuals.
    recompute: Cell<bool>,

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
    /// The per-layer linears of a resident low-precision full fine-tune, their master copies and state
    /// ([`TrainOpts::resident`]); `weights` then holds handles to its device weights and the parameter store
    /// holds none of them.
    resident: Option<RefCell<model::resident_ft::ResidentFullFt>>,
    /// The global gradient norm of the latest clipped resident step (it spans both parameter groups).
    resident_norm: Cell<Option<f32>>,
    /// A layer-streamed build defers its backward to [`Qwen35::adamw_step`], which runs it once per pass over the
    /// layers; set by `backward`, cleared by the step and by `zero_grads`.
    backward_pending: Cell<bool>,

    // ---- LoRA scratch (persistent, reused across every targeted linear) ----
    // Sized once at construction for `cfg.lora`'s rank and the widest output
    // dimension across the 12 targetable leaves (`crate::config::
    // lora_targets`). See [`Self::lora_fwd`]/[`Self::proj_bwd`]'s LoRA
    // branch. Size-1 dummies when `cfg.lora` is `None` (rank forced to 1 in
    // `new_impl_on`, never read) - mirrors `qwen35moe::model::Qwen35`'s own
    // `lora_a`/`lora_da`/`lora_out` fields exactly.
    /// The trainable adapter's execution (forward correction, backward), when
    /// `cfg.lora` is set.
    adapter: Option<AdapterRuntime>,

    // ---- served adapter bank (see `Self::enable_adapter_bank`) ----
    /// Many adapters on this one copy of the weights, each row of a batch
    /// choosing its own. `None` until enabled; a LoRA TRAINING build
    /// (`cfg.lora`) never has one.
    adapter_bank: RefCell<Option<AdapterBank>>,
    /// Rows of the forward about to run that [`Self::set_adapter_rows`]
    /// assigned slots to; `0` when no row of it has an adapter, in which case
    /// the forward records exactly the base's dispatches.
    adapter_rows: Cell<u32>,

    // ---- vision-language embedding splice seam (see `crate::vl::Qwen35Vl`) ------
    /// The image-placeholder runs `run_forward` overwrites and `backward`
    /// routes to [`Self::read_d_img_embeds`], set by
    /// [`Self::enable_mm_splices`]. Empty (the default) makes both a no-op.
    splice: model::vlm::SpliceRuns,

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
    /// The single-sequence caches [`Self::step`] and the chunked prefill
    /// read and update - see [`DecodeState`]. Built with the instance for an
    /// inference build; a TRAINING build allocates it on its first decode
    /// (generating from the model being trained), because the f32 KV cache
    /// alone is `2 * kv_dim * 4` bytes per token per GQA layer, dead weight
    /// for a build that only runs whole-sequence forwards and backwards.
    decode: OnceCell<DecodeState>,

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
    /// Intermediates of every window/memory layer of the last one-shot forward,
    /// named `layer{l}.swa.out` / `layer{l}.mem.{scores,gate,query,key,value,read,refs,out}`
    /// as the reference dumper names them. `None` (the default) keeps nothing;
    /// see [`Self::enable_memory_taps`].
    memory_taps: RefCell<Option<Vec<MemoryTap>>>,

    /// This instance's [`CHUNK_ARENA_MIN_ROWS`] - see
    /// [`Self::set_chunk_arena_min_rows`].
    chunk_arena_min_rows: Cell<u32>,
    /// Whether a pooled round drains the device between layers even where submissions are ordered - see
    /// [`Self::set_chunk_arena_drain`]. `None`: only where the backend does not order its submissions.
    chunk_arena_drain: Cell<Option<bool>>,
    /// Whether a short chunk round runs as plain decode steps - see
    /// [`Self::set_exact_rounds`].
    exact_rounds: Cell<bool>,
    /// The exact-decode regime's configured per-sequence context, when on:
    /// every batched decode step runs in groups whose int8 projections are the
    /// bit-identical kernel and sizes its attention from this context rather
    /// than the batch - see [`Self::set_exact_decode`].
    exact_decode: Cell<Option<u32>>,
    /// Whether a decode step may use the native fused kernels where the device
    /// offers them - see [`Self::set_decode_fusion`].
    decode_fusion: Cell<bool>,
    /// The element type of the serving activation stream between the matrix products ([`Self::set_act_precision`]).
    act_precision: Cell<gpu_core::ActPrecision>,
    /// Two one-word buffers the fused front ends bind in place of the packed int8 rows and their scales when no weight
    /// reads them ([`Self::packs_activations`]): the kernel writes neither, but a dispatch may not bind one buffer twice.
    unpacked_slots: std::cell::OnceCell<(DeviceBuffer, DeviceBuffer)>,
    /// Whether `slot_decode` replays a recorded step ([`Self::set_slot_decode_tape`]).
    slot_decode_tapes: Cell<bool>,
    /// Whether the pooled Gated-DeltaNet decode runs the occupancy-built candidate ([`Self::set_gdn_pool_wide`]).
    gdn_pool_wide: Cell<bool>,
}

/// Which per-sequence GQA cache / GDN recurrent state one [`Qwen35::
/// run_decode_step`] call reads and updates - this instance's own persistent
/// decode state (`self.gqa_kv`/`self.gdn_state`/
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
/// irrelevant" convention this struct's own `gqa_kv`/`gdn_state` fields
/// already use.
pub(crate) struct DecodeCaches<'a> {
    /// Per-layer `[cap, kv_dim]` KV cache for GQA layers (placeholder at GDN
    /// indices), each layer's K and V planes in the cache's [`KvTier`].
    pub gqa_kv: &'a [KvLayer],
    /// Cache row capacity, shared by every GQA layer's cache in this call
    /// (one per-sequence capacity, not a per-layer one).
    pub gqa_cap: u32,
    /// The physical blocks that back this sequence's KV in the bound buffers.
    /// `Block(0)` - the only value every caller but a paged serving engine ever
    /// uses - means the buffers above are this sequence's own dedicated flat
    /// cache. A serving engine that keeps ONE pool per layer binds the whole pool
    /// and names the sequence's block (`Block(physical_block)`, one block holding
    /// the whole sequence) or its blocks (`Table`, `gqa_cap` rows each).
    pub gqa_window: KvWindow<'a>,
    /// Per-layer Gated-DeltaNet recurrent `state`/conv `hist` for GDN layers
    /// (dummy at GQA indices) - `gdn_recurrent_step`'s `state`,
    /// `gdn_causal_conv1d_step`'s `hist`.
    pub gdn_state: &'a [DeviceBuffer],
    pub gdn_hist: &'a [DeviceBuffer],
    /// The sequence's row of the anchor-memory pools below (and of a ring-KV pool: its block is named by
    /// [`Self::gqa_window`]).
    pub slot: u32,
    /// Per-layer anchor-memory banks (EdgeRock; `None` entries at layers without a memory, an empty slice for a
    /// model without one).
    pub memory: &'a [Option<model::anchor_memory::AnchorBankPool>],
}

/// One sequence's coordinates in a [`Qwen35::run_decode_batch`] call: where
/// its KV history lives, where this step's token goes, and its own
/// Gated-DeltaNet state. The batched counterpart of the per-sequence half of
/// [`DecodeCaches`]: the GQA POOL is shared by the whole batch
/// ([`BatchDecodeCaches`]), this is everything that is not.
pub(crate) struct BatchSeq<'a> {
    /// The physical block(s) backing this sequence's KV history in every
    /// layer's pool (`BatchDecodeCaches::gqa_cap` rows each).
    pub window: KvWindow<'a>,
    /// The sequence's row of the recurrent-state pools ([`GdnPoolStore`]); unused
    /// where the state is not a pool row (`gdn_pool: None`). Independent of the KV
    /// blocks: a sequence's state is one row however many blocks its KV spans.
    pub slot: u32,
    /// Absolute decode position of this step's token for this sequence. The
    /// rows of one batch are unrelated positions.
    pub pos: u32,
    /// Per-layer recurrent state / conv history, indexed by ABSOLUTE layer
    /// index with a dummy at full-attention indices - [`DecodeCaches`]'s own
    /// convention, and the same buffers.
    pub gdn_state: &'a [DeviceBuffer],
    pub gdn_hist: &'a [DeviceBuffer],
}

/// A serving engine's recurrent state for its whole resident set: per layer one
/// pool of rows of state and one of conv history (a size-1 placeholder at GQA
/// layers), row `p` belonging to the sequence on physical block `p`. A batched
/// decode step finds each sequence's row by [`BatchSeq::phys`], so the whole
/// batch is staged by one dispatch in and one out per layer however many
/// sequences it has - and, where the device offers it, runs as ONE native
/// launch on those rows in place (`model::gdn_mixer::gdn_mixer_decode_pooled_fused`).
pub(crate) struct GdnPoolStore<'a> {
    pub state: &'a [DeviceBuffer],
    pub hist: &'a [DeviceBuffer],
    /// The pools store bf16 ([`crate::decode_state::SlotTiers::recurrent`]); only a decode step reads them so.
    pub bf16: bool,
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
    pub seqs: &'a [BatchSeq<'a>],
    /// The rows are CONSECUTIVE tokens of ONE sequence, not unrelated
    /// sequences: every entry of `seqs` names the same `phys` and the same
    /// recurrent buffers, at successive `pos`. A Gated-DeltaNet layer then has
    /// to take the rows in order, each from the state the previous one left,
    /// which is what makes this a verify round and not a batch - see
    /// [`Qwen35::advance_exact_round`].
    pub one_sequence: bool,
    /// Where the sequences' Gated-DeltaNet state lives when it is pooled by
    /// physical block. `None`: each [`BatchSeq`] carries its own buffers.
    pub gdn_pool: Option<GdnPoolStore<'a>>,
    /// Per-layer anchor-memory banks, one pool row per sequence slot ([`BatchSeq::slot`]); see
    /// [`DecodeCaches::memory`].
    pub memory: &'a [Option<model::anchor_memory::AnchorBankPool>],
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

/// A copy of one sequence's whole fixed-size decode state - recurrent state, sliding-window ring, anchor bank - and
/// its position, taken by [`Qwen35::decode_snapshot`] and put back by [`Qwen35::decode_restore`]. Its size does not
/// depend on how long the sequence is.
pub struct DecodeSnapshot {
    gdn: GdnSnapshot,
    /// `(k, v)` ring copies per layer (1-word dummies where the layer has none).
    rings: Vec<(DeviceBuffer, DeviceBuffer)>,
    banks: Vec<Option<model::anchor_memory::AnchorBankPool>>,
    pos: u32,
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
    // The anchor memory's projections share leaf names (`memory.q_proj.weight`) but are plain fp32 parameters.
    !name.contains(".memory.") && LEAVES.iter().any(|leaf| name.ends_with(leaf))
}

/// The precision of each tensor a resident full fine-tune of `shard` leaves in the parameter store (the norms and
/// recurrent-mixer parameters), checked against what the store trains: the run's precision governs the resident
/// linears and tables, not these ([`model::TrainPrecision::resident_store`]). A choice the optimizer does not run is
/// an `Err` naming the tensor.
pub fn check_resident_store_precision(cfg: &Qwen35Config, shard: &Shard, precision: &model::TrainPrecision) -> Result<Vec<paramstore::precision::ParamPrecision>, String> {
    let resident: std::collections::HashSet<String> = resident_specs(cfg, shard).into_iter().map(|s| s.name).collect();
    let store: Vec<(String, usize)> = shard_param_list(cfg, shard).into_iter().filter(|(n, _)| !resident.contains(n)).collect();
    let store_precision = precision.resident_store();
    store_precision.resolve(&store, store_precision.wants_host_moments())
}

/// The tensors of `shard` a resident full fine-tune holds in its own state: every matrix
/// [`Qwen35Config::layer_leaves`] names, layer by layer, in the run's encoding, then the vocabulary tables the shard
/// owns (the embedding table, and an untied LM head) in bf16. The norms stay in the parameter store.
pub fn resident_specs(cfg: &Qwen35Config, shard: &Shard) -> Vec<model::resident_ft::ResidentSpec> {
    use model::resident_ft::ResidentSpec;
    let mut specs = Vec::new();
    for (l, ty) in cfg.layer_types().iter().enumerate().filter(|(l, _)| shard.owns(*l)) {
        for (suffix, n, k) in cfg.layer_leaves(*ty) {
            specs.push(ResidentSpec::linear(format!("blocks.{l}.{suffix}"), n, k, l as u32));
        }
    }
    let head = cfg.head_weight();
    let tied = head == "tok.weight";
    if shard.embed || (tied && shard.head) {
        specs.push(ResidentSpec::table("tok.weight", cfg.vocab, cfg.d_model));
    }
    if !tied && shard.head {
        specs.push(ResidentSpec::table(head, cfg.vocab, cfg.d_model));
    }
    specs
}

impl Qwen35 {
    pub fn new_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), None, shard)
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
    ///
    /// `init` is any [`checkpoint::TensorSource`] - an in-memory map or a
    /// streaming checkpoint source - so a model loads tensor by tensor with no
    /// whole-model host copy.
    pub fn new_on_dt(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &dyn checkpoint::TensorSource, tier: &TierPolicy) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        Qwen35::new_impl_on(gpu, cfg, b, t, init, tier, None, shard)
    }

    /// Build a fully trainable instance (every weight `Role::Trainable`,
    /// full-parameter, or - when `cfg.lora` is `Some` - frozen base +
    /// trainable LoRA adapters). See the struct's own `is_train` doc.
    pub fn new_train_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        Qwen35::new_train_opts_on(gpu, cfg, b, t, init, &TrainOpts::default())
    }

    /// [`Self::new_train_on`] with explicit [`TrainOpts`]: the storage tier of
    /// the frozen base (a LoRA build may hold it in bf16, see
    /// [`TrainOpts::base`]). `src` is any [`checkpoint::TensorSource`], so a
    /// narrow base streams tensor by tensor with no whole-model fp32 copy.
    pub fn new_train_opts_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, opts: &TrainOpts) -> Qwen35 {
        let shard = Shard::whole(cfg.n_layers as usize);
        let m = Qwen35::new_impl_on(gpu, cfg, b, t, src, &opts.base, Some(opts), shard);
        m.set_recompute(opts.saved.recomputes());
        m.gdn_segment_tokens.set(opts.gdn_segment_tokens);
        m.mlp_rows.set(opts.mlp_rows);
        if let Some(min_t) = opts.flash_min_t {
            m.flash_min_t.set(min_t);
        }
        m
    }

    /// A resident low-precision full fine-tune build (see [`TrainOpts::resident`]): the planner's arithmetic over the
    /// resident weights, gradients, host-side master and state and the plan's other demands is checked before
    /// anything is allocated, and a plan that does not fit is an `Err` naming the category.
    pub fn new_resident(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, opts: &TrainOpts) -> Result<Qwen35, String> {
        let plan = opts.resident.as_ref().ok_or("qwen35: new_resident needs TrainOpts::resident")?;
        if cfg.lora.is_some() || cfg.mtp {
            return Err("qwen35: a resident full fine-tune is adapter-free and has no multi-token head".into());
        }
        let shard = Shard::whole(cfg.n_layers as usize);
        check_resident_store_precision(&cfg, &shard, &opts.precision)?;
        model::resident_ft::check_fits(&plan.cfg, &resident_specs(&cfg, &shard), &plan.budget())?;
        Ok(Qwen35::new_train_opts_on(gpu, cfg, b, t, src, opts))
    }

    /// Switch activation recompute ([`model::saved::SavedPolicy::Keep`] against
    /// [`model::saved::SavedPolicy::Recompute`], see [`TrainOpts::saved`]) for the next
    /// `forward()`. A forward and its backward must run under the same
    /// setting: the backward follows what the forward kept. A build with
    /// quantized boundaries has no kept activations to switch back to.
    pub fn set_recompute(&self, on: bool) {
        assert!(on || self.res_saved.is_none(), "qwen35: a build that holds its boundary residuals quantized always recomputes");
        self.recompute.set(on);
    }

    /// How many layers' saved activations the last `forward()` left for
    /// `backward()`: every layer without recompute, none with it. The
    /// quantity that decides whether training memory scales with depth.
    pub fn retained_layer_activations(&self) -> usize {
        self.train_acts.borrow().as_ref().map_or(0, |ta| ta.layers.len())
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
        Qwen35::new_impl_on(shard_gpu(&shard), cfg, b, t, init, &TierPolicy::uniform(Dtype::F32), Some(&TrainOpts::default()), shard)
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
        Qwen35::new_impl_on(shard_gpu(&shard), cfg, b, t, src, tier, None, shard)
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
        Qwen35::new_impl_on(shard_gpu(&shard), cfg, b, t, src, &TierPolicy::uniform(Dtype::F32), Some(&TrainOpts::default()), shard)
    }

    fn new_impl_on(
        gpu: Gpu,
        cfg: Qwen35Config,
        b: u32,
        t: u32,
        src: &dyn checkpoint::TensorSource,
        tier: &TierPolicy,
        train_opts: Option<&TrainOpts>,
        shard: Shard,
    ) -> Qwen35 {
        let train = train_opts.is_some();
        // The Gated DeltaNet form of the forward that saves nothing (the serving prefill, and a checkpointed training
        // forward) is measured now, not inside the first recorded step. A training forward that saves for the backward
        // keeps the exact TF32 kernels, so the gradients come from the exact recompute.
        if cfg.linear_key_head_dim == 128 && cfg.linear_value_head_dim == 128 {
            gpu.prepare_gdn_prefill();
        }
        if let Err(e) = cfg.check_window_and_memory() {
            panic!("{e}");
        }
        assert!(
            !(cfg.mtp && (cfg.swa_window.is_some() || cfg.anchor_memory.is_some())),
            "qwen35: the multi-token head reuses the full-attention layer and has no sliding window / anchor memory"
        );
        let head_budget = train_opts.and_then(|o| o.head_budget_words);
        let resident_plan = train_opts.and_then(|o| o.resident.as_ref());
        assert!(
            resident_plan.is_none() || (train && cfg.lora.is_none() && !cfg.mtp && shard.is_whole(cfg.n_layers as usize)),
            "qwen35: a resident low-precision full fine-tune is a whole-model, adapter-free build without a multi-token head"
        );
        assert!(
            !(tier.quantizes_anything() && train && cfg.lora.is_none()),
            "qwen35: a narrow tier needs LoRA when training: without adapters there is no frozen base \
             to hold at it (full-parameter training is fp32-only)"
        );
        assert!(
            !train || crate::base_source::LORA_BASES.contains(&tier.default_dtype()),
            "qwen35: a {} frozen base does not train; the trainable formats are {:?}",
            model::policy::format_name(tier.default_dtype()),
            crate::base_source::LORA_BASES.iter().map(|d| model::policy::format_name(*d)).collect::<Vec<_>>()
        );
        // A training sequence of an awkward length (4097 tokens would otherwise
        // run the Gated-DeltaNet recurrence in chunks of ONE token) is padded
        // at its end to a whole chunk of 64: everything is causal, so the
        // padding changes no real token's activation, and its targets are
        // IGNOREd so it adds no loss and no gradient. `t_user` is the length
        // callers see.
        let t_user = t;
        let t = if train && t > GDN_PAD_QUANTUM && !t.is_multiple_of(GDN_PAD_QUANTUM) {
            assert!(!cfg.mtp, "qwen35: MTP needs t={t} to be a multiple of {GDN_PAD_QUANTUM} (the multi-token head has no padding support)");
            t.next_multiple_of(GDN_PAD_QUANTUM)
        } else {
            t
        };
        if train {
            crate::train_base::check_batch_fits(&cfg, b, t, gpu.max_storage_binding_bytes());
        }
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
        // fp32 copy is ever uploaded. A narrow tier with training implies LoRA
        // (the `assert!` above), whose frozen base lives in `weights`; the
        // adapters are never quantizable leaves, so they stay in the store.
        let roles: Vec<(String, usize, Role)> = shard_param_list(&cfg, &shard)
            .into_iter()
            .filter(|(n, _)| !(is_quantizable_linear(n) && (tier.want(n) != Dtype::F32 || resident_plan.is_some())) && table_tier(tier, &cfg, train, n).is_none() && memory_linear_tier(tier, &cfg, train, n).is_none() && !(resident_plan.is_some() && is_table(&cfg, n)))
            .map(|(n, c)| {
                // The admission scorer is outside the loss's gradient (a discrete sort reads it): never optimised.
                let role = if !train || n.contains(".memory.admission_") {
                    Role::Frozen
                } else if cfg.lora.is_some() {
                    model::adapter::device::role_of(&n, &cfg.lora.as_ref().expect("checked by the branch").hp())
                } else {
                    Role::Trainable
                };
                (n, c, role)
            })
            .collect();
        let mut ps = if train && cfg.lora.is_none() {
            // A full fine-tune keeps its AdamW state with the formats the run's policy names: on the device where the
            // fused kernels run them (f32, bf16, int8 blocks), else (or when asked) in host memory, built offloaded so
            // no device moment is ever allocated. Anything else is refused with the parameter's name.
            let params: Vec<(String, usize)> = roles.iter().map(|(n, c, _)| (n.clone(), *c)).collect();
            let precision = train_opts.map(|o| o.precision.clone()).unwrap_or_default();
            // A resident build's store holds only what the resident state does not: those tensors train as the store runs them.
            let precision = if resident_plan.is_some() { precision.resident_store() } else { precision };
            let host = precision.wants_host_moments();
            precision.resolve(&params, host).unwrap_or_else(|e| panic!("qwen35 full fine-tune (unvalidated build): {e}"));
            if host {
                let roles = roles.into_iter().map(|(n, c, role)| (n, c, if role == Role::Trainable { Role::Offload } else { role })).collect();
                ParamStore::new_with_roles_src(&gpu, roles, src)
            } else {
                ParamStore::new_with_policy_src(&gpu, roles, src, &precision.policy)
            }
        } else {
            ParamStore::new_with_roles_src(&gpu, roles, src)
        };
        if let Some(lora) = cfg.lora.as_ref().filter(|_| train) {
            model::adapter::device::apply_groups(&mut ps, &lora.hp());
        }
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
        let mut ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("qwen35: Ops::new: {e}"));
        if let Some(o) = train_opts {
            ops.set_gemm_precision(o.gemm_precision);
            // A step allocates and frees thousands of activations; on CUDA each free is a
            // driver call that waits for the whole device. Keeping the freed blocks makes
            // a free a bookkeeping entry, at the price of the garbage between two drains
            // staying reserved (about 14 GiB at 8K tokens through the 64 layers): taken
            // when the base leaves that room. A bf16 27B base takes 54 GB, and with the
            // freed blocks held its 8K step peaks at about 81 GiB of the 95 (knowledge 311).
            if o.base.want("blocks.0.mlp.gate.weight") != Dtype::F32 || cfg.n_layers <= 32 {
                for handle in [&gpu, ops.gpu()] {
                    handle.hold_freed_blocks(true);
                    // The step cycles tens of gigabytes: without this the pool hands everything above its
                    // default bound back to the driver at every synchronisation and the next kernel waits
                    // for the driver to map it again.
                    handle.retain_freed_blocks(u64::MAX);
                }
            }
        }
        let resident_ft = resident_plan.map(|plan| {
            let specs = resident_specs(&cfg, &shard);
            model::resident_ft::ResidentFullFt::new(&ops, plan.cfg.clone(), specs, src, &plan.budget()).unwrap_or_else(|e| panic!("qwen35: {e}"))
        });
        let mut weights: HashMap<String, Weight> = HashMap::new();
        // Host staging for tiled uploads: one tile at a time, independent of the largest tensor.
        let staging = checkpoint::packed::BoundedWorkspace::new(checkpoint::packed::MemoryBudget::new(64 << 20, 0));
        let mut upload = |name: String, wn: u32, wk: u32, want: Dtype| {
            let resident_weight = resident_ft.as_ref().and_then(|ft| ft.index_of(&name).map(|i| ft.weight(i).clone()));
            let w = if let Some(w) = resident_weight {
                w
            } else if want == Dtype::F32 {
                Weight::dense_f32(ps.w(&name).clone(), wn, wk)
            } else {
                gguf::upload_weight(&ops, src, &name, wn as usize, wk as usize, want, &staging)
                    .unwrap_or_else(|| panic!("qwen35: {name} is missing, or not stored in a format {want:?} can hold"))
            };
            weights.insert(name, w);
        };
        for (l, ty) in cfg.layer_types().iter().enumerate() {
            if !shard.owns(l) {
                continue;
            }
            for (suffix, n, k) in cfg.layer_leaves(*ty) {
                let name = format!("blocks.{l}.{suffix}");
                let want = tier.want(&name);
                upload(name, n, k, want);
            }
        }
        // The anchor memory's read-side projections follow the weight tier of an inference build; the admission
        // scorer, the norms and the read gate stay in the parameter store at fp32.
        for (l, _) in cfg.layer_types().iter().enumerate().filter(|(l, _)| shard.owns(*l) && cfg.layer_has_memory(*l)) {
            for (leaf, n, k) in cfg.memory_linears() {
                let name = format!("blocks.{l}.memory.{leaf}");
                if let Some(want) = memory_linear_tier(tier, &cfg, train, &name) {
                    upload(name, n as u32, k as u32, want);
                }
            }
        }
        if cfg.mtp && shard.head {
            // MTP's single extra layer is architecturally a GQA layer + MLP
            // (`Self::run_mtp_forward` reuses `layer_gqa_fwd`/`mlp_fwd`
            // unchanged) - `layer_leaves(LayerType::Full)` names the exact
            // same 7 leaves, just under `mtp.layers.0.` instead of
            // `blocks.{l}.`.
            for (suffix, n, k) in cfg.layer_leaves(LayerType::Full) {
                let name = format!("mtp.layers.0.{suffix}");
                let want = tier.want(&name);
                upload(name, n, k, want);
            }
        }
        // The frozen embedding table and LM head of a narrow-base LoRA build:
        // streamed into a `Weight` like the linears (see `table_tier`); the
        // trained bf16 tables of a resident full fine-tune are its own weights.
        let head_name = cfg.head_weight();
        let tied = head_name == "tok.weight";
        let mut tables = vec![("tok.weight", shard.embed || (tied && shard.head))];
        if !tied {
            tables.push((head_name, shard.head));
        }
        for (name, owned) in tables {
            let want = if resident_plan.is_some() { Some(Dtype::BF16) } else { table_tier(tier, &cfg, train, name) };
            if let (true, Some(want)) = (owned, want) {
                upload(name.to_string(), cfg.vocab, cfg.d_model, want);
            }
        }

        // LoRA scratch (rank `r`; max projection output across all 12
        // targetable leaves - `crate::config::lora_targets` - mirrors
        // `qwen35moe::model::Qwen35::new_impl_on`'s own sizing, plus
        // `intermediate_size` for this model's dense-MLP `gate`/`up` leaves,
        // which qwen35moe never targets). `.max(1)` so a `cfg.lora: None`
        // build still gets a valid (unused) 1-element rank.
        let adapter = cfg.lora.as_ref().map(|lora| AdapterRuntime::new(&gpu, lora, b * t, cfg.adapter_max_out()));

        let ones_khd = gpu.storage_init("qwen35.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);

        // Text-only: every axis of the M-RoPE table carries the same plain
        // sequential position, reset per sequence (row = batch*t + pos).
        let (cos, sin) = Self::mrope_tables(&cfg, b, t);
        let cos = gpu.storage_init("qwen35.rope_cos", &cos);
        let sin = gpu.storage_init("qwen35.rope_sin", &sin);

        let n = (b * t) as u64;
        let tokens = gpu.storage(n);
        let tok_uniq = (train && shard.embed).then(|| gpu.storage(n));
        let targets = gpu.storage(n);
        // A non-head shard never reads `logits` (`run_forward` only writes it
        // `if self.shard.head`) - on an inference build this is `n * vocab`
        // (248320) of dead VRAM per non-head card.
        //
        // A training build without MTP never materialises `n * vocab` logits
        // (a 32K-token sequence of this vocabulary is 32 GB in fp32): its loss
        // and logit gradient come from the chunked LM head, which keeps a few
        // hundred rows of logits and recomputes them in the backward. The head
        // table is whatever tier `weights` holds (bf16 for a narrow base), or
        // the fp32 store tensor shared as a `Weight`.
        let flash = match (gpu.kernel_index("flash_attn_causal_gqa"), gpu.kernel_index("flash_attn_causal_gqa_bwd_dq"), gpu.kernel_index("flash_attn_causal_gqa_bwd_dkv")) {
            (Some(fwd), Some(bwd_dq), Some(bwd_dkv)) => Some(model::block::FlashGqaIds { fwd, bwd_dq, bwd_dkv, native: !ops.tensor_gemm().is_reference_pinned() }),
            _ => None,
        };
        let head_ce = (train && shard.head && !cfg.mtp).then(|| {
            let name = cfg.head_weight();
            let fp32_head;
            let head = match weights.get(name) {
                Some(w) => w,
                None => {
                    fp32_head = Weight::dense_f32(ps.w(name).clone(), cfg.vocab, cfg.d_model);
                    &fp32_head
                }
            };
            let rows = model::head_ce::rows_for_budget(n, cfg.vocab as u64, head_budget.unwrap_or(HEAD_BUDGET_WORDS)) as u32;
            let kernels = HeadKernels::from_gpu(&gpu).unwrap_or_else(|e| panic!("qwen35: LM head: {e}"));
            HeadCe::new(&ops, cfg.d_model, cfg.vocab, head, HeadCeConfig { rows_per_chunk: rows, tile_budget_words: None }, kernels)
                .unwrap_or_else(|e| panic!("qwen35: LM head: {e}"))
        });
        let logits = gpu.storage(if shard.head && head_ce.is_none() { n * cfg.vocab as u64 } else { 1 });
        let ce_buf = gpu.storage(n);
        let ce_grad_uni = gpu.uniform_dynamic(4);
        let d = cfg.d_model as u64;
        let res_saved = train_opts.map(|o| o.saved.boundary_format()).filter(|f| *f != SavedFormat::F32).map(|format| {
            assert!(shard.is_whole(cfg.n_layers as usize) && !cfg.mtp, "qwen35: quantized boundary residuals need a whole-model build without MTP (no stage seam or MTP head reads the fp32 stream)");
            model::saved::BoundaryStore::new(&gpu, format, cfg.n_layers as usize + 1, b * t, cfg.d_model).unwrap_or_else(|e| panic!("qwen35: {e}"))
        });
        // Two fp32 buffers (a layer's input and output) when the boundaries are
        // kept packed, one per boundary otherwise.
        let res = RefCell::new((0..if res_saved.is_some() { 2 } else { cfg.n_layers + 1 }).map(|_| gpu.storage(n * d)).collect());

        let mtp_input = gpu.storage(if cfg.mtp { n } else { 1 });
        let mtp_target = gpu.storage(if cfg.mtp { n } else { 1 });
        let mtp_logits = gpu.storage(if cfg.mtp { n * cfg.vocab as u64 } else { 1 });
        let mtp_ce_buf = gpu.storage(if cfg.mtp { n } else { 1 });

        // Single-sequence incremental decode state - see `Qwen35::decode`.
        // `dec_cap = t`: this pass's decode capacity is this instance's own
        // fixed prefill length (see `dec_cap`'s own doc).
        let decode = OnceCell::new();
        if !train {
            let _ = decode.set(DecodeState::new(&gpu, &cfg, &shard, t));
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
            t_user,
            t_run: Cell::new(t),
            chunk,
            is_train: train,
            opt,
            tokens,
            tok_uniq,
            targets,
            count: Cell::new(1.0),
            ones_khd,
            cos,
            sin,
            rope_compact: Cell::new(false),
            logits,
            head_ce,
            ce_buf,
            ce_grad_uni,
            objectives: Default::default(),
            res,
            train_acts: RefCell::new(None),
            res_saved,
            recompute: Cell::new(false),
            flash,
            flash_min_t: Cell::new(model::block::FlashGqaIds::MIN_TRAIN_T),
            gdn_segment_tokens: Cell::new(None),
            mlp_rows: Cell::new(None),
            mtp_input,
            mtp_target,
            mtp_logits,
            mtp_ce_buf,
            mtp_acts: RefCell::new(None),
            ops,
            weights,
            resident: resident_ft.map(RefCell::new),
            resident_norm: Cell::new(None),
            backward_pending: Cell::new(false),
            adapter,
            adapter_bank: RefCell::new(None),
            adapter_rows: Cell::new(0),
            splice: model::vlm::SpliceRuns::none(),
            dec_pos: Cell::new(0),
            dec_cap: t,
            decode,
            dres_boundary_in,
            dres_boundary_out,
            shard,
            taps: RefCell::new(Vec::new()),
            tapped: RefCell::new(Vec::new()),
            memory_taps: RefCell::new(None),
            chunk_arena_min_rows: Cell::new(CHUNK_ARENA_MIN_ROWS),
            chunk_arena_drain: Cell::new(None),
            exact_rounds: Cell::new(true),
            exact_decode: Cell::new(None),
            decode_fusion: Cell::new(true),
            act_precision: Cell::new(gpu_core::ActPrecision::F32),
            unpacked_slots: std::cell::OnceCell::new(),
            slot_decode_tapes: Cell::new(false),
            gdn_pool_wide: Cell::new(true),
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

    /// Force the pooled round's per-layer device drain on (`Some(true)`), or off (`Some(false)`), instead of the
    /// default (`None`): drain only on a backend whose submissions are not executed in order
    /// ([`Gpu::submissions_ordered`]). On an ordered backend the drain is not what keeps a reused scratch slot safe
    /// (the stream is), so the default lets the host record the next layer while the device runs this one.
    /// A knob for an A/B and for the test that holds the two bit-identical.
    pub fn set_chunk_arena_drain(&self, drain: Option<bool>) {
        self.chunk_arena_drain.set(drain);
    }

    /// **Run a short chunk round as plain decode steps, or as a chunk.** On
    /// (the default), a round of up to [`DECODE_REGIME_MAX_ROWS`] rows is
    /// computed by the decode tape's own machinery and agrees with decoding the
    /// same tokens one at a time to the last bit (see
    /// [`Self::advance_exact_round`]); off, every round takes the chunk tape, whose
    /// attention and recurrence are different arithmetic. Only a device that is
    /// offered the native recurrent kernel has the exact form at all.
    ///
    /// The off position exists to MEASURE the chunk tape: the distance between
    /// two correct renderings of the same model is the floor a numerical change
    /// elsewhere (a narrower KV tier, say) is judged against, and with the
    /// exact round on that distance is zero.
    pub fn set_exact_rounds(&self, on: bool) {
        self.exact_rounds.set(on);
    }

    /// **Serve every decode row independently of its co-batch**, or go back to
    /// one batched step.
    ///
    /// A decode step normally sends all `bsz` rows through one set of
    /// projections and one paged attention dispatch, sharing the weight reads.
    /// That makes a row's arithmetic a function of the batch: the int8 GEMV
    /// switches to its int8 tensor-core block from
    /// [`gpu_core::select::I8_GEMV_MMA_MIN_ROWS`] rows, and the fused decode's split
    /// count and key partition are chosen from the batch and the longest
    /// sequence. On, [`Self::run_decode_batch`] runs the rows in groups of
    /// [`Self::exact_group_rows`] (each on the bit-identical block) and sizes
    /// every row's attention from `configured_ctx`, so the same request
    /// produces the same bits alone, beside one neighbour, or beside seven.
    ///
    /// `configured_ctx` is the per-sequence `prompt + max_new` ceiling the KV
    /// cache was built for; it becomes every row's attention `cap`, so
    /// over-splitting a short sequence is the only cost of not knowing its
    /// neighbours' lengths. On a device running the fused decode that is a few
    /// empty split records per (sequence, kv head); on one running the
    /// scores/softmax/apply triad it also sizes their scratch and dispatch, so
    /// a large configured context is paid for there.
    ///
    /// The price is the weight traffic the grouping adds: one pass over every
    /// linear's weights per group, on the bit-identical dp4a block. At `bsz` 8
    /// on a device with int8 tensor cores (groups of 4) that is two passes
    /// where the step made one, and those passes stream at the dp4a rate the
    /// kernel was measured at rather than the batched tensor-core block's, so
    /// the projection weight time grows by roughly 2.4-3x and batch-8 aggregate
    /// throughput falls to about the batch-4 level
    /// (`crates/gpu-core/tests/i8_gemv_batched_variants.rs` has the rates).
    /// A speculative verify pass is not grouped
    /// (its rows are a sequence's consecutive tokens, not unrelated
    /// sequences); a serving caller turns speculation off in this regime.
    pub fn set_exact_decode(&self, configured_ctx: Option<u32>) {
        self.exact_decode.set(configured_ctx.map(|c| c.max(1)));
    }

    /// The exact-decode group size on this device, `None` when the regime is
    /// off.
    pub(crate) fn exact_decode_step(&self) -> Option<u32> {
        self.exact_decode.get().map(|_| gpu_core::select::exact_decode_group_rows(&self.gpu.caps()))
    }

    /// **Turn the decode step's native fused kernels off (or back on).** On by
    /// default, and on only matters where the device is offered them (a CUDA
    /// device; `gpu_core::set_native_kernels(false)` withholds them
    /// process-wide). Off,
    /// a decode step is the chain of WGSL dispatches the fused kernels replace,
    /// which is what their output is gated bit-for-bit against - so this is the
    /// A/B switch for that gate and for measuring the fusion, not a tuning knob.
    pub fn set_decode_fusion(&self, on: bool) {
        self.decode_fusion.set(on);
    }

    /// **Choose the element type of the activation stream between the FP8 products** (see [`gpu_core::ActPrecision`]):
    /// with bf16 the gate and up projections of the MLP store 2 bytes an element and the SwiGLU product reads them so. F32
    /// until set; a build that trains, serves adapters, or whose device or shape has no bf16-output form keeps f32 whatever
    /// is asked, per product, with the same results as before.
    pub fn set_act_precision(&self, precision: gpu_core::ActPrecision) {
        self.act_precision.set(precision);
    }

    /// The precision the serving activation stream was asked for.
    pub fn act_precision(&self) -> gpu_core::ActPrecision {
        self.act_precision.get()
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

    /// **Keep the intermediates of the sliding-window and anchor-memory layers**
    /// of every following one-shot forward ([`Self::logits_all`]), for parity
    /// debugging: see [`Self::take_memory_taps`]. Costs a readback per
    /// intermediate per layer, so it is off unless asked for.
    pub fn enable_memory_taps(&self) {
        *self.memory_taps.borrow_mut() = Some(Vec::new());
    }

    /// Take what the last forward tapped, `(name, values)` in the order the
    /// layers ran; the bank `layer{l}.mem.refs` holds the token index of each
    /// slot (`[blocks, groups, capacity]`, `-1` for an empty one). Draining, so
    /// a caller cannot read a previous forward's values.
    pub fn take_memory_taps(&self) -> Vec<MemoryTap> {
        self.memory_taps.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default()
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

    /// The adapter runtime's view of this build: the adapter tensors in the parameter store and the
    /// frozen linears in the tier they are held at, so a kind that reads its base (DoRA) works over any tier.
    fn adapter_params(&self) -> model::adapter::exec::TieredBase<'_> {
        model::adapter::exec::TieredBase { params: &self.ps, ops: &self.ops, weights: &self.weights }
    }

    /// The training LoRA's correction of a targeted linear: `y += scale *
    /// (x A^t) B^t`. No-op for an untargeted leaf (including `fc_e`, `fc_h` and
    /// `lm_head`, which are never targetable). `m` x `k` is the input, `nout`
    /// the output.
    #[allow(clippy::too_many_arguments)]
    fn lora_fwd(&self, s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, wname: &str, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        if let Some(rt) = self.adapter.as_ref().filter(|rt| rt.hits(leaf)) {
            rt.forward(self.ops.tensor_gemm(), s, &self.adapter_params(), wname, x, y, m, k, nout);
        }
    }

    /// The training LoRA's correction of one targeted linear whose base output
    /// `y` is already recorded in `s` (`float_tier` only: its fp32 path needs a
    /// float-read base weight). The served bank corrects after the whole group
    /// of linears that read `x` is recorded: [`Self::bank_fwd`].
    #[allow(clippy::too_many_arguments)]
    fn adapt(&self, float_tier: bool, s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, wname: &str, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        if float_tier {
            self.lora_fwd(s, leaf, x, wname, y, m, k, nout);
        }
    }

    /// The served bank's correction of the linears `items` (`(base weight name,
    /// its output)`, outputs recorded in `s`), all read from the same fp32
    /// activation `x` whatever their weight tier: `y[i] += scale * B_slot(i)
    /// (A_slot(i) x[i])`. Two dispatches for the whole group and batch. Nothing
    /// when no row of this forward has an adapter, and a row without one is
    /// untouched bit for bit.
    fn bank_fwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, items: &[(&str, &DeviceBuffer)], m: u32) {
        let rows = self.adapter_rows.get();
        if rows == 0 {
            return;
        }
        assert_eq!(m, rows, "qwen35: this forward has {m} rows but {rows} adapter slots were set for it");
        if let Some(bank) = self.adapter_bank.borrow().as_ref() {
            bank.correct_named_many(&self.gpu, s, x, m, items);
        }
    }

    /// Give this model an adapter bank over the sites of its own layers
    /// ([`LoraSites`]), for forwards of at most `rows` rows (the larger of the
    /// decode batch and the prefill round). Once, before serving.
    ///
    /// # Errors
    ///
    /// A bank already present, a LoRA training build, or a config the bank
    /// refuses ([`AdapterBank::new_for_layers`]).
    pub fn enable_adapter_bank(&self, cfg: AdapterBankConfig, rows: u32) -> Result<(), AdapterError> {
        if self.cfg.lora.is_some() {
            return Err(AdapterError::BadConfig("a LoRA training build corrects its linears with its own adapter; serve adapters from a base build".into()));
        }
        let mut slot = self.adapter_bank.borrow_mut();
        if slot.is_some() {
            return Err(AdapterError::BadConfig("the adapter bank is already enabled".into()));
        }
        *slot = Some(AdapterBank::new_for_layers(&self.gpu, &self.cfg.lora_sites(), cfg, rows, self.shard.start..self.shard.end)?);
        Ok(())
    }

    /// Run `f` on this model's adapter bank and its device; `None` without one.
    pub fn with_adapter_bank<R>(&self, f: impl FnOnce(&mut AdapterBank, &Gpu) -> R) -> Option<R> {
        self.adapter_bank.borrow_mut().as_mut().map(|bank| f(bank, &self.gpu))
    }

    /// Choose the adapter slot of each row of the forward about to run
    /// ([`model::dispatch::LORA_SLOT_NONE`] for a base row); an empty slice,
    /// or no bank, makes the forward the base's. Must come before the forward
    /// starts (a host write inside a replayed pass would split it) and be
    /// cleared with an empty slice when it is done.
    pub fn set_adapter_rows(&self, slots: &[u32]) {
        let bank = self.adapter_bank.borrow();
        let Some(bank) = bank.as_ref().filter(|_| !slots.is_empty()) else {
            self.adapter_rows.set(0);
            return;
        };
        bank.write_slots(&self.gpu, slots);
        self.adapter_rows.set(slots.len() as u32);
    }

    /// Dispatch one of the 12 per-layer mixer/MLP linears via `self.ops`/
    /// `self.weights`. Returns whether the weight's tier reads fp32
    /// activations ([`reads_float_acts`]): LoRA applies to every such tier
    /// (its correction is computed from the same fp32 activation), so a
    /// caller only runs `lora_fwd` when this is `true`.
    fn ops_linear(&self, s: &mut Vec<Step>, act: &Act, wname: &str, out: &DeviceBuffer) -> bool {
        let w = self.weights.get(wname).unwrap_or_else(|| panic!("qwen35: no Ops weight for {wname}"));
        self.ops.matmul(s, w, act, out, 0);
        reads_float_acts(w)
    }

    /// [`Self::ops_linear`] for several linears that read the SAME activation,
    /// each into its own buffer: a GDN layer's qkv/b/a/z, a GQA layer's q/k/v,
    /// the SwiGLU's gate/up. [`Ops::matmul_group`] makes the group one launch
    /// where the device offers it and an int8 GEMV serves every member, and a
    /// matmul per member otherwise (`set_decode_fusion(false)` always). Returns,
    /// per member, whether LoRA applies to it, exactly as [`Self::ops_linear`]
    /// reports it.
    fn ops_linear_group(&self, s: &mut Vec<Step>, act: &Act, group: &[(&str, &DeviceBuffer)]) -> Vec<bool> {
        let members: Vec<(&Weight, &DeviceBuffer)> =
            group.iter().map(|(name, out)| (self.weights.get(*name).unwrap_or_else(|| panic!("qwen35: no Ops weight for {name}")), *out)).collect();
        if self.decode_fusion.get() {
            self.ops.matmul_group(s, &members, act);
        } else {
            for (w, out) in &members {
                self.ops.matmul(s, w, act, out, 0);
            }
        }
        members.iter().map(|(w, _)| reads_float_acts(w)).collect()
    }

    /// Which arithmetic the bf16/NF4-weight GEMMs and the LoRA products of this
    /// build run: the vendor library's tensor cores, or the fp32 kernels (and
    /// why). Decided once from the device and [`TrainOpts::gemm_precision`].
    pub fn gemm_path(&self) -> &library::GemmPath {
        self.ops.gemm_path()
    }

    /// The format every parameter of this build is held in on the device, with its resident bytes: the tier of
    /// each `Ops` weight (frozen linears, narrow embedding tables), f32 for everything the parameter store
    /// holds (norms, recurrence parameters, adapters). A tier that is a storage encoding is named by its
    /// encoding, so the rows compare directly with a checkpoint's
    /// [`PackedSource::precision_report`](checkpoint::packed::PackedSource::precision_report).
    pub fn precision_report(&self) -> checkpoint::packed::PrecisionReport {
        let mut report = checkpoint::packed::PrecisionReport::default();
        for (name, numel) in self.cfg.param_list() {
            if let Some(w) = self.weights.get(&name).filter(|_| !self.ps.weight.contains_key(&name)) {
                let format = model::ops::encoding_for(w.dtype()).map_or_else(|| w.dtype().name().to_string(), |e| e.to_string());
                report.push(&name, vec![w.n() as u64, w.k() as u64], format, w.resident_bytes());
            } else if self.ps.weight.contains_key(&name) {
                report.push(&name, vec![numel as u64], checkpoint::packed::QuantEncoding::f32().to_string(), numel as u64 * 4);
            }
        }
        report
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
        if self.weights.values().all(reads_float_acts) {
            return self.ops.act_f32(x, 0, rows, k);
        }
        self.ops.act(s, x, 0, rows, k)
    }

    /// The gradient buffer for a trainable weight - only valid on a
    /// [`Self::new_train_on`] instance.
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }

    /// `sum = a + b; out = rmsnorm(sum) * w` (the residual add and the norm that follows it), as the one native
    /// launch `add_rmsnorm_fwd` where a CUDA device offers it, else `add2` and the norm.
    fn add_then_rmsnorm(&self, a: &DeviceBuffer, b: &DeviceBuffer, wname: &str, sum: &DeviceBuffer, out: &DeviceBuffer, d: u32, n: u32) {
        let g = &self.gpu;
        let w = self.w(wname);
        if let Some(fused) = native_add_rmsnorm(g, [a, b, w, sum, out], d, n, self.cfg.rms_eps) {
            g.submit(&[], &[fused]);
            return;
        }
        g.submit(&[], &[g.step(ADD2, &[a, b, sum], &[n * d], n * d)]);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), sum, w, out, d, n, self.cfg.rms_eps)]);
    }

    /// [`Self::rmsnorm_bwd_step`] followed by `total = res + dx`: one native launch (`rmsnorm_dx_add`) when the
    /// norm weight is frozen (no weight gradient to form) and a CUDA device offers it, else the two dispatches.
    /// The portable path needs a scratch `dx`; the fused one does not allocate it.
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_bwd_add_step(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, res: &DeviceBuffer, total: &DeviceBuffer, d: u32, n: u32) {
        if !self.trainable(wname) {
            if let Some(fused) = native_rmsnorm_dx_add(&self.gpu, [x, self.w(wname), dy, res, total], d, n, self.cfg.rms_eps) {
                steps.push(fused);
                return;
            }
        }
        let dx = self.gpu.storage(n as u64 * d as u64);
        self.rmsnorm_bwd_step(steps, x, wname, dy, &dx, d, n);
        steps.push(self.gpu.step(ADD2, &[res, &dx, total], &[n * d], n * d));
    }

    /// RMSNorm backward via the shared builder: input grad always, gain grad
    /// only when the gain is trainable.
    fn rmsnorm_bwd_step(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        let inv = self.gpu.storage(rows as u64);
        let gw = self.trainable(wname).then(|| self.g(wname));
        steps.extend(rmsnorm_bwd(&self.gpu, &kernel_ids(), x, self.w(wname), dy, dx, &inv, gw, dim, rows, self.cfg.rms_eps));
    }

    /// Tokens per sequence the current batch computes (see the `t_run` field).
    pub fn run_len(&self) -> u32 {
        self.t_run.get()
    }

    /// The GDN chunk size of [`Self::run_len`] tokens.
    fn run_chunk(&self) -> u32 {
        if self.run_len() == self.t {
            self.chunk
        } else {
            gdn_chunk_size(self.run_len())
        }
    }

    /// The per-row M-RoPE tables `rows` sequences of `row_len` tokens read,
    /// each sequence's positions starting at zero: one table row per token, at
    /// the token's index in the batch, which is what
    /// [`rope2d_partial_fwd`](model::block::rope2d_partial_fwd) addresses.
    /// Text-only, so every axis carries the same plain sequential position.
    /// `rope_table()` is `None` unless `cfg.rope_scaling` is set, which makes
    /// this bit-for-bit the plain `mrope_tables` output - see
    /// [`Qwen35Config::rope_table`]'s own doc.
    fn mrope_tables(cfg: &Qwen35Config, rows: u32, row_len: u32) -> (Vec<f32>, Vec<f32>) {
        let positions: Vec<[u32; 3]> = (0..rows).flat_map(|_| (0..row_len).map(|ti| [ti, ti, ti])).collect();
        let yarn = cfg.rope_table();
        qwen3vl::mrope::mrope_tables_scaled(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta, yarn.as_ref().map(|(f, a)| (f.as_slice(), *a)))
    }

    /// The tokens per sequence worth computing for `targets` (`b` rows of `t_user`):
    /// a training micro-batch needs nothing past the last supervised position of
    /// its LONGEST row. The model is causal and every row attends only to
    /// itself, so a row's tail past its last supervised position changes no
    /// earlier activation and adds no loss and no gradient; skipping it makes a
    /// record padded to the longest one's block cost its own length, and a
    /// micro-batch of several records cost the longest record in it. Rounded up
    /// to whole GDN chunks (the same padding `new_impl_on` applies to an awkward
    /// `t`), and never past the caller's own row length. MTP builds (their head
    /// shifts across the whole row), weighted-loss builds (their weights are laid
    /// out at `t`) and inference builds keep the full row.
    fn needed_len(&self, targets: &[u32]) -> u32 {
        let trimmable = self.is_train && !self.cfg.mtp && !self.objectives.is_weighted() && self.t > GDN_PAD_QUANTUM;
        if !trimmable {
            return self.t;
        }
        let row = self.t_user as usize;
        let last = (0..self.b as usize)
            .map(|s| targets[s * row..(s + 1) * row].iter().rposition(|&v| v != model::IGNORE).map_or(0, |i| i + 1))
            .max()
            .expect("a batch has at least one row") as u32;
        last.max(1).next_multiple_of(GDN_PAD_QUANTUM).min(self.t_user)
    }

    /// Upload the token ids every forward embeds, and on a training build the distinct-id list the embedding
    /// gradient scatters over.
    fn write_tokens(&self, ids: &[u32]) {
        self.gpu.write(&self.tokens, ids);
        if let Some(uniq) = &self.tok_uniq {
            self.gpu.write(uniq, &model::ops::uniq_padded(ids, (self.b * self.t) as usize));
        }
    }

    /// The gradient of table `name` (the embedding, an untied head) and its dtype: the resident state's on a resident
    /// full fine-tune, the parameter store's fp32 one where the table trains there, `None` for a frozen table.
    fn table_grad(&self, name: &str) -> Option<(DeviceBuffer, model::ops::GradDtype)> {
        if let Some(ft) = &self.resident {
            let ft = ft.borrow();
            if let Some(i) = ft.index_of(name) {
                return Some((ft.grad_buffer(i).clone(), ft.grad_dtype()));
            }
        }
        self.trainable(name).then(|| (self.g(name).clone(), model::ops::GradDtype::F32))
    }

    pub fn set_batch(&self, tokens: &[u32], targets: &[u32]) {
        let c = targets.iter().filter(|&&v| v != model::IGNORE).count();
        let (tu, b) = (self.t_user as usize, self.b as usize);
        assert!(tokens.len() == b * tu && targets.len() == b * tu, "qwen35::set_batch: {} tokens / {} targets for {b} sequences of {tu}", tokens.len(), targets.len());
        let run = self.needed_len(targets) as usize;
        self.t_run.set(run as u32);
        if run == self.t as usize {
            if self.t_user == self.t {
                self.write_tokens(tokens);
                self.gpu.write(&self.targets, targets);
            } else {
                // The construction's own padding to a whole GDN chunk, nothing
                // trimmed: the row the caller sent, in a `t`-long device row.
                let (tp, b) = (self.t as usize, b);
                let (mut padded_tokens, mut padded_targets) = (vec![0u32; b * tp], vec![model::IGNORE; b * tp]);
                for s in 0..b {
                    padded_tokens[s * tp..s * tp + tu].copy_from_slice(&tokens[s * tu..(s + 1) * tu]);
                    padded_targets[s * tp..s * tp + tu].copy_from_slice(&targets[s * tu..(s + 1) * tu]);
                }
                self.write_tokens(&padded_tokens);
                self.gpu.write(&self.targets, &padded_targets);
            }
        } else {
            // The rows are laid out at the run length, which is what every
            // forward and backward reads: `n = b * run_len`, row after row.
            let (mut packed_tokens, mut packed_targets) = (Vec::with_capacity(b * run), Vec::with_capacity(b * run));
            for s in 0..b {
                packed_tokens.extend_from_slice(&tokens[s * tu..s * tu + run]);
                packed_targets.extend_from_slice(&targets[s * tu..s * tu + run]);
            }
            self.write_tokens(&packed_tokens);
            self.gpu.write(&self.targets, &packed_targets);
        }
        // A multi-row batch packed to its run length addresses the M-RoPE table
        // by the token's own index, so the table becomes one block of `run`
        // positions per row; a full-length batch puts the `t`-row table back.
        // A one-row build addresses the construction table directly either way
        // (its run length is the table length the kernel is told).
        if self.b > 1 && (run < self.t as usize || self.rope_compact.get()) {
            let (cos, sin) = Self::mrope_tables(&self.cfg, self.b, run as u32);
            self.gpu.write_f32(&self.cos, &cos);
            self.gpu.write_f32(&self.sin, &sin);
            self.rope_compact.set(run < self.t as usize);
        }
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
        match self.adapter.as_ref().filter(|rt| rt.hits(leaf)) {
            Some(rt) => {
                // base: dx += d_out*W (frozen weight - no dW).
                rt.backward(self.ops.tensor_gemm(), steps, &self.adapter_params(), wname, d_out, x, dx, m, k, nout, &|s, dy| self.base_dx(s, dy, wname, dx, m, k, nout, acc));
            }
            None => {
                if self.trainable(wname) {
                    steps.push(g.step(MATMUL_DW, &[d_out, x, self.g(wname)], &[m, k, nout], nout * k));
                } else if let Some(ft) = &self.resident {
                    // A resident low-precision linear: its dW comes from the shared op against the stored tier.
                    let ft = ft.borrow();
                    if let Some(i) = ft.index_of(wname) {
                        ft.weight_grad(&self.ops, steps, i, x, d_out, m);
                    }
                }
                self.base_dx(steps, d_out, wname, dx, m, k, nout, acc);
            }
        }
    }

    /// `dx (+)= d_out * W` through the tier weight `wname` is held at: the 12
    /// per-layer linears live in `self.weights` (any float-read tier, decoded
    /// on the fly - the same value the forward multiplied by); the few other
    /// projections (`lm_head`, MTP's `fc_e`/`fc_h`) are plain fp32 store
    /// tensors.
    #[allow(clippy::too_many_arguments)]
    fn base_dx(&self, steps: &mut Vec<Step>, d_out: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        match self.weights.get(wname) {
            Some(w) => self.ops.matmul_dx(steps, w, d_out, m, dx, acc != 0),
            None => steps.push(self.gpu.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k)),
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
    /// enabling the splice is pure buffer allocation - call once after
    /// construction, before the first `forward()`.
    pub fn enable_mm_splice(&mut self, row0: u32, n_rows: u32) {
        self.enable_mm_splices(&[(row0, n_rows)]);
    }

    /// [`Self::enable_mm_splice`] for a prompt with several image runs (several
    /// images, or the views of a sliced one): each `(row0, n_rows)` run is
    /// overwritten by its own rows. [`Self::write_img_embeds`] takes the runs'
    /// rows concatenated in this order and [`Self::read_d_img_embeds`] returns
    /// their gradient the same way. The runs must not overlap.
    pub fn enable_mm_splices(&mut self, runs: &[(u32, u32)]) {
        self.splice = model::vlm::SpliceRuns::new(&self.gpu, runs, self.cfg.d_model);
    }

    /// Turn the splice off again: the next `forward()` reads every row from
    /// the token embedding (a text-only example after an image one).
    pub fn disable_mm_splice(&mut self) {
        self.splice = model::vlm::SpliceRuns::none();
    }

    /// The AdamW moments of trainable parameter `name` (`None` before the
    /// first step or for a frozen one) - what a resumable checkpoint saves.
    pub fn read_moments(&self, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
        self.ps.read_moments(&self.gpu, name)
    }

    /// Restore [`Self::read_moments`]'s values.
    pub fn write_moments(&self, name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
        self.ps.write_moments(&self.gpu, name, m, v)
    }

    /// Write the projected image tokens of every run (`[sum n_rows, d_model]`,
    /// row-major, in run order) to splice into the residual stream on the next
    /// `forward()`.
    pub fn write_img_embeds(&self, data: &[f32]) {
        self.splice.write(&self.gpu, data);
    }

    /// Read the gradient of the spliced image embeddings after `backward()`,
    /// in run order - feeds the vision tower/connector backward. Requires a
    /// `new_train_on` build (see [`Self::backward`]'s splice-gradient step).
    pub fn read_d_img_embeds(&self) -> Vec<f32> {
        self.splice.read_grad(&self.gpu)
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
        self.layer_gdn_fwd_pre(l, xn1, None, false, n, call)
    }

    /// [`Self::layer_gdn_fwd`] given `xn1`'s activation already quantised
    /// (`pre`), by a fused kernel that produced `xn1` and its int8 packing in
    /// one launch. `None` quantises here, as every other caller does.
    /// `epilogue` fuses the quantisation of what the layer hands `out_proj`
    /// into one launch ([`Self::quant_epilogue`]); only a decode step, after
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

        // in_proj_qkv (LoRA/int8 dispatch stays local - see `model::gdn_mixer`'s
        // own module doc). `xn1` quantized once here (`Ops::act`), reused
        // unchanged by in_proj_b/a/z below (no further `act` call on `xn1`
        // happens in between).
        // The four projections write every row of their outputs (the adapter then adds into them).
        // A round of prompt segments whose mixer runs in bf16 ([`Self::bf16_gdn_ok`]) takes `mixed_qkv` and `z` as the bf16 rows
        // the FP8 projections store; the other buffers stay f32 (the gates are tiny).
        let mut bf16_gdn = matches!(&call, GdnCall::Segments(r) if r.gdn.is_some()) && self.bf16_gdn_ok(n, value_dim);
        let rows_buffer = |width: u32, bf16: bool| g.storage_uninit(if bf16 { (n * width / 2) as u64 } else { (n * width) as u64 });
        let mut mixed_qkv = rows_buffer(conv_dim, bf16_gdn);
        let bproj = g.storage_uninit((n * nvh) as u64);
        let aproj = g.storage_uninit((n * nvh) as u64);
        let mut z = rows_buffer(value_dim, bf16_gdn);
        let mut s1 = Vec::new();
        let act1 = match pre {
            Some(a) => a,
            None => self.ops_act(&mut s1, xn1, n, d),
        };
        // in_proj_qkv/b/a/z all read `act1` (xn1 quantized once, shared): one
        // group, which is one launch where the device allows it.
        let (qkv_w, b_w, a_w, z_w) = (p("in_proj_qkv.weight"), p("in_proj_b.weight"), p("in_proj_a.weight"), p("in_proj_z.weight"));
        // The two gate projections are dense bf16 whatever the big projections' tier is, and a group fuses only members of one tier:
        // grouped apart, the pair is one launch of the multi-GEMV (six blocks each) beside the two big products.
        if bf16_gdn {
            // Both or neither; the steps pushed stay either way (they may hold the activation's quantisation, which it remembers).
            let mut t = Vec::new();
            let ok = [(&qkv_w, &mixed_qkv), (&z_w, &z)].iter().all(|(name, out)| self.ops.matmul_bf16_out(&mut t, &self.weights[name.as_str()], &act1, out));
            s1.append(&mut t);
            if !ok {
                bf16_gdn = false;
                mixed_qkv = rows_buffer(conv_dim, false);
                z = rows_buffer(value_dim, false);
            }
        }
        let f32s_big = if bf16_gdn { vec![false; 2] } else { self.ops_linear_group(&mut s1, &act1, &[(&qkv_w, &mixed_qkv), (&z_w, &z)]) };
        let f32s_gates = self.ops_linear_group(&mut s1, &act1, &[(&b_w, &bproj), (&a_w, &aproj)]);
        // (No adapter corrects a bf16 round: [`Self::keeps_fp32_product`] is its precondition.)
        self.adapt(f32s_big[0], &mut s1, "in_proj_qkv", xn1, &qkv_w, &mixed_qkv, n, d, conv_dim);
        self.adapt(f32s_gates[0], &mut s1, "in_proj_b", xn1, &b_w, &bproj, n, d, nvh);
        self.adapt(f32s_gates[1], &mut s1, "in_proj_a", xn1, &a_w, &aproj, n, d, nvh);
        self.adapt(f32s_big[1], &mut s1, "in_proj_z", xn1, &z_w, &z, n, d, value_dim);
        self.bank_fwd(&mut s1, xn1, &[(qkv_w.as_str(), &mixed_qkv), (z_w.as_str(), &z), (b_w.as_str(), &bproj), (a_w.as_str(), &aproj)], n);
        g.submit(&[], &s1);

        // conv+split+l2norm+decay-gate+recurrence+gated-norm - LoRA/dtype-
        // agnostic, shared with `crates/qwen35moe` (`model::gdn_mixer`).
        // A chunked round is ONE sequence's `n` rows, with its own chunk size
        // (`n` is a round's length, unrelated to this instance's construction
        // -time `t`); a batched decode step is `n` DIFFERENT sequences' one
        // token each; the whole-sequence forward keeps this instance's shape.
        let gdn_shape = match &call {
            GdnCall::Whole => GdnShape { b: self.b, h: nvh, t: self.run_len(), dk: khd, dv: vhd, chunk: self.run_chunk() },
            GdnCall::Chunk(_) | GdnCall::Segments(_) => GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: gdn_chunk_size(n) },
            GdnCall::Decode(_) | GdnCall::DecodePool(_) | GdnCall::VerifyPool { .. } => GdnShape { b: n, h: nvh, t: 1, dk: khd, dv: vhd, chunk: 1 },
            GdnCall::Rows(_) => GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: 1 },
        };
        let shape = model::gdn_mixer::GdnMixerShape { gdn: gdn_shape, nkh: c.linear_num_key_heads, conv_kernel: c.linear_conv_kernel_dim, rms_eps: c.rms_eps };
        let weights = model::gdn_mixer::GdnMixerWeights {
            conv1d_weight: self.w(&p("conv1d.weight")),
            a_log: self.w(&p("A_log")),
            dt_bias: self.w(&p("dt_bias")),
            norm_weight: self.w(&p("norm.weight")),
            ones_khd: &self.ones_khd,
        };
        // Set by the segments arm when the gated norm also wrote the E4M3 rows the output projection reads.
        let mut out_proj_act: Option<Act> = None;
        let (gated, internals) = match call {
            GdnCall::Decode(streams) => {
                // One sequence's step as a single native launch where the device
                // is offered it; the nineteen-kernel chain otherwise (and for a
                // batch, which the kernel does not serve).
                let fused = match streams {
                    [only] if self.decode_fusion.get() => model::gdn_mixer::gdn_mixer_decode_fused(g, &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, only),
                    _ => None,
                };
                let gated = fused.unwrap_or_else(|| {
                    model::gdn_mixer::gdn_mixer_decode_fwd(g, &gdn_mixer_ids(), &gdn_mixer_decode_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, streams)
                });
                (gated, None)
            }
            GdnCall::DecodePool(pool) => {
                let gated = model::gdn_mixer::gdn_mixer_decode_state_fwd(
                    g,
                    &gdn_mixer_ids(),
                    &gdn_mixer_decode_ids(),
                    &shape,
                    &weights,
                    &mixed_qkv,
                    &bproj,
                    &aproj,
                    &z,
                    &model::gdn_mixer::GdnDecodeState::Pool(pool),
                );
                (gated, None)
            }
            GdnCall::VerifyPool { pool, layout, snaps, n_tokens } => {
                let gated = model::gdn_verify::gdn_mixer_verify_pooled_fused(g, &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, &pool, &layout, n_tokens, snaps)
                    .expect("the verify kernel was offered at engine construction (Qwen35GgufInstance::speculation_supported)");
                (gated, None)
            }
            GdnCall::Rows(stream) => {
                let gated = model::gdn_mixer::gdn_mixer_rows_fused(g, &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, &stream, n)
                    .expect("exact_round_ok said the device serves the multi-row recurrent kernel at this shape");
                (gated, None)
            }
            GdnCall::Segments(round) => {
                let c = round.caches;
                let (sl, hl) = (self.cfg.gdn_state_len() as u32, self.cfg.gdn_hist_len() as u32);
                let whole = round.segs.len() == 1 && round.decode.is_none() && gdn_stream_pieces(round.segs[0].rows).count() == 1;
                // Every segment in one pass of the batched kernels on the pools' own rows, where the device serves it.
                let pools = model::gdn_segments::GdnSegmentPools { state: &c.gdn_state[l], hist: &c.gdn_hist[l] };
                let lead = round.decode.as_ref().map_or(0, |d| d.rows);
                // The output projection reads E4M3 rows on this device: the gated norm writes them beside its product.
                let e4m3 = !self.packs_activations() && self.ops.fp8_reads_e4m3(n, value_dim, d);
                let batched = round.gdn.as_ref().and_then(|sr| model::gdn_segments::gdn_mixer_segments_fwd(g, &gdn_mixer_ids(), &shape, &weights, sr, &pools, &mixed_qkv, &bproj, &aproj, &z, n, lead, e4m3, bf16_gdn));
                // The decode rows' recurrent step and the per-segment path have an f32 form only: the rows they read are
                // widened (the leading decode rows alone while the batched kernels took the round, every row if they declined).
                let widen = |buf: &DeviceBuffer, width: u32, rows: u32| {
                    let mut steps = Vec::new();
                    let wide = gpu_core::provider::library::widen_bf16(g, &mut steps, buf, u64::from(rows) * u64::from(width)).expect("the widening kernel was offered with the bf16 round");
                    g.submit(&[], &steps);
                    wide
                };
                let (mixed_qkv, z) = match (bf16_gdn, batched.is_some()) {
                    (true, false) => (widen(&mixed_qkv, conv_dim, n), widen(&z, value_dim, n)),
                    _ => (mixed_qkv, z),
                };
                let bf16_lead = bf16_gdn && batched.is_some();
                let (qkv_dec, z_dec) = match (&round.decode, bf16_lead) {
                    (Some(dec), true) => (Some(widen(&mixed_qkv, conv_dim, dec.rows)), Some(widen(&z, value_dim, dec.rows))),
                    _ => (None, None),
                };
                let (dec_qkv, dec_z) = (qkv_dec.as_ref().unwrap_or(&mixed_qkv), z_dec.as_ref().unwrap_or(&z));
                // The decode rows and the segments cover every row, so `all` needs no clear.
                let all = batched.as_ref().map(|b| b.gated.clone()).or_else(|| (!whole).then(|| g.storage_uninit((n * value_dim) as u64)));
                let mut only = None;
                // A mixed round's leading decode rows: one recurrent step each, from and back into the sequence's pool row.
                if let (Some(dec), Some(all)) = (&round.decode, &all) {
                    let dec_shape = model::gdn_mixer::GdnMixerShape { gdn: GdnShape { b: dec.rows, t: 1, chunk: 1, ..shape.gdn }, ..shape };
                    let pool = model::gdn_mixer::GdnPoolRows { state: &c.gdn_state[l], hist: &c.gdn_hist[l], rows: dec.slot_rows, gather: POOL_ROWS_GATHER2, scatter: POOL_ROWS_SCATTER2, fuse: self.decode_fusion.get(), bf16: false, wide: self.gdn_pool_wide.get() };
                    // The decode rows lead every row block: the fused step reads them in place and writes its rows into `all`.
                    let in_place = dec.verify.is_none() && batched.is_some() && pool.fuse && model::gdn_mixer::gdn_mixer_decode_pooled_fused_into(g, &dec_shape, &weights, dec_qkv, &bproj, &aproj, dec_z, &pool, all);
                    if !in_place {
                        let (qkv_d, b_d, a_d, z_d) = match (&qkv_dec, &z_dec) {
                            (Some(qkv), Some(z)) => (qkv.clone(), self.take_rows(&bproj, nvh, n, 0, dec.rows), self.take_rows(&aproj, nvh, n, 0, dec.rows), z.clone()),
                            _ => (self.take_rows(&mixed_qkv, conv_dim, n, 0, dec.rows), self.take_rows(&bproj, nvh, n, 0, dec.rows), self.take_rows(&aproj, nvh, n, 0, dec.rows), self.take_rows(&z, value_dim, n, 0, dec.rows)),
                        };
                        let gated_d = match &dec.verify {
                            // A sequence's rows in order from its pool row, the state after each row but the last snapshotted.
                            Some(v) => {
                                let layout = model::gdn_verify::VerifyLayout { rows: &v.layout.rows, counts: &v.layout.counts, offs: &v.layout.offs, b: v.batch.sequences() };
                                let seq_pool = model::gdn_mixer::GdnPoolRows { rows: &v.layout.rows, fuse: true, ..pool };
                                model::gdn_verify::gdn_mixer_verify_pooled_fused(g, &dec_shape, &weights, &qkv_d, &b_d, &a_d, &z_d, &seq_pool, &layout, dec.rows, v.batch.snapshots(l))
                                    .expect("the verify kernel was offered at engine construction (Qwen35GgufInstance::speculation_supported)")
                            }
                            None => model::gdn_mixer::gdn_mixer_decode_state_fwd(g, &gdn_mixer_ids(), &gdn_mixer_decode_ids(), &dec_shape, &weights, &qkv_d, &b_d, &a_d, &z_d, &model::gdn_mixer::GdnDecodeState::Pool(pool)),
                        };
                        self.put_rows_copy(&gated_d, all, value_dim, 0, dec.rows);
                    }
                    // The decode rows' E4M3 rows: the gated norm wrote the segments' only.
                    if let Some((q, s)) = batched.as_ref().and_then(|b| b.e4m3.as_ref()) {
                        let mut steps = Vec::new();
                        let x = gpu_core::provider::library::Operand { buf: all, range: (0, dec.rows as u64 * value_dim as u64) };
                        gpu_core::provider::library::quantize_e4m3_rows_into(g, &mut steps, &x, dec.rows, value_dim, q, s).expect("the segments' kernels were offered the E4M3 form on this device");
                        g.submit(&[], &steps);
                    }
                }
                for seg in round.segs.iter().filter(|_| batched.is_none()) {
                    // The sequence's state row becomes the working copy, the rows run on it piece by piece, and it is
                    // written back.
                    g.submit(&[], &[g.step(c.gather, &[&c.gdn_state[l], &c.gdn_hist[l], &seg.slot, &c.work_state[l], &c.work_hist[l]], &[1, sl, hl], sl + hl)]);
                    for (off, rows, chunk) in gdn_stream_pieces(seg.rows) {
                        let cont = model::gdn_mixer::GdnStream { state: &c.work_state[l], hist: &c.work_hist[l] };
                        let (qkv_i, b_i, a_i, z_i) = if whole {
                            (mixed_qkv.clone(), bproj.clone(), aproj.clone(), z.clone())
                        } else {
                            let row0 = seg.row0 + off;
                            (
                                self.take_rows(&mixed_qkv, conv_dim, n, row0, rows),
                                self.take_rows(&bproj, nvh, n, row0, rows),
                                self.take_rows(&aproj, nvh, n, row0, rows),
                                self.take_rows(&z, value_dim, n, row0, rows),
                            )
                        };
                        let seg_shape = model::gdn_mixer::GdnMixerShape { gdn: GdnShape { t: rows, chunk, ..shape.gdn }, ..shape };
                        let (gated_i, _) = model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &seg_shape, &weights, &qkv_i, &b_i, &a_i, &z_i, rows, false, Some(cont));
                        match &all {
                            Some(all) => self.put_rows_copy(&gated_i, all, value_dim, seg.row0 + off, rows),
                            None => only = Some(gated_i),
                        }
                    }
                    g.submit(&[], &[g.step(c.scatter, &[&c.work_state[l], &c.work_hist[l], &seg.slot, &c.gdn_state[l], &c.gdn_hist[l]], &[1, sl, hl], sl + hl)]);
                }
                if let Some((q, s)) = batched.and_then(|b| b.e4m3) {
                    out_proj_act = Some(self.ops.act_f32_with_e4m3(all.as_ref().expect("a batched round has its output"), n, value_dim, q, s));
                }
                (all.or(only).expect("a round has at least one sequence"), None)
            }
            GdnCall::Whole => match self.gdn_segment_tokens.get().filter(|_| self.is_train) {
                Some(seg) => {
                    let inp = model::gdn_segment::GdnMixerInputs { mixed_qkv: &mixed_qkv, bproj: &bproj, aproj: &aproj, z: &z };
                    let (gated, bounds) = model::gdn_segment::gdn_mixer_seg_fwd(g, &gdn_mixer_ids(), &shape, &weights, &inp, seg);
                    let saved = GdnSaved::Segmented { bounds, mixed_qkv: mixed_qkv.clone(), bproj: bproj.clone(), aproj: aproj.clone(), z: z.clone() };
                    (gated, Some(saved))
                }
                None => {
                    let (gated, acts) = model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, n, self.is_train, None);
                    (gated, acts.map(|a| GdnSaved::Full(Box::new(a))))
                }
            },
            GdnCall::Chunk(cont) if !self.is_train => {
                // The round in whole chunks of 64 and a last short chunk, as a mixed round's segments run.
                let pieces: Vec<_> = gdn_stream_pieces(n).collect();
                let piece = |off: u32, rows: u32, chunk: u32| {
                    let seg_shape = model::gdn_mixer::GdnMixerShape { gdn: GdnShape { t: rows, chunk, ..shape.gdn }, ..shape };
                    let cont = model::gdn_mixer::GdnStream { state: cont.state, hist: cont.hist };
                    match pieces.len() {
                        1 => model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &seg_shape, &weights, &mixed_qkv, &bproj, &aproj, &z, rows, false, Some(cont)).0,
                        _ => {
                            let (qkv_i, b_i, a_i, z_i) = (self.take_rows(&mixed_qkv, conv_dim, n, off, rows), self.take_rows(&bproj, nvh, n, off, rows), self.take_rows(&aproj, nvh, n, off, rows), self.take_rows(&z, value_dim, n, off, rows));
                            model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &seg_shape, &weights, &qkv_i, &b_i, &a_i, &z_i, rows, false, Some(cont)).0
                        }
                    }
                };
                let gated = match pieces[..] {
                    [(off, rows, chunk)] => piece(off, rows, chunk),
                    _ => {
                        let all = g.storage_uninit((n * value_dim) as u64);
                        for &(off, rows, chunk) in &pieces {
                            self.put_rows_copy(&piece(off, rows, chunk), &all, value_dim, off, rows);
                        }
                        all
                    }
                };
                (gated, None)
            }
            GdnCall::Chunk(cont) => {
                let (gated, acts) = model::gdn_mixer::gdn_mixer_stream_fwd(g, &gdn_mixer_ids(), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, n, self.is_train, Some(cont));
                (gated, acts.map(|a| GdnSaved::Full(Box::new(a))))
            }
        };

        // out_proj (LoRA/int8 dispatch stays local). Fresh `Ops::act` call:
        // `gated` is a different activation from `xn1` above.
        let out = g.storage_uninit((n * d) as u64);
        {
            let mut s = Vec::new();
            let act3 = match out_proj_act {
                Some(act) => act,
                None if epilogue => self.quant_epilogue(0, &gated, None, n, value_dim).1,
                None => self.ops_act(&mut s, &gated, n, value_dim),
            };
            let float_tier = self.ops_linear(&mut s, &act3, &p("out_proj.weight"), &out);
            self.adapt(float_tier, &mut s, "out_proj", &gated, &p("out_proj.weight"), &out, n, value_dim, d);
            self.bank_fwd(&mut s, &gated, &[(p("out_proj.weight").as_str(), &out)], n);
            g.submit(&[], &s);
        }

        let acts = internals.map(|saved| GdnLayerActs { saved, gated });
        (out, acts)
    }

    // ---- one GQA (Full) layer ----------------------------------------------

    /// The attention kernels for `layer`'s KV tier, resolved by name on this
    /// device. A failure means the tier's kernels are not in [`pipelines`],
    /// which is a build defect, not a runtime condition.
    fn kv_kernels(&self, layer: &KvLayer) -> KvKernels {
        KvKernels::resolve(&self.gpu, layer.k.tier()).unwrap_or_else(|e| panic!("qwen35: {e}"))
    }

    /// `prefix` is the weight-name prefix up to (not including) the leaf
    /// name - `"blocks.{l}.self_attn"` for a normal layer, `"mtp.layers.0.
    /// self_attn"` for the MTP head's own full-attention sublayer -
    /// reusing this function unchanged, since the mechanism is identical.
    fn layer_gqa_fwd(&self, prefix: &str, xn1: &DeviceBuffer, n: u32, cached: Option<GqaCached>) -> (DeviceBuffer, Option<GqaLayerActs>) {
        self.layer_gqa_fwd_pre(prefix, xn1, None, false, n, cached)
    }

    /// [`Self::layer_gqa_fwd`] given `xn1`'s activation already quantised - see
    /// [`Self::layer_gdn_fwd_pre`].
    fn layer_gqa_fwd_pre(&self, prefix: &str, xn1: &DeviceBuffer, pre: Option<Act>, epilogue: bool, n: u32, cached: Option<GqaCached>) -> (DeviceBuffer, Option<GqaLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, kvd) = (c.q_proj_dim(), c.kv_dim());
        let p = |s: &str| format!("{prefix}.{s}");

        // q/k/v proj (LoRA/int8 dispatch stays local - see `model::gqa_mixer`'s
        // own module doc). xn1 quantized once (`Ops::act`), shared by q/k/v.
        let q_full = g.storage_uninit((n * qpd) as u64);
        let k = g.storage_uninit((n * kvd) as u64);
        let v = g.storage_uninit((n * kvd) as u64);
        let mut s1 = Vec::new();
        let act1 = match pre {
            Some(a) => a,
            None => self.ops_act(&mut s1, xn1, n, d),
        };
        let (q_w, k_w, v_w) = (p("q_proj.weight"), p("k_proj.weight"), p("v_proj.weight"));
        let f32s = self.ops_linear_group(&mut s1, &act1, &[(&q_w, &q_full), (&k_w, &k), (&v_w, &v)]);
        self.adapt(f32s[0], &mut s1, "q_proj", xn1, &q_w, &q_full, n, d, qpd);
        self.adapt(f32s[1], &mut s1, "k_proj", xn1, &k_w, &k, n, d, kvd);
        self.adapt(f32s[2], &mut s1, "v_proj", xn1, &v_w, &v, n, d, kvd);
        self.bank_fwd(&mut s1, xn1, &[(q_w.as_str(), &q_full), (k_w.as_str(), &k), (v_w.as_str(), &v)], n);
        g.submit(&[], &s1);

        // split+qknorm+rope+attention+gating - LoRA/dtype-agnostic, shared
        // with `crates/qwen35moe` (`model::gqa_mixer`). Both SERVING arms
        // supply their OWN M-RoPE table (real absolute positions, not `0..t`)
        // and attend the persistent KV cache instead of an isolated `[T,T]`
        // causal block - see `GqaCached`.
        let shape = model::gqa_mixer::GqaMixerShape { b: self.b, t: self.run_len(), n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: c.rotary_dim() / 2, rms_eps: c.rms_eps };
        let (cos, sin) = match &cached {
            None => (&self.cos, &self.sin),
            Some(GqaCached::Chunk(ch)) => (ch.cos, ch.sin),
            Some(GqaCached::Ring(rc)) => (rc.cos, rc.sin),
            Some(GqaCached::Decode(dc)) => (dc.cos, dc.sin),
            // Each sequence of a multi-sequence round brings its own tables below.
            Some(GqaCached::Segments(round, _)) => (&round.rows.cos, &round.rows.sin),
        };
        let weights = model::gqa_mixer::GqaMixerWeights { q_norm: self.w(&p("q_norm.weight")), k_norm: self.w(&p("k_norm.weight")), cos, sin };
        // Set by the decode arm that fuses the sigmoid gate into the quantiser.
        let mut o_proj_act: Option<Act> = None;
        // A sliding-window layer attends its ring (`GqaCached::Ring`, or a decode step over a ring window); the
        // paged-prefill shapes below address a cache that grows with the sequence and would attend all of it.
        if c.swa_window.is_some() && matches!(cached, Some(GqaCached::Chunk(_)) | Some(GqaCached::Segments(..))) {
            c.require_plain_decoder("multi-round and multi-sequence prefill over a paged cache");
        }
        let (ctx_gated, internals) = match cached {
            None => match c.swa_window {
                Some(window) => {
                    let scores_kernel = g.kernel_index("gqa_scores_win").unwrap_or_else(|| panic!("qwen35: gqa_scores_win is not registered on this Gpu"));
                    model::gqa_mixer::gqa_mixer_fwd_windowed(g, &gqa_mixer_ids(), model::gqa_mixer::SlidingWindow { scores_kernel, window }, &shape, &weights, &q_full, &k, &v, n, self.is_train)
                }
                None => model::gqa_mixer::gqa_mixer_fwd_tc(g, &gqa_mixer_ids(), self.training_flash(), Some(self.ops.tensor_gemm()), &shape, &weights, &q_full, &k, &v, n, self.is_train),
            },
            Some(GqaCached::Ring(rc)) => (
                model::gqa_mixer::gqa_mixer_ring_chunk_fwd(
                    g,
                    &gqa_mixer_ids(),
                    &model::gqa_mixer::RingChunk {
                        layer: rc.layer,
                        row0: rc.row0,
                        window: rc.window,
                        start: rc.start,
                        scores_win: g.kernel_index("gqa_scores_win").unwrap_or_else(|| panic!("qwen35: gqa_scores_win is not registered on this Gpu")),
                        splice: SPLICE,
                        ring_attn: g.kernel_index("gqa_ring_chunk_attn"),
                        pack: g.kernel_index("pack_bf16").unwrap_or_else(|| panic!("qwen35: pack_bf16 is not registered on this Gpu")),
                        unpack: g.kernel_index("unpack_bf16").unwrap_or_else(|| panic!("qwen35: unpack_bf16 is not registered on this Gpu")),
                    },
                    &shape,
                    &weights,
                    &q_full,
                    &k,
                    &v,
                    n,
                ),
                None,
            ),
            Some(GqaCached::Chunk(ch)) => (
                model::gqa_mixer::gqa_mixer_chunk_kv_fwd_paged(
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
                    ch.start,
                    ch.layer,
                    &model::block::ChunkPaging { block_size: ch.block_size, max_bt: ch.max_bt, blocks: &ch.blocks, offsets: &ch.offsets, tables: &ch.tables },
                    &ch.seq_lens,
                ),
                None,
            ),
            Some(GqaCached::Segments(round, layer)) => {
                // The front half (split, QK norm, rotation) and the K/V append run once over every row of the round,
                // whichever sequences they belong to; only the attention, which reads a sequence's own keys, is
                // per sequence. A row's arithmetic does not depend on its neighbours, so each row is what it was
                // when every sequence ran the chain on its own rows.
                let kv = self.kv_kernels(layer);
                let ids = gqa_mixer_ids();
                let prep = model::gqa_mixer::qkv_prepare(g, &ids, &shape, &weights, &q_full, &k, n);
                g.submit(&[], &model::block::kv_append_rows(g, &kv, nkv, hd, round.caches.gqa_cap, n, &prep.k_normed, &v, layer, &round.rows.blocks, &round.rows.offsets));
                // The decode rows and the segments cover every row, so `ctx` needs no clear.
                let ctx = g.storage_uninit((n * shape.qd()) as u64);
                // The decode rows lead the round, so their queries and outputs are the heads of the round's buffers.
                if let Some(dec) = &round.decode {
                    let p = dec.paged;
                    g.submit(
                        &[],
                        &model::block::gqa_decode_batched_attend_steps_kv(g, &kv, DECODE_SOFTMAX_BATCHED, nh, nkv, hd, dec.rows, p.block_size, p.max_bt, p.cap, p.pin_splits, &prep.q_normed, layer, p.block_tables, p.seq_lens, &ctx),
                    );
                }
                for seg in &round.segs {
                    let rows = seg.rows;
                    let whole = rows == n;
                    let (q_i, ctx_i) = if whole { (prep.q_normed.clone(), ctx.clone()) } else { (self.take_rows(&prep.q_normed, shape.qd(), n, seg.row0, rows), g.storage_uninit((rows * shape.qd()) as u64)) };
                    g.submit(
                        &[],
                        &model::block::gqa_chunk_attend_steps_kv_paged(g, &kv, DECODE_SOFTMAX_BATCHED, nh, nkv, hd, seg.start, rows, &q_i, layer, round.caches.gqa_cap, seg.max_bt, &seg.tables, &seg.seq_lens, &ctx_i),
                    );
                    if !whole {
                        self.put_rows_copy(&ctx_i, &ctx, shape.qd(), seg.row0, rows);
                    }
                }
                if epilogue {
                    // The sigmoid output gate rides on the quantisation for `o_proj`, as in a decode step.
                    let (gated, act) = self.quant_epilogue(2, &ctx, Some(&prep.q_gate), n, shape.qd());
                    o_proj_act = Some(act);
                    (gated.expect("a gated epilogue hands back its product"), None)
                } else {
                    (model::gqa_mixer::gate_ctx(g, &ids, &ctx, &prep.q_gate, n, shape.qd()).1, None)
                }
            }
            Some(GqaCached::Decode(dc)) if epilogue => {
                // The sigmoid output gate rides on the quantisation below. The
                // front half of the attention (split, QK norm, rotation, KV
                // append) is one native launch for a single sequence on an f32
                // KV tier where the device is offered it, and the chain of eight
                // otherwise.
                let kv = self.kv_kernels(dc.layer);
                let (ctx, q_gate) = self
                    .decode_fusion
                    .get()
                    .then(|| {
                        model::gqa_mixer::gqa_mixer_decode_fused_kv_attend(g, &kv, DECODE_SOFTMAX_BATCHED, &shape, &weights, &q_full, &k, &v, dc.layer, n, dc.paged)
                    })
                    .flatten()
                    .unwrap_or_else(|| {
                        model::gqa_mixer::gqa_mixer_decode_batched_kv_attend(
                            g,
                            &gqa_mixer_ids(),
                            &kv,
                            DECODE_SOFTMAX_BATCHED,
                            &shape,
                            &weights,
                            &q_full,
                            &k,
                            &v,
                            dc.layer,
                            n,
                            dc.paged,
                        )
                    });
                let (gated, act) = self.quant_epilogue(2, &ctx, Some(&q_gate), n, shape.qd());
                o_proj_act = Some(act);
                (gated.expect("a gated epilogue hands back its product"), None)
            }
            Some(GqaCached::Decode(dc)) => (
                model::gqa_mixer::gqa_mixer_decode_batched_kv_fwd(
                    g,
                    &gqa_mixer_ids(),
                    &self.kv_kernels(dc.layer),
                    DECODE_SOFTMAX_BATCHED,
                    &shape,
                    &weights,
                    &q_full,
                    &k,
                    &v,
                    dc.layer,
                    n,
                    dc.paged,
                ),
                None,
            ),
        };

        // o_proj (LoRA/int8 dispatch stays local). Fresh `Ops::act` call:
        // `ctx_gated` is a different activation from `xn1` above.
        let out = g.storage_uninit((n * d) as u64);
        {
            let mut s = Vec::new();
            let act2 = match o_proj_act {
                Some(a) => a,
                None => self.ops_act(&mut s, &ctx_gated, n, shape.qd()),
            };
            let float_tier = self.ops_linear(&mut s, &act2, &p("o_proj.weight"), &out);
            self.adapt(float_tier, &mut s, "o_proj", &ctx_gated, &p("o_proj.weight"), &out, n, shape.qd(), d);
            self.bank_fwd(&mut s, &ctx_gated, &[(p("o_proj.weight").as_str(), &out)], n);
            g.submit(&[], &s);
        }

        let acts = internals.map(|internals| GqaLayerActs { internals, ctx_gated, memory: None });
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
        self.mlp_fwd_pre(prefix, xn2, None, false, n)
    }

    /// The feed-forward sublayer of decoder layer `l` as the serving layer loops
    /// (decode, verify, prefill) run it: `xn2` is the normalised mixer output
    /// (`pre` its int8 packing when a fused front made one) and the result the
    /// `[n, d_model]` product the residual adds. This is the ONE place those
    /// loops reach the feed-forward block, so a layer kind with a different
    /// block (the sparse-MoE sibling's router and experts)
    /// is a new arm here and nothing else in them changes.
    fn ffn_fwd_pre(&self, l: usize, xn2: &DeviceBuffer, pre: Option<Act>, epilogue: bool, n: u32) -> DeviceBuffer {
        self.mlp_fwd_pre(&format!("blocks.{l}.mlp"), xn2, pre, epilogue, n).0
    }

    /// [`Self::mlp_fwd`] given `xn2`'s activation already quantised - see
    /// [`Self::layer_gdn_fwd_pre`].
    fn mlp_fwd_pre(&self, prefix: &str, xn2: &DeviceBuffer, pre: Option<Act>, epilogue: bool, n: u32) -> (DeviceBuffer, Option<MlpLayerActs>) {
        let (out, acts) = self.mlp_fwd_core(prefix, xn2, pre, epilogue, true, n);
        (out.expect("qwen35: mlp_fwd_core was asked for the output"), acts)
    }

    /// The MLP's activations alone (`gate_pre`, `up`, `h`), without the `down`
    /// product: what the backward needs to rebuild for a chunk whose output
    /// the forward already produced and nothing reads again.
    fn mlp_acts_only(&self, prefix: &str, xn2: &DeviceBuffer, n: u32) -> MlpLayerActs {
        self.mlp_fwd_core(prefix, xn2, None, false, false, n).1.expect("qwen35: backward() requires a training build")
    }

    /// [`Self::mlp_fwd_pre`] with the `down` projection optional (`want_out`).
    fn mlp_fwd_core(&self, prefix: &str, xn2: &DeviceBuffer, pre: Option<Act>, epilogue: bool, want_out: bool, n: u32) -> (Option<DeviceBuffer>, Option<MlpLayerActs>) {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.intermediate_size;
        let p = |s: &str| format!("{prefix}.{s}");

        // The gate and up projections write every row of their outputs (the adapter then adds into them), so
        // neither needs the clear a zeroed allocation costs. On the serving path with FP8 weights they store bf16 rows
        // when the device has the bf16-output form of the product ([`Self::bf16_gate_up_ok`]): the SwiGLU pass then reads
        // half the bytes and the product writes half.
        let mut bf16_rows = epilogue && self.bf16_gate_up_ok(n, ff);
        let mut gate_pre = g.storage_uninit(if bf16_rows { (n * ff / 2) as u64 } else { (n * ff) as u64 });
        let mut up = g.storage_uninit(if bf16_rows { (n * ff / 2) as u64 } else { (n * ff) as u64 });
        {
            let mut s = Vec::new();
            let act1 = match pre {
                Some(a) => a,
                None => self.ops_act(&mut s, xn2, n, d),
            };
            let (gate_w, up_w) = (p("gate.weight"), p("up.weight"));
            if bf16_rows {
                // Both or neither: the steps pushed stay either way (they may hold the activation's quantisation, which the
                // activation remembers for the products after).
                let mut t = Vec::new();
                let ok = [(&gate_w, &gate_pre), (&up_w, &up)].iter().all(|(name, out)| self.ops.matmul_bf16_out(&mut t, &self.weights[name.as_str()], &act1, out));
                s.append(&mut t);
                if !ok {
                    bf16_rows = false;
                    gate_pre = g.storage_uninit((n * ff) as u64);
                    up = g.storage_uninit((n * ff) as u64);
                }
            }
            if !bf16_rows {
                let f32s = self.ops_linear_group(&mut s, &act1, &[(&gate_w, &gate_pre), (&up_w, &up)]);
                self.adapt(f32s[0], &mut s, "gate", xn2, &gate_w, &gate_pre, n, d, ff);
                self.adapt(f32s[1], &mut s, "up", xn2, &up_w, &up, n, d, ff);
                self.bank_fwd(&mut s, xn2, &[(gate_w.as_str(), &gate_pre), (up_w.as_str(), &up)], n);
            }
            g.submit(&[], &s);
        }
        let (h, down_act) = if epilogue {
            // `silu(gate) * up` and its quantisation for `down` in one launch.
            let input = if bf16_rows { gpu_core::ActPrecision::Bf16 } else { gpu_core::ActPrecision::F32 };
            let (h, act) = self.silu_mul_e4m3(&gate_pre, &up, input, n, ff).unwrap_or_else(|| {
                assert!(!bf16_rows, "qwen35: bf16 gate/up rows were produced for a SwiGLU pass the device does not run");
                let (h, act) = self.quant_epilogue(1, &gate_pre, Some(&up), n, ff);
                (h.expect("a silu_mul epilogue hands back its product"), act)
            });
            (h, Some(act))
        } else {
            let h = g.storage_uninit((n * ff) as u64); // both silu_mul forms write all n * ff words
            // On the vendor-library path the product and its bf16 rounding come out of one pass, and the
            // `down` projection takes the rounding as its activation's cached cast.
            let mut s = Vec::new();
            let rounded = if self.weights.values().all(reads_float_acts) {
                self.ops.tensor_gemm().silu_mul_bf16(g, &mut s, &gate_pre, &up, &h, n as u64 * ff as u64)
            } else {
                None
            };
            match rounded {
                Some(bf16) => {
                    g.submit(&[], &s);
                    (h.clone(), Some(self.ops.act_f32_with_bf16(&h, n, ff, bf16)))
                }
                None => {
                    g.submit(&[], &[g.step(SILU_MUL, &[&gate_pre, &up, &h], &[n * ff], n * ff)]);
                    (h, None)
                }
            }
        };
        let down = want_out.then(|| {
            let down = g.storage_uninit((n * d) as u64);
            let mut s = Vec::new();
            let act2 = match down_act {
                Some(a) => a,
                None => self.ops_act(&mut s, &h, n, ff),
            };
            let float_tier = self.ops_linear(&mut s, &act2, &p("down.weight"), &down);
            self.adapt(float_tier, &mut s, "down", &h, &p("down.weight"), &down, n, ff, d);
            self.bank_fwd(&mut s, &h, &[(p("down.weight").as_str(), &down)], n);
            g.submit(&[], &s);
            down
        });

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
                rmsnorm_fwd(g, &kernel_ids(), &e, self.w("mtp.pre_fc_norm_embedding.weight"), &en, d, n, self.cfg.rms_eps),
                rmsnorm_fwd(g, &kernel_ids(), res_last, self.w("mtp.pre_fc_norm_hidden.weight"), &hn, d, n, self.cfg.rms_eps),
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
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &ehp, self.w("mtp.layers.0.ln1.weight"), &xn1, d, n, self.cfg.rms_eps)]);
        let (mixer_out, mixer_acts) = self.layer_gqa_fwd("mtp.layers.0.self_attn", &xn1, n, None);

        let xmid = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(ADD2, &[&ehp, &mixer_out, &xmid], &[n * d], n * d)]);

        let xn2 = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &xmid, self.w("mtp.layers.0.ln2.weight"), &xn2, d, n, self.cfg.rms_eps)]);
        let (mlp_out, mlp_acts) = self.mlp_fwd("mtp.layers.0.mlp", &xn2, n);

        let block_out = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &block_out], &[n * d], n * d)]);

        let final_h = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &block_out, self.w("mtp.norm.weight"), &final_h, d, n, self.cfg.rms_eps)]);
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
                mlp: MlpSaved::Full(mlp_acts.expect("qwen35: is_train but mlp_fwd returned no acts for the MTP layer")),
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
        let d_xn2 = self.mlp_bwd_saved("mtp.layers.0.mlp", &ma.layer.mlp, &d_block_out, n);
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

    /// The anchor-memory shape of this build's current batch.
    fn memory_shape(&self) -> model::anchor_memory::AnchorMemoryShape {
        let c = &self.cfg;
        let mem = c.anchor_memory.as_ref().expect("a layer with a memory implies an anchor_memory config");
        model::anchor_memory::AnchorMemoryShape {
            n_seq: self.b,
            t: self.run_len(),
            d_model: c.d_model,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            head_dim: c.head_dim,
            rotary_half: c.rotary_dim() / 2,
            rms_eps: c.rms_eps,
            capacity: mem.capacity,
            commit_block: c.memory_commit_block().expect("a memory has a commit block"),
            admission_rank: mem.admission_rank,
            writer_intermediate: mem.writer_intermediate,
        }
    }

    fn memory_ids(&self) -> model::anchor_memory::AnchorMemoryIds {
        model::anchor_memory::AnchorMemoryIds {
            kernels: kernel_ids(),
            silu: SILU,
            silu_mul: SILU_MUL,
            add2: ADD2,
            scale_row: SCALE_ROW,
            rope2d_partial: ROPE2D_PARTIAL,
            row_dot: ROW_DOT,
            memory: model::anchor_memory::AnchorMemoryKernels::resolve(&self.gpu).unwrap_or_else(|e| panic!("qwen35: {e}")),
        }
    }

    fn memory_weights(&self, l: usize) -> model::anchor_memory::AnchorMemoryWeights<'_> {
        self.memory_weights_at(l, &self.cos, &self.sin)
    }

    /// [`Self::memory_weights`] with the rope tables of the step's own rows (decode positions, a chunk's range).
    fn memory_weights_at<'a>(&'a self, l: usize, cos: &'a DeviceBuffer, sin: &'a DeviceBuffer) -> model::anchor_memory::AnchorMemoryWeights<'a> {
        let p = |leaf: &str| self.w(&format!("blocks.{l}.memory.{leaf}"));
        model::anchor_memory::AnchorMemoryWeights {
            input_norm: p("input_norm.weight"),
            q_norm: p("q_norm.weight"),
            k_norm: p("k_norm.weight"),
            read_scale: p("read_scale"),
            read_gate_weight: p("read_gate_weight"),
            read_gate_bias: p("read_gate_bias"),
            cos,
            sin,
        }
    }

    /// The memory's projections of layer `l` over `n` rows: plain fp32 matmuls over the parameter store's
    /// tensors, corrected by the layer's adapters where it has them.
    fn memory_project<'a>(&'a self, l: usize, shape: &'a model::anchor_memory::AnchorMemoryShape, n: u32) -> impl Fn(model::anchor_memory::MemoryProj, &DeviceBuffer, &DeviceBuffer) + 'a {
        move |which, x, out| {
            let g = &self.gpu;
            let (n_out, n_in) = which.dims(shape);
            let wname = format!("blocks.{l}.memory.{}.weight", which.leaf());
            let mut s = Vec::new();
            match self.weights.get(&wname) {
                // Held in a narrower tier: the same product through the library path the layer's linears take.
                Some(w) => {
                    let act = self.ops_act(&mut s, x, n, n_in);
                    self.ops.matmul(&mut s, w, &act, out, 0);
                }
                // The scorer and anything not held in a tier: exact fp32. A few rows take the skinny-M kernel (one
                // workgroup per output column, the weight read once), which a single decode row needs; it has no
                // tensor-core variant, so the product stays exact.
                None => match g.kernel_index("matmul_gemv") {
                    // Many rows: the exact tiled product, one output per thread, whose grid is as wide as the batch
                    // (the skinny-M kernel below would launch one narrow dispatch per 32 rows).
                    Some(_) if n > MEMORY_TILED_MIN_ROWS && g.caps().workgroup_reductions && g.kernel_index("matmul_tile16").is_some() => {
                        let tile = g.kernel_index("matmul_tile16").expect("checked");
                        s.push(g.dispatch(tile, &[x, self.w(&wname), out], &[n, n_in, n_out], gpu_core::Dispatch::Workgroups(n.div_ceil(16) * n_out.div_ceil(16))));
                    }
                    // Groups of up to 32 rows, each a view of the activations and the output.
                    Some(gemv) => {
                        let wbuf = self.w(&wname);
                        let wlen = u64::from(n_in) * u64::from(n_out);
                        for first in (0..n).step_by(32) {
                            let rows = (n - first).min(32);
                            let (xo, yo) = (u64::from(first) * u64::from(n_in), u64::from(first) * u64::from(n_out));
                            s.push(g.dispatch_sliced(
                                gemv,
                                &[x, wbuf, out],
                                &[(xo, u64::from(rows) * u64::from(n_in)), (0, wlen), (yo, u64::from(rows) * u64::from(n_out))],
                                &[rows, n_in, n_out],
                                gpu_core::Dispatch::Workgroups(n_out),
                            ));
                        }
                    }
                    None => s.push(g.step(MATMUL, &[x, self.w(&wname), out], &[n, n_in, n_out], n * n_out)),
                },
            }
            self.lora_fwd(&mut s, &format!("memory.{}", which.leaf()), x, &wname, out, n, n_in, n_out);
            g.submit(&[], &s);
        }
    }

    /// The memory's shape for `n_seq` sequences of `t` rows each.
    /// The pipeline index registered under `name`: a kernel of this crate's list that no hand-numbered const names.
    fn kernel_by_name(&self, name: &str) -> usize {
        self.gpu.kernel_index(name).unwrap_or_else(|| panic!("qwen35: kernel '{name}' is not registered on this Gpu"))
    }

    fn memory_shape_of(&self, n_seq: u32, t: u32) -> model::anchor_memory::AnchorMemoryShape {
        model::anchor_memory::AnchorMemoryShape { n_seq, t, ..self.memory_shape() }
    }

    /// The anchor memory of full-attention layer `l` (`model::anchor_memory`):
    /// reads the layer's residual input `xres` and returns `attn_out +
    /// memory(xres)`, the mixer output the residual add takes, and (on a
    /// training build) what its backward needs. The memory's projections are
    /// plain fp32 matmuls over the parameter store's tensors, never a
    /// quantized tier, which also keeps the admission scores fp32 whatever
    /// tier the rest of the layer runs at; an adapter corrects the read-side
    /// ones exactly as it does the attention's.
    fn add_anchor_memory(&self, l: usize, xres: &DeviceBuffer, attn_out: DeviceBuffer, n: u32) -> (DeviceBuffer, Option<model::anchor_memory::AnchorMemoryActs>) {
        use model::anchor_memory::anchor_memory_fwd;
        let g = &self.gpu;
        let c = &self.cfg;
        let shape = self.memory_shape();
        let ids = self.memory_ids();
        let weights = self.memory_weights(l);
        let project = self.memory_project(l, &shape, n);
        let (memory_out, acts) = anchor_memory_fwd(g, &ids, &shape, &weights, &project, xres);

        if let Some(taps) = self.memory_taps.borrow_mut().as_mut() {
            let d = c.d_model as usize;
            let rows = n as usize;
            let mut push = |name: &str, buf: &DeviceBuffer, len: usize| taps.push((format!("layer{l}.{name}"), g.read(buf, len)));
            push("swa.out", &attn_out, rows * d);
            push("mem.scores", &acts.scores, rows * c.n_kv_heads as usize);
            push("mem.gate", &acts.gate, rows * c.n_heads as usize);
            push("mem.query", &acts.query, rows * c.q_dim() as usize);
            push("mem.key", &acts.key, rows * c.kv_dim() as usize);
            push("mem.value", &acts.value, rows * c.kv_dim() as usize);
            push("mem.read", &acts.read, rows * c.q_dim() as usize);
            push("mem.out", &memory_out, rows * d);
            let refs = model::anchor_memory::decode_refs(&g.read(&acts.refs, shape.refs_len() as usize));
            taps.push((format!("layer{l}.mem.refs"), refs.into_iter().map(|r| r as f32).collect()));
        }

        let sum = g.storage_uninit((n * c.d_model) as u64);
        g.submit(&[], &[g.step(ADD2, &[&attn_out, &memory_out, &sum], &[n * c.d_model], n * c.d_model)]);
        (sum, self.is_train.then_some(acts))
    }

    /// Reverse of [`Self::add_anchor_memory`]'s memory term: the gradient of
    /// the layer's residual input `xres` through the memory, given `d_mixer`
    /// (the gradient of the mixer output the memory is added to). Parameter
    /// gradients accumulate where the parameter trains; the admission scorer
    /// is frozen by construction (see `model::anchor_memory::anchor_memory_bwd`).
    fn anchor_memory_bwd(&self, l: usize, xres: &DeviceBuffer, acts: &model::anchor_memory::AnchorMemoryActs, d_mixer: &DeviceBuffer, n: u32) -> DeviceBuffer {
        use model::anchor_memory::{anchor_memory_bwd, AnchorMemoryGrads, MemoryProj, ReadGateGrads};
        let g = &self.gpu;
        let shape = self.memory_shape();
        let ids = self.memory_ids();
        let weights = self.memory_weights(l);
        let name = |leaf: &str| format!("blocks.{l}.memory.{leaf}");
        let grad = |leaf: &str| self.trainable(&name(leaf)).then(|| self.g(&name(leaf)));
        let read_gate = match (grad("read_scale"), grad("read_gate_weight"), grad("read_gate_bias")) {
            (Some(scale), Some(weight), Some(bias)) => Some(ReadGateGrads { scale, weight, bias }),
            (None, None, None) => None,
            _ => panic!("qwen35: blocks.{l}.memory.read_scale/read_gate_weight/read_gate_bias must train together or all stay frozen"),
        };
        let grads = AnchorMemoryGrads { input_norm: grad("input_norm.weight"), q_norm: grad("q_norm.weight"), k_norm: grad("k_norm.weight"), read_gate };
        let project_bwd = |which: MemoryProj, d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, acc: bool| {
            let (n_out, n_in) = which.dims(&shape);
            let mut s = Vec::new();
            self.proj_bwd(&mut s, &format!("memory.{}", which.leaf()), d_out, x, &name(&format!("{}.weight", which.leaf())), dx, n, n_in, n_out, acc as u32);
            g.submit(&[], &s);
        };
        anchor_memory_bwd(g, &ids, &shape, &weights, &grads, &project_bwd, acts, xres, d_mixer)
    }

    /// One decoder layer's forward: `out = layer_l(xres)`. On a training
    /// build it returns the activations the backward needs; the caller keeps
    /// them (no recompute) or drops them at once (recompute, which calls this
    /// again from `res[l]` inside the backward - the same kernels over the
    /// same inputs, so the rebuilt activations are the ones that were dropped).
    ///
    /// `rebuild` is the backward's replay: when the MLP is saved as rows, the
    /// backward rebuilds each chunk itself ([`Self::mlp_bwd_saved`]), so the
    /// replay stops after `xn2` and leaves `out` unwritten.
    fn layer_forward(&self, l: usize, xres: &DeviceBuffer, out: &DeviceBuffer, rebuild: bool) -> Option<LayerTrainActs> {
        let g = &self.gpu;
        let n = self.b * self.run_len();
        let d = self.cfg.d_model;
        let xn1 = g.storage_uninit((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), xres, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n, self.cfg.rms_eps)]);

        let (mixer_out, mixer_acts) = match self.cfg.layer_types()[l] {
            LayerType::Linear => {
                let (o, a) = self.layer_gdn_fwd(l, &xn1, n, GdnCall::Whole);
                (o, a.map(|a| MixerActs::Gdn(Box::new(a))))
            }
            LayerType::Full => {
                let (o, a) = self.layer_gqa_fwd(&format!("blocks.{l}.self_attn"), &xn1, n, None);
                // The anchor memory reads the layer's residual input, not its normalised copy.
                let (o, memory) = if self.cfg.layer_has_memory(l) { self.add_anchor_memory(l, xres, o, n) } else { (o, None) };
                (o, a.map(|a| MixerActs::Gqa(GqaLayerActs { memory: memory.map(Box::new), ..a })))
            }
        };

        let xmid = g.storage_uninit((n * d) as u64);
        let xn2 = g.storage_uninit((n * d) as u64);
        self.add_then_rmsnorm(xres, &mixer_out, &format!("blocks.{l}.ln2.weight"), &xmid, &xn2, d, n);

        let prefix = format!("blocks.{l}.mlp");
        let chunk_rows = self.mlp_chunk_rows(n);
        let (mlp_out, mlp_saved) = match chunk_rows {
            // Rows are independent: run the MLP a chunk at a time and keep
            // none of its activations (rebuilt by the backward).
            Some(_) if rebuild => (None, Some(MlpSaved::Rows { xn2: xn2.clone() })),
            Some(rows) => (Some(self.mlp_fwd_rows(&prefix, &xn2, n, rows)), Some(MlpSaved::Rows { xn2: xn2.clone() })),
            None => {
                let (out, acts) = self.mlp_fwd(&prefix, &xn2, n);
                (Some(out), acts.map(MlpSaved::Full))
            }
        };
        if let Some(mlp_out) = &mlp_out {
            g.submit(&[], &[g.step(ADD2, &[&xmid, mlp_out, out], &[n * d], n * d)]);
        }

        if !self.is_train {
            return None;
        }
        Some(LayerTrainActs {
            xn1,
            mixer: mixer_acts.expect("qwen35: is_train but layer_gdn_fwd/layer_gqa_fwd returned no acts"),
            xmid,
            mlp: mlp_saved.expect("qwen35: is_train but the MLP returned no acts"),
        })
    }

    /// Gather the embedding rows of `n` token ids into `out` (`[n, d_model]`).
    /// A narrow-base LoRA build holds the frozen table as an `Ops` weight
    /// ([`table_tier`]) rather than as an fp32 store tensor; every path that
    /// embeds (training forward, decode step, chunked prefill) goes through
    /// here so none of them can assume the fp32 one.
    fn embed_rows(&self, tokens: &DeviceBuffer, n: u32, out: &DeviceBuffer) {
        let g = &self.gpu;
        match self.weights.get("tok.weight") {
            Some(table) => {
                let mut s = Vec::new();
                self.ops.embed(&mut s, table, tokens, n, out);
                g.submit(&[], &s);
            }
            None => {
                let d = self.cfg.d_model;
                g.submit(&[], &[g.step(EMBED, &[tokens, self.w("tok.weight"), out], &[d, n], n * d)]);
            }
        }
    }

    pub(crate) fn run_forward(&self) {
        let g = &self.gpu;
        let n = self.b * self.run_len();
        let d = self.cfg.d_model;
        let res = self.res.borrow();
        let mut layer_acts: Vec<LayerTrainActs> = Vec::new();

        if self.shard.embed {
            self.embed_rows(&self.tokens, n, &res[0]);

            // Vision-language splice: overwrite the image-placeholder rows of
            // the freshly-gathered residual stream with the projected image
            // tokens (see `Self::enable_mm_splice`'s doc). No-op unless
            // enabled. Only meaningful on the embed stage (operates on
            // `res[0]`, right after the gather above).
            if !self.splice.is_empty() {
                g.submit(&[], &self.splice.fwd(g, SPLICE, &res[0], 0, n));
            }
        }

        let keep = self.is_train && !self.recompute.get();
        if let Some(saved) = &self.res_saved {
            // Each boundary is stored as bf16 and the next layer reads it back
            // from there, so the forward and the backward's recompute agree.
            saved.store(g, 0, &res[0]);
            saved.load(g, 0, &res[0]);
        }
        for l in self.shard.start..self.shard.end {
            self.stream_load(l);
            // Each frozen weight of the layer is decoded once for all the products that read it (the MLP's row chunks).
            let _decoded = self.ops.decoded_weights();
            let _pass = g.pass_scope();
            let (x, y) = if self.res_saved.is_some() { (&res[0], &res[1]) } else { (&res[l], &res[l + 1]) };
            let acts = self.layer_forward(l, x, y, false);
            if let Some(saved) = &self.res_saved {
                saved.store(g, l + 1, y);
                saved.load(g, l + 1, &res[0]);
            }
            if keep {
                layer_acts.push(acts.expect("qwen35: is_train but layer_forward returned no acts"));
            }
        }
        // The stream entering the final norm: with packed boundaries it is the
        // last layer's rounded output, which the loop left in `res[0]`.
        let last = if self.res_saved.is_some() { &res[0] } else { &res[self.cfg.n_layers as usize] };

        // Head epilogue (final norm + lm_head/logits) and MTP: head stage
        // only. `cfg.mtp` requires a whole shard (asserted in `new_impl_on`),
        // so it is always true here whenever `self.cfg.mtp` is set. On a
        // non-head stage `xn_final` is never read (a size-1 dummy stands in,
        // matching this file's own "size-1 dummy where a value doesn't
        // apply" convention used elsewhere).
        let xn_final = if self.shard.head {
            let xn_final = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), last, self.w("norm.weight"), &xn_final, d, n, self.cfg.rms_eps)]);
            if self.head_ce.is_none() {
                self.head_matmul(&xn_final, &self.logits, n);
            }

            if self.cfg.mtp {
                let mtp_acts = self.run_mtp_forward(last, n);
                *self.mtp_acts.borrow_mut() = mtp_acts;
            }
            xn_final
        } else {
            g.storage(1)
        };

        if self.is_train {
            *self.train_acts.borrow_mut() = Some(TrainActs { layers: layer_acts, recomputed: !keep, xn_final });
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
            gdn: GdnShape { b: self.b, h: nvh, t: self.run_len(), dk: c.linear_key_head_dim, dv: c.linear_value_head_dim, chunk: self.run_chunk() },
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
        let (d_mixed_qkv, d_bproj, d_aproj, d_z) = match &la.saved {
            GdnSaved::Full(acts) => model::gdn_mixer::gdn_mixer_bwd(g, &gdn_mixer_ids(), &shape, &weights, &grads, acts, &d_gated, n),
            GdnSaved::Segmented { bounds, mixed_qkv, bproj, aproj, z } => {
                let inp = model::gdn_segment::GdnMixerInputs { mixed_qkv, bproj, aproj, z };
                model::gdn_segment::gdn_mixer_seg_bwd(g, &gdn_mixer_ids(), &shape, &weights, &grads, &inp, bounds, &d_gated)
            }
        };

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
        let shape = model::gqa_mixer::GqaMixerShape { b: self.b, t: self.run_len(), n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: c.rotary_dim() / 2, rms_eps: c.rms_eps };
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
    /// Rows per MLP chunk for an `n`-row pass, or `None` to run it whole: only
    /// when asked for ([`TrainOpts::mlp_rows`]), on a recomputing training
    /// build (the chunks' activations are dropped, so the backward must be
    /// able to rebuild them), and only when there is more than one chunk.
    fn mlp_chunk_rows(&self, n: u32) -> Option<u32> {
        self.mlp_rows.get().filter(|&rows| self.is_train && self.recompute.get() && rows > 0 && rows < n)
    }

    /// Rows `[row0, row0 + rows)` of a `[total_rows, width]` buffer as their own buffer.
    fn take_rows(&self, src: &DeviceBuffer, width: u32, total_rows: u32, row0: u32, rows: u32) -> DeviceBuffer {
        // Every word is written by the copy below, so the buffer needs no clear.
        let dst = self.gpu.storage_uninit(rows as u64 * width as u64);
        // Rows are the channels of one batch entry and a plane is a whole row, so a
        // block copies a contiguous run instead of one element.
        let count = rows * width;
        self.gpu.submit(&[], &[self.gpu.step(CONCAT_SPLIT, &[src, &dst], &[1, total_rows, rows, row0, 1, width], count)]);
        dst
    }

    /// `dst[row0 ..] = src` for a `[rows, width]` source and a zero-fresh `dst`.
    fn put_rows(&self, src: &DeviceBuffer, dst: &DeviceBuffer, width: u32, row0: u32, rows: u32) {
        let count = rows * width;
        self.gpu.submit(&[], &[self.gpu.step(SPLICE_ADD, &[src, dst], &[count, row0 * width], count)]);
    }

    /// `dst[row0 ..] = src` for a `[rows, width]` source: a copy, so `dst` need not
    /// be zero-fresh (the rows of its sequences cover it).
    fn put_rows_copy(&self, src: &DeviceBuffer, dst: &DeviceBuffer, width: u32, row0: u32, rows: u32) {
        let count = rows * width;
        self.gpu.submit(&[], &[self.gpu.step(SPLICE, &[src, dst], &[count, row0 * width], count)]);
    }

    /// [`Self::mlp_fwd`] over `rows`-row chunks of `xn2`, keeping nothing but the output.
    fn mlp_fwd_rows(&self, prefix: &str, xn2: &DeviceBuffer, n: u32, rows: u32) -> DeviceBuffer {
        let d = self.cfg.d_model;
        let out = self.gpu.storage((n * d) as u64);
        for row0 in (0..n).step_by(rows as usize) {
            let rc = rows.min(n - row0);
            let (chunk_out, _) = self.mlp_fwd(prefix, &self.take_rows(xn2, d, n, row0, rc), rc);
            self.put_rows(&chunk_out, &out, d, row0, rc);
        }
        self.gpu.poll_wait();
        out
    }

    /// [`Self::mlp_bwd`] from whatever the forward kept: the activations, or
    /// the input alone, from which each `rows`-row chunk's activations are
    /// rebuilt just before its gradient is taken.
    fn mlp_bwd_saved(&self, prefix: &str, saved: &MlpSaved, d_mlp_out: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let MlpSaved::Rows { xn2 } = saved else {
            let MlpSaved::Full(la) = saved else { unreachable!() };
            return self.mlp_bwd(prefix, la, d_mlp_out, n);
        };
        let d = self.cfg.d_model;
        let rows = self.mlp_rows.get().expect("qwen35: an MLP saved as rows needs mlp_rows");
        // The row chunks tile all n rows, and each is copied in whole: a copy (`put_rows_copy`), because
        // `put_rows` adds and an uncleared buffer holds whatever the allocator last freed there.
        let d_xn2 = self.gpu.storage_uninit((n * d) as u64);
        for row0 in (0..n).step_by(rows as usize) {
            let rc = rows.min(n - row0);
            let acts = self.mlp_acts_only(prefix, &self.take_rows(xn2, d, n, row0, rc), rc);
            let d_chunk = self.mlp_bwd(prefix, &acts, &self.take_rows(d_mlp_out, d, n, row0, rc), rc);
            self.put_rows_copy(&d_chunk, &d_xn2, d, row0, rc);
        }
        self.gpu.poll_wait();
        d_xn2
    }

    fn mlp_bwd(&self, prefix: &str, la: &MlpLayerActs, d_mlp_out: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.intermediate_size;
        let p = |s: &str| format!("{prefix}.{s}");

        // Each of these is written whole before it is read: the first projection into a buffer overwrites it
        // (acc 0), and swiglu_bwd writes both of its outputs.
        let d_h = g.storage_uninit((n * ff) as u64);
        {
            let mut s = Vec::new();
            self.proj_bwd(&mut s, "down", d_mlp_out, &la.h, &p("down.weight"), &d_h, n, ff, d, 0);
            g.submit(&[], &s);
        }

        let d_gate_pre = g.storage_uninit((n * ff) as u64);
        let d_up = g.storage_uninit((n * ff) as u64);
        g.submit(&[], &swiglu_bwd(g, &kernel_ids(), &la.gate_pre, &la.up, &d_h, &d_gate_pre, &d_up, n * ff));

        let d_xn2 = g.storage_uninit((n * d) as u64);
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
        if self.layer_streamed() {
            // The weights of a layer-streamed build are not all on the device: the backward runs layer by layer in
            // `adamw_step`, where the layers it loads are the ones it updates. The forward's activations stay.
            assert!(
                !self.backward_pending.replace(true),
                "qwen35 layer-streamed full fine-tune: a second backward before the update (gradient accumulation) is not supported; every micro-batch is its own step"
            );
            return;
        }
        let ta = self.train_acts.borrow_mut().take().expect(
            "qwen35: backward() called without an immediately preceding forward() -- \
             every forward() call reallocates its activation cache fresh (this file's \
             own convention throughout), so backward() must run against the SAME call",
        );
        self.backward_over(&ta, &mut |_| Ok(())).unwrap_or_else(|e| panic!("qwen35: backward: {e}"));
    }

    /// The backward over the activations `ta` the forward kept: the head and final norm first, then the layers from
    /// the last down, reporting a [`model::resident_ft::LayerEvent`] at each boundary (a layer-streamed build loads
    /// a layer's weights before its backward and updates it after). The first error from `on` stops the pass.
    fn backward_over(&self, ta: &TrainActs, on: &mut dyn FnMut(model::resident_ft::LayerEvent) -> std::io::Result<()>) -> std::io::Result<()> {
        use model::resident_ft::LayerEvent;
        let g = &self.gpu;
        let n = self.b * self.run_len();
        let d = self.cfg.d_model;
        let v = self.cfg.vocab;

        // ---- head epilogue backward (head stage only): CE-grad, lm_head,
        // final norm -- a non-head stage starts instead from the externally
        // supplied gradient at `res[shard.end]` (`Self::write_out_dres`).
        let mut d_res_next = if self.shard.head {
            let d_xn_final = g.storage((n * d) as u64);
            if let Some(head) = &self.head_ce {
                // Chunked: recompute each chunk's logits, turn them into the
                // logit gradient and apply the head, never holding n * vocab.
                let dw = self.table_grad(self.cfg.head_weight());
                let mut s = Vec::new();
                head.backward_steps(
                    &self.ops,
                    &mut s,
                    &HeadBackward {
                        hidden: &ta.xn_final,
                        d_hidden: &d_xn_final,
                        n,
                        labels: &self.targets,
                        grad_scale: GradScale::Const(self.objectives.grad_factor(self.count.get())),
                        logit_scale: 1.0,
                        row_weights: self.objectives.row_weights(),
                        dw: dw.as_ref().map(|(buf, dtype)| model::head_ce::HeadGrad { buf, dtype: *dtype }),
                        forward_state_held: false,
                    },
                );
                g.submit(&[], &s);
            } else {
                g.write(&self.ce_grad_uni, &[n, v, model::IGNORE, f(self.count.get())]);
                let d_logits = g.storage((n * v) as u64);
                g.submit(&[], &[g.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &d_logits], n * v)]);
                let mut s = Vec::new();
                self.proj_bwd(&mut s, "lm_head", &d_logits, &ta.xn_final, self.cfg.head_weight(), &d_xn_final, n, d, v, 0);
                g.submit(&[], &s);
            }

            let d_res_next = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                let res = self.res.borrow();
                let last = match &self.res_saved {
                    Some(saved) => {
                        saved.load(g, self.cfg.n_layers as usize, &res[0]);
                        &res[0]
                    }
                    None => &res[self.cfg.n_layers as usize],
                };
                self.rmsnorm_bwd_step(&mut s, last, "norm.weight", &d_xn_final, &d_res_next, d, n);
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

        on(LayerEvent::AfterHead)?;
        let res = self.res.borrow();
        // Where a recomputed layer writes its (unused) output residual.
        let recompute_out = ta.recomputed.then(|| g.storage((n * d) as u64));
        for l in (self.shard.start..self.shard.end).rev() {
            on(LayerEvent::BeforeBackward(l))?;
            // The rebuild, the MLP row chunks and the input gradients of the layer read each frozen weight
            // from one decode.
            let _decoded = self.ops.decoded_weights();
            let _pass = g.pass_scope();
            // This layer's input boundary (decoded into `res[0]` when packed).
            let x = match &self.res_saved {
                Some(saved) => {
                    saved.load(g, l, &res[0]);
                    &res[0]
                }
                None => &res[l],
            };
            let rebuilt: Option<LayerTrainActs> =
                recompute_out.as_ref().map(|out| self.layer_forward(l, x, out, true).expect("qwen35: backward() requires a training build"));
            let la = match &rebuilt {
                Some(rebuilt) => rebuilt,
                None => &ta.layers[l - self.shard.start],
            };

            // ---- second residual add backward: res[l+1] = xmid + mlp_out ----
            let d_mlp_out = &d_res_next;
            let d_xn2 = self.mlp_bwd_saved(&format!("blocks.{l}.mlp"), &la.mlp, d_mlp_out, n);

            let d_xmid = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.rmsnorm_bwd_add_step(&mut s, &la.xmid, &format!("blocks.{l}.ln2.weight"), &d_xn2, &d_res_next, &d_xmid, d, n);
                g.submit(&[], &s);
            }

            // ---- first residual add backward: xmid = res[l] + mixer_out ----
            let d_xn1 = g.storage((n * d) as u64);
            match &la.mixer {
                MixerActs::Gdn(acts) => self.gdn_mixer_bwd(l, &la.xn1, acts, &d_xmid, &d_xn1, n),
                MixerActs::Gqa(acts) => self.gqa_mixer_bwd(&format!("blocks.{l}.self_attn"), &la.xn1, acts, &d_xmid, &d_xn1, n),
            }

            // ---- ln1 backward: xn1 = rmsnorm(res[l]) -> d_res[l] = d_xmid + d_tmp ----
            let mut d_res_l = g.storage((n * d) as u64);
            {
                let mut s = Vec::new();
                self.rmsnorm_bwd_add_step(&mut s, x, &format!("blocks.{l}.ln1.weight"), &d_xn1, &d_xmid, &d_res_l, d, n);
                g.submit(&[], &s);
            }
            // The anchor memory read the same residual input: its gradient joins d_res[l].
            if let MixerActs::Gqa(GqaLayerActs { memory: Some(memory), .. }) = &la.mixer {
                let d_h = self.anchor_memory_bwd(l, x, memory, &d_xmid, n);
                let joined = g.storage((n * d) as u64);
                g.submit(&[], &[g.step(ADD2, &[&d_res_l, &d_h, &joined], &[n * d], n * d)]);
                d_res_l = joined;
            }
            d_res_next = d_res_l;
            if rebuilt.is_some() {
                // Return this layer's rebuilt activations before the next
                // layer allocates its own.
                drop(rebuilt);
                g.poll_wait();
            }
            on(LayerEvent::AfterBackward(l))?;
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
            if !self.splice.is_empty() {
                g.submit(&[], &self.splice.bwd(g, SPLICE_BWD, &d_res_next));
            }

            // ---- embedding backward (tok.weight), over the distinct looked-up rows ----
            if let Some((dw, grad)) = self.table_grad("tok.weight") {
                let mut s = Vec::new();
                let ids = self.tok_uniq.as_ref().expect("a training build's embed stage keeps the distinct ids");
                let uniq = model::ops::EmbeddingGrad { tokens: &self.tokens, uniq: ids, n_uniq: self.b * self.t, d_x: &d_res_next, n_rows: n, width: d, dw: &dw, grad };
                self.ops.embedding_grad(&mut s, &uniq);
                g.submit(&[], &s);
            }
        }
        Ok(())
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
        let dec = self.decode_state();
        let mut clears: Vec<&DeviceBuffer> = Vec::new();
        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            if *ty == LayerType::Linear {
                clears.push(&dec.gdn_state[l]);
                clears.push(&dec.gdn_hist[l]);
            }
        }
        // A sliding-window config also forgets its ring and its anchor bank: nothing of the previous sequence may be
        // read by the next, whatever the position arithmetic says about which rows are live.
        if self.cfg.swa_window.is_some() {
            for (l, ty) in self.cfg.layer_types().iter().enumerate() {
                if *ty == LayerType::Full && self.shard.owns(l) {
                    clears.push(dec.gqa_kv[l].k.data());
                    clears.push(dec.gqa_kv[l].v.data());
                }
            }
            for pool in dec.memory.iter().flatten() {
                pool.reset_row(&self.gpu, 0);
            }
        }
        self.gpu.submit(&clears, &[]);
    }

    /// The flash-attention trio a TRAINING forward over this instance's `t`
    /// tokens runs instead of materialising the `[B,H,T,T]` scores and
    /// probabilities: only on a device that can run it, from
    /// `TrainOpts::flash_min_t` tokens up (below that the chain is cheaper and
    /// is the reference the trio is tested against). Inference keeps the chain.
    fn training_flash(&self) -> Option<model::block::FlashGqaIds> {
        let ids = self.flash?;
        (self.is_train && self.run_len() >= self.flash_min_t.get() && model::block::FlashGqaIds::supported(&self.gpu, self.cfg.head_dim)).then_some(ids)
    }

    /// Whether this instance's training attention runs the flash kernels
    /// (see [`TrainOpts::flash_min_t`]) rather than the materialised chain.
    pub fn attention_is_flash(&self) -> bool {
        self.training_flash().is_some()
    }

    /// The decode caches, built now if this is a training build's first decode.
    fn decode_state(&self) -> &DecodeState {
        self.decode.get_or_init(|| DecodeState::new(&self.gpu, &self.cfg, &self.shard, self.dec_cap))
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
        let caches = self.own_caches();
        let hidden = self.run_decode_step(token_id, pos, &caches, input.map(StageRows::Host));
        self.dec_pos.set(pos + 1);
        self.gpu.read(&hidden, self.cfg.d_model as usize)
    }

    /// Whether a decode step of `rows` rows may use the fused front end
    /// ([`Self::rms_quant_front`]): the device is offered `add_rms_quant`, some
    /// weight actually reads an int8 activation, the shape is one the kernel
    /// serves, and the unfused chain would have normalised with the cooperative
    /// RMSNorm - the only order the kernel reproduces. The last clause is what
    /// makes the fusion invisible rather than merely close.
    fn fused_front_ok(&self, rows: u32) -> bool {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let coop = block::rms_variant(g, RMSNORM, Some(RMSNORM_ROWS), rows, d).0 == RMSNORM_ROWS;
        self.decode_fusion.get() && coop && !self.weights.values().all(|w| matches!(w, Weight::F32 { .. }))
            && g.has_fused(gpu_core::Fused::AddRmsQuant)
            && gpu_core::Fused::AddRmsQuant.serves(&[d, rows, f(RMS_EPS), 1])
    }

    /// Whether a decode step of `rows` rows may use [`Self::quant_epilogue`]
    /// at every width the layers hand it: the device is offered the kernel and
    /// some weight actually reads an int8 activation. The kernel's arithmetic
    /// is the unfused chain's own (nothing in it is order-sensitive), so unlike
    /// the front end there is no variant to match.
    fn fused_epilogue_ok(&self, rows: u32) -> bool {
        let g = &self.gpu;
        let c = &self.cfg;
        let widths = [c.intermediate_size, c.linear_value_dim(), c.q_proj_dim() / 2];
        self.decode_fusion.get()
            && !self.weights.values().all(|w| matches!(w, Weight::F32 { .. }))
            && g.has_fused(gpu_core::Fused::QuantEpilogue)
            && widths.iter().all(|&k| gpu_core::Fused::QuantEpilogue.serves(&[k, rows, 0, 0]))
    }

    /// Whether some weight of this build reads packed int8 activation rows. FP8, bf16, NF4 and f32 tiers read float rows
    /// (`ops_act` hands them [`Ops::act_f32`]), so the fused front ends skip the packing there and the `Act` they
    /// return is the float one.
    fn unpacked_slots(&self) -> (&DeviceBuffer, &DeviceBuffer) {
        let (xq, sx) = self.unpacked_slots.get_or_init(|| (self.gpu.storage(1), self.gpu.storage(1)));
        (xq, sx)
    }

    fn packs_activations(&self) -> bool {
        !self.weights.values().all(reads_float_acts)
    }

    /// Whether the fused front ends also write the bf16 rows of their activation (`cast_f32_bf16`'s result, in the same
    /// pass): the FP8 products the tuned table has read bf16 rows then need no cast launch. A build that packs int8
    /// activations leaves the slots to the packing.
    fn fused_writes_bf16(&self) -> bool {
        !self.packs_activations() && self.weights.values().any(|w| w.dtype() == Dtype::F8E4M3)
    }

    /// The MLP product `silu(gate) * up` with the E4M3 rows of the FP8 `down` projection written in the same pass (the
    /// product is kept for the consumers of its fp32 rows). `None` unless this build packs no int8 rows, `down` reads E4M3
    /// activations at this row count on this device and the device offers the kernel: the callers then take
    /// [`Self::quant_epilogue`] and the product's own quantisation launch.
    fn silu_mul_e4m3(&self, gate: &DeviceBuffer, up: &DeviceBuffer, input: gpu_core::ActPrecision, rows: u32, k: u32) -> Option<(DeviceBuffer, Act)> {
        if !self.silu_mul_e4m3_serves(rows, k) {
            return None;
        }
        let g = &self.gpu;
        // The fp32 product has a consumer only in a backward or under an adapter; otherwise the pass does not write it and the
        // buffer handed back is the one-word slot every caller of this path leaves unread.
        let product = self.keeps_fp32_product().then(|| g.storage_uninit(rows as u64 * k as u64));
        let mut steps = Vec::new();
        let (q, s) = gpu_core::provider::library::silu_mul_quantize_e4m3_rows(g, &mut steps, gate, up, input, product.as_ref(), rows, k)?;
        g.submit(&[], &steps);
        let product = product.unwrap_or_else(|| self.unpacked_slots().0.clone());
        let act = self.ops.act_f32_with_e4m3(&product, rows, k, q, s);
        Some((product, act))
    }

    /// Whether [`Self::silu_mul_e4m3`] runs for this shape: no int8 packing in this build, `down` reads E4M3 rows at this
    /// row count and the device offers the kernel.
    fn silu_mul_e4m3_serves(&self, rows: u32, k: u32) -> bool {
        !self.packs_activations() && self.ops.fp8_reads_e4m3(rows, k, self.cfg.d_model) && gpu_core::provider::library::silu_mul_quantize_e4m3_serves(&self.gpu, rows, k)
    }

    /// Whether anything reads the MLP product as fp32 rows: the saved activations of a backward, a training adapter or the
    /// served adapter bank (their corrections of `down` read it).
    fn keeps_fp32_product(&self) -> bool {
        self.is_train || self.adapter.is_some() || self.adapter_bank.borrow().is_some()
    }

    /// Whether the Gated DeltaNet mixer of a round of `rows` rows may run in bf16: the stream was asked for bf16, nothing reads
    /// the layer's fp32 rows (no backward, no adapter correcting a projection in place), and the output projection takes the
    /// E4M3 rows the gated norm writes (the fp32 product of the segments is then not stored).
    fn bf16_gdn_ok(&self, rows: u32, value_dim: u32) -> bool {
        self.act_precision.get() == gpu_core::ActPrecision::Bf16 && !self.keeps_fp32_product() && !self.packs_activations() && self.ops.fp8_reads_e4m3(rows, value_dim, self.cfg.d_model)
    }

    /// Whether the gate and up products of an MLP of `rows` rows may store bf16: the stream was asked for bf16, nothing reads
    /// them as fp32 (no backward, no adapter correcting them in place), and the SwiGLU pass that consumes them runs.
    fn bf16_gate_up_ok(&self, rows: u32, ff: u32) -> bool {
        self.act_precision.get() == gpu_core::ActPrecision::Bf16 && !self.keeps_fp32_product() && self.silu_mul_e4m3_serves(rows, ff)
    }

    /// One launch that produces the activation an int8 linear reads AND
    /// quantises it: `mode` 0 quantises `a` as it is, 1 produces `silu(a) * b`,
    /// 2 produces `a * sigmoid(b)`. Returns `(product, act)`; `product` is
    /// `None` for mode 0, where the activation is `a` itself.
    ///
    /// Only call after [`Self::fused_epilogue_ok`].
    fn quant_epilogue(&self, mode: u32, a: &DeviceBuffer, b: Option<&DeviceBuffer>, rows: u32, k: u32) -> (Option<DeviceBuffer>, Act) {
        let g = &self.gpu;
        let n = (rows * k) as u64;
        let y = (mode != 0).then(|| g.storage_uninit(n));
        // No weight of this build reads packed int8 rows (FP8, bf16, NF4 tiers): the kernel then writes the product
        // only, and mode 0, which would write nothing, is no launch at all.
        if !self.packs_activations() {
            if let Some(y) = &y {
                let (xq, sx) = self.unpacked_slots();
                // The `xq` slot takes the bf16 rows of the product when an FP8 product will read them (flags bit 1).
                let bf16 = self.fused_writes_bf16().then(|| g.storage_uninit(n / 2));
                let step = g
                    .fused_step(gpu_core::Fused::QuantEpilogue, &[a, b.unwrap_or(a), y, bf16.as_ref().unwrap_or(xq), sx], &[k, rows, mode, 1 | if bf16.is_some() { 2 } else { 0 }])
                    .expect("fused_epilogue_ok said the device serves this shape");
                g.submit(&[], &[step]);
                if let Some(bf16) = bf16 {
                    return (Some(y.clone()), self.ops.act_f32_with_bf16(y, rows, k, bf16));
                }
            }
            return (y.clone(), self.ops.act_f32(y.as_ref().unwrap_or(a), 0, rows, k));
        }
        let xq = g.storage_uninit(n / 4);
        let sx = g.storage_uninit(model::int8::act_scale_len(rows as usize, k as usize) as u64);
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
    /// `b` is given), `xn = rmsnorm(sum) * w`, and the group-scaled int8 packing of
    /// `xn` as a ready [`Act`]. Returns `(sum, xn, act)`; `sum` is `None` when
    /// there was no `b`, and the caller keeps `a` as the residual.
    ///
    /// `uses` says who reads the result. When every product that reads it takes E4M3 rows at this row count
    /// ([`FrontUse::reads_e4m3`]) the pass writes those rows and their scales itself - the quantisation launch and its re-read of
    /// `xn` are gone - and `xn` (fp32) is stored only if something reads it as fp32 ([`FrontUse::reads_f32`]: the dense
    /// gates of a Gated DeltaNet layer, an adapter, a backward); otherwise the buffer handed back is the one-word slot
    /// nothing reads.
    ///
    /// Only call after [`Self::fused_front_ok`].
    fn rms_quant_front(&self, a: &DeviceBuffer, b: Option<&DeviceBuffer>, w: &DeviceBuffer, rows: u32, uses: FrontUse) -> (Option<DeviceBuffer>, DeviceBuffer, Act) {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let n = (rows * d) as u64;
        let sum = b.map(|_| g.storage_uninit(n));
        let pack = self.packs_activations();
        let widths = uses.product_widths(&self.cfg);
        let e4m3 = !pack
            && d.is_multiple_of(128)
            && !widths.is_empty()
            && widths.iter().all(|&width| self.ops.fp8_reads_e4m3(rows, d, width))
            && g.has_fused(gpu_core::Fused::AddRmsQuantE4m3)
            && gpu_core::Fused::AddRmsQuantE4m3.serves(&[d, rows, f(RMS_EPS), 1]);
        let store_xn = !e4m3 || uses.reads_f32() || self.keeps_fp32_product();
        let xn = if store_xn { g.storage_uninit(n) } else { self.unpacked_slots().0.clone() };
        // Nothing reads the packed rows of an FP8/bf16/NF4 build: the kernel then writes `sum` and `xn` only, and the
        // two buffers it would have written are one word each.
        // The `xq` slot takes the bf16 rows of `xn` when an FP8 product will read them (flags bit 2), or, in the E4M3 variant of the
        // kernel, its E4M3 rows (and `sx` their scales).
        let bf16 = (!e4m3 && self.fused_writes_bf16()).then(|| g.storage_uninit(n / 2));
        let (xq, sx) = if pack {
            (g.storage_uninit(n / 4), g.storage_uninit(model::int8::act_scale_len(rows as usize, d as usize) as u64))
        } else if e4m3 {
            (g.storage_uninit(n / 4), g.storage_uninit(n / 128))
        } else {
            let (xq, sx) = self.unpacked_slots();
            (bf16.clone().unwrap_or_else(|| xq.clone()), sx.clone())
        };
        // Without an addend the kernel reads neither `b` nor writes `sum`; the
        // slots still have to be bound, so they alias what is already there.
        // (The E4M3 kernel is its own variant: its flags are the add and whether `xn` is stored.)
        let flags = if e4m3 { b.is_some() as u32 | if store_xn { 0 } else { 16 } } else { b.is_some() as u32 | if pack { 0 } else { 2 } | if bf16.is_some() { 4 } else { 0 } };
        let kernel = if e4m3 { gpu_core::Fused::AddRmsQuantE4m3 } else { gpu_core::Fused::AddRmsQuant };
        let step = g
            .fused_step(kernel, &[a, b.unwrap_or(a), w, sum.as_ref().unwrap_or(&xn), &xn, &xq, &sx], &[d, rows, f(RMS_EPS), flags])
            .expect("fused_front_ok said the device serves this shape");
        g.submit(&[], &[step]);
        let act = match (pack, e4m3, bf16) {
            (true, _, _) => self.ops.act_prequantized(&xn, &sx, &xq, rows, d),
            (false, true, _) => self.ops.act_f32_with_e4m3(&xn, rows, d, xq, sx),
            (false, false, Some(bf16)) => self.ops.act_f32_with_bf16(&xn, rows, d, bf16),
            (false, false, None) => self.ops.act_f32(&xn, 0, rows, d),
        };
        (sum, xn, act)
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
    ///
    /// Deliberately does NOT apply an `lm_head` projection, even on a head
    /// stage. At the real Qwen3.8-27B shape a `[248320, 5120]` fp32 head is
    /// 5.09 GB - past `max_buffer_size` on a 24 GB P40, so it cannot be a
    /// device buffer at all and a shard that owned it would fail to load.
    /// Both real-weight callers therefore keep the head OUTSIDE the shard, as
    /// an int8 `model::ops::Weight` (1.27 GB packed, inside the binding
    /// limit), and project with `crate::stream::head_logits_on`.
    pub(crate) fn run_decode_step(&self, token_id: u32, pos: u32, caches: &DecodeCaches, input_override: Option<StageRows<'_>>) -> DeviceBuffer {
        assert!(caches.gqa_cap > 0, "qwen35::run_decode_step: a KV block of 0 rows");
        let seqs = [BatchSeq {
            window: caches.gqa_window,
            slot: caches.slot,
            pos,
            gdn_state: caches.gdn_state,
            gdn_hist: caches.gdn_hist,
        }];
        let batch = BatchDecodeCaches { gqa_kv: caches.gqa_kv, gqa_cap: caches.gqa_cap, seqs: &seqs, one_sequence: false, gdn_pool: None, memory: caches.memory };
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
    pub(crate) fn run_decode_batch(&self, tokens: &[u32], caches: &BatchDecodeCaches, input_override: Option<StageRows<'_>>) -> DeviceBuffer {
        self.run_decode_batch_with(tokens, caches, input_override, None)
    }

    /// Buffers for one decode step's per-step inputs over `bsz` rows ([`DecodeInputs`]), drawn from the scratch arena
    /// when one is open. Their contents are set by [`Self::write_decode_inputs`].
    fn alloc_decode_inputs(&self, bsz: u32, caches: &BatchDecodeCaches, verify: bool) -> DecodeInputs {
        // The exact-decode regime sizes every row's attention from the configured context, not from the longest
        // sequence that happens to share this step: `cap` and the block-table width are then the same for a row
        // whatever its neighbours are, so its split count and key partition cannot move. Unused block-table entries
        // are never read (`seq_lens` bounds every access), and a split past a sequence's live length is one
        // degenerate record.
        let max_bt = match self.exact_decode.get().filter(|_| !verify) {
            Some(ctx) => ctx.div_ceil(caches.gqa_cap).max(1),
            // A decode step of a growing cache is built at the bucketed capacity, so the step (its grid, its scratch and
            // its table width) is the same for every token of a bucket: the shape a recording is keyed on.
            None if !verify && !caches.seqs.iter().any(|s| s.window.is_ring()) => self.paged_step_shape(caches).1,
            None => caches.seqs.iter().map(|s| s.window.blocks().len()).max().unwrap_or(1).max(1) as u32,
        };
        self.alloc_decode_inputs_at(bsz, max_bt)
    }

    /// `(keys, block-table width)` a decode step over growing (paged) caches is built at: the longest live sequence
    /// rounded up to [`DECODE_CAP_BUCKET`] keys, and the blocks that many keys span. Everything in the step's dispatch
    /// list that depends on the context (the attention grid and its key splits, the scores stride, the table width) is
    /// a function of this pair, so a step is one recording per pair; the keys past a row's live length are never read
    /// (`seq_lens` bounds every access) and its table entries past its last block are never followed.
    pub(crate) fn paged_step_shape(&self, caches: &BatchDecodeCaches) -> (u32, u32) {
        let longest = caches.seqs.iter().map(|s| s.pos + 1).max().unwrap_or(1);
        let cap = longest.next_multiple_of(DECODE_CAP_BUCKET);
        (cap, cap.div_ceil(caches.gqa_cap).max(1))
    }

    /// [`Self::alloc_decode_inputs`] for a block-table width of `max_bt`.
    pub(crate) fn alloc_decode_inputs_at(&self, bsz: u32, max_bt: u32) -> DecodeInputs {
        let g = &self.gpu;
        let rows = u64::from(bsz);
        let memory = self.cfg.anchor_memory.as_ref().map(|_| {
            let shape = self.memory_shape_of(bsz, 1);
            (shape, model::anchor_memory::ReadPlan::alloc(g, &shape, bsz), model::anchor_memory::PushPlan::alloc(g, bsz))
        });
        DecodeInputs {
            bsz,
            max_bt,
            tok: g.storage(rows),
            cos: g.storage(rows * u64::from(self.cfg.rotary_dim() / 2)),
            sin: g.storage(rows * u64::from(self.cfg.rotary_dim() / 2)),
            blocks: g.storage(rows),
            offsets: g.storage(rows),
            slot_rows: g.storage(rows),
            block_tables: g.storage(rows * u64::from(max_bt)),
            seq_lens: g.storage(rows),
            memory,
        }
    }

    /// The paged-attention view of a decode step's per-step inputs.
    ///
    /// A ring is a fixed-size window, so its attention scratch has one size and one split count at every position:
    /// the step is the same dispatch list from the first token to the last, which is what a recording needs. Any
    /// other window sizes it to the longest live sequence, in buckets; `exact_ctx` (the exact-decode regime) sizes
    /// it from the configured context instead.
    fn paged_decode_batch<'a>(&self, caches: &BatchDecodeCaches, inputs: &'a DecodeInputs, exact_ctx: Option<u32>) -> model::gqa_mixer::PagedDecodeBatch<'a> {
        let live = |s: &BatchSeq| if s.window.is_ring() { model::gqa_mixer::ring_live(s.pos, caches.gqa_cap) } else { s.pos + 1 };
        let longest = caches.seqs.iter().map(live).max().unwrap_or(1);
        let cap_bound = caches.gqa_cap * inputs.max_bt;
        let all_ring = caches.seqs.iter().all(|s| s.window.is_ring());
        model::gqa_mixer::PagedDecodeBatch {
            blocks: &inputs.blocks,
            offsets: &inputs.offsets,
            block_tables: &inputs.block_tables,
            seq_lens: &inputs.seq_lens,
            block_size: caches.gqa_cap,
            max_bt: inputs.max_bt,
            // The scores/probs stride, and with it those buffers' size and the
            // scores kernel's grid. Pure addressing (see
            // `gqa_decode_batched_step`'s own doc), so it only has to cover the
            // longest live sequence - and it is rounded UP to a bucket so that
            // it, and everything sized from it, is the same from one token to
            // the next. Sized exactly it changed every token: a fresh buffer
            // and a new grid per attention layer, which defeats both the arena
            // and the replayed graph. Now both change once per bucket.
            cap: if all_ring { cap_bound } else { exact_ctx.unwrap_or(longest).next_multiple_of(DECODE_CAP_BUCKET).min(cap_bound) },
            pin_splits: exact_ctx.is_some(),
        }
    }

    /// Set a decode step's per-step inputs for `caches`' rows: the one place the host's view of the batch (positions,
    /// rings, slots) becomes the data the step's kernels read. `tokens` is `None` on a step that starts from a residual.
    /// Every write comes before the step's first dispatch.
    pub(crate) fn write_decode_inputs(&self, inputs: &DecodeInputs, tokens: Option<&[u32]>, caches: &BatchDecodeCaches) {
        let (g, c) = (&self.gpu, &self.cfg);
        let seqs = caches.seqs;
        assert_eq!(seqs.len(), inputs.bsz as usize, "qwen35: decode inputs for {} rows written for {}", inputs.bsz, seqs.len());
        if let Some(tokens) = tokens {
            g.write(&inputs.tok, tokens);
        }
        // The batch's M-RoPE table at each sequence's OWN position, and the four paged-KV index buffers. The rows of
        // one batch are unrelated positions and unrelated physical blocks, which is the whole difference from a chunk.
        let yarn = c.rope_table();
        let positions: Vec<[u32; 3]> = seqs.iter().map(|s| [s.pos, s.pos, s.pos]).collect();
        let (cos_rows, sin_rows) = qwen3vl::mrope::mrope_tables_scaled(&positions, c.mrope_section, c.rotary_dim(), c.rope_theta, yarn.as_ref().map(|(f, a)| (f.as_slice(), *a)));
        g.write_f32(&inputs.cos, &cos_rows);
        g.write_f32(&inputs.sin, &sin_rows);
        // Each row's block table is its sequence's blocks in logical order, padded to the widest (a single entry for a
        // one-block sequence).
        let windows: Vec<Vec<u32>> = seqs.iter().map(|s| s.window.blocks()).collect();
        let bs = caches.gqa_cap;
        g.write(&inputs.blocks, &seqs.iter().zip(&windows).map(|(s, w)| if s.window.is_ring() { w[0] } else { w[(s.pos / bs) as usize] }).collect::<Vec<u32>>());
        g.write(&inputs.offsets, &seqs.iter().map(|s| if s.window.is_ring() { model::gqa_mixer::ring_row(s.pos, bs) } else { s.pos % bs }).collect::<Vec<u32>>());
        g.write(&inputs.slot_rows, &seqs.iter().map(|s| s.slot).collect::<Vec<u32>>());
        g.write(&inputs.block_tables, &windows.iter().flat_map(|w| (0..inputs.max_bt as usize).map(|i| w.get(i).copied().unwrap_or(0))).collect::<Vec<u32>>());
        g.write(&inputs.seq_lens, &seqs.iter().map(|s| if s.window.is_ring() { model::gqa_mixer::ring_live(s.pos, bs) } else { s.pos + 1 }).collect::<Vec<u32>>());
        // The anchor memory's per-step plans, shared by every memory layer: which bank slots each sequence reads, and
        // the record it pushes.
        if let Some((shape, read, push)) = &inputs.memory {
            read.write(g, shape, &seqs.iter().map(|s| (s.slot, s.pos)).collect::<Vec<_>>());
            push.write(g, shape, &seqs.iter().enumerate().map(|(i, s)| model::anchor_memory::Push { slot: s.slot, pos: s.pos, src_row: i as u32 }).collect::<Vec<_>>());
        }
    }

    /// The bank commits a decode step over `seqs` owes: one per sequence whose token completes a commit block (the
    /// host knows every position). Empty on most steps, and then nothing moves.
    fn decode_memory_commits(&self, seqs: &[BatchSeq]) -> Vec<model::anchor_memory::Commit> {
        let Some(cb) = self.cfg.memory_commit_block() else { return Vec::new() };
        seqs.iter().filter(|s| s.pos % cb == cb - 1).map(|s| model::anchor_memory::Commit { slot: s.slot, first_pos: s.pos + 1 - cb, count: cb }).collect()
    }

    /// Commit `commits` into every layer's bank (a decode step's pushes are already in the pending blocks). Issued after
    /// the step so the step's own dispatches do not depend on which sequences complete a block.
    fn commit_memory_blocks(&self, pools: &[Option<model::anchor_memory::AnchorBankPool>], commits: &[model::anchor_memory::Commit]) {
        if commits.is_empty() {
            return;
        }
        let g = &self.gpu;
        let shape = self.memory_shape_of(commits.len() as u32, 1);
        let plan = model::anchor_memory::CommitPlan::upload(g, &shape, commits);
        let ids = self.memory_ids();
        for pool in pools.iter().flatten() {
            g.submit(&[], &model::anchor_memory::state_commit_steps(g, &ids.memory, &shape, pool, &plan));
        }
    }

    /// [`Self::run_decode_batch`] in exact-decode groups of at most `step` rows.
    ///
    /// One `run_decode_batch` per group, each a complete pass over every layer
    /// (so the group's state hand-off is a decode step's own) and each sized
    /// from the configured context rather than the batch. Groups are read back
    /// and concatenated in the caller's row order. The extra device round trip
    /// per group is exact mode's own cost on top of the second weight pass: it
    /// is opt-in, and a caller that needs a row to be a row pays for it.
    fn run_decode_batch_exact_groups(&self, tokens: &[u32], caches: &BatchDecodeCaches, input_override: Option<StageRows<'_>>, step: u32) -> DeviceBuffer {
        let d = self.cfg.d_model as usize;
        let step = step as usize;
        let mut out = Vec::with_capacity(tokens.len() * d);
        for start in (0..tokens.len()).step_by(step) {
            let end = (start + step).min(tokens.len());
            let seqs: Vec<BatchSeq> = caches.seqs[start..end]
                .iter()
                .map(|s| BatchSeq { window: s.window, slot: s.slot, pos: s.pos, gdn_state: s.gdn_state, gdn_hist: s.gdn_hist })
                .collect();
            let pool = caches.gdn_pool.as_ref().map(|p| GdnPoolStore { state: p.state, hist: p.hist, bf16: p.bf16 });
            let group = BatchDecodeCaches { gqa_kv: caches.gqa_kv, gqa_cap: caches.gqa_cap, seqs: &seqs, one_sequence: false, gdn_pool: pool, memory: caches.memory };
            let x = input_override.map(|x| x.slice(start * d, end * d));
            let hidden = self.run_decode_batch_with(&tokens[start..end], &group, x, None);
            out.extend(self.gpu.read(&hidden, (end - start) * d));
        }
        self.gpu.storage_init("qwen35.exact_decode.hidden", &out)
    }

    /// [`Self::run_decode_batch`], optionally as a speculative VERIFY step
    /// ([`verify::VerifyBatch`]): the rows are then several consecutive tokens
    /// of each sequence rather than one token of each, and the Gated-DeltaNet
    /// layers take a sequence's rows in order and snapshot the state between
    /// them. Everything else in the pass is row-agnostic.
    pub(crate) fn run_decode_batch_with(&self, tokens: &[u32], caches: &BatchDecodeCaches, input_override: Option<StageRows<'_>>, verify: Option<&verify::VerifyBatch>) -> DeviceBuffer {
        // The exact-decode regime (see `set_exact_decode`): a batch wider than
        // one bit-identical group runs as several whole steps, in order. Each
        // group's hidden block is read before the next is issued - the same
        // drain every caller of this function already makes between steps -
        // and the rows are put back in the caller's order.
        // Adapter rows are indexed by position within the dispatch, so a
        // group's rows would read the wrong adapter's correction: exact
        // grouping stays off while a bank corrects this step (`adapter_rows`
        // is 0 otherwise, and a serving caller refuses the combination).
        let input_override = input_override.and_then(StageRows::provided);
        let group = if verify.is_none() && !caches.one_sequence && self.adapter_rows.get() == 0 { self.exact_decode_step() } else { None };
        if let Some(step) = group.filter(|&step| tokens.len() as u32 > step) {
            return self.run_decode_batch_exact_groups(tokens, caches, input_override, step);
        }
        let hidden = self.decode_batch_dispatches(tokens, caches, input_override, verify, None);
        // The step's banks that filled a block commit once its dispatches are issued (see `decode_memory_commits`).
        self.commit_memory_blocks(caches.memory, &self.decode_memory_commits(caches.seqs));
        hidden
    }

    /// The dispatches of one decode step over `caches`' rows, without the anchor-memory bank commits (which the caller
    /// issues after, see [`Self::commit_memory_blocks`]). With `persist` the step's per-step inputs live in those
    /// buffers (the caller rewrites them before every replay of a recording of this call), the scratch arena and the
    /// pass are left to the recording, and nothing in the dispatch list depends on the positions.
    fn decode_batch_dispatches(&self, tokens: &[u32], caches: &BatchDecodeCaches, input_override: Option<StageRows<'_>>, verify: Option<&verify::VerifyBatch>, persist: Option<&DecodeInputs>) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let bsz = tokens.len() as u32;
        assert!(bsz > 0, "qwen35::run_decode_batch: empty batch");
        assert_eq!(tokens.len(), caches.seqs.len(), "qwen35::run_decode_batch: {} tokens for {} sequences", tokens.len(), caches.seqs.len());
        for s in caches.seqs {
            let capacity = s.window.capacity(caches.gqa_cap);
            assert!(s.pos < capacity, "qwen35::run_decode_batch: decode position {} exceeds the per-sequence capacity {capacity}", s.pos);
        }
        // A sliding-window model attends rings of exactly `window` rows; a cache that grows with the sequence would
        // silently attend all of it, so any other window is refused. A model without one never takes a ring.
        let ring = c.swa_window;
        for s in caches.seqs {
            match ring {
                Some(win) => assert!(s.window.is_ring() && caches.gqa_cap == win, "qwen35::run_decode_batch: this config's attention is a {win}-row sliding window and needs ring KV windows of that size (got {:?}, block of {} rows)", s.window, caches.gqa_cap),
                None => assert!(!s.window.is_ring(), "qwen35::run_decode_batch: a ring KV window for a model without a sliding window"),
            }
        }
        // A verify round puts several rows of ONE sequence in a batch. Its rows write the ring before any attends, so
        // a row could overwrite a position an earlier row still needs, and its memory rows would commit blocks
        // between each other: neither is implemented.
        assert!(ring.is_none() || (!caches.one_sequence && verify.is_none()), "qwen35::run_decode_batch: a verify round (several rows of one sequence) does not implement this config's sliding window");
        if c.anchor_memory.is_some() {
            assert!(!caches.memory.is_empty(), "qwen35::run_decode_batch: this config has an anchor memory and the caches carry no bank pool");
            for s in caches.seqs {
                assert!(caches.memory.iter().flatten().all(|pool| s.slot < pool.slots), "qwen35::run_decode_batch: slot {} is outside the anchor-bank pool", s.slot);
            }
        }

        // The whole token is one replayable submission. Two things make that so,
        // and both have to be in force before the first buffer is asked for:
        //
        // * the scratch arena hands every temporary - and every per-token input
        //   below - the same device buffer it handed the previous token, so no
        //   allocation (and no free, which on CUDA waits for the whole device)
        //   happens in a steady-state token and the recorded dispatches name the
        //   same addresses every time;
        // * the pass holds the per-layer submissions and issues them as one, so a
        //   backend that can record the token whole replays it instead of
        //   launching ~2500 dispatches one at a time.
        //
        // Declared arena first: guards drop in reverse, so the pass issues its
        // work while the arena is still open. The arena's contract is that the
        // previous token's work has drained before this token's dispatches are
        // submitted, which every caller meets by reading the step's output.
        //
        // The linears' activation scratch is allocated through `self.ops`, which
        // runs on its own handle of the same device, so it has an arena of its
        // own to open.
        //
        // A round of one sequence's rows has an arena per row count: the arena
        // replays a fixed request sequence, and a different count is a
        // different one. One row is the decode step itself and shares its arena
        // (and, with it, its graph).
        //
        // A recording (`persist`) needs neither: it keeps every buffer it names alive itself, and a replay is one
        // submission by construction.
        let arena = if verify.is_some() {
            verify::VERIFY_ARENA
        } else if caches.one_sequence {
            round_arena(bsz)
        } else {
            DECODE_ARENA
        };
        let _ops_scratch = persist.is_none().then(|| self.ops.gpu().scratch_scope_in(arena));
        let _scratch = persist.is_none().then(|| g.scratch_scope_in(arena));
        let _pass = persist.is_none().then(|| g.pass_scope());

        // Every host->device write of the token comes BEFORE the first
        // dispatch is submitted. A write is ordered after all earlier
        // submissions, so one between dispatches would close the pass early and
        // split the token into two graphs.
        // `input_override` is the batch's input rows: on a non-embed stage the
        // previous stage's residual, on the embed stage the caller's own input
        // embeddings in place of the token gather (`Qwen35::prefill_embeds`).
        assert!(
            persist.is_none() || (input_override.is_none() && verify.is_none() && self.exact_decode.get().is_none() && self.adapter_rows.get() == 0),
            "qwen35::decode_batch_dispatches: a recorded step is a plain token step (no verify round, exact-decode regime or adapter rows)"
        );
        let owned_inputs;
        let inputs = match persist {
            Some(inputs) => {
                assert_eq!(inputs.bsz, bsz, "qwen35::decode_batch_dispatches: inputs for {} rows used for {bsz}", inputs.bsz);
                inputs
            }
            None => {
                owned_inputs = self.alloc_decode_inputs(bsz, caches, verify.is_some());
                &owned_inputs
            }
        };
        let mut res = match input_override {
            Some(x) => {
                assert_eq!(x.words(), (bsz * d) as usize, "qwen35: run_decode_batch input_override must be the batch's whole [bsz, d_model] block");
                let res = g.storage((bsz * d) as u64);
                x.load_into(g, &res);
                res
            }
            None => {
                assert!(self.shard.embed, "qwen35: run_decode_batch on a NON-EMBED stage needs the previous stage's residual as `input_override` (this stage holds no `tok.weight`)");
                g.storage((bsz * d) as u64)
            }
        };
        let tok_buf = input_override.is_none().then_some(&inputs.tok);
        self.write_decode_inputs(inputs, if tok_buf.is_some() { Some(tokens) } else { None }, caches);
        let DecodeInputs { cos, sin, slot_rows, memory, .. } = inputs;
        let memory_step = memory.as_ref().map(|(shape, read, push)| (shape, read, push));
        // A verify step's per-sequence layout, uploaded with the rest of the
        // step's inputs (a write after the first dispatch would split the pass).
        let verify_layout = verify.map(|v| v.upload(g));
        let exact_ctx = self.exact_decode.get().filter(|_| verify.is_none());
        let paged = self.paged_decode_batch(caches, inputs, exact_ctx);

        if let Some(tok_buf) = tok_buf {
            self.embed_rows(tok_buf, bsz, &res);
        }

        // The staging kernels of a pooled recurrent state: the f32 pair, or the bf16 builds of a compact pool.
        let (pool_gather, pool_scatter) = match &caches.gdn_pool {
            Some(pool) if pool.bf16 => (self.kernel_by_name(POOL_ROWS_GATHER2_BF16), self.kernel_by_name(POOL_ROWS_SCATTER2_BF16)),
            _ => (POOL_ROWS_GATHER2, POOL_ROWS_SCATTER2),
        };
        let types = c.layer_types();

        // Where the residual add, the norm and the activation quantisation that
        // sit between two linears are ONE launch instead of four (`add2`,
        // `rmsnorm_rows`, `max_abs_rows`, `quant_pack`) - and bit-identical to
        // those four, which is why the gate is "the device offers the kernel and
        // the unfused chain would have used the cooperative RMSNorm", never a
        // tolerance. Off, the loop below is the unfused chain it always was.
        let fused = self.fused_front_ok(bsz);
        let epilogue = self.fused_epilogue_ok(bsz);
        let first_ln1 = format!("blocks.{}.ln1.weight", self.shard.start);
        // The next layer's `(xn1, activation)`, when the previous step produced it.
        let mut front = (fused && self.shard.start < self.shard.end).then(|| {
            let (_, xn, act) = self.rms_quant_front(&res, None, self.w(&first_ln1), bsz, FrontUse::Mixer(self.cfg.layer_types()[self.shard.start]));
            (xn, act)
        });

        // The final norm's output, when the last layer's residual add produced it.
        let mut final_norm: Option<DeviceBuffer> = None;
        // `l` is the ABSOLUTE layer index (into `types`, `caches.*` and the
        // `blocks.{l}.*` weight names below), not just a loop counter -
        // exactly as in `Self::run_forward`.
        #[allow(clippy::needless_range_loop)]
        for l in self.shard.start..self.shard.end {
            let ty = types[l];
            let (xn1, pre1) = match front.take() {
                Some((xn, act)) => (xn, Some(act)),
                None => {
                    let xn1 = g.storage((bsz * d) as u64);
                    g.submit(&[], &[rms_step(g, &res, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, bsz)]);
                    (xn1, None)
                }
            };

            let mixer_out = match ty {
                LayerType::Linear => {
                    let streams: Vec<model::gdn_mixer::GdnStream> = if caches.gdn_pool.is_some() {
                        Vec::new()
                    } else {
                        caches.seqs.iter().map(|s| model::gdn_mixer::GdnStream { state: &s.gdn_state[l], hist: &s.gdn_hist[l] }).collect()
                    };
                    let call = if let (Some(pool), Some(v), Some(layout)) = (&caches.gdn_pool, verify, &verify_layout) {
                        assert!(!pool.bf16, "qwen35: a speculative verify round needs an f32 recurrent-state pool");
                        GdnCall::VerifyPool {
                            pool: model::gdn_mixer::GdnPoolRows {
                                state: &pool.state[l],
                                hist: &pool.hist[l],
                                rows: &layout.rows,
                                gather: POOL_ROWS_GATHER2,
                                scatter: POOL_ROWS_SCATTER2,
                                fuse: true,
                                bf16: false,
                                wide: false,
                            },
                            layout: model::gdn_verify::VerifyLayout { rows: &layout.rows, counts: &layout.counts, offs: &layout.offs, b: v.sequences() },
                            snaps: v.snapshots(l),
                            n_tokens: bsz,
                        }
                    } else if let Some(pool) = &caches.gdn_pool {
                        GdnCall::DecodePool(model::gdn_mixer::GdnPoolRows {
                            state: &pool.state[l],
                            hist: &pool.hist[l],
                            rows: slot_rows,
                            gather: pool_gather,
                            scatter: pool_scatter,
                            fuse: self.decode_fusion.get(),
                            bf16: pool.bf16,
                            wide: self.gdn_pool_wide.get(),
                        })
                    } else if caches.one_sequence {
                        GdnCall::Rows(model::gdn_mixer::GdnStream { state: &caches.seqs[0].gdn_state[l], hist: &caches.seqs[0].gdn_hist[l] })
                    } else {
                        GdnCall::Decode(&streams)
                    };
                    self.layer_gdn_fwd_pre(l, &xn1, pre1, epilogue, bsz, call).0
                }
                LayerType::Full => {
                    let dctx = GqaDecodeCtx { paged: &paged, layer: &caches.gqa_kv[l], cos, sin };
                    let attn = self.layer_gqa_fwd_pre(&format!("blocks.{l}.self_attn"), &xn1, pre1, epilogue, bsz, Some(GqaCached::Decode(&dctx))).0;
                    match (&memory_step, c.layer_has_memory(l)) {
                        (Some((shape, read, push)), true) => {
                            let project = self.memory_project(l, shape, bsz);
                            let pool = caches.memory[l].as_ref().expect("a memory layer has a bank pool");
                            // The memory reads the layer's residual input, like the attention it sits beside.
                            let memory = model::anchor_memory::anchor_memory_decode_fwd(g, &self.memory_ids(), shape, &self.memory_weights_at(l, cos, sin), &project, &res, pool, read, push);
                            let sum = g.storage((bsz * d) as u64);
                            g.submit(&[], &[g.step(ADD2, &[&attn, &memory, &sum], &[bsz * d], bsz * d)]);
                            sum
                        }
                        _ => attn,
                    }
                }
            };

            let ln2 = format!("blocks.{l}.ln2.weight");
            let (xmid, xn2, pre2) = if fused {
                let (sum, xn, act) = self.rms_quant_front(&res, Some(&mixer_out), self.w(&ln2), bsz, FrontUse::Mlp);
                (sum.expect("an add front hands back its sum"), xn, Some(act))
            } else {
                let xmid = g.storage((bsz * d) as u64);
                g.submit(&[], &[g.step(ADD2, &[&res, &mixer_out, &xmid], &[bsz * d], bsz * d)]);
                let xn2 = g.storage((bsz * d) as u64);
                g.submit(&[], &[rms_step(g, &xmid, self.w(&ln2), &xn2, d, bsz)]);
                (xmid, xn2, None)
            };

            let mlp_out = self.ffn_fwd_pre(l, &xn2, pre2, epilogue, bsz);
            if fused && l + 1 < self.shard.end {
                // The next layer's first norm rides on this residual add.
                let (sum, xn, act) = self.rms_quant_front(&xmid, Some(&mlp_out), self.w(&format!("blocks.{}.ln1.weight", l + 1)), bsz, FrontUse::Mixer(types[l + 1]));
                res = sum.expect("an add front hands back its sum");
                front = Some((xn, act));
            } else if fused && self.shard.head {
                // The last layer: the final norm rides on the residual add too (the generated one-row RMSNorm is 130 us a call).
                let (sum, xn, _) = self.rms_quant_front(&xmid, Some(&mlp_out), self.w("norm.weight"), bsz, FrontUse::Head);
                res = sum.expect("an add front hands back its sum");
                final_norm = Some(xn);
            } else {
                let res_next = g.storage((bsz * d) as u64);
                g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &res_next], &[bsz * d], bsz * d)]);
                res = res_next;
            }

            // A hand-off point per layer. On a backend that issues each
            // submission when it is made this is where the card was kept fed
            // while the host built the next layer (without it the whole step
            // was recorded before the first dispatch ran, and removing the
            // perf-number: measured by `qwen35_decode_profile` before and after the M-RoPE flush removal
            // M-RoPE uploads that used to flush here cost 1.45x on the whole
            // perf-number: measured by `qwen35_decode_profile` before and after the M-RoPE flush removal
            // pass: 7.19 -> 4.96 tok/s). On one that holds a pass whole it is
            // a no-op, and the step is issued as one when the pass ends.
            g.flush();
        }

        // Head epilogue (final norm): head stage only - `run_forward`'s own
        // `shard.head` branch, at `bsz` rows. A non-head stage returns its raw
        // residual for the next stage to pick up as `input_override`.
        if self.shard.head {
            final_norm.unwrap_or_else(|| {
                let xn_final = g.storage((bsz * d) as u64);
                g.submit(&[], &[rms_step(g, &res, self.w("norm.weight"), &xn_final, d, bsz)]);
                xn_final
            })
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
    /// * GQA layers: rows `pos_start..pos_start+n` of `gqa_kv`'s K and V planes
    ///   hold this round's QK-normed, RoPE'd K/V, written by
    ///   `model::block::gqa_chunk_step_kv`'s append, and the round's queries
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
    ///
    /// `input` replaces the token gather with the caller's own `[n, d_model]`
    /// input embeddings (`tokens` then only gives the round's length).
    pub(crate) fn run_prefill_chunk(&self, tokens: &[u32], pos_start: u32, caches: &DecodeCaches, input: Option<&[f32]>) -> DeviceBuffer {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let n = tokens.len() as u32;
        assert!(
            self.shard.embed && self.shard.head,
            "qwen35::run_prefill_chunk is whole-model only (this shard has embed={}, head={}) - see this function's own doc",
            self.shard.embed,
            self.shard.head
        );
        let xn_final = self.run_prefill_chunk_stage(tokens, pos_start, caches, input.map(StageRows::Host));
        // Only the LAST row is ever wanted (this round's next-token
        // prediction, or the seam into the next round).
        let last = g.storage(d as u64);
        g.submit(&[], &[g.step(CONCAT_SPLIT, &[&xn_final, &last], &[1, n, 1, n - 1, 1, d], d)]);
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
        input_override: Option<StageRows<'_>>,
    ) -> DeviceBuffer {
        let mut round = self.begin_prefill_chunk(tokens, pos_start, caches, input_override);
        self.advance_prefill_chunk(&mut round, caches, None).expect("an unbounded advance finishes the round")
    }

    /// Start a chunk round: everything [`Self::run_prefill_chunk_stage`] does before its first layer (the
    /// input rows, the round's position tables), left for [`Self::advance_prefill_chunk`] to carry through the
    /// layers a few at a time. A short round that runs as decode steps is finished here.
    pub(crate) fn begin_prefill_chunk(&self, tokens: &[u32], pos_start: u32, caches: &DecodeCaches, input_override: Option<StageRows<'_>>) -> ChunkRound {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let n = tokens.len() as u32;
        assert!(n > 0, "qwen35::run_prefill_chunk_stage: empty chunk (no token to produce a hidden state from)");
        assert!(
            pos_start + n <= caches.gqa_window.capacity(caches.gqa_cap),
            "qwen35::run_prefill_chunk_stage: chunk ends at position {} but the KV cache holds {} rows",
            pos_start + n,
            caches.gqa_window.capacity(caches.gqa_cap)
        );
        // `lora_fwd`'s scratch (`self.lora_a`/`lora_out`) is sized `b*t` rows
        // at construction; a chunk wider than that would write past it.
        assert!(
            self.cfg.lora.is_none() || n <= self.b * self.t,
            "qwen35::run_prefill_chunk_stage: a LoRA build's adapter scratch holds {} rows, chunk is {n}",
            self.b * self.t
        );

        if self.exact_round_ok(n) {
            // The rows of the round are kept, owned: its groups are issued over several calls.
            let input = input_override.map(|x| match x {
                StageRows::Host(rows) => ExactInput::Host(rows.to_vec()),
                device => {
                    let rows = g.storage(device.words() as u64);
                    device.load_into(g, &rows);
                    ExactInput::Device(rows)
                }
            });
            return ChunkRound::Exact(Box::new(ExactRound { tokens: tokens.to_vec(), pos_start, input, done: 0, out: Vec::new() }));
        }

        // `input_override` is the round's input rows: on a non-embed stage the
        // previous stage's residual block, on the embed stage the caller's own
        // input embeddings in place of the token gather.
        let res = match input_override {
            Some(x) => {
                assert_eq!(x.words(), (n * d) as usize, "qwen35: run_prefill_chunk_stage input_override must be the round's whole [n, d_model] block");
                let res = g.storage((n * d) as u64);
                x.load_into(g, &res);
                res
            }
            None => {
                assert!(
                    self.shard.embed,
                    "qwen35: run_prefill_chunk_stage on a NON-EMBED stage needs the previous stage's residual block as `input_override` (this stage holds no `tok.weight`)"
                );
                let tok_buf = g.storage(n as u64);
                g.write(&tok_buf, tokens);
                let res = g.storage((n * d) as u64);
                self.embed_rows(&tok_buf, n, &res);
                res
            }
        };

        // A sliding-window config prefills into a ring window of exactly its rows (the sequence's block is the ring's
        // pool block, and `slot` its bank row); anything else would attend, or fill, a cache that grows.
        let (ring_block, slot) = match (c.swa_window, caches.gqa_window) {
            (Some(win), KvWindow::Ring(block)) => {
                assert_eq!(caches.gqa_cap, win, "qwen35::run_prefill_chunk_stage: the ring is {} rows, the config's window {win}", caches.gqa_cap);
                (block, caches.slot)
            }
            (Some(win), other) => panic!("qwen35::run_prefill_chunk_stage: this config's attention is a {win}-row sliding window and needs a ring KV window (got {other:?})"),
            (None, _) => (0, 0),
        };
        if c.anchor_memory.is_some() {
            assert!(caches.memory.len() == c.n_layers as usize, "qwen35::run_prefill_chunk_stage: this config has an anchor memory and the caches carry no bank pool");
        }
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
        let yarn = c.rope_table();
        let (cos, sin) =
            qwen3vl::mrope::mrope_tables_scaled(&positions, c.mrope_section, c.rotary_dim(), c.rope_theta, yarn.as_ref().map(|(f, a)| (f.as_slice(), *a)));
        let cos = g.storage_init("qwen35.prefill_chunk.cos", &cos);
        let sin = g.storage_init("qwen35.prefill_chunk.sin", &sin);
        let blocks = g.storage(n as u64);
        let offsets = g.storage(n as u64);
        let seq_lens = g.storage(n as u64);
        // Each row's append address and the sequence's block table (one block
        // for every row when the window is a single block).
        // A ring window has no paged rows to address: its chunk attention gathers and writes the ring itself.
        let (row_blocks, row_offsets, row_tables, max_bt) = if caches.gqa_window.is_ring() { (vec![0], vec![0], vec![0], 1) } else { caches.gqa_window.chunk_rows(caches.gqa_cap, pos_start, n) };
        let tables = g.storage(row_tables.len() as u64);
        g.write(&blocks, &row_blocks[..(n as usize).min(row_blocks.len())]);
        g.write(&offsets, &row_offsets[..(n as usize).min(row_offsets.len())]);
        g.write(&tables, &row_tables);
        g.write(&seq_lens, &(0..n).map(|i| pos_start + i + 1).collect::<Vec<u32>>());

        ChunkRound::Layers(Box::new(LayerRound { n, pos_start, ring_block, slot, max_bt, blocks, offsets, tables, seq_lens, cos, sin, stack: self.begin_layer_stack(res, n) }))
    }

    /// Give up a round begun by [`Self::begin_prefill_chunk`] that is not finished: wait for what of it the device
    /// has been given, and hand back the scratch its layers held.
    pub(crate) fn abandon_prefill_chunk(&self, round: ChunkRound) {
        self.gpu.poll_wait();
        if let ChunkRound::Layers(r) = round {
            if r.stack.pooled {
                self.gpu.scratch_release();
            }
        }
    }

    /// Record layers of a round begun by [`Self::begin_prefill_chunk`] until it is complete (`budget` `None`) or
    /// `budget` layers are recorded (always at least one; a short round that runs as decode steps issues one
    /// group). `Some` with the round's `[n, d_model]` block once every layer is recorded; the device may still be
    /// running them.
    ///
    /// `caches` are the ones the round began with. The device time a call puts on the card is its layer count
    /// times a layer's cost, which a caller that also has latency-bound work to issue bounds with `budget`.
    pub(crate) fn advance_prefill_chunk(&self, round: &mut ChunkRound, caches: &DecodeCaches, budget: Option<usize>) -> Option<DeviceBuffer> {
        let r = match round {
            ChunkRound::Exact(r) => return self.advance_exact_round(r, caches, budget),
            ChunkRound::Layers(r) => r,
        };
        let c = &self.cfg;
        let g = &self.gpu;
        let d = c.d_model;
        let (n, pos_start, ring_block, slot, max_bt) = (r.n, r.pos_start, r.ring_block, r.slot, r.max_bt);
        let (blocks, offsets, tables, seq_lens, cos, sin) = (&r.blocks, &r.offsets, &r.tables, &r.seq_lens, &r.cos, &r.sin);
        let mut mixer = |l: usize, ty: LayerType, res_in: &DeviceBuffer, xn1: &DeviceBuffer, pre: Option<Act>, epilogue: bool| -> DeviceBuffer {
            match ty {
                LayerType::Linear => {
                    let cont = model::gdn_mixer::GdnStream { state: &caches.gdn_state[l], hist: &caches.gdn_hist[l] };
                    self.layer_gdn_fwd_pre(l, xn1, pre, epilogue, n, GdnCall::Chunk(cont)).0
                }
                LayerType::Full if c.swa_window.is_some() => {
                    let window = c.swa_window.expect("checked by the guard");
                    let row0 = ring_block * window;
                    let rc = RingChunkCtx { start: pos_start, window, row0, layer: &caches.gqa_kv[l], cos, sin };
                    let attn = self.layer_gqa_fwd_pre(&format!("blocks.{l}.self_attn"), xn1, pre, epilogue, n, Some(GqaCached::Ring(&rc))).0;
                    if !c.layer_has_memory(l) {
                        return attn;
                    }
                    let shape = self.memory_shape_of(1, n);
                    let project = self.memory_project(l, &shape, n);
                    let pool = caches.memory[l].as_ref().expect("a memory layer has a bank pool");
                    let memory = model::anchor_memory::anchor_memory_chunk_fwd(g, &self.memory_ids(), &shape, &self.memory_weights_at(l, cos, sin), &project, res_in, pool, slot, pos_start);
                    let sum = g.storage((n * d) as u64);
                    g.submit(&[], &[g.step(ADD2, &[&attn, &memory, &sum], &[n * d], n * d)]);
                    sum
                }
                LayerType::Full => {
                    let ctx = GqaChunkCtx {
                        start: pos_start,
                        block_size: caches.gqa_cap,
                        max_bt,
                        layer: &caches.gqa_kv[l],
                        blocks: blocks.clone(),
                        offsets: offsets.clone(),
                        tables: tables.clone(),
                        seq_lens: seq_lens.clone(),
                        cos,
                        sin,
                    };
                    self.layer_gqa_fwd_pre(&format!("blocks.{l}.self_attn"), xn1, pre, epilogue, n, Some(GqaCached::Chunk(&ctx))).0
                }
            }
        };
        self.advance_layer_stack(&mut r.stack, n, &format!("pos_start={pos_start}"), &mut mixer, budget)
    }

    /// Whether this build can run [`Self::run_prefill_multi_stage`]: not a LoRA build,
    /// whose adapter scratch is sized for one sequence's round.
    pub(crate) fn multi_prefill_supported(&self) -> bool {
        self.cfg.lora.is_none()
    }

    /// Whether everything this stage submitted in the current lane has finished -
    /// on its own handle and on the one its linears run on - without waiting for it.
    pub(crate) fn lane_idle(&self) -> bool {
        self.gpu.is_idle() && self.ops.gpu().is_idle()
    }

    /// **One forward over several sequences' prefill rounds**: `segs[i].tokens` are
    /// consecutive prompt tokens of sequence `i` starting at `segs[i].pos_start`, the
    /// sequences' rows laid out one after another. Every projection, the MLP and the
    /// norms run once over all the rows - so the weights are read once for the lot and
    /// the GEMMs see the sum of the rows - while each sequence's rows go through the
    /// token mixer on their own: its recurrent state is its row of the pool (gathered
    /// into the working copy and scattered back around the chunk kernels) and its
    /// attention reads and extends its own KV window.
    ///
    /// The contract of [`Self::run_prefill_chunk_stage`] per sequence: the same KV rows
    /// and state afterwards, and the same `[n, d_model]` block (here `n` is the total)
    /// in the same row order. A row's arithmetic does not depend on which other rows
    /// share the GEMM (activations are quantised per row and the tensor-core kernels
    /// accumulate every output the same way at any tile), so a sequence's rows equal
    /// what a round of its own gives, except where a kernel is picked by the number of
    /// rows (a round of at most [`DECODE_REGIME_MAX_ROWS`] rows is a decode replay,
    /// which this function is not for: the caller sends such a round through the
    /// single-sequence path).
    pub(crate) fn run_prefill_multi_stage(&self, segs: &[PrefillSeg<'_>], caches: &PooledCaches<'_>, input_override: Option<StageRows<'_>>) -> DeviceBuffer {
        self.run_mixed_stage(None, None, segs, caches, input_override)
    }

    /// **One forward over decode rows and prompt rows together** - [`Self::run_prefill_multi_stage`] with the
    /// next token of each sequence of `decode` in front of the prompt segments' rows. Every projection, the MLP
    /// and the norms run once over all rows, so the weights are read once for the lot; the token mixers split the
    /// rows: the decode rows take the batched decode kernels (the recurrent step reading each sequence's pool row
    /// in place, paged decode attention) and the segments the chunked ones. A decode row is the row a decode step
    /// computes except where a kernel is picked by the number of rows (the projections see `n` rows, not the
    /// decode batch's).
    ///
    /// With `verify`, the leading rows are a speculative verify step instead: `verify.counts[i]` consecutive tokens of
    /// the sequence in `verify.slots[i]` (`decode`'s sequences are the rows, a sequence's slot repeated and its
    /// positions consecutive), their recurrent states snapshotted between rows for [`Self::commit_verify_stage`].
    pub(crate) fn run_mixed_stage(
        &self,
        decode: Option<(&[u32], &BatchDecodeCaches<'_>)>,
        verify: Option<&verify::VerifyBatch<'_>>,
        segs: &[PrefillSeg<'_>],
        caches: &PooledCaches<'_>,
        input_override: Option<StageRows<'_>>,
    ) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        assert!(!segs.is_empty(), "qwen35::run_prefill_multi_stage: no sequences");
        assert!(c.lora.is_none(), "qwen35::run_prefill_multi_stage: a LoRA build's adapter scratch is sized for one sequence's round");
        assert!(c.anchor_memory.is_none() && c.swa_window.is_none(), "qwen35::run_mixed_stage: neither an anchor memory nor a sliding window is implemented for mixed rows");
        let input_override = input_override.and_then(StageRows::provided);
        let lead = decode.map_or(0, |(tokens, _)| tokens.len() as u32);
        let n: u32 = lead + segs.iter().map(|s| s.tokens.len() as u32).sum::<u32>();
        assert!(segs.iter().all(|s| !s.tokens.is_empty()), "qwen35::run_prefill_multi_stage: a sequence with no rows");
        // The embed stage gathers the round's tokens from its own table unless the caller hands it the rows (the
        // input embeddings of a prompt, say); any other stage takes the previous stage's residual.
        let res = if self.shard.embed && input_override.is_none() {
            let all: Vec<u32> = decode.map_or(&[][..], |(tokens, _)| tokens).iter().copied().chain(segs.iter().flat_map(|s| s.tokens.iter().copied())).collect();
            let tok_buf = g.storage(n as u64);
            g.write(&tok_buf, &all);
            let res = g.storage((n * d) as u64);
            self.embed_rows(&tok_buf, n, &res);
            res
        } else {
            let x = input_override.expect("qwen35: run_prefill_multi_stage on a NON-EMBED stage needs the previous stage's residual block (or hold the table)");
            assert_eq!(x.words(), (n * d) as usize, "qwen35: run_prefill_multi_stage input_override must be the round's whole [n, d_model] block");
            let res = g.storage((n * d) as u64);
            x.load_into(g, &res);
            res
        };

        // The per-sequence tables a chunk round builds once, for each sequence: its
        // absolute M-RoPE positions, the single-block table its KV window is, and
        // its causal lengths.
        let yarn = c.rope_table();
        let mut row0 = lead;
        let mut round = Vec::with_capacity(segs.len());
        // The decode rows' per-step inputs, written before the round's first dispatch like a decode step's.
        let decode_inputs = decode.map(|(tokens, dcaches)| {
            assert_eq!(tokens.len(), dcaches.seqs.len(), "qwen35::run_mixed_stage: {} tokens for {} sequences", tokens.len(), dcaches.seqs.len());
            let longest_blocks = dcaches.seqs.iter().map(|s| s.window.blocks().len()).max().unwrap_or(1).max(1) as u32;
            let inputs = self.alloc_decode_inputs_at(lead, longest_blocks);
            self.write_decode_inputs(&inputs, None, dcaches);
            inputs
        });
        let paged = decode.zip(decode_inputs.as_ref()).map(|((_, dcaches), inputs)| self.paged_decode_batch(dcaches, inputs, None));
        assert!(verify.is_none() || decode.is_some(), "qwen35::run_mixed_stage: a verify step is the round's leading rows");
        assert!(verify.is_none_or(|v| v.total_rows() == lead), "qwen35::run_mixed_stage: the verify step's rows are the round's leading rows");
        // Uploaded with the rest of the round's inputs: a write after the first dispatch would split the pass.
        let verify_layout = verify.map(|v| v.upload(g));
        let words = |w: &[u32]| {
            // The write below fills every word.
            let buf = g.storage_uninit(w.len() as u64);
            g.write(&buf, w);
            buf
        };
        // Every row of the round in order: where its K/V goes and its own position (the decode rows' coordinates are
        // the ones `write_decode_inputs` derives).
        let (mut all_blocks, mut all_offsets, mut all_positions): (Vec<u32>, Vec<u32>, Vec<[u32; 3]>) = (Vec::new(), Vec::new(), Vec::new());
        if let Some((_, dcaches)) = decode {
            for s in dcaches.seqs {
                all_blocks.push(s.window.blocks()[(s.pos / caches.gqa_cap) as usize]);
                all_offsets.push(s.pos % caches.gqa_cap);
                all_positions.push([s.pos; 3]);
            }
        }
        for seg in segs {
            let rows = seg.tokens.len() as u32;
            let slot = words(&[seg.slot]);
            let (row_blocks, row_offsets, row_tables, max_bt) = seg.gqa_window.chunk_rows(caches.gqa_cap, seg.pos_start, rows);
            all_blocks.extend_from_slice(&row_blocks);
            all_offsets.extend_from_slice(&row_offsets);
            all_positions.extend((0..rows).map(|i| [seg.pos_start + i; 3]));
            let tables = words(&row_tables);
            let seq_lens = words(&(0..rows).map(|i| seg.pos_start + i + 1).collect::<Vec<u32>>());
            round.push(SegRound { rows, row0, slot, start: seg.pos_start, max_bt, tables, seq_lens });
            row0 += rows;
        }
        let (cos, sin) = qwen3vl::mrope::mrope_tables_scaled(&all_positions, c.mrope_section, c.rotary_dim(), c.rope_theta, yarn.as_ref().map(|(f, a)| (f.as_slice(), *a)));
        let rows = RoundRows { blocks: words(&all_blocks), offsets: words(&all_offsets), cos: g.storage_init("qwen35.multi.cos", &cos), sin: g.storage_init("qwen35.multi.sin", &sin) };
        let mixed = decode_inputs.as_ref().zip(paged.as_ref()).map(|(inputs, paged)| MixedDecode {
            rows: lead,
            slot_rows: &inputs.slot_rows,
            paged,
            verify: verify.zip(verify_layout.as_ref()).map(|(batch, layout)| MixedVerify { batch, layout }),
        });
        let spans: Vec<_> = round.iter().zip(segs).map(|(r, s)| model::gdn_segments::SegmentSpan { row0: r.row0, rows: r.rows, slot: s.slot }).collect();
        let gdn = (c.layer_types().contains(&LayerType::Linear) && !self.is_train).then(|| model::gdn_segments::GdnSegmentRound::new(g, &spans)).flatten();
        let round = MultiRound { caches, segs: round, rows, gdn, decode: mixed };
        let mut mixer = |l: usize, ty: LayerType, _res: &DeviceBuffer, xn1: &DeviceBuffer, pre: Option<Act>, epilogue: bool| -> DeviceBuffer {
            match ty {
                LayerType::Linear => self.layer_gdn_fwd_pre(l, xn1, pre, epilogue, n, GdnCall::Segments(&round)).0,
                LayerType::Full => self.layer_gqa_fwd_pre(&format!("blocks.{l}.self_attn"), xn1, pre, epilogue, n, Some(GqaCached::Segments(&round, &caches.gqa_kv[l]))).0,
            }
        };
        self.prefill_layer_stack(res, n, &format!("{} sequences", segs.len()), &mut mixer)
    }

    /// The layers of a prefill round, whatever shape its sequences take: this stage's
    /// layers over `res` (`[n, d_model]`), each layer's token mixer supplied by `mixer`
    /// (given the layer index, its type and the normed input), then the head stage's
    /// final norm. The residual, MLP, scratch-arena and drain discipline lives here once
    /// for [`Self::run_prefill_chunk_stage`] (one sequence's rows) and
    /// [`Self::run_prefill_multi_stage`] (several sequences' rows in one forward).
    ///
    /// The loop is resumable: this opens it, [`Self::advance_layer_stack`] records layers from where `StackState`
    /// stands, so a caller with latency-bound work to issue can take a round a few layers at a time.
    fn begin_layer_stack(&self, res: DeviceBuffer, n: u32) -> StackState {
        let g = &self.gpu;
        let pooled = n >= self.chunk_arena_min_rows.get();
        if pooled {
            // The quantised activations and the GEMM outputs are allocated by
            // the `Ops` handle, a `share` of `g` with its own allocator; the
            // arena scope below only covers `g`.
            self.ops.gpu().hold_freed_blocks(true);
        }
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
        let drain_pooled = self.chunk_arena_drain.get().unwrap_or(!g.submissions_ordered());
        let types = self.cfg.layer_types();
        // The residual add, the norm and the activation quantisation between two linears as ONE launch
        // each, bit-identical to the unfused chain (see `run_decode_step`, which shares these gates).
        let fused = self.fused_front_ok(n);
        let epilogue = self.fused_epilogue_ok(n);
        // The next layer's `(xn1, activation)`, when the previous layer produced it.
        let front = (fused && self.shard.start < self.shard.end).then(|| {
            let (_, xn, act) = self.rms_quant_front(&res, None, self.w(&format!("blocks.{}.ln1.weight", self.shard.start)), n, FrontUse::Mixer(types[self.shard.start]));
            (xn, act)
        });
        StackState { res, front, layer: self.shard.start, types, pooled, drain_pooled, fused, epilogue, final_norm: None }
    }

    /// Record layers of the round `st` stands for - one at a time, until the stage's last or until `budget` layers
    /// are recorded (always at least one) - and, on the last, the epilogue. `Some` with the round's block once every
    /// layer is recorded; the device may still be running them.
    fn advance_layer_stack(&self, st: &mut StackState, n: u32, label: &str, mixer: &mut PrefillMixer<'_>, budget: Option<usize>) -> Option<DeviceBuffer> {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let first = st.layer;
        while st.layer < self.shard.end {
            let l = st.layer;
            let ty = st.types[l];

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
            let _scope = st.pooled.then(|| g.scratch_scope());
            // The layer's dispatches are held and issued as one batch when it ends (before the scope above, which
            // drops after it): the backend then uploads their parameters with one copy instead of one in front of
            // every kernel (knowledge 381).
            let _pass = g.pass_scope();

            let (xn1, pre1) = match st.front.take() {
                Some((xn, act)) => (xn, Some(act)),
                None => {
                    let xn1 = g.storage_uninit((n * d) as u64);
                    g.submit(&[], &[rms_step(g, &st.res, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n)]);
                    (xn1, None)
                }
            };

            let mixer_out = mixer(l, ty, &st.res, &xn1, pre1, st.epilogue);

            let ln2 = format!("blocks.{l}.ln2.weight");
            let (xmid, xn2, pre2) = if st.fused {
                let (sum, xn, act) = self.rms_quant_front(&st.res, Some(&mixer_out), self.w(&ln2), n, FrontUse::Mlp);
                (sum.expect("an add front hands back its sum"), xn, Some(act))
            } else {
                let xmid = g.storage_uninit((n * d) as u64);
                g.submit(&[], &[g.step(ADD2, &[&st.res, &mixer_out, &xmid], &[n * d], n * d)]);
                let xn2 = g.storage_uninit((n * d) as u64);
                g.submit(&[], &[rms_step(g, &xmid, self.w(&ln2), &xn2, d, n)]);
                (xmid, xn2, None)
            };

            let mlp_out = self.ffn_fwd_pre(l, &xn2, pre2, st.epilogue, n);
            if st.fused && l + 1 < self.shard.end {
                // The next layer's first norm rides on this residual add.
                let (sum, xn, act) = self.rms_quant_front(&xmid, Some(&mlp_out), self.w(&format!("blocks.{}.ln1.weight", l + 1)), n, FrontUse::Mixer(st.types[l + 1]));
                st.res = sum.expect("an add front hands back its sum");
                st.front = Some((xn, act));
            } else if st.fused && self.shard.head {
                // The last layer: the final norm rides on the residual add too (the generated RMSNorm is 40-130 us a call).
                let (sum, xn, _) = self.rms_quant_front(&xmid, Some(&mlp_out), self.w("norm.weight"), n, FrontUse::Head);
                st.res = sum.expect("an add front hands back its sum");
                st.final_norm = Some(xn);
            } else {
                let res_next = g.storage_uninit((n * d) as u64);
                g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &res_next], &[n * d], n * d)]);
                st.res = res_next;
            }

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
            // perf-number: measured by `tests/gguf_resident_verify_cost_real.rs`
            // behind, a 1-row one has none and spends 60% of the round
            // blocked. That is why it is now gated on `pooled` (M29) rather
            // than paid by every round - it belongs to the POOL, not to the
            // tape. See M6.10's kernel-performance ledger entry for the pool
            // itself and the qwen35 ledger's M29 for the profile that priced
            // its drain.
            //
            // The second disjunct is a different obligation with the same
            // remedy - see `Gpu::reclaim_due`.
            if (st.pooled && st.drain_pooled) || g.reclaim_due() {
                g.poll_wait();
            }
            g.flush();
            // The tap ([`Self::set_hidden_taps`]). Here, not after the loop,
            // because `res` is rebound every iteration and only this point
            // holds layer `l`'s own output. A tapped read is its own drain
            // (`Gpu::read` flushes and blocks), so it is correct on both sides
            // of the `pooled` branch above.
            if !self.taps.borrow().is_empty() && self.taps.borrow().contains(&l) {
                self.tapped.borrow_mut().push((l, g.read(&st.res, (n * d) as usize)));
            }
            if std::env::var_os("QWEN35_DEBUG_LAYER_RMS").is_some() {
                let last_row = g.read(&st.res, (n * d) as usize)[((n - 1) * d) as usize..].to_vec();
                let rms = (last_row.iter().map(|v| v * v).sum::<f32>() / last_row.len() as f32).sqrt();
                eprintln!("qwen35::run_prefill_chunk_stage: {label} n={n} layer {l} ({ty:?}) last-row rms = {rms}");
            }
            st.layer += 1;
            if st.layer < self.shard.end && budget.is_some_and(|b| st.layer - first >= b.max(1)) {
                return None;
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
        if st.pooled {
            g.scratch_release();
        }

        // Head epilogue (final norm): head stage only - `run_decode_step`'s own
        // `shard.head` branch, at `n` rows. A non-head stage hands back its raw
        // residual block for the next stage to pick up as `input_override`.
        if self.shard.head {
            Some(st.final_norm.take().unwrap_or_else(|| {
                let xn_final = g.storage((n * d) as u64);
                g.submit(&[], &[rms_step(g, &st.res, self.w("norm.weight"), &xn_final, d, n)]);
                xn_final
            }))
        } else {
            Some(st.res.clone())
        }
    }

    /// [`Self::begin_layer_stack`] and every layer of it at once.
    fn prefill_layer_stack(&self, res: DeviceBuffer, n: u32, label: &str, mixer: &mut PrefillMixer<'_>) -> DeviceBuffer {
        let mut st = self.begin_layer_stack(res, n);
        self.advance_layer_stack(&mut st, n, label, mixer, None).expect("an unbounded advance finishes the stack")
    }

    /// Rows of one group of an exact round: [`gpu_core::select::exact_decode_group_rows`],
    /// the widest group whose int8 GEMV is the same arithmetic at every row
    /// count. A round of more rows costs one more pass over the weights per
    /// group, and stays bit-identical to the decode steps it stands for
    /// (`tests/chunk_tape_decode_parity.rs`).
    fn exact_group_rows(&self) -> usize {
        gpu_core::select::exact_decode_group_rows(&self.gpu.caps()) as usize
    }

    /// Whether a round of `n` rows runs as plain decode steps
    /// ([`Self::advance_exact_round`]) rather than as a chunk.
    ///
    /// Up to [`DECODE_REGIME_MAX_ROWS`] rows, where the device is offered the
    /// native recurrent kernel the decode tape runs on and
    /// [`Self::set_exact_rounds`] has not withdrawn it. Not under LoRA (its
    /// adapter scratch is sized for the chunk shape) and not while a hidden tap
    /// is armed (the tap reads the residual between layers, which a replayed
    /// pass cannot do) and not for rows that have served-adapter slots (the
    /// bank corrects a forward of exactly the rows its slots were set for, and
    /// this path splits the round into groups). Everything else keeps the chunk tape, which is also
    /// what a device without the native kernels runs, at the tolerance
    /// `tests/decode_step.rs` gates.
    fn exact_round_ok(&self, n: u32) -> bool {
        let c = &self.cfg;
        let nvh = c.linear_num_value_heads;
        let shape = model::gdn_mixer::GdnMixerShape {
            gdn: GdnShape { b: 1, h: nvh, t: n, dk: c.linear_key_head_dim, dv: c.linear_value_head_dim, chunk: 1 },
            nkh: c.linear_num_key_heads,
            conv_kernel: c.linear_conv_kernel_dim,
            rms_eps: c.rms_eps,
        };
        // A short round steps rows one at a time through the decode tape; a windowed, memory-carrying round has its
        // own chunk path (ring and bank), so it never takes this one.
        self.cfg.swa_window.is_none()
            && n <= DECODE_REGIME_MAX_ROWS
            && self.exact_rounds.get()
            && self.decode_fusion.get()
            && c.lora.is_none()
            && self.adapter_rows.get() == 0
            && self.taps.borrow().is_empty()
            && model::gdn_mixer::rows_fused_offered(&self.gpu, &shape)
    }

    /// **A short round as plain decode steps.** `tokens.len()` consecutive
    /// tokens of one sequence, computed by the decode tape's own machinery and
    /// therefore to its own bits: the same fused kernels, the same GEMV, the
    /// same paged attention, and for the recurrent layers one launch that takes
    /// the rows in order from the state the previous row left
    /// ([`GdnCall::Rows`]).
    ///
    /// This is what a speculative verify round has to be. The weights and
    /// activations are int8, so a tape that is merely algebraically equal moves
    /// rounding decisions in late layers and the noise cascades to about a logit
    /// at the head; verifying on that tape accepts and rejects tokens by a
    /// different arithmetic than the plain decode it promises to reproduce
    /// (`tests/chunk_tape_decode_parity.rs`). The chunk tape's attention (an
    /// online-softmax flash kernel) and, past the first row, its chunked-parallel
    /// recurrence are different arithmetic; this path has neither. It is also the
    /// faster one at these row counts - one replayed graph per round, no
    /// per-layer allocation - for the reasons the decode tape is
    /// (`run_decode_batch`).
    ///
    /// The rows go through the batched decode machinery in groups of at most
    /// [`Self::exact_group_rows`]: the widest group whose int8 projections run the
    /// same arithmetic as a single decode step (on a device with int8 tensor cores
    /// the native GEMV switches to a tensor-core block from
    /// `select::I8_GEMV_MMA_MIN_ROWS` rows, which agrees to f32 rounding of
    /// the running sum, not bit for bit); a longer round is several groups, in order. Each group
    /// is a whole pass over every layer, so the state hand-off between them is
    /// the same one a decode step makes.
    ///
    /// Same contract as [`Self::run_prefill_chunk_stage`]: `[n, d_model]`,
    /// final-normed on a head stage, the raw residual block otherwise.
    fn advance_exact_round(&self, r: &mut ExactRound, caches: &DecodeCaches, budget: Option<usize>) -> Option<DeviceBuffer> {
        let d = self.cfg.d_model as usize;
        assert!(caches.gqa_cap > 0, "qwen35::advance_exact_round: a KV block of 0 rows");
        let total = r.tokens.len();
        let group = |start: usize, len: usize| -> DeviceBuffer {
            let seqs: Vec<BatchSeq> = (0..len)
                .map(|i| BatchSeq { window: caches.gqa_window, slot: 0, pos: r.pos_start + (start + i) as u32, gdn_state: caches.gdn_state, gdn_hist: caches.gdn_hist })
                .collect();
            let batch = BatchDecodeCaches { gqa_kv: caches.gqa_kv, gqa_cap: caches.gqa_cap, seqs: &seqs, one_sequence: true, gdn_pool: None, memory: caches.memory };
            let rows = r.input.as_ref().map(|input| match input {
                ExactInput::Host(rows) => StageRows::Host(rows),
                ExactInput::Device(rows) => StageRows::device(&self.gpu, rows, total * d),
            });
            let x = rows.map(|x| x.slice(start * d, (start + len) * d));
            self.run_decode_batch(&r.tokens[start..start + len], &batch, x)
        };
        let step = self.exact_group_rows();
        if total <= step {
            return Some(group(0, total));
        }
        // Each group's output is read before the next group is issued, which is
        // also the drain the scratch arenas require between passes.
        while r.done < total {
            let len = step.min(total - r.done);
            let rows = self.gpu.read(&group(r.done, len), len * d);
            r.out.extend(rows);
            r.done += len;
            if r.done < total && budget.is_some() {
                return None;
            }
        }
        Some(self.gpu.storage_init("qwen35.round.out", &r.out))
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
    // perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
    /// kernel is 22.9 s (68.0 tok/s), against 152.1 s (10.2 tok/s) per
    // perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
    /// token - a ~6.6x win, not a regression, and slightly FASTER than the
    // perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
    /// INT8 two-card stack's own chunked result (64.6-65.4 tok/s). See
    /// `crate::int8_gguf_resident::MAX_PREFILL_TOKENS`'s own doc for the
    /// full before/after.
    pub(crate) fn chunked_prefill_is_profitable(&self) -> bool {
        true
    }

    /// [`Self::run_prefill_chunk_stage`] staged to the HOST - the one call a
    /// multi-stage CHUNKED prefill driver (`crate::int8_gguf_resident`) needs
    /// per stage per round, and the exact counterpart of what
    /// [`Self::run_decode_step`] is for a single-token pass.
    ///
    /// Returns `[n * d_model]` in token order: this stage's boundary residual
    /// block, ready to be handed to the next stage as its `input_override` (or,
    /// on a head stage, already through the final `norm.weight`). The same
    /// "no `lm_head` here, ever" rule [`Self::run_decode_step`]'s doc
    /// explains applies - at the real shape an fp32 head cannot be a device
    /// buffer, so every sharded caller projects with
    /// `crate::stream::head_logits_on` instead.
    ///
    /// One host round trip per stage per ROUND, where the per-token path pays
    /// one per stage per TOKEN: that ratio is the point of the function.
    pub(crate) fn prefill_chunk_stage(&self, tokens: &[u32], pos_start: u32, caches: &DecodeCaches, input_override: Option<&[f32]>) -> Vec<f32> {
        let out = self.run_prefill_chunk_stage(tokens, pos_start, caches, input_override.map(StageRows::Host));
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
        let caches = self.own_caches();
        let mut hidden = None;
        for round in tokens.chunks(max_chunk as usize) {
            hidden = Some(self.run_prefill_chunk(round, pos, &caches, None));
            pos += round.len() as u32;
        }
        self.dec_pos.set(pos);
        let hidden = hidden.expect("prefill_chunked: prompt is non-empty (asserted above)");
        self.gpu.read(&hidden, self.cfg.d_model as usize)
    }

    /// The token-embedding rows of `tokens` (`[n, d_model]`, row-major) - the
    /// same gather every forward starts with, read back so a caller can build
    /// its own input embeddings (replace some rows with image features) for
    /// [`Self::prefill_embeds`].
    pub fn embed_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        assert!(self.shard.embed, "qwen35::embed_tokens needs the stage that holds tok.weight");
        if let Some(&bad) = tokens.iter().find(|&&t| t >= self.cfg.vocab) {
            panic!("qwen35::embed_tokens: token {bad} exceeds vocab {}", self.cfg.vocab);
        }
        let n = tokens.len() as u32;
        let tok_buf = self.gpu.storage(n as u64);
        self.gpu.write(&tok_buf, tokens);
        let out = self.gpu.storage((n * self.cfg.d_model) as u64);
        self.embed_rows(&tok_buf, n, &out);
        self.gpu.read(&out, (n * self.cfg.d_model) as usize)
    }

    /// [`Self::prefill_chunked`] from input EMBEDDINGS (`[n, d_model]`,
    /// row-major) instead of token ids - the reference's `inputs_embeds`. The
    /// rows enter the layer stack as they are, so a vision-language model
    /// passes the token gather ([`Self::embed_tokens`]) with its placeholder
    /// rows replaced by image features. Leaves the decode state exactly as a
    /// token prefill would, so [`Self::step`] continues from it. Returns the
    /// last row's final-norm hidden state.
    pub fn prefill_embeds(&self, embeds: &[f32], max_chunk: u32) -> Vec<f32> {
        assert_eq!(self.b, 1, "qwen35::prefill_embeds requires b==1 (single sequence)");
        assert!(max_chunk > 0, "qwen35::prefill_embeds: max_chunk must be > 0");
        let d = self.cfg.d_model as usize;
        assert!(!embeds.is_empty() && embeds.len().is_multiple_of(d), "qwen35::prefill_embeds: {} values is not a whole number of {d}-wide rows", embeds.len());
        let n = embeds.len() / d;
        let mut pos = self.dec_pos.get();
        assert!(
            pos + n as u32 <= self.dec_cap,
            "qwen35::prefill_embeds: prompt ends at position {} but this instance's decode capacity is {}",
            pos + n as u32,
            self.dec_cap
        );
        let caches = self.own_caches();
        // The ids are unread when the rows are supplied; only the round length is.
        let lengths = vec![0u32; max_chunk as usize];
        let mut hidden = None;
        for rows in embeds.chunks(max_chunk as usize * d) {
            let len = rows.len() / d;
            hidden = Some(self.run_prefill_chunk(&lengths[..len], pos, &caches, Some(rows)));
            pos += len as u32;
        }
        self.dec_pos.set(pos);
        let hidden = hidden.expect("prefill_embeds: the prompt is non-empty (asserted above)");
        self.gpu.read(&hidden, d)
    }

    /// One Gated-DeltaNet layer's recurrent state, `[bh, dk, dv]` flattened -
    /// parity-debugging introspection only, and the buffer
    /// `tests/spec_decode.rs` compares directly rather than through its
    /// effect on a sampled token. Panics at a GQA layer index, where the
    /// buffer is a 1-word dummy and reading it would silently compare
    /// nothing.
    pub fn debug_gdn_state(&self, l: usize) -> Vec<f32> {
        assert_eq!(self.cfg.layer_types()[l], LayerType::Linear, "layer {l} is GQA and has no recurrent state");
        self.gpu.read(&self.decode_state().gdn_state[l], self.cfg.gdn_state_len() as usize)
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

    /// Copy the whole decode state of this instance's sequence - recurrent state, ring, bank, position - so a later
    /// [`Self::decode_restore`] resumes exactly here. Device-side. A plain config has no ring or bank to copy, and
    /// its KV cache grows with the sequence: use [`Self::gdn_snapshot`] and a position rewind there.
    pub fn decode_snapshot(&self) -> DecodeSnapshot {
        assert!(self.cfg.swa_window.is_some(), "qwen35::decode_snapshot: a config without a sliding window keeps a KV cache that grows with the sequence; there is no fixed state to copy");
        let dec = self.decode_state();
        let copy = DecodeSnapshot {
            gdn: self.new_gdn_snapshot(),
            rings: (0..dec.gqa_kv.len()).map(|_| (self.gpu.storage(self.ring_words()), self.gpu.storage(self.ring_words()))).collect(),
            banks: dec.memory.iter().map(|pool| pool.as_ref().map(|pool| model::anchor_memory::AnchorBankPool::like(&self.gpu, pool))).collect(),
            pos: self.dec_pos.get(),
        };
        self.decode_state_xfer(&copy, true);
        copy
    }

    /// Put a [`Self::decode_snapshot`] back, discarding everything the sequence absorbed since.
    pub fn decode_restore(&self, snap: &DecodeSnapshot) {
        self.decode_state_xfer(snap, false);
        self.dec_pos.set(snap.pos);
    }

    /// Copy the decode state into (`save`) or out of (`!save`) `snap`.
    fn decode_state_xfer(&self, snap: &DecodeSnapshot, save: bool) {
        self.gdn_snapshot_xfer(&self.own_caches(), &snap.gdn, save);
        let dec = self.decode_state();
        let mut steps = Vec::new();
        let mut copy = |live: &DeviceBuffer, cold: &DeviceBuffer, words: u64| {
            let (src, dst) = if save { (live, cold) } else { (cold, live) };
            let n = words as u32;
            steps.push(self.gpu.step(REGION_COPY, &[src, dst], &[1, n, n, 0], n));
        };
        for (l, kv) in dec.gqa_kv.iter().enumerate() {
            if self.cfg.layer_types()[l] == LayerType::Full && self.shard.owns(l) {
                copy(kv.k.data(), &snap.rings[l].0, self.ring_words());
                copy(kv.v.data(), &snap.rings[l].1, self.ring_words());
            }
        }
        for (live, cold) in dec.memory.iter().zip(&snap.banks) {
            if let (Some(live), Some(cold)) = (live, cold) {
                for ((live, words), (cold, _)) in live.arrays().into_iter().zip(cold.arrays()) {
                    copy(live, cold, words);
                }
            }
        }
        self.gpu.submit(&[], &steps);
    }

    // ---- slot pools: the fixed state of many sequences, addressed by slot row ------------------------------------

    /// A pool of `slots` empty sequence slots for this (sliding-window) config; see [`SlotPool`]. `tiers` names how the ring,
    /// the anchor banks and the recurrent state are stored (a bare ring [`KvTier`](model::kv_tier::KvTier) takes the default
    /// tiers for the other two).
    pub fn new_slot_pool(&self, slots: u32, tiers: impl Into<SlotTiers>) -> SlotPool {
        assert!(slots > 0, "qwen35::new_slot_pool: no slots");
        assert!(self.shard.is_whole(self.cfg.n_layers as usize), "qwen35::new_slot_pool: a slot pool holds a whole model's state");
        SlotPool::new(&self.gpu, &self.cfg, &self.shard, slots, tiers.into())
    }

    /// Feed `tokens` into slot `slot`, `max_chunk` rows per round, continuing whatever the slot holds. Returns the
    /// last row's final-norm hidden state (`[d_model]`).
    pub fn slot_prefill(&self, pool: &SlotPool, slot: u32, tokens: &[u32], max_chunk: u32) -> Vec<f32> {
        assert!(slot < pool.slots, "qwen35::slot_prefill: slot {slot} of {}", pool.slots);
        assert!(!tokens.is_empty() && max_chunk > 0, "qwen35::slot_prefill: empty prompt or chunk");
        if let Some(&bad) = tokens.iter().find(|&&t| t >= self.cfg.vocab) {
            panic!("qwen35::slot_prefill: token {bad} exceeds vocab {}", self.cfg.vocab);
        }
        let caches = DecodeCaches {
            gqa_kv: &pool.gqa_kv,
            gqa_cap: pool.window,
            gqa_window: KvWindow::Ring(slot),
            gdn_state: &pool.work_state,
            gdn_hist: &pool.work_hist,
            slot,
            memory: &pool.memory,
        };
        self.stage_slot_state(pool, slot, true);
        let mut hidden = None;
        let mut pos = pool.position(slot);
        for round in tokens.chunks(max_chunk as usize) {
            hidden = Some(self.run_prefill_chunk(round, pos, &caches, None));
            pos += round.len() as u32;
        }
        self.stage_slot_state(pool, slot, false);
        pool.pos.borrow_mut()[slot as usize] = pos;
        self.gpu.read(&hidden.expect("non-empty prompt"), self.cfg.d_model as usize)
    }

    /// One decode token for each `(slot, token)` of `items`, all slots in one batch (any positions, in any order):
    /// the final-norm hidden state (`[d_model]`) of each, in order.
    pub fn slot_decode(&self, pool: &SlotPool, items: &[(u32, u32)]) -> Vec<Vec<f32>> {
        let hidden = self.slot_decode_dev(pool, items);
        let d = self.cfg.d_model as usize;
        self.gpu.read(&hidden, items.len() * d).chunks(d).map(<[f32]>::to_vec).collect()
    }

    /// [`Self::slot_decode`] leaving the `[items.len(), d_model]` hidden block on the device, for a head that
    /// consumes it there (argmax, top-k) without the block crossing to the host.
    pub fn slot_decode_dev(&self, pool: &SlotPool, items: &[(u32, u32)]) -> DeviceBuffer {
        assert!(!items.is_empty(), "qwen35::slot_decode: empty batch");
        let mut seen = std::collections::HashSet::new();
        for &(slot, token) in items {
            assert!(slot < pool.slots, "qwen35::slot_decode: slot {slot} of {}", pool.slots);
            assert!(seen.insert(slot), "qwen35::slot_decode: slot {slot} twice in one batch");
            assert!(token < self.cfg.vocab, "qwen35::slot_decode: token {token} exceeds vocab {}", self.cfg.vocab);
        }
        let seqs: Vec<BatchSeq> = items.iter().map(|&(slot, _)| BatchSeq { window: KvWindow::Ring(slot), slot, pos: pool.position(slot), gdn_state: &[], gdn_hist: &[] }).collect();
        let caches = BatchDecodeCaches {
            gqa_kv: &pool.gqa_kv,
            gqa_cap: pool.window,
            seqs: &seqs,
            one_sequence: false,
            gdn_pool: Some(GdnPoolStore { state: &pool.gdn_state, hist: &pool.gdn_hist, bf16: pool.tiers.recurrent == model::kv_tier::KvTier::BF16 }),
            memory: &pool.memory,
        };
        let tokens: Vec<u32> = items.iter().map(|&(_, t)| t).collect();
        let hidden = if self.slot_tapes_usable() { self.replay_slot_decode(pool, &tokens, &caches) } else { self.run_decode_batch(&tokens, &caches, None) };
        for &(slot, _) in items {
            pool.pos.borrow_mut()[slot as usize] += 1;
        }
        hidden
    }

    /// Whether a slot decode step may run as a recording: the dispatches must not depend on anything the host rewrites
    /// per step but the inputs ([`DecodeInputs`]). An adapter bank correcting the step (rows indexed by position within
    /// the dispatch) and the exact-decode regime (its own grouping) are the exceptions.
    fn slot_tapes_usable(&self) -> bool {
        self.slot_decode_tapes.get() && self.adapter_rows.get() == 0 && self.exact_decode.get().is_none()
    }

    /// Choose the pooled Gated-DeltaNet decode kernel: the occupancy-built candidate (default; it agrees with the exact body to
    /// f32 reassociation) or the exact body the WGSL chain is gated against.
    pub fn set_gdn_pool_wide(&self, on: bool) {
        self.gdn_pool_wide.set(on);
    }

    /// Turn the recorded slot decode step on or off (off by default: measured neutral at small batches and behind at 64
    /// through the API, see the roadmap). Off is the reference every recording is gated against:
    /// the same dispatches built and launched the ordinary way.
    pub fn set_slot_decode_tape(&self, on: bool) {
        self.slot_decode_tapes.set(on);
    }

    /// One decode step of `caches`' rows as a replay of the recording for this batch size, made on first use. Returns the
    /// final-norm hidden block, which lives in the recording's own buffer: valid until the next decode step on this pool.
    fn replay_slot_decode(&self, pool: &SlotPool, tokens: &[u32], caches: &BatchDecodeCaches) -> DeviceBuffer {
        let g = &self.gpu;
        let bsz = tokens.len() as u32;
        let fusion = self.decode_fusion.get();
        let (cached, worth_recording) = {
            let mut tapes = pool.tapes.borrow_mut();
            let held = tapes.note(bsz);
            (tapes.take(bsz, fusion), held)
        };
        let step = match cached {
            Some(step) => {
                self.write_decode_inputs(&step.inputs, Some(tokens), caches);
                step
            }
            // A batch size that has not held yet is built the ordinary way (see `RECORD_AFTER_STEPS`).
            None if !worth_recording => {
                return self.run_decode_batch(tokens, caches, None);
            }
            None => {
                let inputs = self.alloc_decode_inputs_at(bsz, 1);
                // Building the step writes the inputs for these rows too (host writes land immediately).
                g.begin_tape();
                let hidden = self.decode_batch_dispatches(tokens, caches, None, None, Some(&inputs));
                let tape = g.end_tape();
                SlotDecodeTape { bsz, fusion, tape, inputs, hidden }
            }
        };
        g.replay_tape(&step.tape);
        let hidden = step.hidden.clone();
        pool.tapes.borrow_mut().put(step);
        // The commits are issued after the replay, in stream order, and read only the pending blocks the replay pushed.
        self.commit_memory_blocks(caches.memory, &self.decode_memory_commits(caches.seqs));
        hidden
    }

    /// Record one decode step over `caches`' rows, and whatever `tail` issues behind it (a head), as a [`Tape`] bound to
    /// the returned [`DecodeInputs`]. The recording is for every step of the same `(rows, `[`Self::paged_step_shape`]`)`:
    /// before each replay the caller rewrites the inputs ([`Self::write_decode_inputs`]), and nothing but those buffers
    /// differs between steps. Nothing has run when this returns (a recording launches nothing): the caller replays it.
    pub(crate) fn record_decode_step<R>(&self, tokens: &[u32], caches: &BatchDecodeCaches, tail: impl FnOnce(&DeviceBuffer) -> R) -> (Tape, DecodeInputs, R) {
        let g = &self.gpu;
        let inputs = self.alloc_decode_inputs_at(tokens.len() as u32, self.paged_step_shape(caches).1);
        g.begin_tape();
        let hidden = self.decode_batch_dispatches(tokens, caches, None, None, Some(&inputs));
        let out = tail(&hidden);
        (g.end_tape(), inputs, out)
    }

    /// Copy slot `slot`'s recurrent state and conv history between its row of the pool (`into_work`: pool to the
    /// work buffers a prefill chunk runs against; otherwise back).
    fn stage_slot_state(&self, pool: &SlotPool, slot: u32, into_work: bool) {
        let (state_len, hist_len) = (self.cfg.gdn_state_len() as u32, self.cfg.gdn_hist_len() as u32);
        let mut steps = Vec::new();
        // A bf16 pool is staged through the row-indexed gather (decode) and scatter (round) of the batched step, for the
        // one row `slot`; an f32 pool is copied word for word.
        let compact = pool.tiers.recurrent == model::kv_tier::KvTier::BF16;
        let row_id = compact.then(|| {
            let ids = self.gpu.storage(1);
            self.gpu.write(&ids, &[slot]);
            ids
        });
        let (gather, scatter) = if compact { (self.kernel_by_name(POOL_ROWS_GATHER2_BF16), self.kernel_by_name(POOL_ROWS_SCATTER2_BF16)) } else { (0, 0) };
        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            if *ty != LayerType::Linear {
                continue;
            }
            if let Some(ids) = &row_id {
                let params = [1, state_len, hist_len];
                let (kernel, bufs) = if into_work {
                    (gather, [&pool.gdn_state[l], &pool.gdn_hist[l], ids, &pool.work_state[l], &pool.work_hist[l]])
                } else {
                    (scatter, [&pool.work_state[l], &pool.work_hist[l], ids, &pool.gdn_state[l], &pool.gdn_hist[l]])
                };
                steps.push(self.gpu.step(kernel, &bufs, &params, state_len + hist_len));
                continue;
            }
            for (pooled, work, len) in [(&pool.gdn_state[l], &pool.work_state[l], state_len), (&pool.gdn_hist[l], &pool.work_hist[l], hist_len)] {
                let row = (u64::from(slot) * u64::from(len), u64::from(len));
                let (bufs, offs) = if into_work { ([pooled, work], [row, (0, u64::from(len))]) } else { ([work, pooled], [(0, u64::from(len)), row]) };
                steps.push(self.gpu.step_sliced(REGION_COPY, &bufs, &offs, &[1, len, len, 0], len));
            }
        }
        self.gpu.submit(&[], &steps);
    }

    /// Make slot `slot` a new sequence: no recurrent state, no ring rows, an empty bank, position 0. Nothing of the
    /// sequence that held it can reach the next.
    pub fn slot_reset(&self, pool: &SlotPool, slot: u32) {
        assert!(slot < pool.slots, "qwen35::slot_reset: slot {slot} of {}", pool.slots);
        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            if *ty == LayerType::Linear {
                let (sl, hl) = (crate::decode_state::state_words(pool.tiers.recurrent, self.cfg.gdn_state_len()), crate::decode_state::state_words(pool.tiers.recurrent, self.cfg.gdn_hist_len()));
                self.gpu.write_at(&pool.gdn_state[l], u64::from(slot) * sl, &vec![0u32; sl as usize]);
                self.gpu.write_at(&pool.gdn_hist[l], u64::from(slot) * hl, &vec![0u32; hl as usize]);
            }
        }
        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            if *ty == LayerType::Full {
                for plane in [&pool.gqa_kv[l].k, &pool.gqa_kv[l].v] {
                    let ring_words = plane.tier().plane_words(u64::from(pool.window), u64::from(self.cfg.kv_dim()), u64::from(self.cfg.head_dim)).data;
                    self.gpu.write_at(plane.data(), u64::from(slot) * ring_words, &vec![0u32; ring_words as usize]);
                }
            }
        }
        for bank in pool.memory.iter().flatten() {
            bank.reset_row(&self.gpu, slot);
        }
        pool.pos.borrow_mut()[slot as usize] = 0;
    }

    /// Copy slot `from`'s whole state over slot `to`, bit for bit: what a snapshot prefix cache stores and restores.
    pub fn slot_copy(&self, pool: &SlotPool, from: u32, to: u32) {
        assert!(from < pool.slots && to < pool.slots && from != to, "qwen35::slot_copy: {from} -> {to} of {} slots", pool.slots);
        let mut steps = Vec::new();
        let ids = model::anchor_memory::AnchorMemoryKernels::resolve(&self.gpu).unwrap_or_else(|e| panic!("qwen35: {e}"));
        let copy_within = self.gpu.kernel_index("copy_within").expect("copy_within is in the qwen35 kernel set");
        let (state_len, hist_len) = (crate::decode_state::state_words(pool.tiers.recurrent, self.cfg.gdn_state_len()) as u32, crate::decode_state::state_words(pool.tiers.recurrent, self.cfg.gdn_hist_len()) as u32);
        for (l, ty) in self.cfg.layer_types().iter().enumerate() {
            match ty {
                LayerType::Linear => {
                    let (sl, hl) = (u64::from(state_len), u64::from(hist_len));
                    steps.push(model::kv_tier::copy_words_within(&self.gpu, copy_within, &pool.gdn_state[l], u64::from(from) * sl, u64::from(to) * sl, sl));
                    steps.push(model::kv_tier::copy_words_within(&self.gpu, copy_within, &pool.gdn_hist[l], u64::from(from) * hl, u64::from(to) * hl, hl));
                }
                LayerType::Full => {
                    for plane in [&pool.gqa_kv[l].k, &pool.gqa_kv[l].v] {
                        steps.extend(plane.copy_rows(&self.gpu, copy_within, u64::from(from) * u64::from(pool.window), u64::from(to) * u64::from(pool.window), u64::from(pool.window), u64::from(self.cfg.kv_dim()), u64::from(self.cfg.head_dim)));
                    }
                }
            }
        }
        for bank in pool.memory.iter().flatten() {
            steps.extend(bank.copy_row(&self.gpu, &ids, from, to));
        }
        self.gpu.submit(&[], &steps);
        let pos = pool.position(from);
        pool.pos.borrow_mut()[to as usize] = pos;
    }

    /// Words of one ring plane: `window` rows of `kv_dim`.
    fn ring_words(&self) -> u64 {
        u64::from(self.kv_block()) * u64::from(self.cfg.kv_dim())
    }

    /// Bytes of persistent per-sequence state this instance holds (recurrent state, ring, bank), summed over the
    /// buffers it allocated, and the identity and size of each: a sequence that grows nothing leaves both unchanged.
    pub fn decode_state_bytes(&self) -> u64 {
        self.decode_state().bytes()
    }

    pub fn decode_state_allocations(&self) -> Vec<(usize, u64)> {
        self.decode_state().allocations()
    }

    /// This instance's own single-sequence caches, as `run_*` wants them -
    /// the same literal `step`/`prefill_chunked` build, factored out because
    /// the speculative loop needs it in three places.
    fn own_caches(&self) -> DecodeCaches<'_> {
        // These are dedicated per-sequence buffers, not a pool window - see
        // `DecodeCaches::gqa_window`.
        let dec = self.decode_state();
        DecodeCaches {
            gqa_kv: &dec.gqa_kv,
            gqa_cap: self.kv_block(),
            gqa_window: self.own_window(),
            gdn_state: &dec.gdn_state,
            gdn_hist: &dec.gdn_hist,
            slot: 0,
            memory: &dec.memory,
        }
    }

    /// Rows of one sequence's attention cache: the window of a sliding-window config (a ring), else the decode
    /// capacity.
    fn kv_block(&self) -> u32 {
        self.cfg.swa_window.unwrap_or(self.dec_cap)
    }

    /// The window the instance's own cache is addressed as: a ring for a sliding-window config.
    fn own_window(&self) -> KvWindow<'static> {
        if self.cfg.swa_window.is_some() {
            KvWindow::Ring(0)
        } else {
            KvWindow::Block(0)
        }
    }

    /// **Speculative decoding**, greedy, against this instance's own
    /// single-sequence decode state. `draft(ctx, want) -> tokens` proposes up
    /// to `want` continuations of `ctx`; `argmax_of` projects one `[d_model]`
    /// final-norm hidden state to its greedy token id (this crate keeps the
    /// vocab head off-device, so the caller owns that matvec - see
    /// [`Self::run_decode_step`]'s doc for why).
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

    /// [`Self::run_decode_step`] for a BATCH: one decode token for each of
    /// `tokens.len()` independent sequences through THIS stage, staged to the
    /// host as the whole `[bsz, d_model]` boundary block.
    ///
    /// One host round trip per stage per STEP, where the per-sequence form
    /// pays one per stage per sequence per step - which is the same ratio, on
    /// the pipeline seam, that [`Self::run_decode_batch`] buys inside a stage.
    pub(crate) fn decode_batch_stage(&self, tokens: &[u32], caches: &BatchDecodeCaches, input_override: Option<&[f32]>) -> Vec<f32> {
        let hidden = self.run_decode_batch(tokens, caches, input_override.map(StageRows::Host));
        self.gpu.read(&hidden, tokens.len() * self.cfg.d_model as usize)
    }

    /// Device-side head epilogue: `logits[vocab] = hidden[d_model] @
    /// head[vocab, d_model]^T` - the SAME `MATMUL` dispatch [`Self::
    /// run_forward`]'s head epilogue and [`Self::logits_all`] already issue,
    /// at `n = 1`. `hidden` must already be through the final `norm.weight`
    /// RMSNorm - exactly what [`Self::run_decode_step`]'s head-stage branch
    /// returns. Unlike [`Self::run_decode_step`]'s doc (which explains why
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
        // The head is the largest weight in the model and its own row-count
        // gate is the same one the decode projections have, so the exact-decode
        // regime projects it in the same groups: a row's logits may not depend
        // on how many other rows share the step.
        if let Some(step) = self.exact_decode_step().filter(|&step| rows > step) {
            let d = self.cfg.d_model as usize;
            let v = self.cfg.vocab as usize;
            let host = self.gpu.read(hidden, rows as usize * d);
            let mut out = Vec::with_capacity(rows as usize * v);
            for start in (0..rows as usize).step_by(step as usize) {
                let end = (start + step as usize).min(rows as usize);
                let x = self.gpu.storage_init("qwen35.exact_decode.head.in", &host[start * d..end * d]);
                let logits = self.head_logits_rows_ungrouped(&x, (end - start) as u32);
                out.extend(self.gpu.read(&logits, (end - start) * v));
            }
            return self.gpu.storage_init("qwen35.exact_decode.head.logits", &out);
        }
        self.head_logits_rows_ungrouped(hidden, rows)
    }

    /// `out = x @ head^T` over `rows` rows: through `Ops` where the head is held in a narrow tier, the fp32 kernel
    /// otherwise.
    fn head_matmul(&self, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32) {
        let g = &self.gpu;
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        match self.weights.get(self.cfg.head_weight()) {
            Some(w) => {
                let mut s = Vec::new();
                let act = self.ops_act(&mut s, x, rows, d);
                self.ops.matmul(&mut s, w, &act, out, 0);
                g.submit(&[], &s);
            }
            None => g.submit(&[], &[g.step(MATMUL, &[x, self.w(self.cfg.head_weight()), out], &[rows, d, v], rows * v)]),
        }
    }

    /// [`Self::head_logits_rows_dev`] for one group: one `MATMUL` over `rows`
    /// rows of `hidden`.
    fn head_logits_rows_ungrouped(&self, hidden: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let v = self.cfg.vocab;
        let logits = g.storage(rows as u64 * v as u64);
        self.head_matmul(hidden, &logits, rows);
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
        let logits = self.head_logits_rows_dev(hidden, rows);
        argmax_rows_of(&self.gpu, &logits, rows, self.cfg.vocab)
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
        let logits = self.head_logits_rows_dev(hidden, rows);
        topk_rows_of(&self.gpu, &logits, rows, self.cfg.vocab, cap)
    }

    // ---- pipeline-parallel cross-stage seam (`model::Shardable`) ----------

    /// Element count of one residual-stream boundary slab (`b*t*d_model`).
    fn res_numel(&self) -> usize {
        (self.b * self.t * self.cfg.d_model) as usize
    }

    /// This stage's OUTPUT residual `res[shard.end]` (input to the next stage).
    pub fn read_out_res(&self) -> Vec<f32> {
        assert!(self.res_saved.is_none(), "qwen35: the residual stream is kept as quantized boundaries in this build; there is no fp32 buffer to read or write");
        self.gpu.read(&self.res.borrow()[self.shard.end], self.res_numel())
    }

    /// This stage's INPUT residual `res[shard.start]` (from the previous stage).
    pub fn write_in_res(&self, data: &[f32]) {
        assert!(self.res_saved.is_none(), "qwen35: the residual stream is kept as quantized boundaries in this build; there is no fp32 buffer to read or write");
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
        self.backward_pending.set(false);
        self.ps.zero_grads(&self.gpu);
        if let Some(ft) = &self.resident {
            self.gpu.flush();
            ft.borrow().zero_grads(&self.ops);
        }
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        // A resident build updates its linears against their host masters through the generic resident step, whose
        // global clip spans them and the device-optimised remainder; that step also updates the remainder.
        if let Some(ft) = &self.resident {
            let streamed = self.layer_streamed();
            assert!(!streamed || self.backward_pending.replace(false), "qwen35 layer-streamed full fine-tune: adamw_step without a backward before it");
            let args = model::resident_ft::StepArgs { step: t, lr, wd, adam, clip, extra_scale };
            let norm = model::resident_ft::adamw_step(self, ft, &self.ops, &args).unwrap_or_else(|e| panic!("qwen35 resident full fine-tune: update: {e}"));
            self.resident_norm.set(norm);
            if streamed {
                // The step consumed the forward's activations (it ran the backward itself).
                *self.train_acts.borrow_mut() = None;
            }
            return;
        }
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, adam, clip, extra_scale);
    }

    /// Whether this build streams its layers' weights through a bounded window of device slots.
    fn layer_streamed(&self) -> bool {
        self.resident.as_ref().is_some_and(|ft| ft.borrow().is_layer_streamed())
    }

    /// Load `layer` into its device slots before the forward or the backward reaches it (a no-op for a build whose
    /// layers are all resident).
    fn stream_load(&self, layer: usize) {
        if let Some(ft) = self.resident.as_ref().filter(|ft| ft.borrow().is_layer_streamed()) {
            self.gpu.flush();
            ft.borrow_mut().load_layer(&self.ops, layer).unwrap_or_else(|e| panic!("qwen35 layer-streamed full fine-tune: loading layer {layer}: {e}"));
        }
    }

    /// Device bytes of the resident linears' weights: every layer's, or the window's slots when layer-streamed
    /// (`None` for a build with no resident state).
    pub fn resident_device_weight_bytes(&self) -> Option<u64> {
        self.resident.as_ref().map(|ft| ft.borrow().device_weight_bytes())
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
            self.ps.trained_params().map(|(n, _)| n.clone()).collect()
        } else {
            let resident = self.resident.iter().flat_map(|ft| ft.borrow().specs().iter().map(|s| s.name.clone()).collect::<Vec<_>>());
            // A training build lists what it trains: a tensor that is frozen by design (the admission scorer) has no gradient.
            self.ps.params.iter().map(|(n, _)| n.clone()).filter(|n| !self.is_train || self.ps.grad.contains_key(n)).chain(resident).collect()
        }
    }

    fn resident_index(&self, name: &str) -> Option<usize> {
        self.resident.as_ref().and_then(|ft| ft.borrow().index_of(name))
    }

    /// The resident tier of linear `name`, `None` for a tensor held outside
    /// the [`Weight`] table (embeddings and norms live in the parameter store).
    pub fn weight_tier(&self, name: &str) -> Option<Dtype> {
        self.weights.get(name).map(Weight::dtype)
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        if let Some(i) = self.resident_index(name) {
            // The master copy: the highest-precision weight, which is what a saved checkpoint should carry.
            self.gpu.flush();
            return self.resident.as_ref().expect("indexed").borrow_mut().read_master(&self.ops, i).unwrap_or_else(|e| panic!("qwen35 resident full fine-tune: reading {name}: {e}"));
        }
        match self.weights.get(name).filter(|_| !self.ps.weight.contains_key(name)) {
            Some(w) => self.ops.read_host(w),
            None => self.ps.read_weight(&self.gpu, name),
        }
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        if let Some(i) = self.resident_index(name) {
            self.gpu.flush();
            self.resident.as_ref().expect("indexed").borrow_mut().write_master(&self.ops, i, data).unwrap_or_else(|e| panic!("qwen35 resident full fine-tune: writing {name}: {e}"));
            return;
        }
        match self.weights.get(name).filter(|_| !self.ps.weight.contains_key(name)) {
            Some(w) => write_weight_host(&self.gpu, w, data),
            None => self.gpu.write(self.w(name), bytemuck::cast_slice(data)),
        }
    }

    /// The accumulated gradient of `name`, as host f32: the resident state's buffer (decoded from bf16 where it is
    /// stored so) for a tensor that state holds, the parameter store's otherwise.
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        if let Some((buf, dtype)) = self.resident.as_ref().and_then(|ft| {
            let ft = ft.borrow();
            ft.index_of(name).map(|i| (ft.grad_buffer(i).clone(), ft.grad_dtype()))
        }) {
            let numel = self.resident.as_ref().expect("checked").borrow().specs().iter().find(|s| s.name == name).map(|s| s.n as usize * s.k as usize).expect("indexed");
            return match dtype {
                model::ops::GradDtype::F32 => self.gpu.read(&buf, numel),
                model::ops::GradDtype::Bf16 => self.gpu.read(&buf, numel / 2).iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xFFFF_0000)]).collect(),
            };
        }
        self.ps.read_grad(&self.gpu, name)
    }

    /// Run the forward graph and return the scalar loss.
    pub fn forward(&self) -> f32 {
        self.run_forward();
        let n = self.b * self.run_len();
        if let Some(head) = &self.head_ce {
            // Recorded on `self.gpu`, not through `HeadCe::forward`: that
            // submits on the `Ops` handle, a separate command stream that is
            // not ordered after this handle's pending layer work.
            let ta = self.train_acts.borrow();
            let xn_final = &ta.as_ref().expect("qwen35: a head_ce build is a training build").xn_final;
            let mut s = Vec::new();
            head.forward_steps_with(&self.ops, &mut s, xn_final, n, &self.targets, &self.ce_buf, self.objectives.argmax_out(), 1.0);
            self.gpu.submit(&[], &s);
            return self.objectives.loss(&self.gpu, &self.gpu.read(&self.ce_buf, n as usize), self.count.get());
        }
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
        assert!(self.res_saved.is_none(), "qwen35: the residual stream is kept as quantized boundaries in this build; there is no fp32 buffer to read or write");
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
        let GdnSaved::Full(ia) = &gdn.saved else { panic!("qwen35: debug_gdn_trace needs the whole-sequence GDN activations (no gdn_segment_tokens)") };
        // The convolution's channel-major copies: kept as they are by the generic forward, transposed here from the
        // token-major tensors the fused forward keeps (a whole sequence, so `ncl_in` has no history rows).
        let to_channel_major = |token_major: Vec<f32>| -> Vec<f32> {
            let mut channel_major = vec![0f32; token_major.len()];
            for row in 0..n {
                for ch in 0..conv_dim {
                    channel_major[ch * n + row] = token_major[row * conv_dim + ch];
                }
            }
            channel_major
        };
        let (ncl_in, ncl_out) = match &ia.conv {
            model::gdn_mixer::ConvActs::ChannelMajor { ncl_in, ncl_out } => (g.read(ncl_in, n * conv_dim), g.read(ncl_out, n * conv_dim)),
            model::gdn_mixer::ConvActs::TokenMajor { x, pre, .. } => (to_channel_major(g.read(x, n * conv_dim)), to_channel_major(g.read(pre, n * conv_dim))),
        };
        vec![
            ("xn1", g.read(&la.xn1, n * d)),
            ("ncl_in", ncl_in),
            ("ncl_out", ncl_out),
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
        // A prefix is enough: the model is causal, so the padded positions
        // cannot reach the logits of the real ones (a rollout asks one prefix
        // after another).
        assert!(
            !tokens.is_empty() && tokens.len() as u32 <= self.t_user,
            "qwen35::logits_all requires 1..={} tokens, got {}",
            self.t_user,
            tokens.len()
        );
        let mut padded = tokens.to_vec();
        padded.resize(self.t as usize, 0);
        self.write_tokens(&padded);
        self.run_forward();
        let real = tokens.len() * self.cfg.vocab as usize;
        let Some(head) = &self.head_ce else {
            return self.gpu.read(&self.logits, (self.t * self.cfg.vocab) as usize)[..real].to_vec();
        };
        // A chunked-head build has no logits buffer: read them chunk by chunk.
        let ta = self.train_acts.borrow();
        let xn_final = &ta.as_ref().expect("qwen35: a head_ce build is a training build").xn_final;
        let vocab = self.cfg.vocab as usize;
        let mut out = Vec::with_capacity(self.t as usize * vocab);
        for (r0, rows) in head.chunks(self.t) {
            let mut s = Vec::new();
            head.logits_steps(&self.ops, &mut s, xn_final, r0, rows, 1.0);
            self.gpu.submit(&[], &s);
            out.extend(self.gpu.read(head.logits(), rows as usize * vocab));
        }
        out.truncate(real);
        out
    }

    /// Rebuild a training model from a file [`Self::save`] wrote (adapters
    /// included), sized for `b` sequences of `t` tokens.
    pub fn load(path: &str, b: u32, t: u32) -> Qwen35 {
        let reader = checkpoint::weightio::WeightReader::open(path).unwrap_or_else(|e| panic!("qwen35: open {path}: {e}"));
        let cfg = Qwen35Config::from_json(&reader.config());
        Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, b, t, &reader, &TrainOpts::default())
    }

    pub fn save(&self, path: &str) {
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            self.ps.params.iter().map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name))).collect();
        // A resident build keeps its linears outside the parameter store: they are saved from their masters.
        if let Some(ft) = &self.resident {
            let names: Vec<(String, u64)> = ft.borrow().specs().iter().map(|s| (s.name.clone(), s.n as u64 * s.k as u64)).collect();
            tensors.extend(names.into_iter().map(|(name, numel)| (name.clone(), vec![numel], self.read_weight(&name))));
        }
        let config = self.cfg.to_json();
        checkpoint::save_carded(path, config, &tensors, &checkpoint::st::ModelCard::new("brain/qwen35", "qwen35"));
    }
}

impl model::resident_ft::ResidentBackward for Qwen35 {
    fn flush_backward(&self) {
        self.gpu.flush();
    }

    fn zero_remainder_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    fn remainder_grad_sum_sq(&self, norm: &optim::gradnorm::DeviceGradNorm) -> f64 {
        let items: Vec<optim::gradnorm::NormItem<'_>> =
            self.ps.opt_params().iter().map(|(name, numel)| optim::gradnorm::NormItem { buf: self.ps.g(name), numel: *numel, dtype: model::ops::GradDtype::F32 }).collect();
        norm.sum_sq(&self.gpu, &items)
    }

    /// The backward over the activations the forward kept (a layer-streamed build leaves them in place for every pass).
    fn backward_pass(&self, on: &mut dyn FnMut(model::resident_ft::LayerEvent) -> std::io::Result<()>) -> std::io::Result<()> {
        let ta = self.train_acts.borrow();
        let ta = ta.as_ref().ok_or_else(|| std::io::Error::other("qwen35: a resident backward pass needs the activations of the forward before it"))?;
        self.backward_over(ta, on)
    }

    fn update_remainder(&self, args: &model::resident_ft::StepArgs, grad_multiplier: f32) {
        self.opt.step(&self.gpu, &self.ps, args.step, args.lr, args.wd, args.adam, None, grad_multiplier);
    }
}

impl model::OffDeviceState for Qwen35 {
    fn commit(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let ft = self.resident.as_ref().ok_or_else(|| std::io::Error::other("qwen35: no resident state to commit"))?;
        let ckpt = optim::checkpoint::CheckpointDir::new(dir)?;
        self.gpu.flush();
        let mut ft = ft.borrow_mut();
        let run = ft.run_state(0, Vec::new());
        if run.step != step {
            return Err(std::io::Error::other(format!("the run has absorbed {step} updates, the resident state {}", run.step)));
        }
        ft.commit(&self.ops, &ckpt, &run).map(drop)
    }

    fn restore(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let ft = self.resident.as_ref().ok_or_else(|| std::io::Error::other("qwen35: no resident state to restore"))?;
        let ckpt = optim::checkpoint::CheckpointDir::new(dir)?;
        self.gpu.flush();
        if ft.borrow_mut().restore_at(&self.ops, &ckpt, step)?.is_some() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("{}: no training state was committed at update {step}", dir.display())))
        }
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
    fn optimizer_state(&mut self) -> Option<model::OptimizerState<'_>> {
        Some(model::OptimizerState { gpu: &self.gpu, store: &mut self.ps, optim: &self.opt })
    }

    fn optimizer_parts(&self) -> Option<model::OptimizerParts<'_>> {
        Some(model::OptimizerParts { gpu: &self.gpu, store: &self.ps, optim: &self.opt })
    }
    /// Keep every layer or recompute it ([`Qwen35::set_recompute`]); residuals below f32 are fixed when the model is
    /// built, so a request for them is answered from the build, never ignored.
    fn set_saved_policy(&mut self, policy: model::saved::SavedPolicy) -> Result<(), String> {
        use model::saved::{SavedFormat, SavedPolicy};
        let built = self.res_saved.as_ref().map_or(SavedFormat::F32, |b| b.format());
        match policy {
            SavedPolicy::Keep | SavedPolicy::Recompute if built == SavedFormat::F32 => {
                self.set_recompute(policy.recomputes());
                Ok(())
            }
            SavedPolicy::Quantized(f) if f == built => Ok(()),
            other => Err(format!("qwen35 was built keeping {built:?} residuals; {other:?} is chosen when the model is built")),
        }
    }

    type Config = Qwen35Config;

    /// The training build (weights trainable, or a frozen base with trainable
    /// adapters when `cfg.lora` is set).
    fn new(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Qwen35::new_train_on(Gpu::new(pipelines()), cfg, b, t, init)
    }
    fn new_inference(cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
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
            model::Batch::Lm { tokens, targets } => {
                Qwen35::set_batch(self, tokens, targets);
                self.reset_loss_weights();
            }
            model::Batch::LmWeighted { tokens, targets, weights } => {
                Qwen35::set_batch(self, tokens, targets);
                self.write_loss_weights(weights);
            }
            _ => panic!("qwen35::Qwen35 only supports Batch::Lm / Batch::LmWeighted"),
        }
    }
    fn token_objectives(&self) -> Option<&dyn model::training::TokenObjectives> {
        self.offers_token_objectives().then_some(self)
    }
    fn token_objectives_mut(&mut self) -> Option<&mut dyn model::training::TokenObjectives> {
        if self.offers_token_objectives() {
            Some(self)
        } else {
            None
        }
    }
    fn forward(&self) -> f32 {
        Qwen35::forward(self)
    }
    fn computed_tokens(&self) -> Option<u64> {
        Some(u64::from(self.b) * u64::from(self.run_len()))
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
    fn last_grad_norm(&self) -> Option<f32> {
        match &self.resident {
            Some(_) => self.resident_norm.get(),
            None => self.opt.last_grad_norm(&self.gpu, &self.ps),
        }
    }
    fn off_device_state(&self) -> Option<&dyn model::OffDeviceState> {
        self.resident.is_some().then_some(self as &dyn model::OffDeviceState)
    }
    fn pin_rounding_seed(&self, seed: u64) {
        // Domain-separated from the batch generator, which the run keys on the same seed.
        self.opt.set_rounding_seed(seed ^ 0x5eed_0000_1234_9876);
    }
    fn poll_wait(&self) {
        Qwen35::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        Qwen35::param_names(self)
    }
    fn frozen_adapter_param_names(&self) -> Vec<String> {
        model::adapter::device::frozen_adapter_params(&self.ps)
    }
    fn adapter_sites(&self) -> Vec<model::adapter::LinearSite> {
        model::adapter::sites::LoraSites::lora_sites(&self.cfg)
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

/// Where a prefill round's layer loop stands between [`Qwen35::advance_layer_stack`] calls: the residual the next
/// layer reads, the fused front it brings, and the round's scratch discipline.
pub(crate) struct StackState {
    res: DeviceBuffer,
    /// The next layer's `(xn1, activation)`, when the previous layer produced it.
    front: Option<(DeviceBuffer, Act)>,
    /// The next layer to record.
    layer: usize,
    types: Vec<LayerType>,
    pooled: bool,
    drain_pooled: bool,
    fused: bool,
    epilogue: bool,
    /// The final norm's output, when the last layer's residual add produced it ([`Qwen35::advance_layer_stack`]).
    final_norm: Option<DeviceBuffer>,
}

/// One sequence's chunk round between [`Qwen35::begin_prefill_chunk`] and the last of its
/// [`Qwen35::advance_prefill_chunk`] calls.
pub(crate) enum ChunkRound {
    /// Layers still to record; the round's tables and where the loop stands.
    Layers(Box<LayerRound>),
    /// A short round, which runs as decode steps a group of rows at a time.
    Exact(Box<ExactRound>),
}

/// A short round on its way through the decode machinery: the rows still to run and what the finished ones gave.
pub(crate) struct ExactRound {
    tokens: Vec<u32>,
    pos_start: u32,
    input: Option<ExactInput>,
    /// Rows run so far.
    done: usize,
    /// Their output rows, `[done, d_model]`.
    out: Vec<f32>,
}

/// A round's input rows, owned so the round can outlive the caller's borrow of them.
enum ExactInput {
    Host(Vec<f32>),
    /// On this stage's own device.
    Device(DeviceBuffer),
}

/// The state of a chunk round that records layer by layer: the position tables every layer shares and the stack.
pub(crate) struct LayerRound {
    n: u32,
    pos_start: u32,
    /// The ring window's block and bank row (`0` for a paged window).
    ring_block: u32,
    slot: u32,
    max_bt: u32,
    blocks: DeviceBuffer,
    offsets: DeviceBuffer,
    tables: DeviceBuffer,
    seq_lens: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    stack: StackState,
}

/// A prefill round's token mixer for one layer: given the layer index, its type,
/// the normed input, the fused epilogue the next GEMM may apply and whether the
/// input is quantised, it returns the mixer output (see `prefill_layer_stack`).
/// A prefill round's token mixer: layer index, its type, the layer's residual input, the normed input, the fused
/// epilogue the next GEMM may apply and whether the input is quantised.
type PrefillMixer<'a> = dyn FnMut(usize, LayerType, &DeviceBuffer, &DeviceBuffer, Option<Act>, bool) -> DeviceBuffer + 'a;

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: a resident full fine-tune accepts the documented resident precisions. The run's `grad`/`master`/`value`
    /// govern the resident linears and tables; the tensors left in the store train f32 with no master, and the tables
    /// are never checked as store tensors.
    #[test]
    fn resident_precision_governs_resident_tensors_and_leaves_the_store_f32() {
        let cfg = Qwen35Config::tiny();
        let shard = Shard::whole(cfg.n_layers as usize);
        for spec in ["weights=bf16,grad=bf16,state=bf16,master=f32", "weights=bf16,state=int8", "weights=bf16"] {
            let run = model::TrainPrecision::parse(spec).unwrap();
            let store = check_resident_store_precision(&cfg, &shard, &run).unwrap_or_else(|e| panic!("{spec}: {e}"));
            assert!(!store.is_empty() && store.iter().all(|p| p.value == paramstore::precision::ValueDtype::F32 && p.master == paramstore::precision::MasterPolicy::None), "{spec}");
        }
    }

    /// One decode step of ONE pipeline stage, driven by hand: the same
    /// `DecodeCaches` [`Qwen35::step`] builds from this instance's own
    /// per-sequence state, but with an explicit `pos` and the cross-stage
    /// `input_override` seam exposed (`step` itself is the whole-shard
    /// convenience wrapper and always passes `None`).
    fn decode_stage(m: &Qwen35, token_id: u32, pos: u32, input: Option<&[f32]>) -> Vec<f32> {
        let caches = m.own_caches();
        let out = m.run_decode_step(token_id, pos, &caches, input.map(StageRows::Host));
        m.gpu.read(&out, m.cfg.d_model as usize)
    }

    /// A chunk round recorded a layer at a time (a host-time budget too small for two layers) is the round recorded
    /// whole: same block, bit for bit, and it really took several calls. The round is pooled, the regime a served
    /// prefill round runs in, so the scratch arena carries across the calls.
    #[test]
    fn a_chunk_round_recorded_layer_by_layer_equals_the_whole_round() {
        let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny_i8() };
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);
        let m = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        m.set_chunk_arena_min_rows(1);
        let tokens: Vec<u32> = (0..12).map(|i| (i * 5 + 3) % cfg.vocab).collect();

        m.reset_decode_cache();
        let want = {
            let caches = m.own_caches();
            let out = m.run_prefill_chunk_stage(&tokens, 0, &caches, None);
            m.gpu.read(&out, tokens.len() * d)
        };

        m.reset_decode_cache();
        let caches = m.own_caches();
        let mut round = m.begin_prefill_chunk(&tokens, 0, &caches, None);
        let mut calls = 0;
        let out = loop {
            calls += 1;
            if let Some(out) = m.advance_prefill_chunk(&mut round, &caches, Some(1)) {
                break out;
            }
            assert!(calls < 100, "the round never finished");
        };
        assert_eq!(calls, cfg.n_layers as usize, "a zero budget records exactly one layer per call");
        assert_eq!(m.gpu.read(&out, tokens.len() * d), want, "recording a round in pieces changed what it computes");
        // The state the round left is the whole round's: one more token decodes the same either way.
        let after = m.gpu.read(&m.run_decode_step(5, tokens.len() as u32, &caches, None), d);
        m.reset_decode_cache();
        let caches = m.own_caches();
        let _ = m.run_prefill_chunk_stage(&tokens, 0, &caches, None);
        assert_eq!(m.gpu.read(&m.run_decode_step(5, tokens.len() as u32, &caches, None), d), after);
    }

    /// A short round that runs as decode steps, group by group, gives the same block when its groups are issued over
    /// several calls (a budget too small for a second group) as when they are issued at once.
    #[test]
    fn a_short_round_issued_a_group_at_a_time_equals_the_whole_round() {
        // Wide recurrent heads: the shape the native recurrent kernel is offered for.
        let cfg = Qwen35Config {
            n_layers: 8,
            d_model: 640,
            intermediate_size: 1280,
            block_size: 64,
            max_position_embeddings: 64,
            head_dim: 256,
            mrope_section: [11, 11, 10],
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            ..Qwen35Config::tiny_i8()
        };
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 11);
        let m = Qwen35::new_i8(cfg.clone(), 1, cfg.block_size, &init);
        let step = m.exact_group_rows();
        let tokens: Vec<u32> = (0..3 * step + 1).map(|i| (i as u32 * 5 + 3) % cfg.vocab).collect();
        if !m.exact_round_ok(tokens.len() as u32) {
            brain_testutil::skip_unavailable("short rounds run as decode steps only where the native recurrent kernel is offered");
            return;
        }

        m.reset_decode_cache();
        let want = {
            let caches = m.own_caches();
            let out = m.run_prefill_chunk_stage(&tokens, 0, &caches, None);
            m.gpu.read(&out, tokens.len() * d)
        };

        m.reset_decode_cache();
        let caches = m.own_caches();
        let mut round = m.begin_prefill_chunk(&tokens, 0, &caches, None);
        let mut calls = 0;
        let out = loop {
            calls += 1;
            if let Some(out) = m.advance_prefill_chunk(&mut round, &caches, Some(1)) {
                break out;
            }
            assert!(calls < 100, "the round never finished");
        };
        assert_eq!(calls, 4, "a zero budget issues one group of {step} rows per call");
        assert_eq!(m.gpu.read(&out, tokens.len() * d), want, "issuing a short round in pieces changed what it computes");
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
        let caches = m.own_caches();
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

    /// The same two stages as [`two_shard_int8_decode_matches_the_whole_shard_model`], with the
    /// boundary residual handed over DEVICE to DEVICE ([`StageRows::Device`]) instead of read to
    /// the host: stage 1 copies it from stage 0's own output buffer. It must still be the
    /// whole-shard model's bits - the hand-off moves bytes and nothing else.
    #[test]
    fn two_shard_int8_decode_with_a_device_boundary_matches_the_whole_shard_model() {
        let cfg = Qwen35Config::tiny_i8();
        let n_layers = cfg.n_layers as usize;
        let cut = 2usize;
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);

        let whole = Qwen35::new_i8(cfg.clone(), 1, t, &init);
        let stage0 = Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: 0, end: cut, embed: true, head: false, gpu_index: Shard::ANY_GPU });
        let stage1 = Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: cut, end: n_layers, embed: false, head: true, gpu_index: Shard::ANY_GPU });
        for m in [&whole, &stage0, &stage1] {
            m.reset_decode_cache();
        }

        let tokens: Vec<u32> = (0..6).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = i as u32;
            let want = whole.step(tok);
            let boundary = stage0.run_decode_step(tok, pos, &stage0.own_caches(), None);
            let rows = StageRows::device(&stage0.gpu, &boundary, d);
            let out = stage1.run_decode_step((tok + 1) % cfg.vocab, pos, &stage1.own_caches(), Some(rows));
            assert_eq!(stage1.gpu.read(&out, d), want, "pos {pos}: a device boundary changed the two-shard decode");
        }
    }

    /// A prefill round across the seam, device to device, against the same round with the
    /// boundary read to the host: a short one (the exact-round path, which slices the boundary into
    /// groups) and one past a GEMM tile (the chunk path). Two instance pairs, one per boundary
    /// kind, so neither inherits the other's recurrent state.
    #[test]
    fn two_shard_int8_prefill_round_with_a_device_boundary_matches_the_host_boundary() {
        let cfg = Qwen35Config { block_size: 160, max_position_embeddings: 160, ..Qwen35Config::tiny_i8() };
        let n_layers = cfg.n_layers as usize;
        let cut = 2usize;
        let t = cfg.block_size;
        let d = cfg.d_model as usize;
        let init = crate::init::init_weights(&cfg, 7);
        let pair = || {
            let first = Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: 0, end: cut, embed: true, head: false, gpu_index: Shard::ANY_GPU });
            let second = Qwen35::new_i8_shard(cfg.clone(), 1, t, &init, Shard { start: cut, end: n_layers, embed: false, head: true, gpu_index: Shard::ANY_GPU });
            (first, second)
        };
        for rows in [3usize, 140] {
            let prompt: Vec<u32> = (0..rows as u32).map(|i| (i * 5 + 3) % cfg.vocab).collect();
            let (host0, host1) = pair();
            let boundary = prefill_stage(&host0, &prompt, 0, None);
            let want = prefill_stage(&host1, &prompt, 0, Some(&boundary));

            let (dev0, dev1) = pair();
            let out0 = dev0.run_prefill_chunk_stage(&prompt, 0, &dev0.own_caches(), None);
            let rows_in = StageRows::device(&dev0.gpu, &out0, rows * d);
            let out1 = dev1.run_prefill_chunk_stage(&prompt, 0, &dev1.own_caches(), Some(rows_in));
            assert_eq!(dev1.gpu.read(&out1, rows * d), want, "{rows}-row round: a device boundary changed the two-shard prefill");
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

        // The portable paths agree to 9.4e-4 here (measured with
        // `BRAIN_NO_PROVIDER=cuda`, against a bar of 1e-3). On a device with
        // tensor cores the chunk round's int8 GEMMs run on int8 MMAs and its
        // attention on fp16 MMAs (`gpu-core/tests/cuda_flash_prefill.rs` states
        // that kernel's own bar), and the same comparison measures 1.08e-3 - the
        // fp16 rounding of attention, which the token-by-token loop does not
        // pay. That tier's bar is therefore 2e-3, still tight enough to catch a
        // wrong head, mask or scale (those are O(1) errors).
        let bar = if gpu_core::provider::cuda::tensor_core_kernels_enabled(&whole.gpu) { 2e-3 } else { 1e-3 };
        let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let mut worst = maxabs(&got_last, &want_last);
        assert!(worst < bar, "chunked int8 prefill at real qwen38_27b dims diverged at the prompt's last token: maxabs={worst}");
        for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
            let err = maxabs(got, want);
            worst = worst.max(err);
            assert!(err < bar, "decode step {i} after real-dims chunked prefill diverged: maxabs={err}");
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

    /// Two sequences' prefill rounds in ONE forward ([`Qwen35::run_prefill_multi_stage`])
    /// leave each sequence's KV rows and recurrent state exactly where its own rounds
    /// do: the second multi-sequence round of each continues from the first's pool rows
    /// and lands on the hidden state a one-sequence replay of the same rounds gives.
    ///
    /// Two rounds per sequence with ragged, unequal lengths (8+6 and 5+4 rows, so the
    /// round shapes differ and a sequence's rows sit at a different offset in each) on
    /// the eight-layer `tiny()`, which has a GQA layer at index 3 and 7. A wrong slot,
    /// a state row not scattered back, a position taken from the round and not from
    /// the sequence, or rows of one sequence attending the other's window all move the
    /// second round's hidden states, which the first round's output alone would hide.
    #[test]
    fn a_multi_sequence_prefill_round_matches_each_sequence_alone() {
        use std::collections::HashMap;
        let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny() };
        let d = cfg.d_model as usize;
        let mut init: HashMap<String, Vec<f32>> = crate::init::init_weights(&cfg, 7);
        // A decay that keeps the recurrent state alive from one token to the next, as a
        // trained checkpoint's does; at the fresh-weight init it is gone in two tokens
        // and a dropped state would go unnoticed.
        for (name, numel) in cfg.param_list() {
            if name.ends_with(".A_log") {
                init.insert(name, vec![0.05f32.ln(); numel]);
            } else if name.ends_with(".dt_bias") {
                init.insert(name, vec![-1.0f32; numel]);
            }
        }
        let a: Vec<u32> = (0..14).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let b: Vec<u32> = (0..9).map(|i| (i * 7 + 1) % cfg.vocab).collect();

        // Each sequence alone, its two rounds, on its own instance.
        let alone = |prompt: &[u32], first: usize| -> (Vec<f32>, Vec<f32>) {
            let m = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, cfg.block_size, &init);
            m.set_chunk_arena_min_rows(u32::MAX);
            m.reset_decode_cache();
            let one = m.prefill_chunked(&prompt[..first], first as u32);
            let two = m.prefill_chunked(&prompt[first..], (prompt.len() - first) as u32);
            (one, two)
        };
        let (a1, a2) = alone(&a, 8);
        let (b1, b2) = alone(&b, 5);

        // The same rounds, the two sequences together, on pooled caches.
        let m = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, cfg.block_size, &init);
        let gpu = &m.gpu;
        m.set_chunk_arena_min_rows(u32::MAX);
        let (cap, slots) = (cfg.block_size, 2u32);
        let (kv_dim, hd) = (cfg.kv_dim() as u64, cfg.head_dim as u64);
        let types = cfg.layer_types();
        let zeros = |n: u64| gpu.storage_init("test.zero", &vec![0f32; n as usize]);
        let (sl, hl) = (cfg.gdn_state_len(), cfg.gdn_hist_len());
        let full = |l: usize| types[l] == LayerType::Full;
        let gqa_kv: Vec<KvLayer> = (0..types.len()).map(|l| if full(l) { KvLayer::new(gpu, model::kv_tier::KvTier::F32, slots as u64 * cap as u64, kv_dim, hd) } else { KvLayer::placeholder(gpu) }).collect();
        let per = |l: usize, n: u64| if full(l) { zeros(1) } else { zeros(n) };
        let pool_state: Vec<DeviceBuffer> = (0..types.len()).map(|l| per(l, slots as u64 * sl)).collect();
        let pool_hist: Vec<DeviceBuffer> = (0..types.len()).map(|l| per(l, slots as u64 * hl)).collect();
        let work_state: Vec<DeviceBuffer> = (0..types.len()).map(|l| per(l, sl)).collect();
        let work_hist: Vec<DeviceBuffer> = (0..types.len()).map(|l| per(l, hl)).collect();
        let pooled = PooledCaches {
            gqa_kv: &gqa_kv,
            gqa_cap: cap,
            gdn_state: &pool_state,
            gdn_hist: &pool_hist,
            work_state: &work_state,
            work_hist: &work_hist,
            gather: gpu.kernel_index("pool_rows_gather2").expect("in the qwen35 kernel set"),
            scatter: gpu.kernel_index("pool_rows_scatter2").expect("in the qwen35 kernel set"),
        };
        let round = |segs: &[PrefillSeg<'_>]| -> Vec<f32> {
            let out = m.run_prefill_multi_stage(segs, &pooled, None);
            let n: usize = segs.iter().map(|s| s.tokens.len()).sum();
            gpu.read(&out, n * d)
        };
        let seg = |tokens: &'static [u32], pos_start: u32, slot: u32| PrefillSeg { tokens, pos_start, gqa_window: KvWindow::Block(slot), slot };
        let (a, b): (&'static [u32], &'static [u32]) = (Box::leak(a.into_boxed_slice()), Box::leak(b.into_boxed_slice()));
        let r1 = round(&[seg(&a[..8], 0, 0), seg(&b[..5], 0, 1)]);
        let r2 = round(&[seg(&a[8..], 8, 0), seg(&b[5..], 5, 1)]);

        let last_of = |block: &[f32], row: usize| block[row * d..(row + 1) * d].to_vec();
        let maxabs = |x: &[f32], y: &[f32]| x.iter().zip(y).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        for (what, got, want) in [("round 1 of A", last_of(&r1, 7), &a1), ("round 1 of B", last_of(&r1, 12), &b1), ("round 2 of A", last_of(&r2, 5), &a2), ("round 2 of B", last_of(&r2, 9), &b2)] {
            let err = maxabs(&got, want);
            assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite hidden state");
            assert!(err < 1e-5, "{what}: the multi-sequence round differs from the sequence alone by {err:e}");
        }
    }

    /// A sequence's rows are addressed the same way whether one block holds it or
    /// several do: row `p` lives in block `table[p / block_size]` at offset
    /// `p % block_size`, and every row sees the whole table.
    #[test]
    fn a_kv_window_addresses_rows_through_its_blocks() {
        let (blocks, offsets, tables, max_bt) = KvWindow::Block(3).chunk_rows(8, 2, 4);
        assert_eq!((blocks, offsets, tables, max_bt), (vec![3; 4], vec![2, 3, 4, 5], vec![3; 4], 1), "one block: the layout every engine had before paging");
        let table = [7u32, 2, 9];
        let (blocks, offsets, tables, max_bt) = KvWindow::Table(&table).chunk_rows(4, 2, 7);
        assert_eq!(blocks, vec![7, 7, 2, 2, 2, 2, 9], "rows 2..9 cross from block 7 into 2 and 9");
        assert_eq!(offsets, vec![2, 3, 0, 1, 2, 3, 0]);
        assert_eq!(max_bt, 3);
        assert_eq!(tables, [7u32, 2, 9].repeat(7), "every row carries the whole table");
        assert_eq!(KvWindow::Table(&table).capacity(4), 12);
    }

    #[test]
    #[should_panic(expected = "past the")]
    fn rows_past_a_windows_blocks_are_refused() {
        KvWindow::Table(&[1, 2]).chunk_rows(4, 6, 3);
    }
}
