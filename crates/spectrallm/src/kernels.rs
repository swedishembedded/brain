// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The kernel pipeline list and the slot of every kernel the graphs use.

use model::block::{FlashGqaIds, GqaDecodeIds, KernelIds};

pub(crate) const EMBED: usize = 0;
pub(crate) const MATMUL: usize = 1;
pub(crate) const MATMUL_REG3: usize = 2;
pub(crate) const MATMUL_DX: usize = 3;
pub(crate) const MATMUL_DX_REG: usize = 4;
pub(crate) const MATMUL_DW: usize = 5;
pub(crate) const MATMUL_DW_REG: usize = 6;
const RMSNORM: usize = 7;
const RMSNORM_ROWS: usize = 8;
const RMS_INV: usize = 9;
const RMSNORM_DX: usize = 10;
const RMSNORM_DX_ROWS: usize = 11;
const RMSNORM_DW: usize = 12;
const ROPE: usize = 13;
const ROPE_BWD: usize = 14;
pub(crate) const ROPE_AT: usize = 15;
const GQA_SCORES: usize = 16;
const ATTN_SOFTMAX: usize = 17;
const GQA_APPLY: usize = 18;
const GQA_DSCORES: usize = 19;
const GQA_DV: usize = 20;
const GQA_DQ: usize = 21;
const GQA_DK: usize = 22;
const SILU_MUL: usize = 23;
const SILU_DA: usize = 24;
const SILU_DB: usize = 25;
pub(crate) const ADD2: usize = 26;
pub(crate) const CE_VALUE: usize = 27;
pub(crate) const EMB_BWD: usize = 28;
pub(crate) const ADAMW: usize = 29;
pub(crate) const GRADNORM_SQ: usize = 30;
pub(crate) const GRAD_SCALE: usize = 31;
pub(crate) const CLIP_COEF: usize = 32;
pub(crate) const GRAD_SCALE_BUF: usize = 33;
const KV_APPEND: usize = 36;
const ATTN_DECODE_SCORES: usize = 37;
const DECODE_SOFTMAX: usize = 38;
const ATTN_DECODE_APPLY: usize = 39;
pub(crate) const CHAN_EXPAND: usize = 40;
pub(crate) const CHAN_CONTRACT: usize = 41;
pub(crate) const CHAN_GRAD: usize = 42;
pub(crate) const RDFT_SYNTH: usize = 43;
pub(crate) const RDFT_ANALYSIS: usize = 44;
pub(crate) const CE_STATS: usize = 45;
pub(crate) const CE_GRAD_STATS: usize = 46;
pub(crate) const CE_VALUE_STATS_ROWS: usize = 47;
const FLASH_GQA_FWD: usize = 48;
const FLASH_GQA_BWD_DQ: usize = 49;
const FLASH_GQA_BWD_DKV: usize = 50;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("rmsnorm", kernels::RMSNORM),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dx_rows", kernels::RMSNORM_DX_ROWS),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("rope_at", kernels::ROPE_AT),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("add2", kernels::ADD2),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    // Cooperative gradient norm; `optim::Optim` resolves these by name.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
    ("kv_append", kernels::KV_APPEND),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("chan_scale_expand", kernels::CHAN_SCALE_EXPAND),
    ("chan_scale_contract", kernels::CHAN_SCALE_CONTRACT),
    ("chan_scale_grad", kernels::CHAN_SCALE_GRAD),
    ("rdft_synth", kernels::RDFT_SYNTH),
    ("rdft_analysis", kernels::RDFT_ANALYSIS),
    // Two-pass cross-entropy: per-row softmax statistics once, then an
    // elementwise gradient that reads them; the row-cooperative variant
    // computes the loss and the statistics together.
    ("ce_stats", kernels::CE_STATS),
    ("ce_grad_stats", kernels::CE_GRAD_STATS),
    ("ce_value_stats_rows", kernels::CE_VALUE_STATS_ROWS),
    // Causal flash attention and its backward, which never materialize the
    // `[b, heads, t, t]` score matrix.
    ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA),
    ("flash_attn_causal_gqa_bwd_dq", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DQ),
    ("flash_attn_causal_gqa_bwd_dkv", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DKV),
    // Narrow boundary residuals (`model::saved::Residuals`), resolved by name.
    ("pack_bf16", kernels::PACK_BF16),
    ("unpack_bf16", kernels::UNPACK_BF16),
    ("pack_fp8_rows", kernels::PACK_FP8_ROWS),
    ("unpack_fp8_rows", kernels::UNPACK_FP8_ROWS),
];

pub(crate) fn block_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
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
        rmsnorm_dx_rows: RMSNORM_DX_ROWS,
    }
}

/// The flash attention kernels, where this device and shape take them
/// (`FlashGqaIds::for_training`: workgroup barriers, enough workgroup memory,
/// and a sequence long enough to beat the materialized path).
pub(crate) fn flash_ids(gpu: &gpu_core::Gpu, head_dim: u32, t: u32) -> Option<FlashGqaIds> {
    FlashGqaIds { fwd: FLASH_GQA_FWD, bwd_dq: FLASH_GQA_BWD_DQ, bwd_dkv: FLASH_GQA_BWD_DKV, native: false }.for_training(gpu, head_dim, t)
}

pub(crate) fn decode_ids() -> GqaDecodeIds {
    GqaDecodeIds { kv_append: KV_APPEND, attn_decode_scores: ATTN_DECODE_SCORES, decode_softmax: DECODE_SOFTMAX, attn_decode_apply: ATTN_DECODE_APPLY }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slots index `PIPELINES` positionally; a kernel inserted without
    /// renumbering would dispatch the wrong WGSL. Pin every slot by name.
    #[test]
    fn slots_match_the_pipeline_list() {
        for (slot, want) in [
            (EMBED, "embed"), (MATMUL, "matmul"), (MATMUL_REG3, "matmul_reg3"), (MATMUL_DX, "matmul_dx"),
            (MATMUL_DX_REG, "matmul_dx_reg"), (MATMUL_DW, "matmul_dw"), (MATMUL_DW_REG, "matmul_dw_reg"),
            (RMSNORM, "rmsnorm"), (RMSNORM_ROWS, "rmsnorm_rows"), (RMS_INV, "rms_inv"), (RMSNORM_DX, "rmsnorm_dx"),
            (RMSNORM_DX_ROWS, "rmsnorm_dx_rows"), (RMSNORM_DW, "rmsnorm_dw"), (ROPE, "rope_base"),
            (ROPE_BWD, "rope_base_bwd"), (ROPE_AT, "rope_at"), (GQA_SCORES, "gqa_scores"),
            (ATTN_SOFTMAX, "attn_softmax"), (GQA_APPLY, "gqa_apply"), (GQA_DSCORES, "gqa_bwd_dscores"),
            (GQA_DV, "gqa_bwd_dv"), (GQA_DQ, "gqa_bwd_dq"), (GQA_DK, "gqa_bwd_dk"), (SILU_MUL, "silu_mul"),
            (SILU_DA, "silu_bwd_da"), (SILU_DB, "silu_bwd_db"), (ADD2, "add2"), (CE_VALUE, "ce_value"),
            (EMB_BWD, "emb_bwd"), (ADAMW, "adamw"), (GRADNORM_SQ, "gradnorm_sq"),
            (GRAD_SCALE, "grad_scale"), (CLIP_COEF, "clip_coef"), (GRAD_SCALE_BUF, "grad_scale_buf"),
            (KV_APPEND, "kv_append"), (ATTN_DECODE_SCORES, "attn_decode_scores"),
            (DECODE_SOFTMAX, "decode_softmax"), (ATTN_DECODE_APPLY, "attn_decode_apply"),
            (CHAN_EXPAND, "chan_scale_expand"), (CHAN_CONTRACT, "chan_scale_contract"),
            (CHAN_GRAD, "chan_scale_grad"), (RDFT_SYNTH, "rdft_synth"), (RDFT_ANALYSIS, "rdft_analysis"),
            (CE_STATS, "ce_stats"), (CE_GRAD_STATS, "ce_grad_stats"), (CE_VALUE_STATS_ROWS, "ce_value_stats_rows"),
            (FLASH_GQA_FWD, "flash_attn_causal_gqa"), (FLASH_GQA_BWD_DQ, "flash_attn_causal_gqa_bwd_dq"),
            (FLASH_GQA_BWD_DKV, "flash_attn_causal_gqa_bwd_dkv"),
        ] {
            assert_eq!(PIPELINES[slot].0, want, "slot {slot} is not {want}");
        }
    }

    #[test]
    fn every_registered_kernel_has_a_cost_formula() {
        for (name, _) in PIPELINES {
            assert!(gpu_core::cost::covers(name), "kernel {name} has no cost formula");
        }
    }
}
