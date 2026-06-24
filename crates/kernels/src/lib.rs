// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Raw WGSL compute kernels — the single source of truth for brain's GPU
//! engine. fp32-only, core-compute-only (single bind group, <=4 storage
//! buffers/kernel, `@workgroup_size(64)`, no atomics/subgroups/f16) so the
//! same text runs on old desktop GPUs and on WebGPU in the browser.
//!
//! Each `.wgsl` file under `wgsl/` is embedded as a `pub const` (UPPER_SNAKE of
//! the file stem) and indexed by name via [`src`]. This crate is pure string
//! data: no GPU, no std beyond core, safe for every target including wasm32.

/// `wgsl/adamw.wgsl`
pub const ADAMW: &str = include_str!("../wgsl/adamw.wgsl");
/// `wgsl/add2.wgsl`
pub const ADD2: &str = include_str!("../wgsl/add2.wgsl");
/// `wgsl/add.wgsl`
pub const ADD: &str = include_str!("../wgsl/add.wgsl");
/// `wgsl/attention.wgsl`
pub const ATTENTION: &str = include_str!("../wgsl/attention.wgsl");
/// `wgsl/attn_apply.wgsl`
pub const ATTN_APPLY: &str = include_str!("../wgsl/attn_apply.wgsl");
/// `wgsl/attn_apply_bidir.wgsl`
pub const ATTN_APPLY_BIDIR: &str = include_str!("../wgsl/attn_apply_bidir.wgsl");
/// `wgsl/attn_bwd_dk.wgsl`
pub const ATTN_BWD_DK: &str = include_str!("../wgsl/attn_bwd_dk.wgsl");
/// `wgsl/attn_bwd_dk_bidir.wgsl`
pub const ATTN_BWD_DK_BIDIR: &str = include_str!("../wgsl/attn_bwd_dk_bidir.wgsl");
/// `wgsl/attn_bwd_dq.wgsl`
pub const ATTN_BWD_DQ: &str = include_str!("../wgsl/attn_bwd_dq.wgsl");
/// `wgsl/attn_bwd_dq_bidir.wgsl`
pub const ATTN_BWD_DQ_BIDIR: &str = include_str!("../wgsl/attn_bwd_dq_bidir.wgsl");
/// `wgsl/attn_bwd_dscores.wgsl`
pub const ATTN_BWD_DSCORES: &str = include_str!("../wgsl/attn_bwd_dscores.wgsl");
/// `wgsl/attn_bwd_dscores_bidir.wgsl`
pub const ATTN_BWD_DSCORES_BIDIR: &str = include_str!("../wgsl/attn_bwd_dscores_bidir.wgsl");
/// `wgsl/attn_bwd_dv.wgsl`
pub const ATTN_BWD_DV: &str = include_str!("../wgsl/attn_bwd_dv.wgsl");
/// `wgsl/attn_bwd_dv_bidir.wgsl`
pub const ATTN_BWD_DV_BIDIR: &str = include_str!("../wgsl/attn_bwd_dv_bidir.wgsl");
/// `wgsl/attn_scores_masked.wgsl`
pub const ATTN_SCORES_MASKED: &str = include_str!("../wgsl/attn_scores_masked.wgsl");
/// `wgsl/attn_scores.wgsl`
pub const ATTN_SCORES: &str = include_str!("../wgsl/attn_scores.wgsl");
/// `wgsl/attn_scores_bidir.wgsl`
pub const ATTN_SCORES_BIDIR: &str = include_str!("../wgsl/attn_scores_bidir.wgsl");
/// `wgsl/attn_softmax_masked.wgsl`
pub const ATTN_SOFTMAX_MASKED: &str = include_str!("../wgsl/attn_softmax_masked.wgsl");
/// `wgsl/attn_softmax.wgsl`
pub const ATTN_SOFTMAX: &str = include_str!("../wgsl/attn_softmax.wgsl");
/// `wgsl/attn_softmax_bidir.wgsl`
pub const ATTN_SOFTMAX_BIDIR: &str = include_str!("../wgsl/attn_softmax_bidir.wgsl");
/// `wgsl/bias_add.wgsl`
pub const BIAS_ADD: &str = include_str!("../wgsl/bias_add.wgsl");
/// `wgsl/bias_grad.wgsl`
pub const BIAS_GRAD: &str = include_str!("../wgsl/bias_grad.wgsl");
/// `wgsl/ce_grad_masked.wgsl`
pub const CE_GRAD_MASKED: &str = include_str!("../wgsl/ce_grad_masked.wgsl");
/// `wgsl/ce_grad.wgsl`
pub const CE_GRAD: &str = include_str!("../wgsl/ce_grad.wgsl");
/// `wgsl/ce_value_masked.wgsl`
pub const CE_VALUE_MASKED: &str = include_str!("../wgsl/ce_value_masked.wgsl");
/// `wgsl/ce_value.wgsl`
pub const CE_VALUE: &str = include_str!("../wgsl/ce_value.wgsl");
/// `wgsl/clip_coef.wgsl`
pub const CLIP_COEF: &str = include_str!("../wgsl/clip_coef.wgsl");
/// `wgsl/emb_bwd.wgsl`
pub const EMB_BWD: &str = include_str!("../wgsl/emb_bwd.wgsl");
/// `wgsl/embed.wgsl`
pub const EMBED: &str = include_str!("../wgsl/embed.wgsl");
/// `wgsl/expert_counts.wgsl`
pub const EXPERT_COUNTS: &str = include_str!("../wgsl/expert_counts.wgsl");
/// `wgsl/gelu_bwd.wgsl`
pub const GELU_BWD: &str = include_str!("../wgsl/gelu_bwd.wgsl");
/// `wgsl/gelu.wgsl`
pub const GELU: &str = include_str!("../wgsl/gelu.wgsl");
/// `wgsl/gradnorm_sq.wgsl`
pub const GRADNORM_SQ: &str = include_str!("../wgsl/gradnorm_sq.wgsl");
/// `wgsl/grad_scale_buf.wgsl`
pub const GRAD_SCALE_BUF: &str = include_str!("../wgsl/grad_scale_buf.wgsl");
/// `wgsl/grad_scale.wgsl`
pub const GRAD_SCALE: &str = include_str!("../wgsl/grad_scale.wgsl");
/// `wgsl/layernorm_dbeta.wgsl`
pub const LAYERNORM_DBETA: &str = include_str!("../wgsl/layernorm_dbeta.wgsl");
/// `wgsl/layernorm_dgamma.wgsl`
pub const LAYERNORM_DGAMMA: &str = include_str!("../wgsl/layernorm_dgamma.wgsl");
/// `wgsl/layernorm_dx.wgsl`
pub const LAYERNORM_DX: &str = include_str!("../wgsl/layernorm_dx.wgsl");
/// `wgsl/layernorm.wgsl`
pub const LAYERNORM: &str = include_str!("../wgsl/layernorm.wgsl");
/// `wgsl/ln_stats.wgsl`
pub const LN_STATS: &str = include_str!("../wgsl/ln_stats.wgsl");
/// `wgsl/matmul_dw.wgsl`
pub const MATMUL_DW: &str = include_str!("../wgsl/matmul_dw.wgsl");
/// `wgsl/matmul_dx.wgsl`
pub const MATMUL_DX: &str = include_str!("../wgsl/matmul_dx.wgsl");
/// `wgsl/matmul.wgsl`
pub const MATMUL: &str = include_str!("../wgsl/matmul.wgsl");
/// `wgsl/pos_add.wgsl`
pub const POS_ADD: &str = include_str!("../wgsl/pos_add.wgsl");
/// `wgsl/pos_bwd.wgsl`
pub const POS_BWD: &str = include_str!("../wgsl/pos_bwd.wgsl");
/// `wgsl/rms_inv.wgsl`
pub const RMS_INV: &str = include_str!("../wgsl/rms_inv.wgsl");
/// `wgsl/rmsnorm_dw.wgsl`
pub const RMSNORM_DW: &str = include_str!("../wgsl/rmsnorm_dw.wgsl");
/// `wgsl/rmsnorm_dx.wgsl`
pub const RMSNORM_DX: &str = include_str!("../wgsl/rmsnorm_dx.wgsl");
/// `wgsl/rmsnorm.wgsl`
pub const RMSNORM: &str = include_str!("../wgsl/rmsnorm.wgsl");
/// `wgsl/rope_train_bwd.wgsl`
pub const ROPE_TRAIN_BWD: &str = include_str!("../wgsl/rope_train_bwd.wgsl");
/// `wgsl/rope_train.wgsl`
pub const ROPE_TRAIN: &str = include_str!("../wgsl/rope_train.wgsl");
/// `wgsl/rope.wgsl`
pub const ROPE: &str = include_str!("../wgsl/rope.wgsl");
/// `wgsl/router_bwd.wgsl`
pub const ROUTER_BWD: &str = include_str!("../wgsl/router_bwd.wgsl");
/// `wgsl/router_gate_train.wgsl`
pub const ROUTER_GATE_TRAIN: &str = include_str!("../wgsl/router_gate_train.wgsl");
/// `wgsl/router_gate.wgsl`
pub const ROUTER_GATE: &str = include_str!("../wgsl/router_gate.wgsl");
/// `wgsl/scale_add_dexp.wgsl`
pub const SCALE_ADD_DEXP: &str = include_str!("../wgsl/scale_add_dexp.wgsl");
/// `wgsl/scale_add_dgate.wgsl`
pub const SCALE_ADD_DGATE: &str = include_str!("../wgsl/scale_add_dgate.wgsl");
/// `wgsl/scale_add.wgsl`
pub const SCALE_ADD: &str = include_str!("../wgsl/scale_add.wgsl");
/// `wgsl/silu_bwd_da.wgsl`
pub const SILU_BWD_DA: &str = include_str!("../wgsl/silu_bwd_da.wgsl");
/// `wgsl/silu_bwd_db.wgsl`
pub const SILU_BWD_DB: &str = include_str!("../wgsl/silu_bwd_db.wgsl");
/// `wgsl/silu_mul.wgsl`
pub const SILU_MUL: &str = include_str!("../wgsl/silu_mul.wgsl");

/// All kernels as `(name, source)` pairs, sorted by name.
pub const ALL: &[(&str, &str)] = &[
    ("adamw", ADAMW),
    ("add2", ADD2),
    ("add", ADD),
    ("attention", ATTENTION),
    ("attn_apply", ATTN_APPLY),
    ("attn_apply_bidir", ATTN_APPLY_BIDIR),
    ("attn_bwd_dk", ATTN_BWD_DK),
    ("attn_bwd_dk_bidir", ATTN_BWD_DK_BIDIR),
    ("attn_bwd_dq", ATTN_BWD_DQ),
    ("attn_bwd_dq_bidir", ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dscores", ATTN_BWD_DSCORES),
    ("attn_bwd_dscores_bidir", ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv", ATTN_BWD_DV),
    ("attn_bwd_dv_bidir", ATTN_BWD_DV_BIDIR),
    ("attn_scores_masked", ATTN_SCORES_MASKED),
    ("attn_scores", ATTN_SCORES),
    ("attn_scores_bidir", ATTN_SCORES_BIDIR),
    ("attn_softmax_masked", ATTN_SOFTMAX_MASKED),
    ("attn_softmax", ATTN_SOFTMAX),
    ("attn_softmax_bidir", ATTN_SOFTMAX_BIDIR),
    ("bias_add", BIAS_ADD),
    ("bias_grad", BIAS_GRAD),
    ("ce_grad_masked", CE_GRAD_MASKED),
    ("ce_grad", CE_GRAD),
    ("ce_value_masked", CE_VALUE_MASKED),
    ("ce_value", CE_VALUE),
    ("clip_coef", CLIP_COEF),
    ("emb_bwd", EMB_BWD),
    ("embed", EMBED),
    ("expert_counts", EXPERT_COUNTS),
    ("gelu_bwd", GELU_BWD),
    ("gelu", GELU),
    ("gradnorm_sq", GRADNORM_SQ),
    ("grad_scale_buf", GRAD_SCALE_BUF),
    ("grad_scale", GRAD_SCALE),
    ("layernorm_dbeta", LAYERNORM_DBETA),
    ("layernorm_dgamma", LAYERNORM_DGAMMA),
    ("layernorm_dx", LAYERNORM_DX),
    ("layernorm", LAYERNORM),
    ("ln_stats", LN_STATS),
    ("matmul_dw", MATMUL_DW),
    ("matmul_dx", MATMUL_DX),
    ("matmul", MATMUL),
    ("pos_add", POS_ADD),
    ("pos_bwd", POS_BWD),
    ("rms_inv", RMS_INV),
    ("rmsnorm_dw", RMSNORM_DW),
    ("rmsnorm_dx", RMSNORM_DX),
    ("rmsnorm", RMSNORM),
    ("rope_train_bwd", ROPE_TRAIN_BWD),
    ("rope_train", ROPE_TRAIN),
    ("rope", ROPE),
    ("router_bwd", ROUTER_BWD),
    ("router_gate_train", ROUTER_GATE_TRAIN),
    ("router_gate", ROUTER_GATE),
    ("scale_add_dexp", SCALE_ADD_DEXP),
    ("scale_add_dgate", SCALE_ADD_DGATE),
    ("scale_add", SCALE_ADD),
    ("silu_bwd_da", SILU_BWD_DA),
    ("silu_bwd_db", SILU_BWD_DB),
    ("silu_mul", SILU_MUL),
];

/// Look up a kernel's WGSL source by file stem (e.g. `"matmul"`).
///
/// # Panics
/// Panics if `name` is not a known kernel — callers reference kernels by
/// compile-time-known names, so an unknown name is a programmer error.
pub fn src(name: &str) -> &'static str {
    ALL.iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("unknown WGSL kernel: {name:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_kernels_present_and_nonempty() {
        assert_eq!(ALL.len(), 61);
        for (n, s) in ALL { assert!(!s.trim().is_empty(), "empty kernel {n}"); }
    }
    #[test]
    fn src_roundtrips() {
        for (n, s) in ALL { assert_eq!(src(n), *s); }
    }
}
