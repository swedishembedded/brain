// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision model's kernel set, and its indices resolved by NAME.
//!
//! The encoder and the head run on ONE device with ONE pipeline list, so they
//! must agree on what index 7 means. Hard-coded per-module constants cannot
//! give that: two lists that each start at zero silently disagree the moment
//! both halves exist, and the symptom is a plausible number from the wrong
//! kernel rather than a crash.
//!
//! So the list is declared once here and every index is looked up by name at
//! build time. Adding a kernel to [`PIPELINES`] cannot renumber anything, and a
//! kernel a model forgot to register fails by name at construction instead of
//! running whatever happens to sit at that index.

use gpu_core::Gpu;

/// Every kernel either half dispatches, forward and backward.
pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("emb_bwd", kernels::EMB_BWD),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("add2", kernels::ADD2),
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    // The cooperative LayerNorm trio. Never indexed directly: `block::
    // LayerNormIds::resolve` finds these by name and selects between them and
    // the reference per device.
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
];

macro_rules! ids {
    ($($f:ident => $n:literal),+ $(,)?) => {
        /// One resolved index per kernel, built once per device.
        #[derive(Clone, Copy, Debug)]
        pub struct Ids { $(pub $f: usize),+ }

        impl Ids {
            /// Resolve against a device built with [`PIPELINES`]. A missing
            /// kernel is named, because the alternative is dispatching
            /// whatever occupies that slot.
            pub fn resolve(g: &Gpu) -> Ids {
                Ids {
                    $($f: g.kernel_index($n).unwrap_or_else(|| {
                        panic!("kernel {:?} is not registered on this device - build it with decide::kern::PIPELINES", $n)
                    })),+
                }
            }
        }
    };
}

ids! {
    embed => "embed",
    emb_bwd => "emb_bwd",
    matmul => "matmul",
    matmul_reg3 => "matmul_reg3",
    matmul_dx => "matmul_dx",
    matmul_dw => "matmul_dw",
    matmul_dx_reg => "matmul_dx_reg",
    matmul_dw_reg => "matmul_dw_reg",
    bias_add => "bias_add",
    bias_grad => "bias_grad",
    add2 => "add2",
    gelu_erf => "gelu_erf",
    gelu_erf_bwd => "gelu_erf_bwd",
    layernorm => "layernorm",
    ln_stats => "ln_stats",
    layernorm_dx => "layernorm_dx",
    ln_dgamma => "layernorm_dgamma",
    ln_dbeta => "layernorm_dbeta",
    scores_cross => "attn_scores_cross",
    softmax_cross => "attn_softmax_cross",
    apply_cross => "attn_apply_cross",
    dscores_cross => "attn_bwd_dscores_cross",
    dq_cross => "attn_bwd_dq_cross",
    dk_cross_acc => "attn_bwd_dk_cross_acc",
    dv_cross_acc => "attn_bwd_dv_cross_acc",
}
