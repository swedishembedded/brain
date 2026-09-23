// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The compute pipelines this crate builds a device with.
//!
//! Indices are positions in [`PIPELINES`] and are referred to by the `K_*`
//! constants below, never by a literal. Appending is safe; reordering is not.

/// `gelu_erf` is paired with `gelu_erf_bwd` and NOT with `gelu_bwd`: the
/// latter is the derivative of the tanh approximation, agrees with this one
/// to about 1e-3, and would therefore train on the gradient of a different
/// function while every tolerance-based gate stayed green.
pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    // The register-tiled twins. Never selected directly: `model::block::
    // pick_gemm` routes by output shape, because a 128x128 tile on an output
    // smaller than one tile leaves most of the card idle and the naive
    // kernel wins.
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("add2", kernels::ADD2),
    ("ce_value", kernels::CE_VALUE),
    ("ce_grad", kernels::CE_GRAD),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

pub const K_MATMUL: usize = 0;
pub const K_MATMUL_DX: usize = 1;
pub const K_MATMUL_DW: usize = 2;
pub const K_MATMUL_REG: usize = 3;
pub const K_MATMUL_DX_REG: usize = 4;
pub const K_MATMUL_DW_REG: usize = 5;
pub const K_BIAS_ADD: usize = 6;
pub const K_BIAS_GRAD: usize = 7;
pub const K_GELU: usize = 8;
pub const K_GELU_BWD: usize = 9;
pub const K_ADD2: usize = 10;
pub const K_CE_VALUE: usize = 11;
pub const K_CE_GRAD: usize = 12;
pub const K_SOFTMAX: usize = 13;
pub const K_ADAMW: usize = 14;
pub const K_GRADNORM_SQ: usize = 15;
pub const K_GRAD_SCALE: usize = 16;
pub const K_CLIP_COEF: usize = 17;
pub const K_GRAD_SCALE_BUF: usize = 18;
