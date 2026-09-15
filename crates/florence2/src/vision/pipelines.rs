// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kernel registration list for the DaViT vision tower - resolved by NAME
//! (`vision::ids::ConvKernelIds::resolve`, `model::vit::VitKernelIds`
//! fields) against whichever subset a given forward actually dispatches.
//! Modeled directly on `fastvlm::encoder::PIPELINES`, the closest existing
//! model (conv stages + NCHW<->sequence attention stages).

pub const PIPELINES: &[(&str, &str)] = &[
    // --- conv / bias ---
    ("conv2d_gd", kernels::CONV2D_GD),
    ("conv2d_gd_reg", kernels::CONV2D_GD_REG),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("bias_add", kernels::BIAS_ADD),
    ("add_chan_bcast", kernels::ADD_CHAN_BCAST),
    // --- transpose + norm + linear ---
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("layernorm", kernels::LAYERNORM),
    ("gelu_erf", kernels::GELU_ERF),
    ("matmul", kernels::MATMUL),
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("add2", kernels::ADD2),
    ("add_inplace", kernels::ADD_INPLACE),
    ("scale_chan", kernels::SCALE_CHAN),
    // --- window attention (self-attention over a fused qkv slab) ---
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
];
