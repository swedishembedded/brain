// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod adapter_kinds_fd;
mod adapter_kinds_fd_qwen35;
mod attn_bidir_fd;
mod attn_cross_fd;
mod backend_parity;
mod col2im_kernels;
mod conv3d_kernels;
mod convtr2d_kernels;
mod decide_fd;
mod decoder_variants_fd;
mod depth_kernels;
mod florence2_fd;
mod gelu_erf_fd;
mod glue;
mod grid_sample_kernels;
mod horizon_fd;
mod huginn_fd;
mod imaging_models;
mod layernorm2d_kernels;
mod maxpool2d_kernels;
mod mix_fd;
mod modernbert_adamw;
mod modernbert_fd;
mod mse_fd;
mod ouro_fd;
mod prelu_kernels;
mod quick_gelu_fd;
mod qwen35_image_record_fd;
mod resize_bicubic_kernels;
mod sam_hd_fd;
mod spectrallm_fd;
mod splitk_kernels;
mod timesfm3_fd;
