// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod act_kernels;
mod audio_io;
mod bench_conv1d_lowering;
mod bias_ncl_kernels;
mod conv1d_gemm_lowering;
mod conv_kernels;
mod resample_rational;
mod snake_kernels;
mod weight_norm;
