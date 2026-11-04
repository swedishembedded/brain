// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `parity` keeps a binary of its own.

mod deepseek_vl_parity;
mod flash_relpos;
mod gradcheck;
mod native_attention;
mod recompute;
mod tensor_core;
mod wgpu_block_count_corruption;
mod wgpu_real_weight_parity;
