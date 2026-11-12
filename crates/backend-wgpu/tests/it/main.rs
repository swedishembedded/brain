// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

#[path = "../common/mod.rs"]
mod common;

mod adapter_enumeration;
mod arch_desc_int8_not_a_speed_claim;
mod batch_bounded;
mod bench_conv;
mod bench_conv_gemm;
mod kernel_timing;
mod native_f16;
mod read_staging_reuse;
mod serialized_batch_count;
mod upload_flush;
