// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `catalogue_coverage`, `leaks` keep binaries of their own.

mod aot_driver_only;
mod arch_specific;
mod backend_contract;
mod block_cache;
mod concurrent_open;
mod cuda_graphs;
mod cuda_pass;
mod large_buffer;
mod mem_pool;
mod native_launch_v2;
mod placement;
mod sparse;
mod staged_writes;
mod tensormap_template;
mod wgsl_cuda_bounds;
mod wgsl_cuda_golden;
mod wgsl_cuda_structs;
mod wgsl_cuda_vectors;
