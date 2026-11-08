// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `bench_forward_p40`, `bench_train_p40`, `cpu_reg_parity` keep binaries of their own.

mod convergence;
mod cuda_backend_parity;
mod device_memory;
mod distributed_train;
mod dp_parity;
mod layer_streamed;
mod optimizer_state;
mod recompute;
mod shard_parity;
mod streamed_construction;
mod warm_start;
