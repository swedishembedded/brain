// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod adafactor_state;
mod adamw_bf16_parity;
mod adamw_q8_parity;
mod bench_gradnorm;
mod bench_host_update;
mod bench_q8;
mod coop_gradnorm;
mod bench_step;
mod full_ft_state;
mod fused_update;
mod named_moments;
mod offload_parity;
mod parallel_update;
mod state_memory;
mod streamed_state;
