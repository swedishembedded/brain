// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod backend_parity;
mod chunked_prefill;
mod decode_throughput;
mod generate;
mod gradcheck;
mod layer_streamed;
mod optimizer_state;
mod parity;
mod recompute;
mod serve;
mod serve_fast;
mod shard_parity;
mod streamed_construction;
mod weight_read_order_bench;
