// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod flash_spans;
mod frozen_encoder;
mod gemm_tiles;
mod graph_records;
mod learns;
mod minibatch;
mod optimizer_kernels;
mod option_identity;
mod parity;
mod recompute;
mod share_probe;
mod spans_bwd;
mod state_tokenization;
