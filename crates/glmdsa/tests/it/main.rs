// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `reg3_parity` keeps a binary of its own.

mod convergence;
mod indexer;
mod layer_streamed;
mod optimizer_state;
mod recompute;
mod streamed_construction;
mod topk_mask_kernel;
