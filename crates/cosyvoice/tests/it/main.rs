// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod cv3_flow_parity;
mod cv3_hift_parity;
mod flow_parity;
mod hift_parity;
mod llm3_parity;
mod llm_parity;
mod lm_overfit;
mod lm_recompute;
mod pipeline_e2e;
