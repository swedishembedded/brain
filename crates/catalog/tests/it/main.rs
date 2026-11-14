// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `served` keeps a binary of its own.

#[path = "../common/mod.rs"]
mod common;

mod cancellation;
mod decision_agreement;
mod decision_lifecycle;
mod decision_served;
mod edgerock_real_served;
mod edgerock_served;
mod mirage_http;
mod mirage_served;
mod multi_adapter;
mod npu_model_parity;
mod resident_packed;
mod ricursive_chat;
mod ricursive_served;
mod scheduled_batch;
mod scored;
mod served_bounds;
mod served_paths;
mod served_training;
