// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod composite;
mod encode_views;
mod prompt_real;
mod real_weight;
mod real_weight_generate;
mod recompute;
mod tensor_core;
mod tiny_ref;
mod train_overfit;
