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

mod finetune_real;
mod generate_parity;
mod peft_export;
mod processor_parity;
mod recompute;
mod serving;
mod text_parity;
mod train_parity;
mod train_tiny;
mod tune_reload;
mod video_parity;
mod vision_parity;
