// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The synthetic Qwen3 chat model directory of the fine-tuning test binaries;
//! the shared parts are in `common`.

pub mod common;
pub use common::{cpu, qwen35_model_dir, TempDir};

/// The model directory: `model.safetensors`, `tokenizer.json`,
/// `tokenizer_config.json` (carrying the chat template).
pub fn chat_model_dir(tag: &str) -> TempDir {
    chat_model_dir_of(tag, qwen3::QwenConfig::tiny())
}

/// [`chat_model_dir`] of the model shape `shape` (the vocabulary and context are the fixture's own).
/// `QwenConfig::tiny_i8()` has every contraction width a whole number of quantisation groups, which
/// the packed base formats need.
pub fn chat_model_dir_of(tag: &str, shape: qwen3::QwenConfig) -> TempDir {
    common::qwen3_model_dir_of(tag, shape)
}
