// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The synthetic Qwen3 chat model directory of the fine-tuning test binaries;
//! the shared parts are in `common`.

mod common;
pub use common::{cpu, write_tokenizer, TempDir};

/// The model directory: `model.safetensors`, `tokenizer.json`,
/// `tokenizer_config.json` (carrying the chat template).
pub fn chat_model_dir(tag: &str) -> TempDir {
    chat_model_dir_of(tag, qwen3::QwenConfig::tiny())
}

/// [`chat_model_dir`] of the model shape `shape` (the vocabulary and context are the fixture's own).
/// `QwenConfig::tiny_i8()` has every contraction width a whole number of quantisation groups, which
/// the packed base formats need.
pub fn chat_model_dir_of(tag: &str, shape: qwen3::QwenConfig) -> TempDir {
    let dir = TempDir::new(tag);
    let cfg = qwen3::QwenConfig { vocab: 151_936, block_size: 128, max_position_embeddings: 128, ..shape };
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, numel)| {
            let data: Vec<f32> = (0..numel).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
            (name, vec![numel as u64], data)
        })
        .collect();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();

    write_tokenizer(&dir);
    dir
}
