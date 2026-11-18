// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The synthetic Qwen3 chat model directory of the fine-tuning test binaries;
//! the shared parts are in `common`.

pub mod common;
pub use common::{cpu, TempDir};
// Only the test of several architectures at once takes it through this module.
#[cfg(feature = "text")]
pub use common::qwen35_model_dir;

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

/// The fixture model as a store entry whose weights are the repo directory (a pulled `transformers` checkout, the form
/// a served trainer names its base in): `(models dir, "Qwen/Tiny")` for a run's `models_dir` and base.
pub fn chat_model_as_repo_dir(model: &std::path::Path) -> (std::path::PathBuf, &'static str) {
    let root = model.join("store");
    let repo = root.join("Qwen").join("Tiny");
    std::fs::create_dir_all(&repo).expect("the repo directory");
    for entry in std::fs::read_dir(model).expect("the fixture directory").flatten() {
        if entry.path().is_file() {
            std::fs::copy(entry.path(), repo.join(entry.file_name())).expect("a fixture file");
        }
    }
    std::fs::write(repo.join("config.json"), b"{}").expect("the config");
    std::fs::write(repo.join("brain.manifest.json"), br#"{"id":"Qwen/Tiny","family":"qwen","roles":{"weights":"."}}"#).expect("the manifest");
    (root, "Qwen/Tiny")
}
