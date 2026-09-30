// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A fully synthetic, forward-capable Qwen3 checkpoint and tokenizer, shared
//! by the `text`-surface test binaries that need a real generation without a
//! real model.
//!
//! The checkpoint is `QwenConfig::tiny()` (`vocab: 23, n_layers: 2,
//! d_model: 16, ...`): its config in the safetensors header plus every tensor
//! `tiny().param_list()` names. The tokenizer is a curated 23-letter vocabulary, not a
//! universal one: it covers [`PROMPT_WITHIN_VOCAB`] and every id the tiny
//! model's 23-wide LM head can sample, so the decode side round-trips too.

use std::path::{Path, PathBuf};

/// A fixture file or directory that deletes itself when the test ends.
pub struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.0.is_dir() {
            std::fs::remove_dir_all(&self.0).ok();
        } else {
            std::fs::remove_file(&self.0).ok();
        }
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

/// A fresh path in the temp directory, unique per process, call and `tag`,
/// so tests running in parallel never share a fixture file.
pub fn scratch_path(tag: &str, ext: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("brain-sdk-{tag}-{}-{n}.{ext}", std::process::id()))
}

/// The fixture vocabulary: one id per letter, exactly as wide as the tiny
/// config's LM head.
pub const VOCAB_LETTERS: std::ops::RangeInclusive<u8> = b'a'..=b'w';
/// A prompt every character of which is a single known vocab entry.
pub const PROMPT_WITHIN_VOCAB: &str = "cabbage";

/// Writes the fixture tokenizer and returns its path.
///
/// `data::bpe::bytes_to_unicode()` is the SAME byte<->char table
/// `QwenBpe::encode_piece` looks up through, so a vocab key built any other
/// way (e.g. the literal ASCII byte) would silently miss on every lookup -
/// the failure would be an empty encode, not a panic.
pub fn tiny_tokenizer(tag: &str) -> Scratch {
    let byte_encoder = data::bpe::bytes_to_unicode();
    let vocab: serde_json::Map<String, serde_json::Value> =
        VOCAB_LETTERS.enumerate().map(|(id, b)| (byte_encoder[b as usize].to_string(), serde_json::json!(id as u32))).collect();
    assert_eq!(vocab.len(), qwen3::QwenConfig::tiny().vocab as usize, "the vocab must exactly cover the tiny config's lm_head width");
    let path = scratch_path(tag, "tokenizer.json");
    std::fs::write(&path, serde_json::to_vec(&serde_json::json!({ "model": { "vocab": vocab, "merges": [] } })).unwrap()).unwrap();
    Scratch(path)
}

/// A real, forward-capable `QwenConfig::tiny()` checkpoint: every tensor
/// `param_list()` names, filled with small deterministic non-constant values:
/// not all-zero, so a degenerate all-equal softmax cannot mask a real
/// indexing bug.
pub fn tiny_qwen3_checkpoint(tag: &str) -> Scratch {
    let path = scratch_path(tag, "safetensors");
    let cfg = qwen3::QwenConfig::tiny();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, numel)| {
            let data: Vec<f32> = (0..numel).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
            (name, vec![numel as u64], data)
        })
        .collect();
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
    Scratch(path)
}
