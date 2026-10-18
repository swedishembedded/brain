// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A synthetic chat model directory shared by the fine-tuning test binaries:
//! a real Qwen3 checkpoint at toy width (2 layers, 16 wide) but the real
//! vocabulary size, because the chat packer terminates every record with
//! Qwen's `<|endoftext|>` id. Its tokenizer knows 23 letters and the ChatML
//! specials, and its template is a minimal ChatML: enough for every stage of
//! a fine-tune to run for real on the CPU backend.

use std::path::{Path, PathBuf};

/// A temp directory removed when the test ends.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("brain-sdk-finetune-{tag}-{}-{n}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

const TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}{{ message.content }}<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}";

pub fn write_tokenizer(dir: &Path) {
    let byte_encoder = data::bpe::bytes_to_unicode();
    let vocab: serde_json::Map<String, serde_json::Value> = (b'a'..=b'w').enumerate().map(|(id, b)| (byte_encoder[b as usize].to_string(), serde_json::json!(id as u32))).collect();
    let added = serde_json::json!([
        {"content": "<|endoftext|>", "id": 151_643},
        {"content": "<|im_start|>", "id": 151_644},
        {"content": "<|im_end|>", "id": 151_645}
    ]);
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({ "model": { "vocab": vocab, "merges": [] }, "added_tokens": added }).to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({ "chat_template": TEMPLATE }).to_string()).unwrap();
}

pub fn cpu() -> brain::Device {
    brain::Device::parse("cpu").unwrap()
}
