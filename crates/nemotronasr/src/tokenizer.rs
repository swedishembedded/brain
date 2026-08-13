// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal id→text detokenizer for the Nemotron RNN-T vocabulary, loaded from the
//! HF `tokenizer.json` (a `BPE` model with a `Metaspace` decoder). Only *decoding*
//! is needed — the acoustic model emits token ids; there is no text→id path here.
//!
//! Decoding is the SentencePiece/Metaspace rule: concatenate the sub-word pieces
//! and turn the metaspace marker `▁` (U+2581) back into a space, dropping the
//! leading one. Special/added tokens (language tags like `<en-US>`, `<unk>`) are
//! skipped so the output is clean transcription text. No new dependency — the JSON
//! is parsed with the `serde_json` the crate already uses.

use std::collections::{HashMap, HashSet};

/// The metaspace replacement character used by the Nemotron tokenizer.
const METASPACE: char = '\u{2581}'; // ▁

/// A decode-only Nemotron tokenizer.
pub struct Detokenizer {
    id_to_piece: HashMap<u32, String>,
    special: HashSet<u32>,
}

impl Detokenizer {
    /// Load from a checkpoint directory containing `tokenizer.json`.
    pub fn from_hf(dir: &str) -> Result<Detokenizer, String> {
        let path = format!("{dir}/tokenizer.json");
        let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_json_bytes(&bytes)
    }

    /// Build from raw `tokenizer.json` bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Detokenizer, String> {
        let j: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("tokenizer.json parse: {e}"))?;
        let vocab = j["model"]["vocab"].as_object().ok_or("tokenizer.json: missing model.vocab object")?;
        let mut id_to_piece = HashMap::with_capacity(vocab.len());
        for (piece, id) in vocab {
            if let Some(i) = id.as_u64() {
                id_to_piece.insert(i as u32, piece.clone());
            }
        }
        // Added/special tokens (control tokens, language tags) — excluded from text.
        let mut special = HashSet::new();
        if let Some(arr) = j["added_tokens"].as_array() {
            for t in arr {
                if let Some(i) = t["id"].as_u64() {
                    special.insert(i as u32);
                }
            }
        }
        Ok(Detokenizer { id_to_piece, special })
    }

    /// Number of vocabulary entries (for sanity checks).
    pub fn vocab_size(&self) -> usize {
        self.id_to_piece.len()
    }

    /// Decode emitted (non-blank) RNN-T token ids to transcription text. Metaspace
    /// `▁` → space (leading one dropped); special/added tokens are skipped.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw = String::new();
        for &id in ids {
            if self.special.contains(&id) {
                continue;
            }
            if let Some(p) = self.id_to_piece.get(&id) {
                raw.push_str(p);
            }
        }
        // Metaspace → space, then trim the single leading space the scheme prepends.
        let mut out = String::with_capacity(raw.len());
        for ch in raw.chars() {
            out.push(if ch == METASPACE { ' ' } else { ch });
        }
        out.trim_start().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metaspace_decode_from_inline_vocab() {
        // A tiny synthetic tokenizer.json: pieces "▁he", "llo", "▁world", "<unk>".
        let json = r#"{
            "model": { "type": "BPE", "vocab": { "▁he": 10, "llo": 11, "▁world": 12, "<unk>": 0 } },
            "added_tokens": [ { "id": 0, "content": "<unk>" } ]
        }"#;
        let d = Detokenizer::from_json_bytes(json.as_bytes()).expect("build");
        assert_eq!(d.vocab_size(), 4);
        // "▁he" + "llo" + "▁world" → "he llo world"? metaspace joins: "▁hello▁world"
        assert_eq!(d.decode(&[10, 11, 12]), "hello world");
        // special token id 0 is dropped
        assert_eq!(d.decode(&[10, 11, 0, 12]), "hello world");
    }
}
