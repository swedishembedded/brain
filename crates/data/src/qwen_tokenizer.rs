// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen byte-level BPE tokenizer, loaded from a HuggingFace `tokenizer.json`.
//!
//! Same byte-level BPE family as [`crate::bpe::Gpt2Bpe`] — it reuses the byte
//! <-> unicode map and the merge loop ([`crate::bpe::bpe_merge`]) — but with
//! Qwen's vocabulary/merges and its own pre-tokenizer. Qwen's pre-tokenizer
//! (from `tokenizer.json`) is the cl100k-style regex
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//!     | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//! reproduced here as a hand-written scanner (no regex crate). Notable vs GPT-2:
//! digits split **one at a time** (`\p{N}`), and a letter run may absorb a
//! single leading non-alphanumeric char (not just a space).
//!
//! Special/added tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, …) are
//! matched as atomic units *before* BPE. `vocab_size()` reports the model vocab
//! used to size the embedding table.

use std::collections::HashMap;

use crate::bpe::{bpe_merge, bytes_to_unicode};
use crate::tokenizer::Tokenizer;

pub struct QwenBpe {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), u32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    /// (content, id) for special/added tokens, longest content first.
    specials: Vec<(String, u32)>,
    vocab_size: usize,
}

impl QwenBpe {
    /// Build from a `tokenizer.json` file path.
    pub fn from_file(path: &str) -> Result<QwenBpe, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_json_bytes(&bytes)
    }

    /// Build from a checkpoint DIRECTORY: `tokenizer.json` when present, else
    /// the split `vocab.json` + `merges.txt` (+ `added_tokens.json`) layout
    /// some Qwen2-family repos ship (FastVLM). Same byte-level BPE either way
    /// — the split files are re-assembled into the unified shape and parsed by
    /// the ONE existing parser, so the formats cannot drift apart.
    pub fn from_dir(dir: &str) -> Result<QwenBpe, String> {
        let unified = format!("{dir}/tokenizer.json");
        if std::path::Path::new(&unified).exists() {
            return Self::from_file(&unified);
        }
        let vocab: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!("{dir}/vocab.json")).map_err(|e| format!("read {dir}/vocab.json: {e}"))?,
        )
        .map_err(|e| format!("vocab.json: {e}"))?;
        let merges_txt = std::fs::read_to_string(format!("{dir}/merges.txt"))
            .map_err(|e| format!("read {dir}/merges.txt: {e}"))?;
        let merges: Vec<serde_json::Value> = merges_txt
            .lines()
            .skip_while(|l| l.starts_with("#version"))
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::Value::String(l.to_string()))
            .collect();
        let added: serde_json::Value = std::fs::read(format!("{dir}/added_tokens.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .map(|m| {
                // added_tokens.json is { content: id }; unified wants records.
                let arr: Vec<serde_json::Value> = m
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .map(|(c, id)| serde_json::json!({"content": c, "id": id}))
                            .collect()
                    })
                    .unwrap_or_default();
                serde_json::Value::Array(arr)
            })
            .unwrap_or(serde_json::Value::Array(Vec::new()));
        let unified = serde_json::json!({
            "model": { "vocab": vocab, "merges": merges },
            "added_tokens": added,
        });
        Self::from_json_bytes(unified.to_string().as_bytes())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<QwenBpe, String> {
        let j: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("tokenizer.json: {e}"))?;
        let model = &j["model"];

        // vocab: { token_string: id }
        let vocab = model["vocab"].as_object().ok_or("tokenizer.json: model.vocab")?;
        let mut encoder = HashMap::with_capacity(vocab.len());
        let mut decoder = HashMap::with_capacity(vocab.len());
        for (tok, id) in vocab {
            let id = id.as_u64().ok_or("vocab id")? as u32;
            encoder.insert(tok.clone(), id);
            decoder.insert(id, tok.clone());
        }

        // merges: array of ["a","b"] pairs (newer) or "a b" strings (older).
        let mut bpe_ranks = HashMap::new();
        if let Some(merges) = model["merges"].as_array() {
            for (rank, m) in merges.iter().enumerate() {
                let (l, r) = if let Some(arr) = m.as_array() {
                    (
                        arr[0].as_str().ok_or("merge pair")?.to_string(),
                        arr[1].as_str().ok_or("merge pair")?.to_string(),
                    )
                } else if let Some(s) = m.as_str() {
                    let mut it = s.splitn(2, ' ');
                    (it.next().unwrap().to_string(), it.next().ok_or("merge str")?.to_string())
                } else {
                    return Err("tokenizer.json: bad merge entry".into());
                };
                bpe_ranks.insert((l, r), rank as u32);
            }
        }

        // added/special tokens (top-level `added_tokens`).
        let mut specials: Vec<(String, u32)> = Vec::new();
        if let Some(at) = j["added_tokens"].as_array() {
            for t in at {
                if let (Some(c), Some(id)) = (t["content"].as_str(), t["id"].as_u64()) {
                    specials.push((c.to_string(), id as u32));
                    encoder.entry(c.to_string()).or_insert(id as u32);
                    decoder.entry(id as u32).or_insert_with(|| c.to_string());
                }
            }
        }
        // Longest content first so e.g. "<|im_start|>" matches before any prefix.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }

        // Vocab size = max id + 1 (covers added tokens beyond the base table).
        let vocab_size = decoder.keys().copied().max().map(|m| m as usize + 1).unwrap_or(0);

        Ok(QwenBpe { encoder, decoder, bpe_ranks, byte_encoder, byte_decoder, specials, vocab_size })
    }

    /// Encode one pre-token (already special-free) into ids via byte-level BPE.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let chars: Vec<String> = piece
            .bytes()
            .map(|b| self.byte_encoder[b as usize].to_string())
            .collect();
        for sub in bpe_merge(&self.bpe_ranks, chars) {
            match self.encoder.get(&sub) {
                Some(&id) => out.push(id),
                // Unknown subword should not occur for a complete byte-level vocab.
                None => {}
            }
        }
    }

    /// Render the Qwen ChatML template for a single-turn (or multi-turn) chat,
    /// optionally appending the assistant generation prompt. Plain string
    /// assembly (no Jinja). Returns the prompt text; encode it for inference.
    pub fn apply_chat_template(&self, msgs: &[(&str, &str)], add_generation_prompt: bool) -> String {
        let mut s = String::new();
        for (role, content) in msgs {
            s.push_str("<|im_start|>");
            s.push_str(role);
            s.push('\n');
            s.push_str(content);
            s.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            s.push_str("<|im_start|>assistant\n");
        }
        s
    }
}

impl Tokenizer for QwenBpe {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_with_specials(text, &mut out);
        out
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        let flush = |bytes: &mut Vec<u8>, s: &mut String| {
            if !bytes.is_empty() {
                s.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };
        let mut s = String::new();
        for &id in ids {
            let tok = match self.decoder.get(&id) {
                Some(t) => t,
                None => continue,
            };
            // A special token decodes to its literal content directly; BPE
            // subwords decode through the byte map.
            if self.specials.iter().any(|(_, sid)| *sid == id) {
                flush(&mut bytes, &mut s);
                s.push_str(tok);
            } else {
                for c in tok.chars() {
                    if let Some(&b) = self.byte_decoder.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        flush(&mut bytes, &mut s);
        s
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl QwenBpe {
    fn encode_with_specials(&self, text: &str, out: &mut Vec<u32>) {
        // Split on special-token literals first (longest-first), BPE the gaps.
        let mut rest = text;
        'outer: while !rest.is_empty() {
            // Find the earliest special-token occurrence.
            let mut best: Option<(usize, &str, u32)> = None;
            for (content, id) in &self.specials {
                if let Some(pos) = rest.find(content.as_str()) {
                    if best.map(|(bp, _, _)| pos < bp).unwrap_or(true) {
                        best = Some((pos, content, *id));
                    }
                }
            }
            if let Some((pos, content, id)) = best {
                if pos > 0 {
                    for piece in qwen_pretokenize(&rest[..pos]) {
                        self.encode_piece(&piece, out);
                    }
                }
                out.push(id);
                rest = &rest[pos + content.len()..];
                continue 'outer;
            }
            // No specials left: BPE the remainder.
            for piece in qwen_pretokenize(rest) {
                self.encode_piece(&piece, out);
            }
            break;
        }
    }
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_digit(c: char) -> bool {
    c.is_numeric()
}
fn is_ws(c: char) -> bool {
    c.is_whitespace()
}
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}
/// `[^\r\n\p{L}\p{N}]` — the optional leading char a letter run may absorb.
fn is_letter_prefix(c: char) -> bool {
    !is_nl(c) && !is_letter(c) && !is_digit(c)
}
/// `[^\s\p{L}\p{N}]` — a punctuation/symbol char.
fn is_symbol(c: char) -> bool {
    !is_ws(c) && !is_letter(c) && !is_digit(c)
}

/// Hand-written reproduction of Qwen's pre-tokenizer regex (see module docs).
/// Returns the list of pre-tokens, in order, reconstructing `text`.
pub fn qwen_pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut toks: Vec<String> = Vec::new();
    let mut i = 0;

    const CONTRACTIONS: [&[char]; 7] = [
        &['s'],
        &['t'],
        &['r', 'e'],
        &['v', 'e'],
        &['m'],
        &['l', 'l'],
        &['d'],
    ];

    while i < n {
        // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if chars[i] == '\'' {
            let mut matched = None;
            for suf in CONTRACTIONS {
                if i + 1 + suf.len() <= n
                    && chars[i + 1..i + 1 + suf.len()]
                        .iter()
                        .zip(suf)
                        .all(|(a, b)| a.to_ascii_lowercase() == *b)
                {
                    matched = Some(1 + suf.len());
                    break;
                }
            }
            if let Some(len) = matched {
                toks.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }

        // 2) [^\r\n\p{L}\p{N}]?\p{L}+
        if is_letter(chars[i]) {
            let mut j = i;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            toks.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if is_letter_prefix(chars[i]) && i + 1 < n && is_letter(chars[i + 1]) {
            let start = i;
            let mut j = i + 1;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            toks.push(chars[start..j].iter().collect());
            i = j;
            continue;
        }

        // 3) \p{N}  (single digit)
        if is_digit(chars[i]) {
            toks.push(chars[i..i + 1].iter().collect());
            i += 1;
            continue;
        }

        // 4) ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            let has_space = chars[i] == ' ';
            let p = if has_space { i + 1 } else { i };
            if p < n && is_symbol(chars[p]) {
                let start = i;
                let mut j = p;
                while j < n && is_symbol(chars[j]) {
                    j += 1;
                }
                while j < n && is_nl(chars[j]) {
                    j += 1;
                }
                toks.push(chars[start..j].iter().collect());
                i = j;
                continue;
            }
        }

        // 5) \s*[\r\n]+   (whitespace run up to & including its last newline)
        if is_ws(chars[i]) {
            let mut j = i;
            while j < n && is_ws(chars[j]) {
                j += 1;
            }
            // last newline index in [i, j)
            let mut last_nl: Option<usize> = None;
            for (k, c) in chars.iter().enumerate().take(j).skip(i) {
                if is_nl(*c) {
                    last_nl = Some(k);
                }
            }
            if let Some(k) = last_nl {
                toks.push(chars[i..k + 1].iter().collect());
                i = k + 1;
                continue;
            }
            // 6) \s+(?!\S) and 7) \s+ : a pure space/tab run [i, j) with no
            // newline. If followed by a non-space char, reserve the last ws char
            // for the next token's optional prefix; else emit the whole run.
            if j == n || j - 1 == i {
                toks.push(chars[i..j].iter().collect());
                i = j;
            } else {
                toks.push(chars[i..j - 1].iter().collect());
                i = j - 1;
            }
            continue;
        }

        // Fallback: emit one char to guarantee progress.
        toks.push(chars[i..i + 1].iter().collect());
        i += 1;
    }
    toks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Option<QwenBpe> {
        let path = std::env::var("QWEN_TOKENIZER").ok()?;
        QwenBpe::from_file(&path).ok()
    }

    #[test]
    fn pinned_reference_vectors() {
        let Some(t) = tok() else {
            eprintln!("QWEN_TOKENIZER unset; skipping");
            return;
        };
        // Ground truth from the HF tokenizer (gen_tok.py).
        assert_eq!(t.encode("The capital of France is"), vec![785, 6722, 315, 9625, 374]);
        assert_eq!(t.encode("Hello, world"), vec![9707, 11, 1879]);
        assert_eq!(t.encode("12345"), vec![16, 17, 18, 19, 20]); // single-digit split
        assert_eq!(t.encode("brain"), vec![53060]);
        assert_eq!(t.encode("  spaced"), vec![220, 63828]);
        assert_eq!(t.encode("def main():\n\tpass"), vec![750, 1887, 3932, 41431]);
    }

    #[test]
    fn special_tokens_and_roundtrip() {
        let Some(t) = tok() else {
            return;
        };
        assert_eq!(t.encode("<|im_start|>"), vec![151644]);
        assert_eq!(t.encode("<|endoftext|>"), vec![151643]);
        for s in ["The capital of France is", "Hello, world", "brain models"] {
            assert_eq!(t.decode(&t.encode(s)), s, "roundtrip {s:?}");
        }
        let prompt = t.apply_chat_template(&[("user", "Hi")], true);
        assert_eq!(prompt, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }
}
