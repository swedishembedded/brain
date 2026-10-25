// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tokenizer and dataset pipelines for training
// language models from scratch for its clients. If your team needs expertise
// in building a vocabulary fitted to a corpus, you can procure our services
// by sending an email to info@swedishembedded.com.

//! A byte-level BPE tokenizer trained on a corpus.
//!
//! [`crate::bpe::Gpt2Bpe`] carries GPT-2's fixed 50257-token vocabulary; a
//! small model trained from scratch spends most of its parameters on a table
//! that large. This trains a vocabulary of a chosen size on the corpus
//! itself, with the same GPT-2 pre-tokenization (merges never cross a
//! pre-token boundary).
//!
//! Ids `0..256` are the raw bytes, ids `256..256+merges` the learned merges
//! in the order they were learned, and the last id is the end-of-text
//! separator. Every token decodes to a fixed byte string, so the UTF-8 byte
//! length of a decoded sequence is the sum of its tokens' byte lengths: the
//! denominator a bits-per-byte score needs is exact.
//!
//! Training is the textbook algorithm: count adjacent symbol pairs over the
//! pre-token frequency table, merge the most frequent pair everywhere (ties
//! go to the smallest pair, so a corpus always yields the same vocabulary),
//! repeat. Encoding applies the merges by rank within each pre-token.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::tokenizer::Tokenizer;

/// The separator between documents, as TinyStories and Qwen spell it.
pub const END_OF_TEXT: &str = "<|endoftext|>";

#[derive(Clone, Debug)]
pub struct ByteBpe {
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    /// Byte string of every non-special token, indexed by id.
    pieces: Vec<Vec<u8>>,
}

impl ByteBpe {
    /// Learn `vocab_size - 257` merges from `text` (256 byte tokens and one
    /// end-of-text id make up the rest). `vocab_size` must exceed 257.
    pub fn train(text: &str, vocab_size: usize) -> ByteBpe {
        assert!(vocab_size > 257, "byte BPE vocabulary {vocab_size} leaves no room for merges");
        // Pre-tokenized a line at a time, so the transient piece list stays a
        // line long however large the training text is.
        let mut freq: HashMap<String, u64> = HashMap::new();
        for line in text.split_inclusive('\n') {
            for piece in crate::bpe::pre_tokenize(line) {
                *freq.entry(piece).or_insert(0) += 1;
            }
        }
        let mut words: Vec<(Vec<u32>, u64)> = freq.into_iter().map(|(w, c)| (w.bytes().map(u32::from).collect(), c)).collect();
        words.sort();
        let mut merges = Vec::with_capacity(vocab_size - 257);
        while merges.len() < vocab_size - 257 {
            let mut pairs: HashMap<(u32, u32), u64> = HashMap::new();
            for (w, c) in &words {
                for p in w.windows(2) {
                    *pairs.entry((p[0], p[1])).or_insert(0) += c;
                }
            }
            let Some((&best, _)) = pairs.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))) else { break };
            let id = 256 + merges.len() as u32;
            for (w, _) in words.iter_mut() {
                merge_in_place(w, best, id);
            }
            merges.push(best);
        }
        ByteBpe::from_merges(merges)
    }

    fn from_merges(merges: Vec<(u32, u32)>) -> ByteBpe {
        let ranks = merges.iter().enumerate().map(|(r, p)| (*p, r as u32)).collect();
        let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for (a, b) in &merges {
            let mut s = pieces[*a as usize].clone();
            s.extend_from_slice(&pieces[*b as usize]);
            pieces.push(s);
        }
        ByteBpe { merges, ranks, pieces }
    }

    /// The end-of-text separator's id (the last id).
    pub fn end_of_text(&self) -> u32 {
        self.pieces.len() as u32
    }

    /// Bytes `id` decodes to (empty for the separator).
    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces.get(id as usize).map_or(&[], |p| p.as_slice())
    }

    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        let mut w: Vec<u32> = word.bytes().map(u32::from).collect();
        loop {
            let best = w.windows(2).filter_map(|p| self.ranks.get(&(p[0], p[1])).map(|r| (*r, (p[0], p[1])))).min();
            let Some((rank, pair)) = best else { break };
            merge_in_place(&mut w, pair, 256 + rank);
        }
        out.extend(w);
    }

    pub fn to_json(&self) -> Value {
        json!({ "type": "byte-bpe", "merges": self.merges, "end_of_text": END_OF_TEXT })
    }

    pub fn from_json(v: &Value) -> Result<ByteBpe, String> {
        if v["type"] != "byte-bpe" {
            return Err(format!("tokenizer type {} is not byte-bpe", v["type"]));
        }
        let merges = v["merges"].as_array().ok_or("byte-bpe tokenizer has no merges array")?;
        let mut out = Vec::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let pair = m.as_array().filter(|p| p.len() == 2).ok_or(format!("merge {i} is not a pair"))?;
            let id = |x: &Value| x.as_u64().map(|v| v as u32).ok_or(format!("merge {i} holds a non-integer"));
            let (a, b) = (id(&pair[0])?, id(&pair[1])?);
            if a >= 256 + i as u32 || b >= 256 + i as u32 {
                return Err(format!("merge {i} refers to a token learned after it"));
            }
            out.push((a, b));
        }
        Ok(ByteBpe::from_merges(out))
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_string(&self.to_json())?)
    }

    pub fn load(path: &Path) -> Result<ByteBpe, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        ByteBpe::from_json(&v)
    }
}

/// Replace every non-overlapping occurrence of `pair` in `w`, left to right.
fn merge_in_place(w: &mut Vec<u32>, pair: (u32, u32), id: u32) {
    let mut i = 0;
    let mut j = 0;
    while i < w.len() {
        if i + 1 < w.len() && w[i] == pair.0 && w[i + 1] == pair.1 {
            w[j] = id;
            i += 2;
        } else {
            w[j] = w[i];
            i += 1;
        }
        j += 1;
    }
    w.truncate(j);
}

impl Tokenizer for ByteBpe {
    /// Encode text; [`END_OF_TEXT`] occurrences become the separator id.
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cache: HashMap<String, Vec<u32>> = HashMap::new();
        for (i, part) in text.split(END_OF_TEXT).enumerate() {
            if i > 0 {
                out.push(self.end_of_text());
            }
            for word in crate::bpe::pre_tokenize(part) {
                if let Some(ids) = cache.get(&word) {
                    out.extend_from_slice(ids);
                    continue;
                }
                let start = out.len();
                self.encode_word(&word, &mut out);
                cache.insert(word, out[start..].to_vec());
            }
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if *id == self.end_of_text() {
                bytes.extend_from_slice(END_OF_TEXT.as_bytes());
            } else {
                bytes.extend_from_slice(self.piece(*id));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.pieces.len() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "Once upon a time, there was a little girl named Lily. She loved to play outside. \
        One day, Lily saw a big red ball. \"Can I play?\" she asked. Ünïcödé too: café, naïve, 你好.";

    #[test]
    fn trained_vocabulary_round_trips_and_has_the_requested_size() {
        let bpe = ByteBpe::train(TEXT, 300);
        assert_eq!(bpe.vocab_size(), 300);
        let ids = bpe.encode(TEXT);
        assert_eq!(bpe.decode(&ids), TEXT);
        assert!(ids.len() < TEXT.len(), "merges must compress the training text");
        let bytes: usize = ids.iter().map(|i| bpe.piece(*i).len()).sum();
        assert_eq!(bytes, TEXT.len(), "token byte lengths sum to the UTF-8 length");
    }

    #[test]
    fn separator_is_one_token_and_text_it_never_saw_still_round_trips() {
        let bpe = ByteBpe::train(TEXT, 280);
        let doc = format!("{END_OF_TEXT}A cat sat.{END_OF_TEXT}Ωmega 🙂");
        let ids = bpe.encode(&doc);
        assert_eq!(ids.iter().filter(|i| **i == bpe.end_of_text()).count(), 2);
        assert_eq!(bpe.decode(&ids), doc);
    }

    #[test]
    fn training_is_deterministic_and_json_round_trips() {
        let a = ByteBpe::train(TEXT, 290);
        let b = ByteBpe::train(TEXT, 290);
        assert_eq!(a.merges, b.merges);
        let c = ByteBpe::from_json(&a.to_json()).unwrap();
        assert_eq!(c.encode(TEXT), a.encode(TEXT));
        assert!(ByteBpe::from_json(&json!({"type": "byte-bpe", "merges": [[300, 1]]})).is_err());
    }
}
