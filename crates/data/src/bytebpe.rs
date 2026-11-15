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
//! itself, with GPT-2's pre-tokenization (merges never cross a pre-token
//! boundary) as a `tokenizer.json` `ByteLevel` step applies it, Unicode letter
//! and number classes included ([`crate::hf_pretok::gpt2_pieces`]).
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
//! repeat; the counts are kept incrementally, so a merge only revisits the
//! words that hold its pair. Encoding applies the merges by rank within each
//! pre-token. [`ByteBpe::to_hf_tokenizer_json`] writes the vocabulary as a
//! Hugging Face `tokenizer.json` with declared special tokens, which every
//! brain loader reads ([`crate::qwen_tokenizer::QwenBpe`]).

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
        ByteBpe::train_reserving(text, vocab_size - 257, &[])
    }

    /// Learn up to `merges` merges from `text`, never one whose bytes spell a `reserved` string (a special token the
    /// vocabulary will carry beside the merges, so no learned piece shadows it) or a piece already learned (two merge
    /// paths to the same bytes would be two ids for one string, which a `tokenizer.json` vocabulary cannot hold).
    /// The `reserved` strings are cut out of the text before pairs are counted: a special token is matched whole
    /// before byte-pair encoding, so its bytes are never part of a word.
    ///
    /// The pair counts are kept incrementally: a merge recounts only the words that hold its pair, so training costs
    /// the occurrences each merge rewrites rather than the whole table per merge. The order is the textbook one (the
    /// most frequent pair, ties to the smallest), which the tests hold it to.
    pub fn train_reserving(text: &str, merges: usize, reserved: &[&str]) -> ByteBpe {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashSet};
        // Pre-tokenized a line at a time, so the transient piece list stays a
        // line long however large the training text is.
        let mut freq: HashMap<String, u64> = HashMap::new();
        for line in text.split_inclusive('\n') {
            for part in split_on_all(line, reserved) {
                for piece in crate::hf_pretok::gpt2_pieces(part) {
                    *freq.entry(piece).or_insert(0) += 1;
                }
            }
        }
        let mut words: Vec<(Vec<u32>, u64)> = freq.into_iter().map(|(w, c)| (w.bytes().map(u32::from).collect(), c)).collect();
        words.sort();
        let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut taken: HashSet<Vec<u8>> = pieces.iter().cloned().chain(reserved.iter().map(|r| r.as_bytes().to_vec())).collect();
        let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
        let mut holders: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (i, (w, c)) in words.iter().enumerate() {
            for p in w.windows(2) {
                *counts.entry((p[0], p[1])).or_insert(0) += c;
                holders.entry((p[0], p[1])).or_default().push(i);
            }
        }
        // A max-heap of (count, smallest pair first); an entry whose count is no longer the pair's is stale and skipped.
        let mut heap: BinaryHeap<(u64, Reverse<(u32, u32)>)> = counts.iter().map(|(p, c)| (*c, Reverse(*p))).collect();
        let mut banned: HashSet<(u32, u32)> = HashSet::new();
        let mut learned = Vec::with_capacity(merges);
        while learned.len() < merges {
            let Some((count, Reverse(pair))) = heap.pop() else { break };
            if count == 0 || counts.get(&pair) != Some(&count) || banned.contains(&pair) {
                continue;
            }
            let mut bytes = pieces[pair.0 as usize].clone();
            bytes.extend_from_slice(&pieces[pair.1 as usize]);
            if !taken.insert(bytes.clone()) {
                banned.insert(pair);
                continue;
            }
            let id = 256 + learned.len() as u32;
            let mut delta: HashMap<(u32, u32), i64> = HashMap::new();
            let mut held = holders.remove(&pair).unwrap_or_default();
            held.sort_unstable();
            held.dedup();
            for i in held {
                let (w, c) = &mut words[i];
                if !w.windows(2).any(|p| (p[0], p[1]) == pair) {
                    continue;
                }
                for p in w.windows(2) {
                    *delta.entry((p[0], p[1])).or_insert(0) -= *c as i64;
                }
                merge_in_place(w, pair, id);
                for p in w.windows(2) {
                    *delta.entry((p[0], p[1])).or_insert(0) += *c as i64;
                    if p[0] == id || p[1] == id {
                        holders.entry((p[0], p[1])).or_default().push(i);
                    }
                }
            }
            for (p, d) in delta {
                if d == 0 {
                    continue;
                }
                let c = counts.entry(p).or_insert(0);
                *c = (*c as i64 + d).max(0) as u64;
                heap.push((*c, Reverse(p)));
            }
            pieces.push(bytes);
            learned.push(pair);
        }
        ByteBpe::from_merges(learned)
    }

    /// The vocabulary as a Hugging Face `tokenizer.json` document: a byte-level BPE whose ids are this tokenizer's
    /// (`0..256` the bytes, then the merges in rank order), the GPT-2 split this tokenizer trains and encodes with,
    /// and `specials` as atomic added tokens taking the ids after the merges in the order given. The end-of-text
    /// separator of [`Self::end_of_text`] is not in the file unless it is one of `specials`.
    ///
    /// # Errors
    /// A special token that is empty, named twice, or spelled by a learned piece (it would shadow it); train with
    /// [`Self::train_reserving`] to keep a special token's bytes out of the merges.
    pub fn to_hf_tokenizer_json(&self, specials: &[&str]) -> Result<Value, String> {
        let byte_char = crate::bpe::bytes_to_unicode();
        let spell = |bytes: &[u8]| bytes.iter().map(|b| byte_char[*b as usize]).collect::<String>();
        let mut vocab = serde_json::Map::with_capacity(self.pieces.len());
        for (id, piece) in self.pieces.iter().enumerate() {
            vocab.insert(spell(piece), json!(id));
        }
        let merges: Vec<Value> = self.merges.iter().map(|(a, b)| json!([spell(self.piece(*a)), spell(self.piece(*b))])).collect();
        let mut added = Vec::with_capacity(specials.len());
        for (i, special) in specials.iter().enumerate() {
            if special.is_empty() {
                return Err("a special token is empty".to_string());
            }
            if specials[..i].contains(special) {
                return Err(format!("the special token {special:?} is named twice"));
            }
            if self.pieces.iter().any(|p| p.as_slice() == special.as_bytes()) {
                return Err(format!("the special token {special:?} is also a learned piece of the vocabulary; train with it reserved"));
            }
            added.push(json!({"id": self.pieces.len() + i, "content": special, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
        }
        Ok(json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges},
        }))
    }

    /// The number of learned merges.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
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

/// `text` cut at every occurrence of any of `separators` (the separators dropped); `text` whole when there are none.
fn split_on_all<'t>(text: &'t str, separators: &[&str]) -> Vec<&'t str> {
    let mut parts = vec![text];
    for sep in separators.iter().filter(|s| !s.is_empty()) {
        parts = parts.into_iter().flat_map(|p| p.split(sep)).collect();
    }
    parts
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
            for word in crate::hf_pretok::gpt2_pieces(part) {
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

    /// The textbook trainer, recounting every pair over every word before each merge: the oracle the incremental
    /// trainer is held to.
    fn textbook_merges(text: &str, merges: usize) -> Vec<(u32, u32)> {
        let mut freq: HashMap<String, u64> = HashMap::new();
        for line in text.split_inclusive('\n') {
            for piece in crate::hf_pretok::gpt2_pieces(line) {
                *freq.entry(piece).or_insert(0) += 1;
            }
        }
        let mut words: Vec<(Vec<u32>, u64)> = freq.into_iter().map(|(w, c)| (w.bytes().map(u32::from).collect(), c)).collect();
        let mut out = Vec::new();
        while out.len() < merges {
            let mut pairs: HashMap<(u32, u32), u64> = HashMap::new();
            for (w, c) in &words {
                for p in w.windows(2) {
                    *pairs.entry((p[0], p[1])).or_insert(0) += c;
                }
            }
            let Some((&best, _)) = pairs.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))) else { break };
            let id = 256 + out.len() as u32;
            for (w, _) in words.iter_mut() {
                merge_in_place(w, best, id);
            }
            out.push(best);
        }
        out
    }

    fn corpus() -> String {
        // Repetitive text with many shared prefixes, so merges compete and ties occur.
        let mut rng = crate::rng::Rng::new(11);
        let words = ["the", "then", "there", "cat", "cats", "catalog", "sat", "on", "mat", "matter", "a", "an", "and", "ant", "tool_call", "{\"x\": 1}", "12", "123", "1234"];
        let mut text = String::from(TEXT);
        for i in 0..4000 {
            text.push_str(words[rng.gen_range_inclusive(0, words.len() as i64 - 1) as usize]);
            text.push(if i % 13 == 12 { '\n' } else { ' ' });
        }
        text
    }

    #[test]
    fn the_trainer_learns_the_textbook_merges_in_their_order() {
        let text = corpus();
        for merges in [1, 40, 400] {
            assert_eq!(ByteBpe::train(&text, 257 + merges).merges, textbook_merges(&text, merges), "{merges} merges");
        }
        // A corpus with fewer distinct pairs than merges asked for stops when nothing is left to merge.
        assert_eq!(ByteBpe::train("abab", 300).merges, textbook_merges("abab", 43));
    }

    #[test]
    fn an_exported_vocabulary_encodes_as_the_trainer_does_with_its_special_tokens_atomic() {
        let text = corpus();
        let specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"];
        let bpe = ByteBpe::train_reserving(&text, 600, &specials);
        let json = bpe.to_hf_tokenizer_json(&specials).unwrap();
        let hf = crate::qwen_tokenizer::QwenBpe::from_json_bytes(json.to_string().as_bytes()).unwrap();
        let merges = bpe.merge_count();
        assert!(merges > 100, "the corpus yields merges: {merges}");
        assert_eq!(hf.vocab_size(), 256 + merges + specials.len());
        let sample = format!("{TEXT} the cats sat\n\n  on 1234 mats {{\"x\": 1}}");
        assert_eq!(hf.encode(&sample), bpe.encode(&sample), "the exported file encodes exactly as the trainer");
        for (i, s) in specials.iter().enumerate() {
            assert_eq!(hf.special_id(s), Some((256 + merges + i) as u32), "{s} takes the id after the merges, in declared order");
        }
        let framed = "<|im_start|>user the cat<|im_end|><tool_call>{\"x\": 1}</tool_call>".to_string();
        let ids = hf.encode(&framed);
        assert_eq!(ids[0], hf.special_id("<|im_start|>").unwrap());
        assert_eq!(hf.decode(&ids), framed);
        for bad in [vec!["<a>", "<a>"], vec![""]] {
            assert!(bpe.to_hf_tokenizer_json(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_reserved_special_token_is_never_learned_as_a_piece() {
        // "the" is a frequent word of the corpus: declared special, no merge may spell it.
        let text = corpus();
        let bpe = ByteBpe::train_reserving(&text, 300, &["the"]);
        assert!((256..256 + bpe.merges.len() as u32).all(|id| bpe.piece(id) != b"the"));
        assert!(bpe.to_hf_tokenizer_json(&["the"]).is_ok());
        // Without the reservation the word is learned, and exporting it as a special token is refused, not shadowed.
        let plain = ByteBpe::train(&text, 600);
        assert!(plain.to_hf_tokenizer_json(&["the"]).unwrap_err().contains("the"));
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
