// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaMA's SentencePiece byte-fallback BPE tokenizer - the text front-end for
//! Vicuna-1.5-13B, LLaVA-1.5's decoder half (`crates/llava`).
//!
//! # Why this is a separate type and not a mode on [`crate::clip_bpe::ClipBpe`]
//! or [`crate::bpe::Gpt2Bpe`]
//!
//! The byte-pair *merge loop* - [`crate::bpe::bpe_merge`] - is reused
//! **verbatim**: same greedy lowest-rank adjacent-pair algorithm, same
//! `(left, right) -> rank` table shape. Everything upstream of the merge loop
//! is genuinely different, and none of it is a knob on the existing types:
//!
//! | | GPT-2 / CLIP | LLaMA |
//! |---|---|---|
//! | pre-tokenization | a regex splits text into words BEFORE merging | **none** - the whole normalized string is one merge sequence |
//! | word boundary | leading byte marker (`Ġ`) or trailing `</w>` | `▁` (U+2581) prepended once, then every literal space replaced |
//! | unknown input | byte-to-unicode remaps every byte up front (GPT-2), so nothing is ever "unknown" | initial symbols are Unicode **characters**; a character absent from the vocab falls back to its UTF-8 **bytes**, each looked up as a `<0xXX>` piece |
//! | specials | BOS+EOS both added, symmetric | **BOS only** (`<s>`, id 1) - the reference checkpoint's `TemplateProcessing` appends nothing for a single sequence |
//!
//! Confirmed empirically against `tokenizers`' `Tokenizer.encode` on the real
//! `NousResearch/Llama-2-13b-hf` `tokenizer.json` (Vicuna-1.5 is a LLaMA-2
//! fine-tune with the identical tokenizer): no pre-tokenizer is registered at
//! all, so merges freely cross what would be word boundaries under a
//! pre-tokenized scheme (`"  double  space"` merges the run of `▁` characters
//! together exactly as any other adjacent pair). This is precisely the shape
//! [`crate::bpe::bpe_merge`] already handles: feed it the WHOLE input as one
//! "word" instead of one per pre-token, and its existing
//! get-best-pair-and-merge-all-occurrences loop does the rest.
//!
//! # Byte-fallback, precisely
//!
//! `byte_fallback: true` on the checkpoint's BPE model means: for every
//! Unicode scalar value in the normalized text, if the single-character
//! string is itself a vocab entry (true for ASCII, digits, `▁`, and a few
//! thousand common non-Latin characters actually seen during training - e.g.
//! CJK ideographs are frequently present as single-char pieces), it becomes
//! one initial symbol. Otherwise the character's UTF-8 encoding is split into
//! individual bytes, each represented by the vocab's `<0xXX>` piece (256 of
//! them, ids 3..258, one per byte value). These byte pieces then enter the
//! SAME merge loop as any other symbol - measured against the real merge
//! table, none of the 61 249 trained merges ever combine two `<0xXX>` pieces
//! (multi-byte UTF-8 sequences for unseen characters, e.g. emoji, stay as
//! separate byte tokens), so in practice byte pieces never merge further, but
//! nothing in this implementation special-cases that; it falls out of feeding
//! them through the same generic loop.
//!
//! Because every one of the 256 byte values has a vocab entry and every
//! Unicode scalar value has a well-defined UTF-8 encoding, `<unk>` (id 0) is
//! unreachable for any well-formed input - unlike [`crate::clip_bpe`], this
//! module has no `unk`/`fuse_unk` path to implement.
//!
//! # A stated gap: literal special-token substrings mid-text
//!
//! `<s>`/`</s>`/`<unk>` typed as literal text are, in the real tokenizer,
//! matched atomically **only in some positions** (empirically: `<s>` at the
//! very start of a normalized string is; `</s>` is not, even adjacent to it).
//!
//! This is an inconsistency in the reference implementation's
//! `AddedVocabulary` matcher that is not worth reproducing exactly for
//! SUPIR's one call site (a fixed caption prompt or a `vicuna_v1`-templated
//! conversation, neither of which embeds these literal strings as content).
//! This module tokenizes such substrings as ordinary text in every position,
//! which is the closest internally-consistent behavior, and is documented
//! here rather than silently assumed correct.
//!
//! # Why `tokenizer.json` and not `tokenizer.model`
//!
//! Same reasoning as [`crate::unigram`]: `tokenizer.json` is what
//! `AutoTokenizer`/`LlamaTokenizerFast` actually loads and already expresses
//! the trained BPE model as `(vocab, merges)` in the same shape
//! [`crate::clip_bpe`] reads - no protobuf decoder needed.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::bpe::bpe_merge;

/// `U+2581` "lower one eighth block" - SentencePiece's word-boundary marker.
const METASPACE: char = '\u{2581}';

// ---------------------------------------------------------------------------
// wire format (validated structurally by serde, semantically below)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WireTokenizer {
    #[serde(default)]
    added_tokens: Vec<WireAdded>,
    normalizer: Option<Value>,
    #[serde(default)]
    pre_tokenizer: Option<Value>,
    post_processor: Option<Value>,
    model: WireModel,
}

#[derive(Deserialize)]
struct WireModel {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    dropout: Option<f32>,
    #[serde(default)]
    byte_fallback: bool,
    /// `(left, right)`-space-joined merge rules, in priority order.
    merges: Vec<String>,
    /// piece -> id.
    vocab: HashMap<String, u32>,
}

#[derive(Deserialize)]
struct WireAdded {
    id: u32,
    content: String,
    #[allow(dead_code)]
    #[serde(default)]
    special: bool,
}

// ---------------------------------------------------------------------------

/// LLaMA/Vicuna SentencePiece byte-fallback BPE tokenizer.
pub struct LlamaBpe {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), u32>,
    bos_id: u32,
    eos_id: u32,
    unk_id: u32,
}

impl LlamaBpe {
    /// Build from the raw contents of a `tokenizer.json`.
    // Named `from_tokenizer_json`, not `from_str`: a single-`&str`-argument
    // `-> Result<Self, _>` method named `from_str` is easily confused for
    // (and shadows the intent of) `std::str::FromStr::from_str`.
    pub fn from_tokenizer_json(tokenizer_json: &str) -> Result<LlamaBpe, String> {
        let j: WireTokenizer =
            serde_json::from_str(tokenizer_json).map_err(|e| format!("tokenizer.json: {e}"))?;

        if j.model.kind != "BPE" {
            return Err(format!("tokenizer.json: model.type is {:?}, not BPE", j.model.kind));
        }
        if !j.model.byte_fallback {
            // Every non-byte-fallback code path in this module assumes an
            // unseen character decomposes to bytes rather than becoming
            // `<unk>` - silently treating a false flag as true would give
            // fabricated ids for any character outside the trained vocab.
            return Err("tokenizer.json: byte_fallback is false - this reader only implements the byte-fallback BPE LLaMA/Vicuna ship".into());
        }
        if j.model.dropout.is_some() {
            return Err("tokenizer.json: BPE dropout is set - that makes encoding nondeterministic, which this reader does not support".into());
        }
        validate_normalizer(j.normalizer.as_ref())?;
        if j.pre_tokenizer.as_ref().is_some_and(|v| !v.is_null()) {
            return Err(format!(
                "tokenizer.json: pre_tokenizer is set ({}) - this reader assumes LLaMA's \"no pre-tokenizer, whole string is one BPE sequence\" shape",
                j.pre_tokenizer.unwrap()
            ));
        }

        let mut encoder: HashMap<String, u32> = HashMap::with_capacity(j.model.vocab.len());
        let mut decoder: HashMap<u32, String> = HashMap::with_capacity(j.model.vocab.len());
        for (piece, id) in j.model.vocab {
            decoder.insert(id, piece.clone());
            encoder.insert(piece, id);
        }

        // Every one of the 256 byte values must be encodable, or byte-fallback
        // could silently drop a byte mid-encode.
        for b in 0u32..256 {
            let piece = format!("<0x{b:02X}>");
            if !encoder.contains_key(&piece) {
                return Err(format!("tokenizer.json: missing byte-fallback piece {piece:?} - every byte must be encodable"));
            }
        }

        let mut bpe_ranks: HashMap<(String, String), u32> = HashMap::with_capacity(j.model.merges.len());
        for (rank, line) in j.model.merges.iter().enumerate() {
            let mut it = line.split(' ');
            let left = it.next().ok_or_else(|| format!("tokenizer.json: merge {rank} is empty"))?.to_string();
            let right = it
                .next()
                .ok_or_else(|| format!("tokenizer.json: merge {rank} ({line:?}) has no right half"))?
                .to_string();
            if !encoder.contains_key(&format!("{left}{right}")) {
                return Err(format!("tokenizer.json: merge {left:?}+{right:?} produces a token absent from vocab"));
            }
            bpe_ranks.insert((left, right), rank as u32);
        }

        let added: HashMap<String, u32> = j.added_tokens.into_iter().map(|a| (a.content, a.id)).collect();
        let id_of = |names: &[&str]| -> Result<u32, String> {
            for n in names {
                if let Some(&id) = added.get(*n).or_else(|| encoder.get(*n)) {
                    return Ok(id);
                }
            }
            Err(format!("tokenizer.json: none of {names:?} present in added_tokens/vocab"))
        };
        let bos_id = bos_id_from_post_processor(j.post_processor.as_ref())?.unwrap_or(id_of(&["<s>"])?);
        let eos_id = id_of(&["</s>"])?;
        let unk_id = id_of(&["<unk>"])?;

        Ok(LlamaBpe { encoder, decoder, bpe_ranks, bos_id, eos_id, unk_id })
    }

    /// Build from a checkpoint DIRECTORY holding `tokenizer.json`.
    pub fn from_dir(dir: &std::path::Path) -> std::io::Result<LlamaBpe> {
        let raw = std::fs::read_to_string(dir.join("tokenizer.json"))?;
        LlamaBpe::from_tokenizer_json(&raw).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {e}", dir.display())))
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Content ids only - no BOS. `normalize` (prepend `▁`, then map every
    /// literal space to `▁`) then the single BPE merge pass over the WHOLE
    /// string, byte-falling-back any character absent from the vocab.
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            // The reference `Prepend` normalizer is a no-op on empty input -
            // verified against `tokenizers`' own `normalize_str("")`; a
            // literal implementation that always prepends would instead emit
            // one lone `▁` token for an empty string.
            return Vec::new();
        }
        let mut normalized = String::with_capacity(text.len() + 4);
        normalized.push(METASPACE);
        for c in text.chars() {
            normalized.push(if c == ' ' { METASPACE } else { c });
        }

        let mut symbols: Vec<String> = Vec::with_capacity(normalized.len());
        let mut byte_buf = [0u8; 4];
        for c in normalized.chars() {
            let s = c.to_string();
            if self.encoder.contains_key(&s) {
                symbols.push(s);
            } else {
                for &b in c.encode_utf8(&mut byte_buf).as_bytes() {
                    symbols.push(format!("<0x{b:02X}>"));
                }
            }
        }

        bpe_merge(&self.bpe_ranks, symbols)
            .into_iter()
            .map(|s| {
                *self
                    .encoder
                    .get(&s)
                    .unwrap_or_else(|| panic!("llama BPE produced {s:?}, absent from the vocab - every base symbol and every merge result is a vocab entry by construction"))
            })
            .collect()
    }

    /// The call site LLaVA/Vicuna actually makes: `[bos] + encode_raw(text)`,
    /// matching the reference checkpoint's `TemplateProcessing` (BOS only, no
    /// EOS on a single sequence).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        ids.push(self.bos_id);
        ids.extend(self.encode_raw(text));
        ids
    }

    /// Inverse of [`Self::encode`]/[`Self::encode_raw`]: BOS/EOS/`<unk>` are
    /// dropped (matching `tokenizer.decode(ids, skip_special_tokens=True)`),
    /// `▁` renders as a space, `<0xXX>` runs are fused back into their
    /// original UTF-8 bytes, and exactly one leading space is stripped (the
    /// `▁` [`Self::encode_raw`] always prepends).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        let mut byte_buf = [0u8; 4];
        for &id in ids {
            if id == self.bos_id || id == self.eos_id || id == self.unk_id {
                continue;
            }
            let Some(piece) = self.decoder.get(&id) else { continue };
            if let Some(b) = byte_fallback_value(piece) {
                bytes.push(b);
                continue;
            }
            for c in piece.chars() {
                if c == METASPACE {
                    bytes.push(b' ');
                } else {
                    bytes.extend_from_slice(c.encode_utf8(&mut byte_buf).as_bytes());
                }
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
    }
}

/// Parse a `<0xXX>` byte-fallback piece into its byte value, if `piece` is one.
fn byte_fallback_value(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

/// The reference normalizer is exactly `Sequence[Prepend("▁"),
/// Replace(" " -> "▁")]` - anything else is rejected by name, since a
/// silently-different normalizer changes ids for every input.
fn validate_normalizer(v: Option<&Value>) -> Result<(), String> {
    let v = v.ok_or("tokenizer.json: no normalizer")?;
    if v["type"] != "Sequence" {
        return Err(format!("tokenizer.json: normalizer.type is {:?}, not Sequence", v["type"]));
    }
    let steps = v["normalizers"].as_array().ok_or("tokenizer.json: normalizer.normalizers is not an array")?;
    if steps.len() != 2 {
        return Err(format!("tokenizer.json: normalizer has {} steps, expected 2 (Prepend, Replace)", steps.len()));
    }
    if steps[0]["type"] != "Prepend" || steps[0]["prepend"] != METASPACE.to_string() {
        return Err(format!("tokenizer.json: normalizer step 0 is {}, expected Prepend(\"▁\")", steps[0]));
    }
    if steps[1]["type"] != "Replace" || steps[1]["pattern"]["String"] != " " || steps[1]["content"] != METASPACE.to_string() {
        return Err(format!("tokenizer.json: normalizer step 1 is {}, expected Replace(\" \" -> \"▁\")", steps[1]));
    }
    Ok(())
}

/// The reference `post_processor` is `TemplateProcessing` with a single
/// template of exactly `[SpecialToken(bos), Sequence]` - BOS only, no EOS
/// (see the module doc). Returns the BOS id the template names, so a
/// checkpoint using a differently-spelled BOS content (never observed, but
/// not assumed) still resolves correctly.
fn bos_id_from_post_processor(v: Option<&Value>) -> Result<Option<u32>, String> {
    let Some(v) = v else { return Ok(None) };
    if v["type"] != "TemplateProcessing" {
        return Err(format!("tokenizer.json: post_processor.type is {:?}, not TemplateProcessing", v["type"]));
    }
    let single = v["single"].as_array().ok_or("tokenizer.json: post_processor.single is not an array")?;
    if single.len() != 2 || single[1].get("Sequence").is_none() {
        return Err("tokenizer.json: post_processor.single is not [SpecialToken, Sequence]".into());
    }
    let name = single[0]["SpecialToken"]["id"].as_str().ok_or("tokenizer.json: post_processor.single[0] is not a SpecialToken")?;
    let ids = v["special_tokens"][name]["ids"].as_array().ok_or_else(|| format!("tokenizer.json: post_processor special token {name:?} has no ids"))?;
    match ids.as_slice() {
        [id] => Ok(Some(id.as_u64().ok_or("tokenizer.json: non-integer BOS id")? as u32)),
        other => Err(format!("tokenizer.json: post_processor BOS template has {} ids, expected 1", other.len())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature tokenizer.json with LLaMA's exact shape: BPE + byte
    /// fallback, the Prepend+Replace normalizer, no pre_tokenizer, a
    /// BOS-only TemplateProcessing, and just enough vocab/merges to exercise
    /// a merge, a lone space, and a byte-fallback character.
    fn tiny_json() -> String {
        let mut vocab = serde_json::Map::new();
        vocab.insert("<unk>".into(), 0.into());
        vocab.insert("<s>".into(), 1.into());
        vocab.insert("</s>".into(), 2.into());
        for b in 0u32..256 {
            vocab.insert(format!("<0x{b:02X}>"), (3 + b).into());
        }
        let base = 259u32;
        let pieces = ["▁", "a", "b", "▁a", "ab", "▁ab"];
        for (i, p) in pieces.iter().enumerate() {
            vocab.insert(p.to_string(), (base + i as u32).into());
        }
        serde_json::json!({
            "added_tokens": [
                {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": true, "special": true},
                {"id": 1, "content": "<s>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": true, "special": true},
                {"id": 2, "content": "</s>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": true, "special": true}
            ],
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": "\u{2581}"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}]},
            "pre_tokenizer": null,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                           {"Sequence": {"id": "A", "type_id": 0}}],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}},
            "model": {
                "type": "BPE",
                "byte_fallback": true,
                // "▁a"+"b" -> "▁ab" beats "▁"+"a"+"b" one merge at a time.
                "merges": ["\u{2581} a", "a b", "\u{2581}a b"],
                "vocab": vocab
            }
        })
        .to_string()
    }

    fn tok() -> LlamaBpe {
        LlamaBpe::from_tokenizer_json(&tiny_json()).unwrap()
    }

    #[test]
    fn specials_resolve_by_content() {
        let t = tok();
        assert_eq!(t.bos_id(), 1);
        assert_eq!(t.eos_id(), 2);
        assert_eq!(t.unk_id(), 0);
    }

    #[test]
    fn merge_prefers_the_lowest_rank_pair_first() {
        let t = tok();
        // "ab" -> "▁ab" (normalize) -> symbols [▁, a, b] -> merges "▁ a"
        // (rank 0) before "a b" (rank 1) -> [▁a, b] -> then "▁a b" (rank 2)
        // -> [▁ab].
        assert_eq!(t.encode_raw("ab"), vec![259 + 5]); // "▁ab"
    }

    #[test]
    fn empty_input_normalizes_to_nothing() {
        let t = tok();
        assert_eq!(t.encode_raw(""), Vec::<u32>::new());
        assert_eq!(t.encode(""), vec![t.bos_id()]);
    }

    #[test]
    fn byte_fallback_for_a_character_outside_the_vocab() {
        let t = tok();
        // 'z' has no single-char vocab entry in this tiny fixture, so it
        // byte-falls-back to its one ASCII byte, id 3 + b'z'; the leading
        // "▁" (normalize always prepends one on non-empty input) stays
        // separate since no merge rule joins it to a byte-fallback piece.
        let ids = t.encode_raw("z");
        assert_eq!(ids, vec![259, 3 + u32::from(b'z')]);
        assert_eq!(t.decode(&ids), "z");
    }

    #[test]
    fn decode_strips_one_leading_space_and_skips_specials() {
        let t = tok();
        let ids = t.encode("ab"); // [bos, ▁ab]
        assert_eq!(t.decode(&ids), "ab");
        assert_eq!(t.decode(&[t.bos_id(), t.eos_id(), t.unk_id()]), "");
    }

    #[test]
    fn unsupported_configurations_error_by_name() {
        let bad = |f: &dyn Fn(&mut Value)| -> String {
            let mut j: Value = serde_json::from_str(&tiny_json()).unwrap();
            f(&mut j);
            match LlamaBpe::from_tokenizer_json(&j.to_string()) {
                Err(e) => e,
                Ok(_) => panic!("expected a rejection"),
            }
        };
        assert!(bad(&|j| j["model"]["type"] = "Unigram".into()).contains("BPE"));
        assert!(bad(&|j| j["model"]["byte_fallback"] = false.into()).contains("byte_fallback"));
        assert!(bad(&|j| j["model"]["dropout"] = 0.1.into()).contains("dropout"));
        assert!(bad(&|j| j["pre_tokenizer"] = serde_json::json!({"type": "Whitespace"})).contains("pre_tokenizer"));
        assert!(bad(&|j| j["normalizer"]["normalizers"][0]["type"] = "Replace".into()).contains("Prepend"));
    }
}
