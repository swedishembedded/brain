// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Byte-level BPE tokenizer for HF `tokenizer.json` checkpoints (Qwen,
//! LFM2.5, DeepSeek, Llama 3, ...).
//!
//! Same byte-level BPE family as [`crate::bpe::Gpt2Bpe`] - it reuses the byte
//! <-> unicode map and the merge loop ([`crate::bpe::bpe_merge`]) - with the
//! checkpoint's vocabulary, merges, and its own declared normalizer and
//! pre-tokenizer ([`crate::hf_pretok`]): the split a file declares is the one
//! applied, and a step brain does not implement is refused at load. A GGUF
//! names its pre-tokenizer instead of declaring it; [`crate::hf_pretok::for_gguf`]
//! supplies the split llama.cpp runs for that name. `ignore_merges` (Llama 3)
//! encodes a piece already in the vocabulary as that one token.
//!
//! Special/added tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, …) are
//! matched as atomic units *before* normalization and BPE. A
//! `TemplateProcessing` post-processor (LFM2.5 prepends `<|startoftext|>`) is
//! captured as [`QwenBpe::template_prefix`] - callers that want HF-equivalent
//! single-sequence encodings prepend it; `encode()` itself stays
//! template-free. `vocab_size()` reports the model vocab used to size the
//! embedding table.

use std::collections::HashMap;

use crate::bpe::{bpe_merge, bytes_to_unicode};
use crate::hf_pretok::{Normalizer, PreTokenizer};
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
    /// The file's normalizer, applied to the text between special tokens.
    normalizer: Option<Normalizer>,
    /// The file's pre-tokenizer: where BPE may not merge across.
    pre: PreTokenizer,
    /// Encode a piece that is itself a vocabulary entry as that one token,
    /// without running the merges (`model.ignore_merges`).
    ignore_merges: bool,
    /// Special-token ids a `TemplateProcessing` post-processor prepends to a
    /// single-sequence encoding (empty when the file declares none).
    template_prefix: Vec<u32>,
}

impl QwenBpe {
    /// Build from a `tokenizer.json` file path.
    pub fn from_file(path: &str) -> Result<QwenBpe, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_json_bytes(&bytes)
    }

    /// Build from a checkpoint DIRECTORY: `tokenizer.json` when present, else
    /// the split `vocab.json` + `merges.txt` (+ special-token source) layout
    /// some Qwen2-family repos ship (FastVLM). Same byte-level BPE either way -
    /// the split files are re-assembled into the unified shape and parsed by
    /// the ONE existing parser, so the formats cannot drift apart.
    ///
    /// The special-token source is checked in order: `added_tokens.json`
    /// (`{content: id}`, the older convention), else `tokenizer_config.json`'s
    /// `added_tokens_decoder` (`{id: {content, special, ...}}`, what a
    /// checkpoint with no standalone `added_tokens.json` at all still
    /// carries - confirmed against a real Qwen3-Omni-30B-A3B-Instruct
    /// checkpoint on disk, whose directory has neither `tokenizer.json` nor
    /// `added_tokens.json`, only `vocab.json`/`merges.txt`/
    /// `tokenizer_config.json`). Skipping this fallback silently drops EVERY
    /// special token (`<|im_end|>`, `<|endoftext|>`, …) for such a
    /// checkpoint: `vocab.json` itself does not contain them (they are
    /// allocated past the base vocab), so a caller's `special_id("<|im_end|>")`
    /// would return `None` - no EOS ever matches, greedy generation runs to
    /// `max_new_tokens` every time, and the chat template's own
    /// `<|im_start|>`/`<|im_end|>` framing text gets BPE'd byte-by-byte
    /// instead of encoded as the single token ids the checkpoint was trained
    /// on. A checkpoint with none of the three sources still loads (`added`
    /// stays empty, matching this function's prior behavior) since a base
    /// tokenizer with no special tokens at all is a real, valid case.
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
        let from_added_tokens_json = std::fs::read(format!("{dir}/added_tokens.json")).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()).map(|m| {
            // added_tokens.json is { content: id }; unified wants records.
            let arr: Vec<serde_json::Value> =
                m.as_object().map(|o| o.iter().map(|(c, id)| serde_json::json!({"content": c, "id": id})).collect()).unwrap_or_default();
            serde_json::Value::Array(arr)
        });
        let added = from_added_tokens_json.or_else(|| Self::added_tokens_from_tokenizer_config(dir)).unwrap_or(serde_json::Value::Array(Vec::new()));
        // The split-file layout is Qwen2's (`Qwen2TokenizerFast`), whose
        // pre-tokenizer is its own pattern: declare it, since these files do
        // not.
        let unified = serde_json::json!({
            "model": { "vocab": vocab, "merges": merges },
            "added_tokens": added,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": crate::hf_pretok::QWEN2_PATTERN}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
            ]},
        });
        Self::from_json_bytes(unified.to_string().as_bytes())
    }

    /// [`Self::from_dir`]'s fallback special-token source when
    /// `added_tokens.json` is absent: `<dir>/tokenizer_config.json`'s
    /// `added_tokens_decoder` (`{"151645": {"content": "<|im_end|>",
    /// "special": true, ...}, ...}`), converted to the same `[{content, id}]`
    /// record shape [`Self::from_json_bytes`] reads. `None` (never an error)
    /// when the file is absent, unparseable, or has no such field - the
    /// caller's own `unwrap_or(empty)` is the real fallback.
    fn added_tokens_from_tokenizer_config(dir: &str) -> Option<serde_json::Value> {
        let text = std::fs::read_to_string(format!("{dir}/tokenizer_config.json")).ok()?;
        let cfg: serde_json::Value = serde_json::from_str(&text).ok()?;
        let decoder = cfg.get("added_tokens_decoder")?.as_object()?;
        let arr: Vec<serde_json::Value> = decoder
            .iter()
            .filter_map(|(id_str, rec)| {
                let id: u64 = id_str.parse().ok()?;
                let content = rec.get("content")?.as_str()?;
                Some(serde_json::json!({"content": content, "id": id}))
            })
            .collect();
        Some(serde_json::Value::Array(arr))
    }

    /// Build from a GGUF's embedded `tokenizer.ggml.*` KV (see
    /// [`checkpoint::gguf::GgufTokenizer`]). Supports the GPT-2-style byte-level
    /// BPE (`model == "gpt2"`) a Qwen3 GGUF ships - the same family as the
    /// `tokenizer.json` path, so it reuses this struct's vocab/merge/special
    /// representation rather than forking a second BPE.
    ///
    /// Mapping: `tokens[id]` is the token text (already in the GPT-2 byte-encoded
    /// domain, i.e. HF `vocab.json` keys) → `encoder`/`decoder` keyed by index;
    /// `merges` are the ranked `"a b"` pairs → `bpe_ranks`; tokens whose
    /// `token_type` is CONTROL(3) or USER_DEFINED(4) - plus the declared
    /// bos/eos/unk/pad ids - become atomic `specials`. A GGUF carries the
    /// pre-tokenizer's NAME (`tokenizer.ggml.pre`), never its regex: the split
    /// comes from [`crate::hf_pretok::for_gguf`], and a name it does not know
    /// is an error.
    ///
    /// Non-gpt2 schemes (llama/bert/…) return a clear `Err` (a documented
    /// follow-up - each needs its own tokenization model).
    pub fn from_gguf(tok: &checkpoint::gguf::GgufTokenizer) -> Result<QwenBpe, String> {
        if tok.model != "gpt2" {
            return Err(format!("gguf tokenizer model '{}' not supported", tok.model));
        }
        if tok.tokens.is_empty() {
            return Err("gguf tokenizer: empty tokens array".to_string());
        }

        // vocab: index is the id (GGUF stores tokens in id order).
        let mut encoder = HashMap::with_capacity(tok.tokens.len());
        let mut decoder = HashMap::with_capacity(tok.tokens.len());
        for (id, t) in tok.tokens.iter().enumerate() {
            let id = id as u32;
            encoder.insert(t.clone(), id);
            decoder.insert(id, t.clone());
        }

        // merges: "a b" strings, ranked by their position in the array.
        let mut bpe_ranks = HashMap::with_capacity(tok.merges.len());
        for (rank, m) in tok.merges.iter().enumerate() {
            let mut it = m.splitn(2, ' ');
            let l = it.next().unwrap().to_string();
            let r = it.next().ok_or_else(|| format!("gguf merge {rank:?}: missing space in {m:?}"))?.to_string();
            bpe_ranks.insert((l, r), rank as u32);
        }

        // Specials: control / user-defined tokens are matched atomically before
        // BPE (their text is literal, not byte-encoded). CONTROL=3, USER_DEFINED=4.
        let mut specials: Vec<(String, u32)> = Vec::new();
        for (id, ty) in tok.token_types.iter().enumerate() {
            if (*ty == 3 || *ty == 4) && id < tok.tokens.len() {
                specials.push((tok.tokens[id].clone(), id as u32));
            }
        }
        // Ensure the declared bos/eos/unk/pad tokens are matchable even if their
        // token_type was NORMAL / the token_type array was absent.
        for id in [tok.bos, tok.eos, tok.unk, tok.pad].into_iter().flatten() {
            if let Some(t) = tok.tokens.get(id as usize) {
                if !specials.iter().any(|(_, sid)| *sid == id) {
                    specials.push((t.clone(), id));
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

        let vocab_size = decoder.keys().copied().max().map(|m| m as usize + 1).unwrap_or(0);
        let (pre, ignore_merges) = crate::hf_pretok::for_gguf(tok.pre.as_deref())?;

        Ok(QwenBpe {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            specials,
            vocab_size,
            normalizer: None,
            pre,
            ignore_merges,
            // Qwen declares no single-sequence template prefix.
            template_prefix: Vec::new(),
        })
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
                    // `.get`, never `arr[0]`: tokenizer.json is an untrusted
                    // user file, and a short merge entry must be an Err, not
                    // a slice-index panic.
                    (
                        arr.first().and_then(|v| v.as_str()).ok_or("merge pair")?.to_string(),
                        arr.get(1).and_then(|v| v.as_str()).ok_or("merge pair")?.to_string(),
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

        let normalizer = Normalizer::from_json(&j["normalizer"])?;
        let pre = PreTokenizer::from_json(&j["pre_tokenizer"])?;
        let ignore_merges = model["ignore_merges"].as_bool().unwrap_or(false);
        let template_prefix = template_prefix_from(&j["post_processor"], &encoder);

        Ok(QwenBpe {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            specials,
            vocab_size,
            normalizer,
            pre,
            ignore_merges,
            template_prefix,
        })
    }

    /// Id of a special/added token by literal content (e.g. `"<|mask|>"`).
    pub fn special_id(&self, content: &str) -> Option<u32> {
        self.specials.iter().find(|(c, _)| c == content).map(|(_, id)| *id)
    }

    /// Append new special/added tokens past the current vocabulary, in the
    /// given order - each gets the next sequential id (`vocab_size`,
    /// `vocab_size + 1`, ...). Mirrors HF `PreTrainedTokenizer.add_special_
    /// tokens`'s behavior: a content already present keeps its existing id
    /// (not re-added, not re-ordered). Needed for checkpoints (Florence-2)
    /// whose custom `*Processor.__init__` adds task-prompt/location tokens
    /// programmatically at load time rather than shipping them in
    /// `tokenizer.json`'s `added_tokens` - see `florence2::tokenizer` for
    /// the caller that needs this.
    pub fn add_special_tokens(&mut self, contents: &[&str]) {
        for &content in contents {
            if self.encoder.contains_key(content) {
                continue;
            }
            let id = self.vocab_size as u32;
            self.encoder.insert(content.to_string(), id);
            self.decoder.insert(id, content.to_string());
            self.specials.push((content.to_string(), id));
            self.vocab_size += 1;
        }
        // Longest content first, same invariant `from_json_bytes`/`from_gguf`
        // establish - a longer added token must match before a shorter one
        // that happens to be one of its prefixes.
        self.specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    }

    /// Number of distinct ids this tokenizer can produce (max id + 1,
    /// covering added tokens beyond the base vocab table) - the embedding /
    /// LM-head row count a checkpoint built against this tokenizer needs.
    /// A synthetic test checkpoint sizing its vocab to a hardcoded
    /// Qwen3-era constant (151936) panics on the first decode of a prompt
    /// encoded with the larger Qwen3.8 table (248k ids) otherwise.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Special-token ids the checkpoint's post-processor prepends to a single
    /// sequence (LFM2.5: `[<|startoftext|>]`; Qwen: empty). `encode()` does not
    /// apply this - callers wanting HF-equivalent encodings prepend it.
    pub fn template_prefix(&self) -> &[u32] {
        &self.template_prefix
    }

    /// Encode one pre-token (already special-free) into ids via byte-level BPE.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let chars: Vec<String> = piece
            .bytes()
            .map(|b| self.byte_encoder[b as usize].to_string())
            .collect();
        if self.ignore_merges {
            if let Some(&id) = self.encoder.get(&chars.concat()) {
                out.push(id);
                return;
            }
        }
        for sub in bpe_merge(&self.bpe_ranks, chars) {
            // A miss cannot occur for a complete byte-level vocab; drop it rather
            // than emitting an UNK the decoder has no way to invert.
            if let Some(&id) = self.encoder.get(&sub) {
                out.push(id);
            }
        }
    }

    /// One ChatML turn: `<|im_start|>{role}\n{content}<|im_end|>\n`. The
    /// single building block `apply_chat_template` folds over -- exposed so a
    /// per-message-boundary encoder (multi-turn SFT with per-message loss
    /// masking, `data::chat::ChatSample::encode`) renders byte-identically to
    /// this batch path rather than a parallel reimplementation that could
    /// drift from it.
    pub fn frame_message(&self, role: &str, content: &str) -> String {
        format!("<|im_start|>{role}\n{content}<|im_end|>\n")
    }

    /// Render the Qwen ChatML template for a single-turn (or multi-turn) chat,
    /// optionally appending the assistant generation prompt. Plain string
    /// assembly (no Jinja). Returns the prompt text; encode it for inference.
    pub fn apply_chat_template(&self, msgs: &[(&str, &str)], add_generation_prompt: bool) -> String {
        let mut s = String::new();
        for (role, content) in msgs {
            s.push_str(&self.frame_message(role, content));
        }
        if add_generation_prompt {
            s.push_str("<|im_start|>assistant\n");
        }
        s
    }

    /// The Qwen3 template with `enable_thinking=false`: the generation prompt
    /// ends with an empty `<think>` block. This is the exact rendering FLUX.2
    /// Klein feeds its text encoder - the suffix is part of the conditioning
    /// and must not be dropped.
    pub fn apply_chat_template_no_think(&self, msgs: &[(&str, &str)]) -> String {
        let mut s = self.apply_chat_template(msgs, true);
        s.push_str("<think>\n\n</think>\n\n");
        s
    }
}

impl Tokenizer for QwenBpe {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_with_specials(text, &mut out);
        out
    }

    /// The `ByteLevel` decoder's rule, for every token alike: one whose
    /// chars are all byte-map chars decodes through the map, any other is its
    /// own UTF-8 text. A special token spelled with a space or fullwidth bar
    /// therefore reads back literally, while an added token that is itself a
    /// byte-map char (coder v1 adds `ü` as one) decodes to that one byte -
    /// exactly as `tokenizers` does.
    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for tok in ids.iter().filter_map(|id| self.decoder.get(id)) {
            let mapped: Option<Vec<u8>> = tok.chars().map(|c| self.byte_decoder.get(&c).copied()).collect();
            match mapped {
                Some(b) => bytes.extend(b),
                None => bytes.extend_from_slice(tok.as_bytes()),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl QwenBpe {
    /// Normalize and pre-tokenize one already-special-free span.
    fn pretokenize(&self, text: &str) -> Vec<String> {
        match &self.normalizer {
            Some(n) => self.pre.split(&n.apply(text)),
            None => self.pre.split(text),
        }
    }

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
                    for piece in self.pretokenize(&rest[..pos]) {
                        self.encode_piece(&piece, out);
                    }
                }
                out.push(id);
                rest = &rest[pos + content.len()..];
                continue 'outer;
            }
            // No specials left: BPE the remainder.
            for piece in self.pretokenize(rest) {
                self.encode_piece(&piece, out);
            }
            break;
        }
    }
}

/// Special-token ids a `TemplateProcessing` post-processor places before the
/// `A` sequence in its `single` template (searched recursively - the processor
/// may sit inside a `Sequence`). Ids resolve through the vocab/added tokens.
fn template_prefix_from(post: &serde_json::Value, encoder: &HashMap<String, u32>) -> Vec<u32> {
    fn find_single(v: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
        if v["type"] == "TemplateProcessing" {
            return v["single"].as_array();
        }
        if let Some(arr) = v["processors"].as_array() {
            return arr.iter().find_map(find_single);
        }
        None
    }
    let mut prefix = Vec::new();
    if let Some(single) = find_single(post) {
        for item in single {
            if item.get("Sequence").is_some() {
                break; // only tokens before the `A` sequence are a prefix
            }
            if let Some(content) = item["SpecialToken"]["id"].as_str() {
                if let Some(&id) = encoder.get(content) {
                    prefix.push(id);
                }
            }
        }
    }
    prefix
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
            brain_testutil::skip("QWEN_TOKENIZER unset");
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

    /// A minimal, self-contained tokenizer (no external file/env var
    /// needed) - just enough base vocab to prove `add_special_tokens`
    /// behavior in isolation: sequential ids past the base table, existing
    /// content keeps its id, and a newly-added special is matched atomically
    /// by `encode()` before BPE, the same as a checkpoint-declared one.
    fn minimal_tok() -> QwenBpe {
        let json = serde_json::json!({
            "model": { "vocab": {"a": 0u32, "b": 1u32}, "merges": [] },
            "added_tokens": [{"content": "<pad>", "id": 2u32}],
        });
        QwenBpe::from_json_bytes(json.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn add_special_tokens_assigns_sequential_ids_past_vocab_size() {
        let mut t = minimal_tok();
        assert_eq!(t.vocab_size(), 3); // ids 0,1,2 used above
        t.add_special_tokens(&["<loc_0>", "<loc_1>", "<od>"]);
        assert_eq!(t.special_id("<loc_0>"), Some(3));
        assert_eq!(t.special_id("<loc_1>"), Some(4));
        assert_eq!(t.special_id("<od>"), Some(5));
        assert_eq!(t.vocab_size(), 6);
    }

    #[test]
    fn add_special_tokens_is_idempotent_for_existing_content() {
        let mut t = minimal_tok();
        let pad_id_before = t.special_id("<pad>");
        t.add_special_tokens(&["<pad>", "<new>"]);
        assert_eq!(t.special_id("<pad>"), pad_id_before, "existing token must keep its id");
        assert_eq!(t.special_id("<new>"), Some(3));
        assert_eq!(t.vocab_size(), 4, "only the genuinely new token grows vocab_size");
    }

    #[test]
    fn newly_added_special_is_matched_atomically_by_encode() {
        let mut t = minimal_tok();
        t.add_special_tokens(&["<loc_5>"]);
        let id = t.special_id("<loc_5>").unwrap();
        assert_eq!(t.encode("<loc_5>"), vec![id]);
    }

    fn lfm_tok() -> Option<QwenBpe> {
        let path = std::env::var("LFM_TOKENIZER").ok()?;
        QwenBpe::from_file(&path).ok()
    }

    #[test]
    fn lfm_pinned_reference_vectors() {
        let Some(t) = lfm_tok() else {
            brain_testutil::skip("LFM_TOKENIZER unset");
            return;
        };
        // Ground truth from HF `tokenizers` on LFM2.5-Encoder-230M/tokenizer.json
        // (add_special_tokens=False). Digit runs group up to 3 (`\p{N}{1,3}`).
        assert_eq!(t.encode("The capital of France is"), vec![1098, 5706, 803, 4481, 856]);
        assert_eq!(t.encode("Hello, world"), vec![36309, 521, 2031]);
        assert_eq!(t.encode("12345"), vec![10293, 2637]); // "123"+"45"
        assert_eq!(t.encode("1234"), vec![10293, 529]); // "123"+"4"
        assert_eq!(t.encode("3.14159"), vec![528, 523, 13888, 5599]);
        assert_eq!(
            t.encode("year 2026, price $1299.99"),
            vec![30721, 730, 1718, 531, 521, 7264, 1058, 12936, 534, 523, 2962]
        );
        // Arabic-Indic digits are \p{N} too.
        assert_eq!(t.encode("١٢٣٤٥"), vec![659, 604, 659, 605, 659, 606, 659, 607, 659, 608]);
        assert_eq!(t.encode("  spaced"), vec![730, 56551]);
        assert_eq!(t.encode("def main():\n\tpass"), vec![3663, 2120, 32711, 707, 9859]);
        assert_eq!(t.encode("über café naïve"), vec![13168, 35499, 2116, 6838, 1124]);
        assert_eq!(t.encode("日本語のテキスト"), vec![62506, 1084, 4374, 4459, 7133]);
        for s in ["The capital of France is", "year 2026, price $1299.99", "über café naïve"] {
            assert_eq!(t.decode(&t.encode(s)), s, "roundtrip {s:?}");
        }
    }

    #[test]
    fn lfm_specials_and_template() {
        let Some(t) = lfm_tok() else {
            return;
        };
        assert_eq!(t.special_id("<|mask|>"), Some(16));
        assert_eq!(t.special_id("<|pad|>"), Some(0));
        assert_eq!(t.special_id("<|startoftext|>"), Some(1));
        assert_eq!(t.special_id("<|im_end|>"), Some(7));
        // TemplateProcessing single = [<|startoftext|>, A] -> prefix [1].
        assert_eq!(t.template_prefix(), &[1]);
        // Specials are atomic mid-string.
        assert_eq!(t.encode("Paris<|mask|>Lyon"), vec![41677, 16, 553, 31862]);
    }

    /// A tiny synthetic gpt2 GGUF tokenizer: build `QwenBpe::from_gguf` directly
    /// from a hand-made [`GgufTokenizer`] and assert encode/decode round-trip and
    /// special-id resolution - no GGUF bytes, no files, no GPU.
    #[test]
    fn from_gguf_gpt2_roundtrip_and_specials() {
        use checkpoint::gguf::GgufTokenizer;
        // ids: 0..2 control specials; 3,4 single-byte tokens; 5 the "hi" merge.
        let gt = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen2".into()),
            tokens: vec![
                "<|endoftext|>".into(),
                "<|im_start|>".into(),
                "<|im_end|>".into(),
                "h".into(),
                "i".into(),
                "hi".into(),
            ],
            merges: vec!["h i".into()],
            token_types: vec![3, 3, 3, 1, 1, 1],
            bos: Some(0),
            eos: Some(2),
            unk: None,
            pad: None,
            ..Default::default()
        };
        let t = QwenBpe::from_gguf(&gt).unwrap();

        assert_eq!(t.vocab_size(), 6);
        // The merge fires: "hi" is one token; "hii" is "hi" + "i".
        assert_eq!(t.encode("hi"), vec![5]);
        assert_eq!(t.encode("hii"), vec![5, 4]);
        // Specials resolve and match atomically mid-string.
        assert_eq!(t.special_id("<|im_end|>"), Some(2));
        assert_eq!(t.special_id("<|im_start|>"), Some(1));
        assert_eq!(t.encode("<|im_start|>hi<|im_end|>"), vec![1, 5, 2]);
        // Round-trips (plain + with specials).
        for s in ["hi", "hii", "<|im_start|>hi<|im_end|>"] {
            assert_eq!(t.decode(&t.encode(s)), s, "roundtrip {s:?}");
        }

        // A non-gpt2 scheme is a clear, deferred error.
        let llama = GgufTokenizer { model: "llama".into(), ..gt.clone() };
        let err = match QwenBpe::from_gguf(&llama) {
            Ok(_) => panic!("expected non-gpt2 scheme to be rejected"),
            Err(e) => e,
        };
        assert!(err.contains("'llama' not supported"), "{err}");
    }

    /// The Qwen2 split takes `\p{N}` exactly: a Roman numeral (Nl) or a
    /// fraction (No) is a digit-family piece and does not glue onto the
    /// letter run after it. ASCII and ordinary text are asserted with it.
    #[test]
    fn roman_numerals_take_the_digit_branch() {
        let qwen = PreTokenizer::isolated_splits(&[crate::hf_pretok::QWEN2_PATTERN]).unwrap();
        assert_eq!(qwen.split("\u{216B}xyz"), vec!["\u{216B}", "xyz"]);
        assert_eq!(qwen.split("\u{2160}\u{00BD}"), vec!["\u{2160}", "\u{00BD}"]);
        let (llama3, _) = crate::hf_pretok::for_gguf(Some("llama-bpe")).unwrap();
        assert_eq!(llama3.split("\u{2160}\u{2160}\u{2160}\u{2160}"), vec!["\u{2160}\u{2160}\u{2160}", "\u{2160}"]);
        assert_eq!(qwen.split("Hello, world"), vec!["Hello", ",", " world"]);
        assert_eq!(qwen.split("a1b"), vec!["a", "1", "b"]);
        assert_eq!(qwen.split("über café"), vec!["über", " café"]);
    }

    /// A bare `ByteLevel` pre-tokenizer (Laya's and ModernBERT's own shape)
    /// splits by GPT-2's pattern, pinned against the REAL `tokenizers`
    /// library's `pre_tokenizer.pre_tokenize_str` on Laya's actual
    /// `tokenizer.json`. Every case differs from the Qwen2 split on the same
    /// input: contractions are case-sensitive, digit runs are uncapped, and
    /// only a literal space may prefix a letter/digit/symbol run.
    #[test]
    fn a_bare_byte_level_pre_tokenizer_splits_like_the_real_tokenizers_library() {
        let bare = PreTokenizer::from_json(&serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true})).unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("don't", &["don", "'t"]),
            ("DON'T", &["DON", "'", "T"]), // case-sensitive: no 'T contraction
            ("Don'T", &["Don", "'", "T"]),
            ("can't", &["can", "'t"]),
            ("CAN'T", &["CAN", "'", "T"]),
            ("I'm", &["I", "'m"]),
            ("I'M", &["I", "'", "M"]),
            ("it's", &["it", "'s"]),
            ("IT'S", &["IT", "'", "S"]),
            ("1234567890", &["1234567890"]), // uncapped, unlike cl100k's K=1/3
            ("12345", &["12345"]),
            ("hello world", &["hello", " world"]),
            ("multiple    spaces", &["multiple", "   ", " spaces"]),
            ("newline\ntest", &["newline", "\n", "test"]),
            ("cafe naive uber", &["cafe", " naive", " uber"]),
            ("(hello", &["(", "hello"]), // no non-space letter-prefix absorption
            ("3(hello world)", &["3", "(", "hello", " world", ")"]),
            ("foo(bar", &["foo", "(", "bar"]),
            ("a 123", &["a", " 123"]), // digit branch DOES take a leading space
            ("a  123", &["a", " ", " 123"]),
            ("end of line \n", &["end", " of", " line", " \n"]),
            ("trailing space \n next", &["trailing", " space", " \n", " next"]),
            ("multi\n\n\n\nline", &["multi", "\n\n\n", "\n", "line"]),
            ("a\n\n\nb", &["a", "\n\n", "\n", "b"]), // no `\s*[\r\n]+` special-case
            ("!!!\n\nx", &["!!!", "\n", "\n", "x"]), // symbol branch: no trailing nl absorption
            ("a!!!\nb", &["a", "!!!", "\n", "b"]),
            ("hello!!!", &["hello", "!!!"]),
            ("a: b", &["a", ":", " b"]),
            ("3!", &["3", "!"]),
            (
                "Testing punctuation: ,.;:!?()[]{}",
                &["Testing", " punctuation", ":", " ,.;:!?()[]{}"],
            ),
            (
                "under_score and-dash and.dot",
                &["under", "_", "score", " and", "-", "dash", " and", ".", "dot"],
            ),
            ("192.168.1.1", &["192", ".", "168", ".", "1", ".", "1"]),
            ("user@example.com", &["user", "@", "example", ".", "com"]),
        ];
        for (text, want) in cases {
            assert_eq!(bare.split(text), *want, "{text:?}");
        }
    }

    /// A GGUF's pre-tokenizer NAME selects the split llama.cpp runs for it,
    /// end to end through `from_gguf`, on a vocab that can express both
    /// readings of "123": one token, or three single digits.
    #[test]
    fn the_gguf_pre_tokenizer_name_selects_the_split() {
        use checkpoint::gguf::GgufTokenizer;
        let gt = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("deepseek-v3".into()),
            tokens: vec!["1".into(), "2".into(), "3".into(), "12".into(), "123".into()],
            merges: vec!["1 2".into(), "12 3".into()],
            token_types: vec![1, 1, 1, 1, 1],
            bos: None,
            eos: None,
            unk: None,
            pad: None,
            ..Default::default()
        };
        assert_eq!(QwenBpe::from_gguf(&gt).unwrap().encode("123"), vec![4]);
        let qwen = GgufTokenizer { pre: Some("qwen2".into()), ..gt.clone() };
        assert_eq!(QwenBpe::from_gguf(&qwen).unwrap().encode("123"), vec![0, 1, 2]);
        let unknown = GgufTokenizer { pre: Some("falcon".into()), ..gt };
        assert!(QwenBpe::from_gguf(&unknown).is_err());
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

    /// SPEC: `from_dir`'s split-file layout (`vocab.json` + `merges.txt`, no
    /// `tokenizer.json`) resolves special tokens from `tokenizer_config.json`'s
    /// `added_tokens_decoder` when there is no standalone `added_tokens.json`
    /// -- the real shape a Qwen3-Omni-30B-A3B-Instruct checkpoint on disk
    /// ships (confirmed this session: it has neither `tokenizer.json` nor
    /// `added_tokens.json`). Without this fallback `special_id("<|im_end|>")`
    /// silently returns `None` (vocab.json itself has no entry for it), which
    /// breaks EOS detection for every model built from such a directory.
    #[test]
    fn from_dir_resolves_specials_from_tokenizer_config_when_added_tokens_json_is_absent() {
        let dir = std::env::temp_dir().join(format!("brain-qwen-tok-added-from-tcfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A trimmed, real-shaped vocab: a few base BPE entries plus the
        // special-token ids allocated PAST the base vocab (as a real
        // checkpoint does) -- vocab.json itself never lists them.
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"b":1,"ab":2}"#).unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.1\na b\n").unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {
                "3": {"content": "<|endoftext|>", "special": true},
                "4": {"content": "<|im_start|>", "special": true},
                "5": {"content": "<|im_end|>", "special": true}
            }}"#,
        )
        .unwrap();
        // Deliberately no added_tokens.json and no tokenizer.json.
        assert!(!dir.join("added_tokens.json").exists());
        assert!(!dir.join("tokenizer.json").exists());

        let t = QwenBpe::from_dir(dir.to_str().unwrap()).expect("load from split files + tokenizer_config.json fallback");
        assert_eq!(t.special_id("<|im_end|>"), Some(5));
        assert_eq!(t.special_id("<|endoftext|>"), Some(3));
        assert_eq!(t.encode("<|im_start|>"), vec![4]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// SPEC: with NEITHER `added_tokens.json` NOR a usable
    /// `tokenizer_config.json`, `from_dir` still loads (a base tokenizer with
    /// no special tokens is a real, valid case) rather than erroring --
    /// `special_id` then correctly reports `None`, not a stale/wrong id.
    #[test]
    fn from_dir_loads_with_no_special_token_source_at_all() {
        let dir = std::env::temp_dir().join(format!("brain-qwen-tok-no-specials-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"b":1,"ab":2}"#).unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.1\na b\n").unwrap();
        let t = QwenBpe::from_dir(dir.to_str().unwrap()).expect("load with no special-token source");
        assert_eq!(t.special_id("<|im_end|>"), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
