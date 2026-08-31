// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Byte-level BPE tokenizer for HF `tokenizer.json` checkpoints (Qwen, LFM2.5).
//!
//! Same byte-level BPE family as [`crate::bpe::Gpt2Bpe`] - it reuses the byte
//! <-> unicode map and the merge loop ([`crate::bpe::bpe_merge`]) - but with
//! the checkpoint's vocabulary/merges and a cl100k-style pre-tokenizer
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,K}
//!     | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//! reproduced here as a hand-written scanner (no regex crate). Notable vs GPT-2:
//! digits split in runs of at most K (K read from the file's own `pre_tokenizer`
//! pattern: Qwen K=1, LFM2.5 K=3), and a letter run may absorb a single leading
//! non-alphanumeric char (not just a space).
//!
//! Special/added tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, …) are
//! matched as atomic units *before* BPE. A `TemplateProcessing`
//! post-processor (LFM2.5 prepends `<|startoftext|>`) is captured as
//! [`QwenBpe::template_prefix`] - callers that want HF-equivalent single-sequence
//! encodings prepend it; `encode()` itself stays template-free. `vocab_size()`
//! reports the model vocab used to size the embedding table.

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
    /// Max digits per pre-token (`\p{N}{1,K}` in the pre-tokenizer pattern).
    digit_run_max: usize,
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
        let unified = serde_json::json!({
            "model": { "vocab": vocab, "merges": merges },
            "added_tokens": added,
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
    /// bos/eos/unk/pad ids - become atomic `specials`. The digit-run cap comes
    /// from `tokenizer.ggml.pre` (see [`digit_run_max_from_pre`]) - a GGUF
    /// carries the pre-tokenizer's NAME, never its regex.
    ///
    /// **The pre-tokenizer is reproduced only as far as that cap.** This
    /// scanner is the cl100k/Qwen pattern; a checkpoint whose `pre` names a
    /// different family agrees with it on ASCII prose, spaces, punctuation runs
    /// and newline handling, and can still differ elsewhere. For
    /// `pre = "deepseek-v3"` (DeepSeek-V3/R1, DeepSeek-OCR) the two known
    /// remaining divergences are its **isolated CJK/kana branch**
    /// (`[一-龥぀-ゟ゠-ヿ]+`, which splits CJK away from adjacent Latin where
    /// this scanner keeps one letter run) and its use of `\p{P}\p{S}` where
    /// this one writes `[^\s\p{L}\p{N}]` (they differ on combining marks and
    /// control characters). `crates/deepseekocr`'s prompt test pins the real
    /// vocab against HF ground truth on the strings that model actually sends.
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

        Ok(QwenBpe {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            specials,
            vocab_size,
            digit_run_max: digit_run_max_from_pre(tok.pre.as_deref()),
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

        let digit_run_max = digit_run_max_from(&j["pre_tokenizer"]);
        let template_prefix = template_prefix_from(&j["post_processor"], &encoder);

        Ok(QwenBpe {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            specials,
            vocab_size,
            digit_run_max,
            template_prefix,
        })
    }

    /// Id of a special/added token by literal content (e.g. `"<|mask|>"`).
    pub fn special_id(&self, content: &str) -> Option<u32> {
        self.specials.iter().find(|(c, _)| c == content).map(|(_, id)| *id)
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
                    for piece in pretokenize_digits(&rest[..pos], self.digit_run_max) {
                        self.encode_piece(&piece, out);
                    }
                }
                out.push(id);
                rest = &rest[pos + content.len()..];
                continue 'outer;
            }
            // No specials left: BPE the remainder.
            for piece in pretokenize_digits(rest, self.digit_run_max) {
                self.encode_piece(&piece, out);
            }
            break;
        }
    }
}

/// Read `K` from a `\p{N}{1,K}` branch anywhere in the pre-tokenizer's Split
/// pattern(s); a bare `\p{N}` (Qwen) - or no pattern at all - means K = 1.
fn digit_run_max_from(pre: &serde_json::Value) -> usize {
    let s = pre.to_string(); // JSON text of the whole pre_tokenizer subtree
    if let Some(pos) = s.find("p{N}{1,") {
        let rest = &s[pos + "p{N}{1,".len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(k) = digits.parse::<usize>() {
            return k.max(1);
        }
    }
    1
}

/// `K` in `\p{N}{1,K}` for a GGUF's `tokenizer.ggml.pre` scheme.
///
/// A GGUF names the pre-tokenizer llama.cpp compiles in; it never stores the
/// regex, so the cap cannot be read the way [`digit_run_max_from`] reads it out
/// of a `tokenizer.json`. Only the schemes whose cap is *known* are listed:
///
/// * `deepseek-v3` - `\p{N}{1,3}`. Confirmed against the shipped
///   `deepseek-ai/DeepSeek-OCR` `tokenizer.json`, whose first `Split` really is
///   `\p{N}{1,3}`; with `K = 1` this vocab tokenizes `12345` as five ids
///   instead of the reference's `["123", "45"]`.
/// * everything else (Qwen's `qwen2`, GPT-2, `deepseek-coder`, …) - a bare
///   `\p{N}`, one digit per pre-token, which is also the safe default because
///   it is what every caller of this constructor got before the mapping
///   existed.
///
/// Deliberately NOT mapped: `deepseek-llm`, whose `\p{N}+` is an *unbounded*
/// run and so is not expressible as a `K` at all.
fn digit_run_max_from_pre(pre: Option<&str>) -> usize {
    match pre {
        Some("deepseek-v3") => 3,
        _ => 1,
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

/// `\p{L}` - NOT `char::is_alphabetic`, which is the Unicode `Alphabetic`
/// property, a strict superset of `\p{L}` that also contains `Nl` (Roman
/// numerals, other letter-numbers) and `Other_Alphabetic` marks. The regex
/// this scanner transcribes says `\p{L}`, so e.g. `Ⅻ` must start a
/// digit-family token, not a letter run - subtracting [`is_digit`]
/// (`char::is_numeric` = `Nd | Nl | No`, which is exactly `\p{N}`) removes the
/// `Nl` part exactly. This is byte-for-byte the mismatch `clip_bpe.rs`'s
/// `is_letter` carries the long note about; the two definitions are kept
/// identical on purpose, and `roman_numerals_take_the_digit_branch` below is
/// this file's own regression test for it.
///
/// **Known residual gap, identical to the clip-side one and measured there over
/// every scalar value: 1510 codepoints where this says letter and `\p{L}` does
/// not** (905 `Mn` + 423 `Mc` combining marks, 130 `So`, 52 unassigned), and
/// **0** in the other direction. Latin, Greek, Cyrillic, CJK, Hangul, kana,
/// digits, punctuation and emoji are unaffected. Closing it needs the
/// `Other_Alphabetic` range table; until then it is a stated limitation.
fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_numeric()
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
/// `[^\r\n\p{L}\p{N}]` - the optional leading char a letter run may absorb.
fn is_letter_prefix(c: char) -> bool {
    !is_nl(c) && !is_letter(c) && !is_digit(c)
}
/// `[^\s\p{L}\p{N}]` - a punctuation/symbol char.
fn is_symbol(c: char) -> bool {
    !is_ws(c) && !is_letter(c) && !is_digit(c)
}

/// Hand-written reproduction of Qwen's pre-tokenizer regex (see module docs).
/// Returns the list of pre-tokens, in order, reconstructing `text`.
pub fn qwen_pretokenize(text: &str) -> Vec<String> {
    pretokenize_digits(text, 1)
}

/// The cl100k-style scanner with a configurable digit-run cap (`\p{N}{1,K}`):
/// Qwen K=1, LFM2.5 K=3. Everything else is identical between the two patterns.
pub fn pretokenize_digits(text: &str, digit_run_max: usize) -> Vec<String> {
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

        // 3) \p{N}{1,K}  (digit run, greedy up to K)
        if is_digit(chars[i]) {
            let mut j = i;
            while j < n && j - i < digit_run_max && is_digit(chars[j]) {
                j += 1;
            }
            toks.push(chars[i..j].iter().collect());
            i = j;
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

    /// `\p{L}` is category `L*`, but `char::is_alphabetic` is the Unicode
    /// `Alphabetic` *property* = `L* | Nl | Other_Alphabetic`. `Nl` is the
    /// letter-number class (Roman numerals), and it is the half this scanner
    /// subtracts: a Roman numeral must take the `\p{N}{1,K}` branch, exactly as
    /// `clip_bpe`'s twin test asserts for CLIP's pattern. Without the
    /// subtraction `Ⅻxyz` is ONE letter run and every id downstream of it moves.
    ///
    /// ASCII and the common cases must be untouched by that rule, so they are
    /// asserted in the same test - this is the whole risk of the definition.
    #[test]
    fn roman_numerals_take_the_digit_branch() {
        // U+216B ROMAN NUMERAL TWELVE (Nl): a digit-family pre-token, and at
        // K = 1 it does not glue onto the following letter run.
        assert_eq!(qwen_pretokenize("\u{216B}xyz"), vec!["\u{216B}", "xyz"]);
        // U+2160 ROMAN NUMERAL ONE (Nl) next to a `No` fraction: both digits.
        assert_eq!(qwen_pretokenize("\u{2160}\u{00BD}"), vec!["\u{2160}", "\u{00BD}"]);
        // K > 1 groups them like any other `\p{N}` run.
        assert_eq!(pretokenize_digits("\u{2160}\u{2160}\u{2160}\u{2160}", 3), vec!["\u{2160}\u{2160}\u{2160}", "\u{2160}"]);
        // …and nothing about ASCII or ordinary text moves.
        assert_eq!(qwen_pretokenize("Hello, world"), vec!["Hello", ",", " world"]);
        assert_eq!(qwen_pretokenize("a1b"), vec!["a", "1", "b"]);
        assert_eq!(qwen_pretokenize("über café"), vec!["über", " café"]);
    }

    /// The digit-run cap is read from the GGUF's `tokenizer.ggml.pre` NAME (a
    /// GGUF never stores the regex). DeepSeek-V3's is `\p{N}{1,3}`; every other
    /// scheme keeps the single-digit default.
    #[test]
    fn the_gguf_digit_run_cap_follows_the_pre_tokenizer_name() {
        assert_eq!(digit_run_max_from_pre(Some("deepseek-v3")), 3);
        assert_eq!(digit_run_max_from_pre(Some("qwen2")), 1);
        assert_eq!(digit_run_max_from_pre(Some("deepseek-coder")), 1);
        assert_eq!(digit_run_max_from_pre(None), 1);

        // End to end through `from_gguf`, on a vocab that can express both
        // readings: "123" exists as one token AND as three single digits.
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
        };
        assert_eq!(QwenBpe::from_gguf(&gt).unwrap().encode("123"), vec![4]);
        let qwen = GgufTokenizer { pre: Some("qwen2".into()), ..gt };
        assert_eq!(QwenBpe::from_gguf(&qwen).unwrap().encode("123"), vec![0, 1, 2]);
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
