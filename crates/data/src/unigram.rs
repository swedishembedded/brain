// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SentencePiece **unigram** tokenizer for HF `tokenizer.json` checkpoints
//! (umT5, T5, ALBERT, XLNet - the whole `Unigram` family).
//!
//! This is the first non-BPE tokenization model in the workspace.
//! [`crate::bpe`], [`crate::clip_bpe`] and [`crate::qwen_tokenizer`] all build a
//! word up by *merging* adjacent pieces in a fixed rank order; a unigram model
//! instead scores every possible segmentation and takes the best one:
//!
//! ```text
//! score(segmentation) = sum over pieces of log p(piece)
//! ```
//!
//! which is a shortest-path problem over the lattice of pieces that match at
//! each byte offset, solved by Viterbi in one left-to-right sweep. There is no
//! merge table at all - the file ships `(piece, log-probability)` pairs and
//! nothing else.
//!
//! ## Why `tokenizer.json` and not `spiece.model`
//!
//! Both ship in a umT5 checkpoint and both describe the same 256k pieces, but
//! they are not equally authoritative for *this* pipeline. `spiece.model` is a
//! protobuf holding the trainer's view (pieces, scores, piece TYPES, and a
//! `precompiled_charsmap` normalizer); `tokenizer.json` is the artifact
//! `AutoTokenizer.from_pretrained` actually loads, and it is what upstream Wan
//! tokenizes with (`wan/modules/tokenizers.py` wraps `AutoTokenizer`). Reading
//! the protobuf would mean reimplementing sentencepiece's normalizer and its
//! piece-type semantics and then hoping the result agrees with the JSON;
//! reading the JSON reproduces the real pipeline directly, needs no protobuf
//! decoder, and every rule it applies is written down in the file itself. The
//! goldens are dumped from both the library and an independent Viterbi over
//! this same JSON, so "the file" is checked, not trusted.
//!
//! ## The pipeline, in order
//!
//! 1. **Normalizer** - a `Sequence` of rules read from the file itself. umT5
//!    ships exactly one, `" {2,}" -> " "`; T5-XXL (FLUX.1's `tokenizer_2`)
//!    ships three, `Precompiled` then `Strip{right}` then `" {2,}" -> "\u{2581}"`.
//!    Anything else is an error naming what was found, never a silent skip: a
//!    normalizer that is quietly dropped changes ids for a whole class of
//!    inputs and nothing downstream can tell.
//!
//!    [`NormStep::Precompiled`] is sentencepiece's own charsmap - a Darts
//!    double-array trie plus a blob of replacement strings, packed into one
//!    base64 field. It is the NFKC-ish folding every released T5 depends on
//!    (`\u{FB01}` -> `fi`, `\u{FF21}` -> `A`, `\u{2160}` -> `I`), so a reader that
//!    rejects it cannot tokenize T5-XXL at all. See [`Precompiled`] for the
//!    binary format and for why its grapheme loop looks the way it does.
//! 2. **Added tokens** (`<pad>`, `</s>`, `<extra_id_0>`, ...) are matched
//!    atomically, longest content first, and split the text into segments.
//! 3. **Metaspace pre-tokenizer** - every space becomes `U+2581`, one is
//!    prepended, and the segment splits so that each pre-token *starts* with
//!    `U+2581` (`SplitDelimiterBehavior::MergedWithNext`).
//! 4. **Viterbi** per pre-token, over BYTES with char-boundary starts. A
//!    character no piece covers becomes `unk`, scored `min_score - 10`, and
//!    consecutive unknown characters **fuse** into one `unk` token.
//! 5. **Post-processor** - `TemplateProcessing`'s single-sequence prefix and
//!    suffix (umT5: append `</s>`).
//!
//! [`UnigramTokenizer::encode_padded`] adds the last two things a text encoder
//! wants: truncation to `max_len` *including* the post-processor's specials,
//! and right padding with the pad id, returning the attention mask alongside.

use std::collections::HashMap;

use serde::Deserialize;
use unicode_segmentation::UnicodeSegmentation;

use crate::tokenizer::Tokenizer;

/// The `U+2581` "lower one eighth block" sentencepiece uses for a space.
const METASPACE: char = '\u{2581}';

/// `tokenizers`' `Unigram::unk_penalty` - the unknown piece scores
/// `min_score - 10`, low enough that any real segmentation wins.
const UNK_PENALTY: f64 = 10.0;

// ---------------------------------------------------------------------------
// wire format (validated structurally by serde, semantically below)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTokenizer {
    #[allow(dead_code)]
    version: Option<String>,
    #[allow(dead_code)]
    truncation: Option<serde_json::Value>,
    #[allow(dead_code)]
    padding: Option<serde_json::Value>,
    #[allow(dead_code)]
    decoder: Option<serde_json::Value>,
    #[serde(default)]
    added_tokens: Vec<WireAdded>,
    normalizer: Option<serde_json::Value>,
    pre_tokenizer: serde_json::Value,
    post_processor: Option<serde_json::Value>,
    model: WireModel,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireModel {
    #[serde(rename = "type")]
    kind: String,
    unk_id: u32,
    /// `[[piece, log-probability], ...]`, indexed by token id.
    vocab: Vec<(String, f64)>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAdded {
    id: u32,
    content: String,
    #[allow(dead_code)]
    single_word: bool,
    #[allow(dead_code)]
    lstrip: bool,
    #[allow(dead_code)]
    rstrip: bool,
    #[allow(dead_code)]
    normalized: bool,
    #[allow(dead_code)]
    special: bool,
}

/// One normalizer step this reader understands. A file asking for anything
/// else is rejected by name rather than silently normalized differently.
#[derive(Clone, Debug, PartialEq)]
enum NormStep {
    /// Collapse each run of two or more spaces into the carried replacement
    /// (the `" {2,}"` regex both umT5 and T5-XXL ship, recognised as a rule
    /// rather than run through a regex engine). The replacement is NOT always
    /// a space: umT5 writes `" "`, T5-XXL writes `"\u{2581}"`.
    CollapseSpaces(String),
    /// Literal string replacement.
    Replace(String, String),
    /// `tokenizers`' `Strip`: drop leading and/or trailing whitespace.
    Strip { left: bool, right: bool },
    /// SentencePiece's `Precompiled` charsmap.
    Precompiled(Precompiled),
}

// ---------------------------------------------------------------------------
// SentencePiece's `Precompiled` charsmap
// ---------------------------------------------------------------------------

/// SentencePiece's `Precompiled` normalizer, read from the base64
/// `precompiled_charsmap` a `tokenizer.json` ships.
///
/// This is a **direct port** of HuggingFace's own `spm_precompiled` crate
/// (v0.1.4), which is what their `tokenizers` uses in production for this
/// normalizer, which in turn emulates `google/sentencepiece`'s
/// `Darts::DoubleArray`. The binary format is:
///
/// ```text
/// [u32 LE trie byte length][that many bytes of u32 LE trie units][normalized blob]
/// ```
///
/// The trie keys on the input's BYTES and its values are byte offsets into the
/// normalized blob, where each replacement string is terminated by `\0`. So the
/// blob `"abc\0"` serves both the entry that maps to `"abc"` (offset 0) and the
/// one that maps to `"bc"` (offset 1) - the replacements deliberately overlap.
///
/// The only deliberate departures from the reference are **bounds checks**: it
/// indexes its arrays and slices its blob directly and therefore panics on a
/// corrupt charsmap, whereas everything crossing into brain from a file is
/// validated at the boundary. On a well-formed charsmap the two are identical -
/// a check can only fire where the reference would already have panicked.
/// Units are also kept at their on-file `u32` width (the reference widens every
/// one to `usize` at parse time) and widened only inside the arithmetic, which
/// halves the resident trie. That cannot change a result: `has_leaf`, `value`
/// and `label` are masks of bits the unit already has, and `offset` widens
/// BEFORE its shift, so its one shift-past-32-bits case is carried the same way
/// the reference carries it.
#[derive(Clone, PartialEq)]
pub struct Precompiled {
    /// The Darts double-array trie units, in file order.
    trie: Vec<u32>,
    /// The `\0`-separated replacement strings the trie's values index into.
    normalized: String,
}

/// A trie unit carries four fields in its 32 bits. Transcribed from
/// `spm_precompiled`'s `ArrayUnitTrait`; the constants are load-bearing.
mod unit {
    pub fn has_leaf(u: u32) -> bool {
        (u >> 8) & 1 == 1
    }

    pub fn value(u: u32) -> usize {
        (u & ((1u32 << 31) - 1)) as usize
    }

    /// Includes bit 31, so a unit with it set can never match a byte label.
    pub fn label(u: u32) -> usize {
        (u & ((1u32 << 31) | 0xFF)) as usize
    }

    pub fn offset(u: u32) -> usize {
        ((u >> 10) as usize) << ((u & (1 << 9)) >> 6)
    }
}

impl Precompiled {
    /// Parse a decoded `precompiled_charsmap`.
    pub fn from_charsmap(charsmap: &[u8]) -> Result<Precompiled, String> {
        let Some(header) = charsmap.get(..4) else {
            return Err(format!("precompiled_charsmap: {} bytes is too short for a header", charsmap.len()));
        };
        let trie_bytes = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        // The reference truncates with `trie_size / 4`; a ragged trie length is
        // a corrupt file, so say so instead of silently dropping a partial unit.
        if !trie_bytes.is_multiple_of(4) {
            return Err(format!("precompiled_charsmap: trie length {trie_bytes} is not a multiple of 4"));
        }
        // `4 + trie_bytes` is computed once, checked: the header is attacker-
        // controlled and `u32::MAX + 4` overflows a 32-bit `usize` (this crate
        // builds for wasm32).
        let blob_at = 4usize
            .checked_add(trie_bytes)
            .ok_or_else(|| format!("precompiled_charsmap: trie length {trie_bytes} overflows"))?;
        let Some(trie_raw) = charsmap.get(4..blob_at) else {
            return Err(format!(
                "precompiled_charsmap: trie wants {trie_bytes} bytes but only {} follow the header",
                charsmap.len().saturating_sub(4)
            ));
        };
        if trie_raw.is_empty() {
            // `common_prefix_search` starts by reading unit 0.
            return Err("precompiled_charsmap: empty trie".into());
        }
        let trie: Vec<u32> =
            trie_raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let normalized = String::from_utf8(charsmap[blob_at..].to_vec())
            .map_err(|e| format!("precompiled_charsmap: normalized blob is not UTF-8: {e}"))?;
        Ok(Precompiled { trie, normalized })
    }

    /// Every value on the trie walk over `key`'s bytes, in visit order.
    ///
    /// Stops at a NUL byte, and at the first byte whose label does not match -
    /// so the returned offsets are those of the key's own prefixes, longest
    /// last.
    fn common_prefix_search(&self, key: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let mut node_pos = 0usize;
        let Some(&first) = self.trie.first() else { return results };
        node_pos ^= unit::offset(first);
        for &c in key {
            if c == 0 {
                break;
            }
            node_pos ^= c as usize;
            let Some(&u) = self.trie.get(node_pos) else { return results };
            if unit::label(u) != c as usize {
                return results;
            }
            node_pos ^= unit::offset(u);
            if unit::has_leaf(u) {
                let Some(&leaf) = self.trie.get(node_pos) else { return results };
                results.push(unit::value(leaf));
            }
        }
        results
    }

    /// The replacement for `chunk`, or `None` when the trie has no entry
    /// starting at its first byte (i.e. leave `chunk` alone).
    fn transform(&self, chunk: &str) -> Option<&str> {
        let start = *self.common_prefix_search(chunk.as_bytes()).first()?;
        let bytes = self.normalized.as_bytes();
        if start > bytes.len() {
            return None;
        }
        let end = start + bytes[start..].iter().position(|&b| b == 0).unwrap_or(bytes.len() - start);
        self.normalized.get(start..end)
    }

    /// Apply the charsmap to a whole string.
    ///
    /// Future reader, from `spm_precompiled`'s @Narsil, on the shape of this
    /// loop - a whole grapheme cluster tried first, but only under 6 bytes, and
    /// otherwise a per-CHARACTER fallback:
    ///
    /// > Yes, this is weird, Yes, this seems broken, No, I don't know why
    /// > Google did this. If you question this code, check this normalizer
    /// > against XNLI database (all languages) with Unigram model against
    /// > Mbart, XLMRoberta *AND* Marian. If you don't get 100% or break a
    /// > single test. You don't pass.
    ///
    /// It is reproduced exactly rather than tidied, because it is what decides
    /// real token ids. Halfwidth `\u{FF76}\u{FF9E}` is the case that shows why:
    /// it is one grapheme of exactly 6 bytes, so it takes the per-character
    /// path and comes out as `\u{30AB}` + combining `\u{3099}` rather than the
    /// precomposed `\u{30AC}` its own grapheme entry would have given.
    pub fn normalize_string(&self, original: &str) -> String {
        let mut out = String::with_capacity(original.len());
        for grapheme in original.graphemes(true) {
            if grapheme.len() < 6 {
                if let Some(norm) = self.transform(grapheme) {
                    out.push_str(norm);
                    continue;
                }
            }
            for (at, c) in grapheme.char_indices() {
                let part = &grapheme[at..at + c.len_utf8()];
                match self.transform(part) {
                    Some(norm) => out.push_str(norm),
                    None => out.push(c),
                }
            }
        }
        out
    }
}

/// Sizes only: the real trie is tens of thousands of units and the blob tens of
/// kilobytes, so the derived `Debug` would bury whatever printed it.
impl std::fmt::Debug for Precompiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Precompiled")
            .field("trie_units", &self.trie.len())
            .field("normalized_bytes", &self.normalized.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------

/// A SentencePiece unigram tokenizer built from a `tokenizer.json`.
pub struct UnigramTokenizer {
    /// piece -> (id, log-probability). Scores stay `f64` because the lattice
    /// tie-breaks on `>` and `tokenizers` accumulates in `f64`; rounding them
    /// to `f32` could pick a different (equally scored) segmentation.
    pieces: HashMap<String, (u32, f64)>,
    id_to_piece: Vec<String>,
    unk_id: u32,
    unk_score: f64,
    /// Longest piece in BYTES - the lattice never looks further ahead.
    max_piece_bytes: usize,
    /// (content, id) for added/special tokens, longest content first.
    added: Vec<(String, u32)>,
    /// First byte of every added token, so the scan skips most positions.
    added_first: [bool; 256],
    normalizer: Vec<NormStep>,
    /// Single-sequence template specials, e.g. umT5's trailing `</s>`.
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    pad_id: u32,
    vocab_size: usize,
}

impl UnigramTokenizer {
    /// Build from a `tokenizer.json` path.
    pub fn from_file(path: &str) -> Result<UnigramTokenizer, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_json_bytes(&bytes).map_err(|e| format!("{path}: {e}"))
    }

    /// Build from a checkpoint DIRECTORY holding `tokenizer.json`.
    pub fn from_dir(dir: &str) -> Result<UnigramTokenizer, String> {
        Self::from_file(&format!("{dir}/tokenizer.json"))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<UnigramTokenizer, String> {
        let j: WireTokenizer =
            serde_json::from_slice(bytes).map_err(|e| format!("tokenizer.json: {e}"))?;
        if j.model.kind != "Unigram" {
            return Err(format!("tokenizer.json: model.type is {:?}, not Unigram", j.model.kind));
        }
        if j.model.byte_fallback {
            // The `<0x..>` byte pieces would have to take over for unknown
            // characters instead of `unk`. umT5 sets this false; supporting it
            // silently as if it were false would mis-tokenize every unknown.
            return Err("tokenizer.json: byte_fallback is not implemented".into());
        }
        if j.model.vocab.is_empty() {
            return Err("tokenizer.json: empty Unigram vocab".into());
        }

        let mut pieces = HashMap::with_capacity(j.model.vocab.len());
        let mut id_to_piece = Vec::with_capacity(j.model.vocab.len());
        let mut min_score = f64::INFINITY;
        let mut max_piece_bytes = 0usize;
        for (id, (piece, score)) in j.model.vocab.iter().enumerate() {
            // A duplicated piece keeps its FIRST id, matching `tokenizers`.
            pieces.entry(piece.clone()).or_insert((id as u32, *score));
            id_to_piece.push(piece.clone());
            min_score = min_score.min(*score);
            max_piece_bytes = max_piece_bytes.max(piece.len());
        }
        if j.model.unk_id as usize >= id_to_piece.len() {
            return Err(format!("tokenizer.json: unk_id {} is out of vocab", j.model.unk_id));
        }

        let mut added: Vec<(String, u32)> = Vec::new();
        let mut added_first = [false; 256];
        for a in &j.added_tokens {
            if a.content.is_empty() {
                return Err("tokenizer.json: an added token has empty content".into());
            }
            added.push((a.content.clone(), a.id));
            added_first[a.content.as_bytes()[0] as usize] = true;
        }
        added.sort_by(|x, y| y.0.len().cmp(&x.0.len()).then_with(|| x.0.cmp(&y.0)));

        let normalizer = parse_normalizer(j.normalizer.as_ref())?;
        parse_metaspace(&j.pre_tokenizer)?;
        let (prefix, suffix) = parse_post_processor(j.post_processor.as_ref())?;

        // The pad id is the one a text encoder right-pads with. umT5 (like
        // every T5) uses id 0; taking it from the added tokens rather than
        // hardcoding keeps a checkpoint that renumbers them honest.
        let pad_id = added.iter().find(|(c, _)| c == "<pad>").map(|(_, i)| *i).unwrap_or(0);

        Ok(UnigramTokenizer {
            vocab_size: id_to_piece.len(),
            pieces,
            id_to_piece,
            unk_id: j.model.unk_id,
            unk_score: min_score - UNK_PENALTY,
            max_piece_bytes,
            added,
            added_first,
            normalizer,
            prefix,
            suffix,
            pad_id,
        })
    }

    /// The id right padding uses (`<pad>`, 0 for every released T5/umT5).
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// The unknown-piece id.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Id of an added/special token by literal content (e.g. `"</s>"`).
    pub fn special_id(&self, content: &str) -> Option<u32> {
        self.added.iter().find(|(c, _)| c == content).map(|(_, id)| *id)
    }

    /// `wan/modules/tokenizers.py`'s `whitespace_clean`: collapse every run of
    /// whitespace to a single space, then trim. Wan applies this (as
    /// `clean='whitespace'`) to every prompt before tokenizing, and it is not
    /// cosmetic - it is what makes the tokenizer's own `" {2,}" -> " "`
    /// normalizer a no-op and what removes newlines from a pasted prompt.
    ///
    /// The `ftfy.fix_text` + double `html.unescape` that upstream's
    /// `basic_clean` runs FIRST is deliberately not reproduced: both are
    /// repairs for text that arrived damaged (mojibake, HTML-escaped entities),
    /// both are the identity on well-formed UTF-8 prompts, and reimplementing
    /// either badly would be worse than not having it. A caller feeding brain
    /// scraped HTML should unescape it before it gets here.
    pub fn clean_whitespace(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut in_space = false;
        for c in text.chars() {
            if c.is_whitespace() {
                in_space = true;
            } else {
                if in_space && !out.is_empty() {
                    out.push(' ');
                }
                in_space = false;
                out.push(c);
            }
        }
        out
    }

    /// Encode, then truncate to `max_len` (specials included, exactly as
    /// `padding='max_length', truncation=True, max_length=max_len` does) and
    /// right-pad with [`Self::pad_id`]. Returns `(ids, mask)`, both `max_len`
    /// long, with `mask[i] == 1` on the real tokens.
    pub fn encode_padded(&self, text: &str, max_len: usize) -> (Vec<u32>, Vec<u32>) {
        let mut ids = self.encode(text);
        if ids.len() > max_len {
            // Truncation drops CONTENT, never the template's specials: keep the
            // prefix, refill from the middle, keep the suffix.
            let keep = max_len.saturating_sub(self.prefix.len() + self.suffix.len());
            let body: Vec<u32> =
                ids[self.prefix.len()..ids.len() - self.suffix.len()].iter().take(keep).copied().collect();
            ids = self.prefix.iter().copied().chain(body).chain(self.suffix.iter().copied()).collect();
            ids.truncate(max_len);
        }
        let mut mask = vec![1u32; ids.len()];
        ids.resize(max_len, self.pad_id);
        mask.resize(max_len, 0);
        (ids, mask)
    }

    // ---- the pipeline -----------------------------------------------------

    fn normalize(&self, text: &str) -> String {
        let mut s = text.to_string();
        for step in &self.normalizer {
            s = match step {
                NormStep::CollapseSpaces(to) => {
                    let mut out = String::with_capacity(s.len());
                    let mut run = 0usize;
                    let flush = |run: usize, out: &mut String| match run {
                        0 => {}
                        // ONE space is not a run of two or more: the regex does
                        // not match it, so it survives verbatim even when the
                        // replacement is something else (T5-XXL's `U+2581`).
                        1 => out.push(' '),
                        _ => out.push_str(to),
                    };
                    for c in s.chars() {
                        if c == ' ' {
                            run += 1;
                        } else {
                            flush(run, &mut out);
                            run = 0;
                            out.push(c);
                        }
                    }
                    flush(run, &mut out);
                    out
                }
                NormStep::Replace(from, to) => s.replace(from.as_str(), to.as_str()),
                NormStep::Strip { left, right } => {
                    let t = if *left { s.trim_start_matches(char::is_whitespace) } else { &s };
                    let t = if *right { t.trim_end_matches(char::is_whitespace) } else { t };
                    t.to_string()
                }
                NormStep::Precompiled(p) => p.normalize_string(&s),
            };
        }
        s
    }

    /// The longest added token matching at byte offset `at`, if any.
    fn added_at(&self, s: &str, at: usize) -> Option<(usize, u32)> {
        if !self.added_first[s.as_bytes()[at] as usize] {
            return None;
        }
        let rest = &s[at..];
        self.added
            .iter()
            .find(|(c, _)| rest.starts_with(c.as_str()))
            .map(|(c, id)| (c.len(), *id))
    }

    /// Metaspace + Viterbi over one added-token-free segment.
    fn encode_segment(&self, seg: &str, out: &mut Vec<u32>) {
        if seg.is_empty() {
            return;
        }
        let mut s: String = seg.replace(' ', &METASPACE.to_string());
        if !s.starts_with(METASPACE) {
            s.insert(0, METASPACE);
        }
        // `SplitDelimiterBehavior::MergedWithNext`: every pre-token begins with
        // the replacement character, and the (empty) piece before the first one
        // is dropped.
        let mut start = 0usize;
        let bounds: Vec<usize> = s
            .char_indices()
            .filter(|&(i, c)| c == METASPACE && i > 0)
            .map(|(i, _)| i)
            .chain(std::iter::once(s.len()))
            .collect();
        for end in bounds {
            self.viterbi(&s[start..end], out);
            start = end;
        }
    }

    /// One Viterbi sweep over `text`, appending the best segmentation's ids.
    fn viterbi(&self, text: &str, out: &mut Vec<u32>) {
        let n = text.len();
        if n == 0 {
            return;
        }
        // best[k] = (score of the best path ending at byte k, its start, its id)
        let mut best: Vec<Option<(f64, usize, u32)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, u32::MAX));
        let mut i = 0usize;
        while i < n {
            let here = best[i].expect("lattice has no gaps: every step fills i+mblen").0;
            let mblen = utf8_len(text.as_bytes()[i]);
            let mut has_single = false;
            let hi = (i + self.max_piece_bytes).min(n);
            for j in (i + 1)..=hi {
                if !text.is_char_boundary(j) {
                    continue;
                }
                if let Some(&(id, score)) = self.pieces.get(&text[i..j]) {
                    let cand = here + score;
                    if best[j].is_none_or(|(s, _, _)| cand > s) {
                        best[j] = Some((cand, i, id));
                    }
                    if j - i == mblen {
                        has_single = true;
                    }
                }
            }
            if !has_single {
                // The single character is out of vocabulary: spend one `unk`.
                let j = i + mblen;
                let cand = here + self.unk_score;
                if best[j].is_none_or(|(s, _, _)| cand > s) {
                    best[j] = Some((cand, i, self.unk_id));
                }
            }
            i += mblen;
        }

        let mut path: Vec<u32> = Vec::new();
        let mut k = n;
        while k > 0 {
            let (_, start, id) = best[k].expect("backtrack hit an unreachable byte");
            path.push(id);
            k = start;
        }
        path.reverse();
        // fuse_unk: a run of unknown characters is ONE unk token, not several.
        for id in path {
            if id == self.unk_id && out.last() == Some(&self.unk_id) {
                continue;
            }
            out.push(id);
        }
    }
}

fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

fn parse_normalizer(v: Option<&serde_json::Value>) -> Result<Vec<NormStep>, String> {
    let Some(v) = v else { return Ok(Vec::new()) };
    if v.is_null() {
        return Ok(Vec::new());
    }
    let kind = v["type"].as_str().ok_or("normalizer: no type")?;
    match kind {
        "Sequence" => {
            let list = v["normalizers"].as_array().ok_or("normalizer: Sequence has no list")?;
            let mut out = Vec::new();
            for n in list {
                out.extend(parse_normalizer(Some(n))?);
            }
            Ok(out)
        }
        "Replace" => {
            let to = v["content"].as_str().ok_or("normalizer: Replace has no content")?.to_string();
            if let Some(s) = v["pattern"]["String"].as_str() {
                return Ok(vec![NormStep::Replace(s.to_string(), to)]);
            }
            let re = v["pattern"]["Regex"].as_str().ok_or("normalizer: Replace has no pattern")?;
            if re == " {2,}" {
                // The replacement is read from the file, not assumed: umT5
                // writes a space here and T5-XXL writes `U+2581`.
                return Ok(vec![NormStep::CollapseSpaces(to)]);
            }
            Err(format!("normalizer: Replace pattern {re:?} is not implemented"))
        }
        "Strip" => {
            let flag = |k: &str| {
                v[k].as_bool().ok_or_else(|| format!("normalizer: Strip has no boolean {k}"))
            };
            Ok(vec![NormStep::Strip { left: flag("strip_left")?, right: flag("strip_right")? }])
        }
        "Precompiled" => {
            let b64 = v["precompiled_charsmap"]
                .as_str()
                .ok_or("normalizer: Precompiled has no precompiled_charsmap string")?;
            let raw = crate::base64::decode(b64)
                .map_err(|e| format!("normalizer: Precompiled charsmap is not base64: {e}"))?;
            let p = Precompiled::from_charsmap(&raw).map_err(|e| format!("normalizer: {e}"))?;
            Ok(vec![NormStep::Precompiled(p)])
        }
        other => Err(format!("normalizer: {other:?} is not implemented")),
    }
}

/// Validate the Metaspace pre-tokenizer. Nothing is returned because the only
/// configuration this reader implements is the one umT5 (and every T5) ships;
/// anything else is an error rather than a quietly different pre-tokenization.
fn parse_metaspace(v: &serde_json::Value) -> Result<(), String> {
    let kind = v["type"].as_str().ok_or("pre_tokenizer: no type")?;
    if kind != "Metaspace" {
        return Err(format!("pre_tokenizer: {kind:?} is not implemented"));
    }
    let rep = v["replacement"].as_str().ok_or("pre_tokenizer: no replacement")?;
    if !rep.starts_with(METASPACE) || rep.chars().count() != 1 {
        return Err(format!("pre_tokenizer: replacement {rep:?} is not U+2581"));
    }
    // Two spellings of the same thing: the current `prepend_scheme`/`split`
    // pair (the diffusers export) and the legacy `add_prefix_space` bool
    // (`google/umt5-xxl`), which `tokenizers` widens to `always` + split.
    if let Some(scheme) = v["prepend_scheme"].as_str() {
        if scheme != "always" {
            return Err(format!("pre_tokenizer: prepend_scheme {scheme:?} is not implemented"));
        }
        if v["split"].as_bool() != Some(true) {
            return Err("pre_tokenizer: Metaspace without split is not implemented".into());
        }
    } else if v["add_prefix_space"].as_bool() != Some(true) {
        return Err("pre_tokenizer: Metaspace without a prefix space is not implemented".into());
    }
    Ok(())
}

/// `TemplateProcessing`'s single-sequence specials, as `(prefix, suffix)` id
/// lists around the one `Sequence` slot.
fn parse_post_processor(v: Option<&serde_json::Value>) -> Result<(Vec<u32>, Vec<u32>), String> {
    let Some(v) = v else { return Ok((Vec::new(), Vec::new())) };
    if v.is_null() {
        return Ok((Vec::new(), Vec::new()));
    }
    let kind = v["type"].as_str().ok_or("post_processor: no type")?;
    if kind != "TemplateProcessing" {
        return Err(format!("post_processor: {kind:?} is not implemented"));
    }
    let single = v["single"].as_array().ok_or("post_processor: no single template")?;
    let (mut prefix, mut suffix, mut seen_seq) = (Vec::new(), Vec::new(), false);
    for slot in single {
        if slot.get("Sequence").is_some() {
            if seen_seq {
                return Err("post_processor: more than one Sequence slot".into());
            }
            seen_seq = true;
            continue;
        }
        let name = slot["SpecialToken"]["id"]
            .as_str()
            .ok_or("post_processor: template slot is neither Sequence nor SpecialToken")?;
        let ids = v["special_tokens"][name]["ids"]
            .as_array()
            .ok_or_else(|| format!("post_processor: special token {name:?} has no ids"))?;
        for id in ids {
            let id = id.as_u64().ok_or("post_processor: non-integer special id")? as u32;
            if seen_seq {
                suffix.push(id);
            } else {
                prefix.push(id);
            }
        }
    }
    if !seen_seq {
        return Err("post_processor: the single template has no Sequence slot".into());
    }
    Ok((prefix, suffix))
}

impl Tokenizer for UnigramTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let s = self.normalize(text);
        let mut out: Vec<u32> = self.prefix.clone();
        let mut seg_start = 0usize;
        let mut i = 0usize;
        while i < s.len() {
            if !s.is_char_boundary(i) {
                i += 1;
                continue;
            }
            if let Some((len, id)) = self.added_at(&s, i) {
                self.encode_segment(&s[seg_start..i], &mut out);
                out.push(id);
                i += len;
                seg_start = i;
            } else {
                i += utf8_len(s.as_bytes()[i]);
            }
        }
        self.encode_segment(&s[seg_start..], &mut out);
        out.extend_from_slice(&self.suffix);
        out
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            let Some(p) = self.id_to_piece.get(id as usize) else { continue };
            s.push_str(p);
        }
        // The Metaspace decoder maps `U+2581` back to a space and drops the one
        // the pre-tokenizer prepended.
        let s = s.replace(METASPACE, " ");
        s.strip_prefix(' ').unwrap_or(&s).to_string()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature Unigram file with umT5's exact shape: the same normalizer,
    /// the legacy Metaspace spelling, `</s>` appended, and scores chosen so the
    /// lattice has a decidable answer.
    fn tiny_json() -> String {
        // "▁ab" beats "▁a" + "b" (-1.0 vs -2.5); "▁a" + "bc" (-4.0) loses to
        // "▁ab" + "c" (-3.5), so a greedy longest-match would pick differently.
        serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "</s>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]},
            "pre_tokenizer": {"type": "Metaspace", "replacement": "\u{2581}",
                              "add_prefix_space": true},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"Sequence": {"id": "A", "type_id": 0}},
                           {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                "special_tokens": {"</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}}},
            "model": {
                "type": "Unigram",
                "unk_id": 3,
                "byte_fallback": false,
                "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["<s>", 0.0], ["<unk>", 0.0],
                          ["\u{2581}a", -2.0], ["b", -0.5], ["c", -1.5],
                          ["\u{2581}ab", -1.0], ["bc", -2.0], ["\u{2581}", -3.0]]
            }
        })
        .to_string()
    }

    fn tiny() -> UnigramTokenizer {
        UnigramTokenizer::from_json_bytes(tiny_json().as_bytes()).unwrap()
    }

    #[test]
    fn viterbi_beats_greedy_longest_match() {
        let t = tiny();
        // "abc" -> "▁abc": "▁ab"(-1.0) + "c"(-1.5) = -2.5 beats
        // "▁a"(-2.0) + "bc"(-2.0) = -4.0 and "▁a"+"b"+"c" = -4.0.
        assert_eq!(t.encode("abc"), vec![7, 6, 1]);
        // "ab" alone is one piece.
        assert_eq!(t.encode("ab"), vec![7, 1]);
    }

    #[test]
    fn metaspace_prepends_and_splits_on_words() {
        let t = tiny();
        // Two words: each pre-token starts with U+2581, so "▁ab" then "▁ab".
        assert_eq!(t.encode("ab ab"), vec![7, 7, 1]);
        // The normalizer collapses the double space first.
        assert_eq!(t.encode("ab  ab"), t.encode("ab ab"));
    }

    #[test]
    fn unknown_characters_fuse_into_one_unk() {
        let t = tiny();
        // 'z' and 'y' are both out of vocabulary and adjacent -> ONE unk.
        assert_eq!(t.encode("abzy"), vec![7, 3, 1]);
        // Separated by a known piece -> two.
        assert_eq!(t.encode("abzcy"), vec![7, 3, 6, 3, 1]);
    }

    #[test]
    fn added_tokens_are_atomic_and_the_template_is_applied() {
        let t = tiny();
        assert_eq!(t.special_id("</s>"), Some(1));
        assert_eq!(t.pad_id(), 0);
        // "<pad>" in the middle of text is one token, not tokenized text.
        assert_eq!(t.encode("ab<pad>ab"), vec![7, 0, 7, 1]);
        // Empty input is the template alone.
        assert_eq!(t.encode(""), vec![1]);
    }

    #[test]
    fn encode_padded_pads_and_truncates_keeping_the_template() {
        let t = tiny();
        let (ids, mask) = t.encode_padded("abc", 8);
        assert_eq!(ids, vec![7, 6, 1, 0, 0, 0, 0, 0]);
        assert_eq!(mask, vec![1, 1, 1, 0, 0, 0, 0, 0]);
        // Truncation keeps the `</s>` the post-processor appended.
        let (ids, mask) = t.encode_padded("ab ab ab ab", 3);
        assert_eq!(ids, vec![7, 7, 1]);
        assert_eq!(mask, vec![1, 1, 1]);
    }

    #[test]
    fn decode_inverts_a_round_trip() {
        let t = tiny();
        assert_eq!(t.decode(&[7, 6]), "abc");
        assert_eq!(t.decode(&[7, 7]), "ab ab");
        assert_eq!(t.vocab_size(), 10);
    }

    #[test]
    fn whitespace_clean_collapses_and_trims() {
        assert_eq!(UnigramTokenizer::clean_whitespace("  a \t\n b  "), "a b");
        assert_eq!(UnigramTokenizer::clean_whitespace("a\u{00a0}b"), "a b");
    }

    /// Everything crossing in from a file is validated at the boundary; a
    /// scheme this reader does not implement must be an error naming it, never
    /// a silently different tokenization.
    #[test]
    fn unsupported_configurations_error_by_name() {
        let bad = |f: &dyn Fn(&mut serde_json::Value)| -> String {
            let mut j: serde_json::Value = serde_json::from_str(&tiny_json()).unwrap();
            f(&mut j);
            match UnigramTokenizer::from_json_bytes(j.to_string().as_bytes()) {
                Err(e) => e,
                Ok(_) => panic!("expected a rejection"),
            }
        };
        assert!(bad(&|j| j["model"]["type"] = "BPE".into()).contains("Unigram"));
        assert!(bad(&|j| j["model"]["byte_fallback"] = true.into()).contains("byte_fallback"));
        assert!(bad(&|j| j["pre_tokenizer"]["add_prefix_space"] = false.into())
            .contains("prefix space"));
        assert!(bad(&|j| j["pre_tokenizer"]["type"] = "Whitespace".into())
            .contains("Whitespace"));
        assert!(bad(&|j| j["model"]["unk_id"] = 999.into()).contains("unk_id"));
        // An unknown top-level key is a file this reader has not been checked
        // against, so serde's `deny_unknown_fields` refuses it.
        assert!(bad(&|j| j["surprise"] = 1.into()).contains("surprise"));

        // `Precompiled` is implemented now, so what must be rejected is a
        // MALFORMED one - the reference implementation indexes straight into
        // its arrays and would panic on each of these.
        let with_map = |b64: &str| -> String {
            let b64 = b64.to_string();
            bad(&move |j| {
                j["normalizer"] =
                    serde_json::json!({"type": "Precompiled", "precompiled_charsmap": b64})
            })
        };
        assert!(with_map("!!!!").contains("not base64"));
        assert!(with_map(&crate::base64::encode(&[0u8, 0, 0])).contains("too short"));
        // Header says the trie is 6 bytes: not a whole number of u32 units.
        assert!(with_map(&crate::base64::encode(&[6u8, 0, 0, 0, 1, 2, 3, 4, 5, 6]))
            .contains("multiple of 4"));
        // Header says 64 bytes of trie but only 4 follow.
        assert!(with_map(&crate::base64::encode(&[64u8, 0, 0, 0, 1, 2, 3, 4]))
            .contains("only 4"));
        assert!(with_map(&crate::base64::encode(&[0u8; 4])).contains("empty trie"));
        // A trie is present but the normalized blob is not UTF-8.
        assert!(with_map(&crate::base64::encode(&[4u8, 0, 0, 0, 0, 0, 0, 0, 0xFF]))
            .contains("not UTF-8"));
    }

    // ---- the Precompiled charsmap ----------------------------------------

    /// Build a Darts double-array holding `entries`, as
    /// `(key bytes, offset into the normalized blob)`.
    ///
    /// This is a **test** builder, not a general one: it gives every trie node
    /// its own 256-slot block, so no two children can ever collide and none of
    /// the collision resolution a real sentencepiece builder does is needed.
    /// The point is only to produce bytes in the released format, so the
    /// reader's bit math is checked against a layout derived independently of
    /// it rather than against its own output.
    fn build_charsmap(entries: &[(&[u8], usize)], blob: &[u8]) -> Vec<u8> {
        // Every prefix of every key is a node; node 0 is the root.
        let mut nodes: Vec<Vec<u8>> = vec![Vec::new()];
        for (key, _) in entries {
            for n in 1..=key.len() {
                let pre = key[..n].to_vec();
                if !nodes.contains(&pre) {
                    nodes.push(pre);
                }
            }
        }
        let idx = |p: &[u8]| nodes.iter().position(|n| n.as_slice() == p).expect("node");
        // Node i owns array slots [256*(i+1), 256*(i+2)): its VALUE sits at the
        // block's base and its children at `base ^ byte`, which cannot be the
        // base itself because no key contains a NUL.
        let base = |i: usize| 256 * (i + 1);
        // A unit packs `offset << 10 | has_leaf << 8 | label`. Bit 9 is left
        // clear throughout, so `offset()` is a plain `>> 10`.
        let pack = |offset: usize, leaf: bool, label: u8| {
            ((offset as u32) << 10) | (u32::from(leaf) << 8) | u32::from(label)
        };

        let mut trie = vec![0u32; base(nodes.len())];
        // The root unit lives at index 0, outside every block.
        trie[0] = pack(base(0), false, 0);
        for (i, prefix) in nodes.iter().enumerate().skip(1) {
            let last = prefix[prefix.len() - 1];
            let slot = base(idx(&prefix[..prefix.len() - 1])) ^ (last as usize);
            let terminal = entries.iter().any(|(k, _)| *k == prefix.as_slice());
            trie[slot] = pack(slot ^ base(i), terminal, last);
        }
        for (key, value) in entries {
            trie[base(idx(key))] = *value as u32;
        }

        let mut out = ((trie.len() * 4) as u32).to_le_bytes().to_vec();
        for u in &trie {
            out.extend_from_slice(&u.to_le_bytes());
        }
        out.extend_from_slice(blob);
        out
    }

    /// The fixture the two tests below share.
    ///
    /// The blob's replacements deliberately OVERLAP - `"AB\0"` serves both
    /// `"AB"` at offset 0 and `"B"` at offset 1 - because that overlap is the
    /// whole reason the format stores an offset rather than an index, and a
    /// reader that scanned to the wrong terminator would still look right on
    /// non-overlapping data.
    fn tiny_charsmap() -> Vec<u8> {
        // "AB\0" "é\0"  ->  0:"AB"  1:"B"  3:"é"  5:""
        let blob: &[u8] = b"AB\0\xc3\xa9\0";
        build_charsmap(
            &[
                (b"a", 0),
                (b"b", 1),
                ("e\u{301}".as_bytes(), 3),
                (b"z", 5), // maps to the empty string: a deletion
            ],
            blob,
        )
    }

    #[test]
    fn precompiled_walks_the_trie_and_reads_overlapping_replacements() {
        let p = Precompiled::from_charsmap(&tiny_charsmap()).expect("charsmap parses");

        // Single-byte keys, including the overlapping pair and the deletion.
        assert_eq!(p.normalize_string("a"), "AB");
        assert_eq!(p.normalize_string("b"), "B");
        assert_eq!(p.normalize_string("z"), "");
        // A byte with no entry is passed through untouched, as is a character
        // whose first byte leads nowhere.
        assert_eq!(p.normalize_string("q"), "q");
        assert_eq!(p.normalize_string("cat"), "cABt");
        assert_eq!(p.normalize_string("\u{1F600}"), "\u{1F600}");
        assert_eq!(p.normalize_string(""), "");

        // A 3-byte key reached over three trie hops, and the same bytes as a
        // grapheme cluster: "e" has no entry of its own, so this can only work
        // if the whole cluster was looked up.
        assert_eq!(p.normalize_string("e\u{301}"), "é");
        assert_eq!(p.normalize_string("ze\u{301}a"), "éAB");
    }

    /// The 6-byte threshold in [`Precompiled::normalize_string`] is not a
    /// performance guard - it changes the OUTPUT, and sentencepiece's real
    /// charsmaps depend on which side of it a cluster falls.
    #[test]
    fn precompiled_switches_from_grapheme_to_per_character_at_six_bytes() {
        let p = Precompiled::from_charsmap(&tiny_charsmap()).expect("charsmap parses");

        // One grapheme, 5 bytes: the WHOLE cluster is looked up, the trie's
        // prefix search matches its leading "a", and the combining marks are
        // dropped on the floor.
        let short = "a\u{301}\u{302}";
        assert_eq!(short.len(), 5);
        assert_eq!(p.normalize_string(short), "AB");

        // The same cluster one mark longer is 7 bytes, so each character is
        // transformed on its own and the marks now survive.
        let long = "a\u{301}\u{302}\u{303}";
        assert_eq!(long.len(), 7);
        assert_eq!(p.normalize_string(long), "AB\u{301}\u{302}\u{303}");
    }


    /// End to end through the tokenizer: a `Sequence` of all three steps a real
    /// T5-XXL `tokenizer.json` ships, in its order.
    #[test]
    fn a_precompiled_strip_and_collapse_sequence_runs_in_order() {
        let mut j: serde_json::Value = serde_json::from_str(&tiny_json()).unwrap();
        j["normalizer"] = serde_json::json!({"type": "Sequence", "normalizers": [
            {"type": "Precompiled",
             "precompiled_charsmap": crate::base64::encode(&tiny_charsmap())},
            {"type": "Strip", "strip_left": false, "strip_right": true},
            {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": "\u{2581}"}]});
        let t = UnigramTokenizer::from_json_bytes(j.to_string().as_bytes()).unwrap();

        // "za" -> Precompiled deletes the "z" and rewrites "a" -> "AB"; nothing
        // in the tiny vocabulary covers "A"/"B", so the ids are unk-fused - the
        // point here is the normalized STRING that reached the lattice.
        assert_eq!(t.normalize("za"), "AB");
        // Strip only touches the right, and only after the charsmap ran.
        assert_eq!(t.normalize(" b \t"), " B");
        // A single space survives; two or more become U+2581, not a space.
        assert_eq!(t.normalize("b b"), "B B");
        assert_eq!(t.normalize("b  b"), "B\u{2581}B");
        assert_eq!(t.normalize("b   b"), "B\u{2581}B");
        // The whole chain: charsmap, then right-strip, then collapse.
        assert_eq!(t.normalize("ze\u{301}a  b   "), "éAB\u{2581}B");
    }
}
