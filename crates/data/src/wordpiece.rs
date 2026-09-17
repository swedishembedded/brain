// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WordPiece tokenizer for BERT-family HF `tokenizer.json` checkpoints
//! (all-MiniLM-L6-v2, and every uncased BERT that shares its pipeline).
//!
//! The workspace's **third tokenization family**. [`crate::bpe`],
//! [`crate::clip_bpe`], [`crate::llama_bpe`] and [`crate::qwen_tokenizer`] build
//! a word up by *merging* adjacent pieces in a rank order; [`crate::unigram`]
//! scores every segmentation and takes the best; WordPiece instead eats the
//! word left to right, taking the **longest vocabulary entry that fits** at each
//! position and marking every piece after the first with a continuation prefix
//! (`##`). A word with no match at some position is not partially covered - the
//! whole word becomes `[UNK]`.
//!
//! Four stages, in this order, because that is the order `BertNormalizer` +
//! `BertPreTokenizer` + `WordPiece` + `TemplateProcessing` compose in:
//!
//! 1. **Added tokens** are matched against the RAW input, longest first, and
//!    emitted atomically. Every added token in this family declares
//!    `"normalized": false`, so they are found before normalization can fold
//!    their case. This is what lets a caller write a literal `[SEP]` in a
//!    template string and get token 102 rather than `[`, `sep`, `]`.
//! 2. **Normalize**: `clean_text` DELETES NUL, U+FFFD and every control or
//!    format character (so `null\0and` is one word, not two) and maps the
//!    remaining whitespace to a plain space; `handle_chinese_chars` pads every
//!    CJK ideograph with spaces, which is what makes one ideograph one word;
//!    `strip_accents` decomposes canonically and drops nonspacing marks; then
//!    `lowercase`.
//! 3. **Pre-tokenize**: split on whitespace (dropped), then on punctuation
//!    (kept as its own word).
//! 4. **Post-process**: the checkpoint's `TemplateProcessing` single-sequence
//!    template, `[CLS] A [SEP]`.
//!
//! **`strip_accents: null` means "follow `lowercase`".** An uncased checkpoint
//! leaves the field null and sets `lowercase: true`, so accent stripping is
//! **on** - `café` and `cafe\u{301}` must produce the same ids. Reading null as
//! "off" yields a tokenizer that is correct on ASCII and wrong on everything
//! else, which no downstream accuracy metric would obviously show; see
//! `crates/data/tests/wordpiece_parity.rs`.
//!
//! Padding and truncation are deliberately NOT implemented here even though the
//! checkpoint's file declares both. They are caller policy, not tokenization:
//! the model owns its sequence budget, so [`WordPiece::encode`] returns the bare
//! sequence.
//!
//! Only the single-sequence template is parsed. The pair template
//! (`[CLS] A [SEP] B [SEP]`, with segment ids 0/1) is unused: `crates/decide`
//! separates its state and slot roles by passing the token-type id at the model
//! level, not by encoding two sequences at once.

use std::collections::HashMap;

use serde_json::Value;
use unicode_general_category::{get_general_category, GeneralCategory};

use crate::tokenizer::Tokenizer;
use crate::wordpiece_unicode::{KEYS, OFFS, VALS};

/// Start of the Hangul syllable block, whose decomposition is arithmetic rather
/// than tabulated (see `crates/data/src/wordpiece_unicode.rs`).
const HANGUL_BASE: u32 = 0xAC00;
const HANGUL_COUNT: u32 = 11172;
const JAMO_L: u32 = 0x1100;
const JAMO_V: u32 = 0x1161;
const JAMO_T: u32 = 0x11A7;
/// Vowel-jamo count times trailing-jamo count: the stride of one leading jamo.
const JAMO_VT: u32 = 588;
/// Trailing-jamo count, including "no trailing jamo" at index 0.
const JAMO_T_COUNT: u32 = 28;

pub struct WordPiece {
    vocab: HashMap<String, u32>,
    /// id -> piece, for [`Tokenizer::decode`]. Sized to the largest id present.
    pieces: Vec<String>,
    unk_id: u32,
    /// `continuing_subword_prefix` - `##` for every checkpoint in this family.
    cont: String,
    max_word_chars: usize,
    /// Added/special tokens as `(content, id)`, **longest content first** so the
    /// scan is leftmost-longest and `[MASK]` can never be shadowed by a shorter
    /// entry that happens to be its prefix.
    added: Vec<(String, u32)>,
    /// `TemplateProcessing`'s single-sequence prefix/suffix ids (`[CLS]`/`[SEP]`).
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    clean_text: bool,
    handle_chinese_chars: bool,
    lowercase: bool,
    strip_accents: bool,
}

/// `is_control` in BERT's sense: the C* categories, but **not** `\t`/`\n`/`\r`,
/// which `clean_text` converts to spaces rather than deleting.
///
/// `Unassigned` is included because BERT's rule is "every Other category". A
/// scalar unassigned in this crate's Unicode version but assigned in a newer one
/// would be deleted here and kept by a newer reference tokenizer; that is a
/// difference in UCD vintage, not in this algorithm.
fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => matches!(
            get_general_category(c),
            GeneralCategory::Control
                | GeneralCategory::Format
                | GeneralCategory::PrivateUse
                | GeneralCategory::Surrogate
                | GeneralCategory::Unassigned
        ),
    }
}

/// `is_whitespace` in BERT's sense: Unicode whitespace plus the three ASCII
/// controls `is_control` deliberately spared.
fn is_bert_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

/// What `BertPreTokenizer` isolates into a word of its own: ASCII punctuation
/// (which includes symbols such as `$`, `+`, `<`, `^`, `` ` ``, `|`, `~` that
/// Unicode files under S*, not P*) plus every Unicode P* category.
fn is_bert_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(
            get_general_category(c),
            GeneralCategory::ConnectorPunctuation
                | GeneralCategory::DashPunctuation
                | GeneralCategory::ClosePunctuation
                | GeneralCategory::FinalPunctuation
                | GeneralCategory::InitialPunctuation
                | GeneralCategory::OtherPunctuation
                | GeneralCategory::OpenPunctuation
        )
}

/// The CJK blocks BERT pads with spaces. Deliberately **not** Hangul or kana:
/// those stay glued to their neighbours and are segmented by WordPiece instead.
fn is_chinese_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B920..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// Append `c`'s canonical decomposition with nonspacing marks already removed.
///
/// The composition of both steps is what the generated table stores, so this is
/// one binary search and no intermediate allocation. Hangul is computed rather
/// than looked up, which keeps 11,172 rows out of the table.
fn push_decomposed(c: char, out: &mut String) {
    let cp = c as u32;
    if (HANGUL_BASE..HANGUL_BASE + HANGUL_COUNT).contains(&cp) {
        let s = cp - HANGUL_BASE;
        push_scalar(JAMO_L + s / JAMO_VT, out);
        push_scalar(JAMO_V + (s % JAMO_VT) / JAMO_T_COUNT, out);
        let t = s % JAMO_T_COUNT;
        if t != 0 {
            push_scalar(JAMO_T + t, out);
        }
        return;
    }
    match KEYS.binary_search(&cp) {
        Ok(i) => {
            let (off, len) = ((OFFS[i] >> 8) as usize, (OFFS[i] & 0xFF) as usize);
            for &v in &VALS[off..off + len] {
                push_scalar(v, out);
            }
        }
        Err(_) => out.push(c),
    }
}

/// Every scalar in the generated table and in the Hangul range is a valid
/// `char` by construction, so a failed conversion is a corrupt table, not input.
fn push_scalar(v: u32, out: &mut String) {
    out.push(char::from_u32(v).expect("wordpiece_unicode table holds only valid scalars"));
}

/// One stretch of input: either raw text to be tokenized, or an added token
/// that was matched whole and must not be.
enum Segment<'a> {
    Plain(&'a str),
    Added(u32),
}

impl WordPiece {
    /// Build from a `tokenizer.json` file path.
    pub fn from_file(path: &str) -> Result<WordPiece, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_json_bytes(&bytes)
    }

    /// Build from the bytes of a `tokenizer.json`.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<WordPiece, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("tokenizer.json: {e}"))?;

        let model = v.get("model").ok_or("tokenizer.json: no `model`")?;
        match model.get("type").and_then(Value::as_str) {
            Some("WordPiece") => {}
            other => return Err(format!("tokenizer.json: model.type is {other:?}, expected \"WordPiece\"")),
        }
        let vocab_obj = model.get("vocab").and_then(Value::as_object).ok_or("tokenizer.json: no `model.vocab`")?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0u32;
        for (k, val) in vocab_obj {
            let id = val.as_u64().ok_or_else(|| format!("vocab[{k}] is not an id"))? as u32;
            max_id = max_id.max(id);
            vocab.insert(k.clone(), id);
        }
        let mut pieces = vec![String::new(); max_id as usize + 1];
        for (k, &id) in &vocab {
            pieces[id as usize] = k.clone();
        }

        let unk = model.get("unk_token").and_then(Value::as_str).unwrap_or("[UNK]").to_string();
        let unk_id = *vocab.get(&unk).ok_or_else(|| format!("unk_token {unk:?} is not in the vocab"))?;
        let cont = model.get("continuing_subword_prefix").and_then(Value::as_str).unwrap_or("##").to_string();
        let max_word_chars = model.get("max_input_chars_per_word").and_then(Value::as_u64).unwrap_or(100) as usize;

        // Normalizer. Only `BertNormalizer` is accepted: silently ignoring an
        // unknown one would produce a tokenizer that is subtly wrong rather than
        // absent, which is the harder failure to notice.
        let norm = v.get("normalizer").ok_or("tokenizer.json: no `normalizer`")?;
        match norm.get("type").and_then(Value::as_str) {
            Some("BertNormalizer") => {}
            other => return Err(format!("tokenizer.json: normalizer.type is {other:?}, expected \"BertNormalizer\"")),
        }
        let flag = |k: &str, d: bool| norm.get(k).and_then(Value::as_bool).unwrap_or(d);
        let clean_text = flag("clean_text", true);
        let handle_chinese_chars = flag("handle_chinese_chars", true);
        let lowercase = flag("lowercase", true);
        // The null-means-follow-lowercase rule this module's docs open with.
        let strip_accents = match norm.get("strip_accents") {
            Some(Value::Bool(b)) => *b,
            _ => lowercase,
        };

        match v.get("pre_tokenizer").and_then(|p| p.get("type")).and_then(Value::as_str) {
            Some("BertPreTokenizer") => {}
            other => return Err(format!("tokenizer.json: pre_tokenizer.type is {other:?}, expected \"BertPreTokenizer\"")),
        }

        let mut added: Vec<(String, u32)> = v
            .get("added_tokens")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|t| {
                        let c = t.get("content")?.as_str()?.to_string();
                        let id = t.get("id")?.as_u64()? as u32;
                        Some((c, id))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Longest first: leftmost-LONGEST, so a short entry cannot shadow a
        // longer one that starts at the same position.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

        let (prefix, suffix) = template_affixes(&v)?;

        Ok(WordPiece {
            vocab,
            pieces,
            unk_id,
            cont,
            max_word_chars,
            added,
            prefix,
            suffix,
            clean_text,
            handle_chinese_chars,
            lowercase,
            strip_accents,
        })
    }

    /// `BertNormalizer`, all four stages, in the order they compose.
    fn normalize(&self, s: &str) -> String {
        let mut cur = String::with_capacity(s.len());
        if self.clean_text {
            for c in s.chars() {
                if c == '\0' || c == '\u{fffd}' || is_control(c) {
                    continue;
                }
                cur.push(if is_bert_whitespace(c) { ' ' } else { c });
            }
        } else {
            cur.push_str(s);
        }
        if self.handle_chinese_chars {
            let mut o = String::with_capacity(cur.len());
            for c in cur.chars() {
                if is_chinese_char(c) {
                    o.push(' ');
                    o.push(c);
                    o.push(' ');
                } else {
                    o.push(c);
                }
            }
            cur = o;
        }
        if self.strip_accents {
            let mut o = String::with_capacity(cur.len());
            for c in cur.chars() {
                push_decomposed(c, &mut o);
            }
            cur = o;
        }
        if self.lowercase {
            cur = cur.to_lowercase();
        }
        cur
    }

    /// `BertPreTokenizer`: whitespace splits and is dropped, punctuation splits
    /// and is kept as a word of its own.
    fn pre_tokenize<'a>(&self, s: &'a str, out: &mut Vec<&'a str>) {
        for chunk in s.split(char::is_whitespace) {
            if chunk.is_empty() {
                continue;
            }
            let mut start = 0;
            for (i, c) in chunk.char_indices() {
                if is_bert_punct(c) {
                    if i > start {
                        out.push(&chunk[start..i]);
                    }
                    out.push(&chunk[i..i + c.len_utf8()]);
                    start = i + c.len_utf8();
                }
            }
            if start < chunk.len() {
                out.push(&chunk[start..]);
            }
        }
    }

    /// Greedy longest-match-first over one pre-tokenized word.
    ///
    /// All-or-nothing by design: a word with no vocabulary entry at some
    /// position emits a single `[UNK]` rather than the pieces matched so far.
    fn wordpiece(&self, word: &str, out: &mut Vec<u32>) {
        // Char offsets, plus the end, so every candidate slice is a valid
        // boundary. Shrinking `end` by BYTES would land mid-scalar.
        let bounds: Vec<usize> = word.char_indices().map(|(i, _)| i).chain(std::iter::once(word.len())).collect();
        let n = bounds.len() - 1;
        if n > self.max_word_chars {
            out.push(self.unk_id);
            return;
        }
        let mut sub = Vec::new();
        let mut start = 0usize;
        let mut key = String::new();
        while start < n {
            let mut end = n;
            let mut found = None;
            while start < end {
                key.clear();
                if start > 0 {
                    key.push_str(&self.cont);
                }
                key.push_str(&word[bounds[start]..bounds[end]]);
                if let Some(&id) = self.vocab.get(&key) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    sub.push(id);
                    start = end;
                }
                None => {
                    out.push(self.unk_id);
                    return;
                }
            }
        }
        out.extend(sub);
    }

    /// Split the RAW input around added tokens, longest match first.
    fn split_added<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut segs = Vec::new();
        if self.added.is_empty() {
            if !text.is_empty() {
                segs.push(Segment::Plain(text));
            }
            return segs;
        }
        let mut plain_from = 0usize;
        let mut i = 0usize;
        while i < text.len() {
            let rest = &text[i..];
            let hit = self.added.iter().find(|(c, _)| !c.is_empty() && rest.starts_with(c.as_str()));
            match hit {
                Some((content, id)) => {
                    if i > plain_from {
                        segs.push(Segment::Plain(&text[plain_from..i]));
                    }
                    segs.push(Segment::Added(*id));
                    i += content.len();
                    plain_from = i;
                }
                None => i += rest.chars().next().map(char::len_utf8).unwrap_or(1),
            }
        }
        if plain_from < text.len() {
            segs.push(Segment::Plain(&text[plain_from..]));
        }
        segs
    }

    /// The token ids for `text` with **no** template applied.
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for seg in self.split_added(text) {
            match seg {
                Segment::Added(id) => out.push(id),
                Segment::Plain(s) => {
                    // `words` borrows from `norm`, so both are scoped to this
                    // segment. Hoisting the buffer to reuse its allocation
                    // would outlive the string its entries point into; there
                    // are a handful of segments per input, so the allocation
                    // is not worth restructuring for.
                    let norm = self.normalize(s);
                    let mut words = Vec::new();
                    self.pre_tokenize(&norm, &mut words);
                    for w in &words {
                        self.wordpiece(w, &mut out);
                    }
                }
            }
        }
        out
    }

    /// The id of a vocabulary entry, if present.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// The template's single-sequence prefix ids (`[CLS]`).
    pub fn prefix_ids(&self) -> &[u32] {
        &self.prefix
    }

    /// The template's single-sequence suffix ids (`[SEP]`).
    pub fn suffix_ids(&self) -> &[u32] {
        &self.suffix
    }
}

/// Read `TemplateProcessing`'s single-sequence template as the special-token ids
/// that surround the sequence. A template with specials on BOTH sides of `A`
/// yields `(before, after)`; one with none yields two empty vectors.
fn template_affixes(v: &Value) -> Result<(Vec<u32>, Vec<u32>), String> {
    let Some(post) = v.get("post_processor") else {
        return Ok((Vec::new(), Vec::new()));
    };
    match post.get("type").and_then(Value::as_str) {
        Some("TemplateProcessing") => {}
        None => return Ok((Vec::new(), Vec::new())),
        other => return Err(format!("tokenizer.json: post_processor.type is {other:?}, expected \"TemplateProcessing\"")),
    }
    let single = post.get("single").and_then(Value::as_array).ok_or("post_processor: no `single`")?;
    let specials = post.get("special_tokens").and_then(Value::as_object);
    let ids_of = |name: &str| -> Vec<u32> {
        specials
            .and_then(|m| m.get(name))
            .and_then(|e| e.get("ids"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect())
            .unwrap_or_default()
    };
    let (mut before, mut after) = (Vec::new(), Vec::new());
    let mut seen_sequence = false;
    for item in single {
        if item.get("Sequence").is_some() {
            seen_sequence = true;
        } else if let Some(name) = item.get("SpecialToken").and_then(|s| s.get("id")).and_then(Value::as_str) {
            let ids = ids_of(name);
            if seen_sequence {
                after.extend(ids);
            } else {
                before.extend(ids);
            }
        }
    }
    Ok((before, after))
}

impl Tokenizer for WordPiece {
    /// The template is applied, matching what `Tokenizer.encode` returns for a
    /// single sequence: a caller who wants the bare pieces uses
    /// [`WordPiece::encode_raw`].
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.prefix.len() + self.suffix.len() + text.len() / 3);
        out.extend_from_slice(&self.prefix);
        out.append(&mut self.encode_raw(text));
        out.extend_from_slice(&self.suffix);
        out
    }

    /// Pieces joined by spaces, with `##` continuations glued to the piece
    /// before them. Lossy, as WordPiece always is: casing, accents and any
    /// `[UNK]`ed word are gone by construction.
    fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else { continue };
            match p.strip_prefix(self.cont.as_str()) {
                Some(rest) if !self.cont.is_empty() && !s.is_empty() => s.push_str(rest),
                _ => {
                    if !s.is_empty() {
                        s.push(' ');
                    }
                    s.push_str(p);
                }
            }
        }
        s
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature WordPiece file with an uncased BERT's exact shape. The
    /// vocabulary is chosen so each rule below has a DECIDABLE answer: a
    /// shortest-match or first-match tokenizer produces different ids, rather
    /// than the same ones by luck.
    fn tiny_json(max_word_chars: u32, strip_accents: Value) -> String {
        serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 2, "content": "[CLS]", "special": true},
                {"id": 3, "content": "[SEP]", "special": true},
                {"id": 12, "content": "[MASK]", "special": true},
                // Shares a prefix with [MASK]: only a longest-first scan finds it.
                {"id": 13, "content": "[MASKED]", "special": true}
            ],
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": true,
                "handle_chinese_chars": true,
                "strip_accents": strip_accents,
                "lowercase": true
            },
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
                ],
                "special_tokens": {
                    "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                    "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
                }
            },
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": max_word_chars,
                "vocab": {
                    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                    "un": 4, "unaff": 5, "##able": 6, "##aff": 7,
                    "cafe": 8, "a": 9, "##b": 10, "[MASK]": 12, "[MASKED]": 13,
                    "\u{4e2d}": 14, "\u{6587}": 15, "!": 16, "caf\u{e9}": 17
                }
            }
        })
        .to_string()
    }

    fn tiny() -> WordPiece {
        WordPiece::from_json_bytes(tiny_json(100, Value::Null).as_bytes()).unwrap()
    }

    /// Longest-match-FIRST, not first-match: `unaff` + `##able` beats the
    /// equally valid `un` + `##aff` + `##able` a shortest-first scan finds.
    #[test]
    fn greedy_takes_the_longest_piece_that_fits() {
        assert_eq!(tiny().encode_raw("unaffable"), vec![5, 6]);
    }

    /// All-or-nothing: `a` and `##b` both match, but `z` has no entry, so the
    /// WORD is unknown - the two matched pieces are discarded.
    #[test]
    fn an_unmatchable_tail_unks_the_whole_word() {
        assert_eq!(tiny().encode_raw("abz"), vec![1]);
    }

    /// The `max_input_chars_per_word` cliff is a hard character-count bound,
    /// checked before any piece is looked up.
    #[test]
    fn an_overlong_word_is_one_unk_not_pieces() {
        // `unaff` is a single vocabulary entry, so the limit is the ONLY thing
        // that differs between these two - not whether the word is coverable.
        let at = WordPiece::from_json_bytes(tiny_json(5, Value::Null).as_bytes()).unwrap();
        assert_eq!(at.encode_raw("unaff"), vec![5], "5 chars, limit 5: at the bound, still piece-matched");
        let over = WordPiece::from_json_bytes(tiny_json(4, Value::Null).as_bytes()).unwrap();
        assert_eq!(over.encode_raw("unaff"), vec![1], "5 chars, limit 4: over the bound, one [UNK]");
    }

    /// `strip_accents: null` means "follow `lowercase`". With `lowercase: true`
    /// the accent is stripped, so the precomposed and decomposed spellings of
    /// `café` - and its uppercase form - all reach the same id.
    #[test]
    fn null_strip_accents_follows_lowercase() {
        let t = tiny();
        assert_eq!(t.encode_raw("café"), vec![8]);
        assert_eq!(t.encode_raw("cafe\u{301}"), vec![8]);
        assert_eq!(t.encode_raw("CAFÉ"), vec![8]);
    }

    /// ...and an EXPLICIT `false` turns it off even when lowercasing stays on,
    /// which is the other half of the rule: the accented form then reaches the
    /// accented vocabulary entry instead.
    #[test]
    fn explicit_false_strip_accents_keeps_the_accent() {
        let t = WordPiece::from_json_bytes(tiny_json(100, Value::Bool(false)).as_bytes()).unwrap();
        assert_eq!(t.encode_raw("café"), vec![17]);
        assert_eq!(t.encode_raw("CAFÉ"), vec![17]);
    }

    /// Added tokens are matched leftmost-LONGEST against the raw input, so a
    /// shorter entry that is a prefix of a longer one cannot shadow it.
    #[test]
    fn added_tokens_match_longest_first() {
        let t = tiny();
        assert_eq!(t.encode_raw("[MASKED]"), vec![13]);
        assert_eq!(t.encode_raw("[MASK]"), vec![12]);
    }

    /// Every CJK ideograph is padded with spaces, so it becomes its own word
    /// rather than gluing to its neighbour.
    #[test]
    fn chinese_characters_are_one_word_each() {
        assert_eq!(tiny().encode_raw("中文"), vec![14, 15]);
    }

    /// `clean_text` DELETES control characters rather than replacing them with
    /// a space: `a\0b` is the single word `ab`, not `a` then `b`. Were it a
    /// space, `b` alone would be an unknown word and the ids would differ.
    #[test]
    fn control_characters_are_deleted_not_spaced() {
        assert_eq!(tiny().encode_raw("a\u{0}b"), vec![9, 10]);
    }

    /// Punctuation splits into a word of its own.
    #[test]
    fn punctuation_is_isolated() {
        assert_eq!(tiny().encode_raw("a!"), vec![9, 16]);
    }

    /// `encode` applies the checkpoint's single-sequence template; `encode_raw`
    /// is the same sequence without it.
    #[test]
    fn the_template_wraps_only_encode() {
        let t = tiny();
        assert_eq!(t.encode("a"), vec![2, 9, 3]);
        assert_eq!(t.encode_raw("a"), vec![9]);
        assert_eq!(t.prefix_ids(), &[2]);
        assert_eq!(t.suffix_ids(), &[3]);
    }

    /// A normalizer or model this parser does not implement is refused by name.
    /// Accepting it and ignoring the difference would yield a tokenizer that is
    /// subtly wrong rather than obviously absent.
    #[test]
    fn an_unsupported_pipeline_is_refused_not_ignored() {
        // `unwrap_err` would demand `Debug` on a type holding a 30k-entry map
        // purely to print it on a path that must not be reached.
        let refused = |v: &Value| match WordPiece::from_json_bytes(v.to_string().as_bytes()) {
            Ok(_) => panic!("an unsupported pipeline was accepted: {v}"),
            Err(e) => e,
        };
        let v: Value = serde_json::from_str(&tiny_json(100, Value::Null)).unwrap();
        let mut bad = v.clone();
        bad["model"]["type"] = Value::String("BPE".into());
        assert!(refused(&bad).contains("WordPiece"));
        let mut bad = v;
        bad["normalizer"]["type"] = Value::String("NFKC".into());
        assert!(refused(&bad).contains("BertNormalizer"));
    }
}
