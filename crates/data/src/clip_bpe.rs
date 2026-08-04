// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP byte-level BPE tokenizer — the text front-end for `crates/clip`'s
//! CLIP-L / OpenCLIP-bigG towers (SDXL's `tokenizer/` and `tokenizer_2/`).
//!
//! # Why this is a separate type and not a parameter on [`Gpt2Bpe`]
//!
//! CLIP and GPT-2 share **exactly one** thing: the byte-level BPE *merge loop*.
//! [`bytes_to_unicode`], [`get_pairs`] and [`bpe_merge`] are reused verbatim
//! from [`crate::bpe`] — this module adds no merge code. Everything else
//! differs, and none of it is a knob:
//!
//! | | GPT-2 | CLIP |
//! |---|---|---|
//! | word boundary | leading-space byte (`Ġ`), no marker | explicit `</w>` suffix on the last symbol |
//! | pre-tokenization | ` ?\p{L}+ \| ?\p{N}+ \| ?[^\s\p{L}\p{N}]+ \| \s+` | `[\p{L}]+ \| [\p{N}] \| [^\s\p{L}\p{N}]+`, whitespace **dropped** |
//! | digits | runs (`4567` is one pre-token) | **one digit per pre-token** |
//! | normalization | none | lowercase + whitespace collapse |
//! | framing | none | `<\|startoftext\|>` … `<\|endoftext\|>`, fixed 77 context |
//! | vocab | embedded, 50257 | loaded, 49408 |
//!
//! A `Gpt2Bpe` with a "clip mode" flag would be a different pre-tokenizer, a
//! different word representation and a different framing behind one type — i.e.
//! two tokenizers sharing a name. The merge loop, which is the part that could
//! actually drift, IS shared.
//!
//! # The two SDXL tokenizers
//!
//! `tokenizer/` (CLIP-L) and `tokenizer_2/` (OpenCLIP-bigG) ship **byte-identical
//! `vocab.json` and `merges.txt`**; they differ only in `pad_token`
//! (`<|endoftext|>` = 49407 vs `!` = 0).
//!
//! That difference is **not** confined to the padding, and assuming it is
//! produces wrong ids. `tokenizer_2` registers its pad token `!` as an *added*
//! token, and HuggingFace splits text on added tokens BEFORE the BPE — so every
//! literal `!` in a prompt becomes id 0 there, while `tokenizer/` BPEs it
//! normally. Verified against `transformers`:
//!
//! ```text
//!            tokenizer/            tokenizer_2/
//! "a!b"      [320, 256, 321]       [320, 0, 321]
//! "wow!!"    [2781, 748]           [2781, 0, 0]
//! "!!!"      [995]                 [0, 0, 0]
//! ```
//!
//! [`ClipBpe::with_pad`] therefore registers the pad token as a splitting
//! special as well as setting the pad id.
//!
//! # Fidelity note
//!
//! HuggingFace's `CLIPTokenizer` runs `ftfy.fix_text` before lowercasing when
//! ftfy is installed. `fix_text` repairs mojibake, un-curls quotes, expands
//! Latin ligatures and NFC-normalizes; on already-well-formed NFC text it is the
//! identity, which is the case this tokenizer reproduces. Text that needs
//! *repair* is out of scope and would need a Rust ftfy — say so rather than
//! pretend the outputs match.

use std::collections::HashMap;

use crate::bpe::{bpe_merge, bytes_to_unicode};

/// `<|startoftext|>` — CLIP's BOS.
pub const BOS: &str = "<|startoftext|>";
/// `<|endoftext|>` — CLIP's EOS, and CLIP-L's pad token.
pub const EOS: &str = "<|endoftext|>";
/// The end-of-word marker CLIP appends to the last symbol of every pre-token.
const WORD_END: &str = "</w>";
/// The context both SDXL text encoders are built for (`model_max_length`).
pub const CONTEXT: usize = 77;

/// CLIP byte-level BPE.
pub struct ClipBpe {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), u32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
    /// Tokens that split the text before pre-tokenization (HF "added tokens"),
    /// longest first so a prefix never wins over a longer match.
    specials: Vec<String>,
}

/// One encoded prompt at the fixed context: ids, mask, and where the EOS landed.
///
/// `eos_index` is what the text towers pool at (CLIP-L takes the hidden state at
/// the EOS position), so it is returned rather than re-derived by each caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    /// `[CONTEXT]` ids: BOS, content, EOS, then padding.
    pub ids: Vec<u32>,
    /// `[CONTEXT]` 1/0 mask — 1 for BOS..=EOS, 0 for padding.
    pub mask: Vec<u32>,
    /// Index of the EOS token (= number of real tokens − 1).
    pub eos_index: usize,
}

impl ClipBpe {
    /// Build from the raw contents of `vocab.json` and `merges.txt`.
    ///
    /// Panics on malformed input: these are checkpoint assets, and a tokenizer
    /// that silently drops merges produces ids that are wrong but plausible.
    pub fn from_str(vocab_json: &str, merges_txt: &str) -> ClipBpe {
        let raw: HashMap<String, u32> = serde_json::from_str(vocab_json).expect("vocab.json is valid JSON");
        let mut encoder: HashMap<String, u32> = HashMap::with_capacity(raw.len());
        let mut decoder: HashMap<u32, String> = HashMap::with_capacity(raw.len());
        for (tok, id) in raw {
            decoder.insert(id, tok.clone());
            encoder.insert(tok, id);
        }

        // `merges.txt`: a `#version` header then "left right" per line, in
        // merge-priority order. The header is only skipped when it IS one — the
        // OpenCLIP dumps of the same table do not always carry it.
        let mut bpe_ranks: HashMap<(String, String), u32> = HashMap::new();
        let mut rank = 0u32;
        for line in merges_txt.lines() {
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let mut it = line.split(' ');
            let left = it.next().expect("merge line has a left token").to_string();
            let right = it.next().expect("merge line has a right token").to_string();
            bpe_ranks.insert((left, right), rank);
            rank += 1;
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }

        let id_of = |t: &str| *encoder.get(t).unwrap_or_else(|| panic!("vocab is missing {t}"));
        let (bos_id, eos_id) = (id_of(BOS), id_of(EOS));
        let specials = vec![BOS.to_string(), EOS.to_string()];
        ClipBpe { encoder, decoder, bpe_ranks, byte_encoder, byte_decoder, bos_id, eos_id, pad_id: eos_id, specials }
    }

    /// Build from a HuggingFace tokenizer directory (`vocab.json` +
    /// `merges.txt`). The path comes from the caller — never a baked-in default.
    pub fn from_dir(dir: &std::path::Path) -> std::io::Result<ClipBpe> {
        let vocab = std::fs::read_to_string(dir.join("vocab.json"))?;
        let merges = std::fs::read_to_string(dir.join("merges.txt"))?;
        Ok(ClipBpe::from_str(&vocab, &merges))
    }

    /// Switch to a different pad token — the whole difference between SDXL's
    /// `tokenizer/` (pad `<|endoftext|>`, the default) and `tokenizer_2/`
    /// (pad `!`, id 0).
    ///
    /// This is NOT only about what fills the tail: the pad token also becomes an
    /// added token, so it splits the text before the BPE and changes the CONTENT
    /// ids of any prompt containing it (see the module docs' table).
    pub fn with_pad(mut self, pad_token: &str) -> ClipBpe {
        self.pad_id = *self.encoder.get(pad_token).unwrap_or_else(|| panic!("vocab is missing pad token {pad_token}"));
        if !self.specials.iter().any(|s| s == pad_token) {
            self.specials.push(pad_token.to_string());
            // Longest-first: `<|endoftext|>` must not lose to a one-char token
            // that happens to start at the same offset.
            self.specials.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        }
        self
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Content ids only — no BOS/EOS, no padding, no truncation.
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pre_tokenize(&clean(text), &self.specials) {
            match self.encoder.get(&piece) {
                // An added token: it stands for itself, no BPE, no `</w>`.
                Some(&id) if self.specials.contains(&piece) => out.push(id),
                _ => self.encode_piece(&piece, &mut out),
            }
        }
        out
    }

    /// The tokenizer call the text encoders actually make: lowercase + collapse
    /// whitespace, BPE, wrap in BOS/EOS, truncate to `CONTEXT` **including** the
    /// two specials, then pad to `CONTEXT`.
    pub fn encode(&self, text: &str) -> Encoded {
        self.encode_with_context(text, CONTEXT)
    }

    /// [`ClipBpe::encode`] at a caller-chosen context (77 for both SDXL towers;
    /// parameterised so a longer-context CLIP variant does not need a fork).
    pub fn encode_with_context(&self, text: &str, context: usize) -> Encoded {
        assert!(context >= 2, "context {context} cannot hold even BOS+EOS");
        let mut ids = self.encode_raw(text);
        ids.truncate(context - 2);
        let mut full = Vec::with_capacity(context);
        full.push(self.bos_id);
        full.extend_from_slice(&ids);
        full.push(self.eos_id);
        let eos_index = full.len() - 1;
        let mask: Vec<u32> = (0..context).map(|i| u32::from(i < full.len())).collect();
        full.resize(context, self.pad_id);
        Encoded { ids: full, mask, eos_index }
    }

    /// Inverse of [`ClipBpe::encode_raw`]: token strings concatenated, `</w>`
    /// rendered as the space it stands for, bytes mapped back and UTF-8 decoded.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(tok) = self.decoder.get(&id) else { continue };
            if self.specials.contains(tok) {
                continue;
            }
            let (body, ends_word) = match tok.strip_suffix(WORD_END) {
                Some(b) => (b, true),
                None => (tok.as_str(), false),
            };
            for c in body.chars() {
                if let Some(&b) = self.byte_decoder.get(&c) {
                    bytes.push(b);
                }
            }
            if ends_word {
                bytes.push(b' ');
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        s.trim_end().to_string()
    }

    /// One pre-token: bytes → byte-map chars, `</w>` onto the LAST symbol, then
    /// the shared merge loop.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<String> = piece.bytes().map(|b| self.byte_encoder[b as usize].to_string()).collect();
        if symbols.is_empty() {
            return;
        }
        let last = symbols.len() - 1;
        symbols[last].push_str(WORD_END);
        for sub in bpe_merge(&self.bpe_ranks, symbols) {
            // Unknown subwords cannot occur: every single byte-map char plus
            // `</w>` is in the vocab, so the merge loop's output always is too.
            let id = *self.encoder.get(&sub).unwrap_or_else(|| panic!("BPE produced {sub:?}, absent from the vocab"));
            out.push(id);
        }
    }
}

// `ClipBpe` deliberately does NOT implement `crate::tokenizer::Tokenizer`. That
// trait's `encode -> Vec<u32>` would collide by name with the inherent
// `ClipBpe::encode`, which returns a framed+padded `Encoded` — and an inherent
// method silently wins over a trait one at every call site, so the two would
// differ by which import happened to be in scope. Callers that want bare ids
// call `ClipBpe::encode_raw` explicitly.
//
// (A `//` block, not `///`: a doc comment here would attach to `clean` below.)

/// CLIP's text normalization: collapse every whitespace run to one space, trim,
/// lowercase (`whitespace_clean(...).lower()` in the reference).
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        out.push(c);
    }
    out.to_lowercase()
}

/// Hand-written reproduction of CLIP's pre-tokenization pattern
/// `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`.
///
/// Three differences from [`crate::bpe`]'s GPT-2 scanner, each load-bearing:
/// there is **no** optional leading space (word boundaries ride on `</w>`),
/// **numbers are single characters** (`[\p{N}]`, not `[\p{N}]+`), and
/// whitespace matches no alternative so it is **dropped**.
///
/// `\p{L}` and `\p{N}` are [`is_letter`] / [`is_number`]; see those for exactly
/// how far they agree with the reference.
/// `\p{N}`. `char::is_numeric` is `Nd | Nl | No`, which is **exactly** `\p{N}` —
/// verified over all 1 114 112 scalar values against Python `regex`: 0
/// disagreements.
fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// `\p{L}` — general category `L*`.
///
/// `char::is_alphabetic` is the `Alphabetic` *property*, which is
/// `L* | Nl | Other_Alphabetic`, so it is a strict superset. Subtracting
/// [`is_number`] removes the `Nl` part exactly (roman numerals: `Ⅶ` is `Nl`, so
/// `is_alphabetic` alone glued it to the following word and moved the `</w>` —
/// that was a real mismatch against `transformers`).
///
/// **Known residual gap, measured over every scalar value: 1510 codepoints where
/// this says letter and `\p{L}` does not** — 905 `Mn` + 423 `Mc` (combining
/// marks: Indic matras, Thai/Hebrew/Arabic vowel signs), 130 `So` (circled
/// letters), 52 unassigned-in-that-Unicode-version. There are **0** codepoints
/// in the other direction. Consequence: for a script whose text carries
/// combining marks, a mark is folded into the preceding letter run instead of
/// starting a `[^\s\p{L}\p{N}]+` pre-token, which moves the `</w>`. Latin, Greek,
/// Cyrillic, CJK, Hangul, kana, digits, punctuation and emoji are unaffected —
/// the fuzz corpora in `tests/clip_tokenizer_parity.rs` cover those exactly.
/// Closing it needs the `Other_Alphabetic` range table (255 ranges); until then
/// this is a stated limitation, not a silent one.
fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_numeric()
}

fn pre_tokenize(text: &str, specials: &[String]) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens: Vec<String> = Vec::new();
    let mut i = 0;

    while i < n {
        // Added tokens first — HF splits on them before the pattern runs, so
        // they must survive `[^\s\p{L}\p{N}]+` swallowing `<|` (and `!`).
        let rest: String = chars[i..].iter().collect();
        if let Some(sp) = specials.iter().find(|s| rest.starts_with(s.as_str())) {
            tokens.push(sp.clone());
            i += sp.chars().count();
            continue;
        }
        if chars[i] == '\'' {
            if let Some(c) = CONTRACTIONS.iter().find(|c| rest.starts_with(*c)) {
                tokens.push((*c).to_string());
                i += c.chars().count();
                continue;
            }
        }
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        if is_letter(chars[i]) {
            let mut j = i;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            tokens.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if is_number(chars[i]) {
            // ONE digit per pre-token — the pattern is `[\p{N}]`, not `[\p{N}]+`.
            tokens.push(chars[i].to_string());
            i += 1;
            continue;
        }
        // `[^\s\p{L}\p{N}]+`, stopping only before an added token.
        //
        // It must NOT stop before a contraction. Alternation priority in
        // Python's `re` applies at the position a match STARTS, not inside a
        // greedy run: `%'s` is `["%'", "s"]` and `''re` is `["''", "re"]`,
        // because once `[^\s\p{L}\p{N}]+` starts at `%`/`'` it swallows the
        // following `'` and the engine never reconsiders the `'s`/`'re`
        // alternatives. Breaking out here produced `["%", "'s"]` — verified
        // wrong against `transformers` on 351/3000 fuzzed strings.
        //
        // Added tokens ARE different: HF splits the text on them *before* the
        // pattern runs, so that break is real (it is what makes `tokenizer_2`'s
        // `!` split `wow!!`).
        let mut j = i;
        while j < n && !chars[j].is_whitespace() && !is_letter(chars[j]) && !is_number(chars[j]) {
            if j > i {
                let tail: String = chars[j..].iter().collect();
                if specials.iter().any(|s| tail.starts_with(s.as_str())) {
                    break;
                }
            }
            j += 1;
        }
        tokens.push(chars[i..j].iter().collect());
        i = j;
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_collapses_and_lowercases() {
        assert_eq!(clean("  Hello,\tWorld!  \n\n MiXeD  "), "hello, world! mixed");
        assert_eq!(clean(""), "");
        assert_eq!(clean("   "), "");
    }

    fn base() -> Vec<String> {
        vec![BOS.to_string(), EOS.to_string()]
    }

    #[test]
    fn pre_tokenize_splits_digits_singly_and_drops_whitespace() {
        let sp = base();
        assert_eq!(pre_tokenize("digits 0123 x", &sp), vec!["digits", "0", "1", "2", "3", "x"]);
        assert_eq!(pre_tokenize("a  b", &sp), vec!["a", "b"]);
        assert_eq!(pre_tokenize("don't", &sp), vec!["don", "'t"]);
        assert_eq!(pre_tokenize("#@$%", &sp), vec!["#@$%"]);
    }

    /// A greedy `[^\s\p{L}\p{N}]+` run must NOT stop for a contraction: in
    /// Python's `re` the `'s`/`'re` alternatives only win where a match STARTS.
    #[test]
    fn a_symbol_run_swallows_the_apostrophe_of_a_contraction() {
        let sp = base();
        assert_eq!(pre_tokenize("50%'s", &sp), vec!["5", "0", "%'", "s"]);
        assert_eq!(pre_tokenize("''re", &sp), vec!["''", "re"]);
        assert_eq!(pre_tokenize(">'rer", &sp), vec![">'", "rer"]);
        // ...but at a match start it does win.
        assert_eq!(pre_tokenize("wow's", &sp), vec!["wow", "'s"]);
    }

    /// `\p{L}` is category `L*`; `char::is_alphabetic` also covers `Nl`, so a
    /// roman numeral would otherwise glue onto the next word.
    #[test]
    fn roman_numerals_are_numbers_not_letters() {
        let sp = base();
        // (`pre_tokenize` runs after `clean`; called bare here, so no lowercasing.)
        assert_eq!(pre_tokenize("\u{2167}xyz", &sp), vec!["\u{2167}", "xyz"]);
        assert_eq!(pre_tokenize("\u{2460}\u{bd}", &sp), vec!["\u{2460}", "\u{bd}"]);
    }

    #[test]
    fn pre_tokenize_keeps_specials_whole() {
        assert_eq!(
            pre_tokenize("<|startoftext|>a<|endoftext|>", &base()),
            vec!["<|startoftext|>", "a", "<|endoftext|>"]
        );
    }

    /// `tokenizer_2`'s pad token `!` is an ADDED token: it splits runs that
    /// `tokenizer/` would have merged (`wow!!` -> one `!!` symbol there).
    #[test]
    fn an_added_pad_token_splits_the_text() {
        let mut sp = base();
        sp.push("!".to_string());
        sp.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        assert_eq!(pre_tokenize("wow!!", &sp), vec!["wow", "!", "!"]);
        assert_eq!(pre_tokenize("a!b", &sp), vec!["a", "!", "b"]);
        assert_eq!(pre_tokenize("wow!!", &base()), vec!["wow", "!!"]);
    }
}
