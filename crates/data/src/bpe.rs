// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GPT-2 byte-level Byte-Pair-Encoding (BPE) tokenizer.
//!
//! A faithful, dependency-free (apart from `serde_json` for parsing the
//! embedded `encoder.json`) reimplementation of OpenAI's GPT-2 tokenizer.
//!
//! The original Python reference is `openai/gpt-2` `encoder.py`. The algorithm
//! has four parts:
//!
//! 1. **`bytes_to_unicode`** — a reversible map from the 256 byte values to
//!    Unicode code points. Printable byte ranges map to themselves; the rest
//!    map to code points `256 + n` so that *every* byte becomes a printable,
//!    non-whitespace character. This lets BPE operate purely on strings while
//!    still being able to represent arbitrary bytes.
//! 2. **Pre-tokenization** — a regex splits the text into chunks (contractions,
//!    space-prefixed word/number/symbol runs, whitespace runs). We have no
//!    regex crate, so [`pre_tokenize`] is a bespoke scanner reproducing
//!    the exact GPT-2 pattern:
//!    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//! 3. **BPE merges** — each pre-token's bytes are mapped through
//!    `bytes_to_unicode`, then greedily merged using the rank table loaded from
//!    `vocab.bpe` (lowest rank merged first), and each resulting subword is
//!    looked up in the encoder to get its id.
//! 4. **Decode** — concatenate the token strings, map each character back to
//!    its original byte via the inverse table, then UTF-8-decode (lossily).

use std::collections::HashMap;

use crate::tokenizer::Tokenizer;

/// Embedded GPT-2 vocabulary: JSON object mapping token string -> id.
const ENCODER_JSON: &str = include_str!("../assets/encoder.json");
/// Embedded GPT-2 merge rules: `#version` line then space-separated pairs in
/// merge-priority order.
const VOCAB_BPE: &str = include_str!("../assets/vocab.bpe");

/// GPT-2 byte-level BPE tokenizer.
pub struct Gpt2Bpe {
    /// token string -> id
    encoder: HashMap<String, u32>,
    /// id -> token string (inverse of `encoder`)
    decoder: HashMap<u32, String>,
    /// (left, right) merge pair -> rank (lower = higher priority)
    bpe_ranks: HashMap<(String, String), u32>,
    /// byte value -> unicode char used in the string domain
    byte_encoder: [char; 256],
    /// unicode char -> byte value (inverse of `byte_encoder`)
    byte_decoder: HashMap<char, u8>,
}

/// Build the reversible GPT-2 `bytes_to_unicode` table.
///
/// Returns an array indexed by byte value giving the `char` it maps to. The
/// "safe" printable ranges (`!`..=`~`, `¡`..=`¬`, `®`..=`ÿ`) map to themselves;
/// every other byte is assigned a fresh code point starting at `256`, in
/// ascending byte order. This guarantees a bijection between the 256 bytes and
/// 256 distinct printable, non-whitespace characters.
///
/// `pub` because it is the exact vocabulary a caller needs to build a
/// merge-free byte-level [`crate::qwen_tokenizer::QwenBpe`] in memory (a
/// checkpoint-free tokenizer fixture, `QwenBpe::from_gguf` over one token per
/// byte). That table has one correct value and it is this one, so a caller
/// that cannot reach it re-derives the reference's own ranges by hand - which
/// is the copy this workspace's one-implementation rule exists to prevent.
pub fn bytes_to_unicode() -> [char; 256] {
    // The directly-mapped, printable byte ranges (inclusive) from the reference.
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xACu32);
    bs.extend(0xAEu32..=0xFFu32);

    // `cs` starts as a copy of `bs`; bytes not already covered get new code
    // points 256, 257, ... in ascending order.
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut table = ['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs) {
        table[b as usize] = char::from_u32(c).expect("valid code point");
    }
    table
}

/// Apply the BPE merge loop to a single pre-token already expressed as a vector
/// of single-character symbol strings (each char is a byte-mapped unicode
/// char), using `ranks` (`(left,right) -> priority`, lower = merged first).
/// Shared by [`Gpt2Bpe`] and the Qwen tokenizer — only the rank table differs.
///
/// The best pair is taken from a min-heap over the adjacent pairs of a linked
/// list of symbols, so a piece of `n` symbols costs `O(n log n)` instead of
/// the `O(n^2)` of rescanning every pair after each merge: a prompt with no
/// whitespace (a minified file, base64, the llama tokenizer's whole-input
/// piece) is one piece, and the quadratic form held the serving lane for
/// seconds. The result is the textbook loop's: the lowest rank first, the
/// leftmost occurrence among equals, so overlapping runs (`a a a`) merge left
/// to right.
pub(crate) fn bpe_merge(ranks: &HashMap<(String, String), u32>, token_chars: Vec<String>) -> Vec<String> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    const NONE: usize = usize::MAX;
    let n = token_chars.len();
    if n <= 1 {
        return token_chars;
    }
    let mut sym = token_chars;
    let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
    prev[0] = NONE;
    let mut next: Vec<usize> = (1..=n).collect();
    next[n - 1] = NONE;
    // Bumped whenever a symbol's text changes, so a queued pair that names an older text is recognised as stale.
    let mut version = vec![0u32; n];
    let mut alive = vec![true; n];

    // (rank, left, right, left version, right version): ordered by rank, then position.
    type Entry = Reverse<(u32, usize, usize, u32, u32)>;
    let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<Entry>, sym: &[String], version: &[u32], l: usize, r: usize| {
        if let Some(&rank) = ranks.get(&(sym[l].clone(), sym[r].clone())) {
            heap.push(Reverse((rank, l, r, version[l], version[r])));
        }
    };
    for l in 0..n - 1 {
        push(&mut heap, &sym, &version, l, l + 1);
    }
    while let Some(Reverse((_, l, r, vl, vr))) = heap.pop() {
        if !alive[l] || !alive[r] || next[l] != r || version[l] != vl || version[r] != vr {
            continue;
        }
        let right = std::mem::take(&mut sym[r]);
        sym[l].push_str(&right);
        version[l] += 1;
        alive[r] = false;
        next[l] = next[r];
        if next[r] != NONE {
            prev[next[r]] = l;
        }
        if prev[l] != NONE {
            push(&mut heap, &sym, &version, prev[l], l);
        }
        if next[l] != NONE {
            push(&mut heap, &sym, &version, l, next[l]);
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i != NONE {
        out.push(std::mem::take(&mut sym[i]));
        i = next[i];
    }
    out
}

impl Gpt2Bpe {
    /// Build from the embedded GPT-2 `encoder.json` + `vocab.bpe`.
    ///
    /// Panics only on (impossible) malformed embedded assets.
    pub fn new() -> Self {
        // Parse encoder.json: { "token": id, ... }.
        let raw: HashMap<String, u32> =
            serde_json::from_str(ENCODER_JSON).expect("embedded encoder.json is valid JSON");
        let mut encoder: HashMap<String, u32> = HashMap::with_capacity(raw.len());
        let mut decoder: HashMap<u32, String> = HashMap::with_capacity(raw.len());
        for (tok, id) in raw {
            decoder.insert(id, tok.clone());
            encoder.insert(tok, id);
        }

        // Parse vocab.bpe: skip the `#version` header line, then each remaining
        // line is "left right"; line index is the merge rank (priority).
        let mut bpe_ranks: HashMap<(String, String), u32> = HashMap::new();
        for (rank, line) in VOCAB_BPE
            .lines()
            .skip(1) // drop "#version: 0.2"
            .filter(|l| !l.is_empty())
            .enumerate()
        {
            let mut it = line.split(' ');
            let left = it.next().expect("merge line has a left token").to_string();
            let right = it.next().expect("merge line has a right token").to_string();
            bpe_ranks.insert((left, right), rank as u32);
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }

        Gpt2Bpe {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
        }
    }

    /// Encode a single pre-token (a substring of the original text) into ids.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        // Map each raw UTF-8 byte of the piece to its unicode char.
        let token_chars: Vec<String> = piece
            .bytes()
            .map(|b| self.byte_encoder[b as usize].to_string())
            .collect();
        for sub in bpe_merge(&self.bpe_ranks, token_chars) {
            let id = *self
                .encoder
                .get(&sub)
                .expect("every BPE subword exists in the encoder");
            out.push(id);
        }
    }
}

impl Default for Gpt2Bpe {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of a character for the pre-tokenization scanner.
///
/// For the ASCII corpora used by the brain (Shakespeare, etc.) we treat
/// `\p{L}` as ASCII alphabetic and `\p{N}` as ASCII digit. Non-ASCII bytes
/// (>= 0x80) are *not* letters/numbers here; they fall into the "other"
/// category and are handled byte-wise via `bytes_to_unicode` afterwards. This
/// matches the reference closely for ASCII text, which is what we tokenize.
fn is_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_ascii_digit()
}
fn is_space(c: char) -> bool {
    // GPT-2's `\s`: ASCII whitespace. Unicode whitespace is irrelevant for the
    // ASCII datasets.
    c.is_whitespace()
}

/// Bespoke reproduction of the GPT-2 pre-tokenization regex:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
///
/// Returns the list of substrings (pre-tokens), in order, that together
/// reconstruct `text`. The scanner is greedy and tries each alternative in the
/// pattern's order at every position.
pub(crate) fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens: Vec<String> = Vec::new();
    let mut i = 0;

    // Recognized contractions (the apostrophe variants). Order matters only in
    // that longer ones must be tried before their prefixes are confused, but
    // since they are disjoint after the leading char we check each fully.
    const CONTRACTIONS: [&[char]; 7] = [
        &['\'', 's'],
        &['\'', 't'],
        &['\'', 'r', 'e'],
        &['\'', 'v', 'e'],
        &['\'', 'm'],
        &['\'', 'l', 'l'],
        &['\'', 'd'],
    ];

    while i < n {
        // 1) Contractions: `'s |'t |'re |'ve |'m |'ll |'d`.
        if chars[i] == '\'' {
            let mut matched = None;
            for c in CONTRACTIONS {
                if i + c.len() <= n && chars[i..i + c.len()] == *c {
                    matched = Some(c.len());
                    break;
                }
            }
            if let Some(len) = matched {
                tokens.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
            // A lone apostrophe not forming a contraction falls through to the
            // "other" category below.
        }

        // Determine whether there is an optional single leading space, i.e.
        // ` ?` followed by a letter/number/other run.
        let has_leading_space = chars[i] == ' ';
        let run_start = i;
        // The character that begins the run (after an optional leading space).
        let cls_idx = if has_leading_space { i + 1 } else { i };

        // 2) ` ?\p{L}+`  3) ` ?\p{N}+`  4) ` ?[^\s\p{L}\p{N}]+`
        if cls_idx < n && (is_letter(chars[cls_idx]) || is_number(chars[cls_idx])) {
            // Letters run or numbers run. The optional leading space is only
            // consumed if a run of the matching class actually follows.
            let letters = is_letter(chars[cls_idx]);
            let mut j = cls_idx;
            while j < n && (if letters { is_letter(chars[j]) } else { is_number(chars[j]) }) {
                j += 1;
            }
            tokens.push(chars[run_start..j].iter().collect());
            i = j;
            continue;
        }
        if cls_idx < n && !is_space(chars[cls_idx]) {
            // "Other" run: non-space, non-letter, non-number characters
            // (punctuation, symbols, and any non-ASCII char under our ASCII
            // letter/number definition). Optional leading space included.
            let mut j = cls_idx;
            while j < n && !is_space(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                j += 1;
            }
            tokens.push(chars[run_start..j].iter().collect());
            i = j;
            continue;
        }

        // 5) Whitespace runs: `\s+(?!\S)` then `\s+`.
        //
        // Find the maximal whitespace run `[i..j)`, then split it per the two
        // alternatives in the GPT-2 pattern:
        //   - `\s+(?!\S)`: a whitespace run NOT followed by a non-space. When
        //     the run reaches end-of-text (`j == n`), the whole run matches as
        //     one token.
        //   - `\s+`: otherwise (run followed by a non-space char). Crucially,
        //     the *last* whitespace char of the run is reserved so it can be
        //     consumed as the optional leading space of the FOLLOWING token by
        //     rules 2-4. So we emit `[i..j-1)` as one token IF that is
        //     non-empty (run length >= 2), and re-scan from `j-1`. If the run
        //     is a single whitespace char followed by a non-space, there is
        //     nothing to emit early; instead this lone whitespace is emitted as
        //     its own token here (the ` ?` rules only ever absorb a literal
        //     leading SPACE, e.g. a `\t` before a word stays standalone). We
        //     must always advance `i` to avoid looping.
        if is_space(chars[i]) {
            let mut j = i;
            while j < n && is_space(chars[j]) {
                j += 1;
            }
            if j == n {
                // Trailing whitespace run at end of text: emit it whole.
                tokens.push(chars[i..j].iter().collect());
                i = j;
            } else if j - 1 > i {
                // Run of >= 2 spaces before a non-space: emit all but the last
                // and leave the final whitespace char for the next token's
                // optional leading space.
                tokens.push(chars[i..j - 1].iter().collect());
                i = j - 1;
            } else {
                // Single whitespace char before a non-space. If it is a literal
                // space and the next char starts a letter/number/other run, the
                // ` ?` rules (2-4) should own it — but those rules already run
                // BEFORE this branch and only reach here when they declined
                // (e.g. the whitespace is a tab/newline, which ` ?` never
                // absorbs). Emit it as a standalone whitespace token and
                // advance to guarantee progress.
                tokens.push(chars[i..j].iter().collect());
                i = j;
            }
            continue;
        }

        // Should be unreachable, but advance to avoid an infinite loop.
        tokens.push(chars[i..i + 1].iter().collect());
        i += 1;
    }

    tokens
}

impl Tokenizer for Gpt2Bpe {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pre_tokenize(text) {
            self.encode_piece(&piece, &mut out);
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> String {
        // Concatenate token strings, then map each unicode char back to a byte.
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.decoder.get(&id) {
                for c in tok.chars() {
                    if let Some(&b) = self.byte_decoder.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.encoder.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
/// All adjacent symbol pairs of `word`, deduplicated, preserving first-seen
/// order is unimportant (we only test membership / pick min rank), so a `Vec`
/// of unique pairs suffices.
fn get_pairs(word: &[String]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for w in word.windows(2) {
        let pair = (w[0].clone(), w[1].clone());
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    pairs
}


    /// The textbook merge loop (rescan every pair, merge every non-overlapping occurrence of the best one): the
    /// definition `bpe_merge` must reproduce.
    fn reference_merge(ranks: &HashMap<(String, String), u32>, token_chars: Vec<String>) -> Vec<String> {
        if token_chars.len() <= 1 {
            return token_chars;
        }
        let mut word = token_chars;
        loop {
            let mut best: Option<((String, String), u32)> = None;
            for pair in get_pairs(&word) {
                if let Some(&rank) = ranks.get(&pair) {
                    if best.as_ref().is_none_or(|(_, br)| rank < *br) {
                        best = Some((pair, rank));
                    }
                }
            }
            let Some(((first, second), _)) = best else { break };
            let mut merged_word = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged_word.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged_word;
            if word.len() == 1 {
                break;
            }
        }
        word
    }

    /// Ranks as a trained merge table has them (a merge's result exists before any merge that uses it), on a small
    /// alphabet so equal pairs repeat and overlap.
    fn trained_ranks(seed: u64, merges: usize) -> HashMap<(String, String), u32> {
        let mut state = seed;
        let mut next = move |m: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % m
        };
        let mut vocab: Vec<String> = "abc".chars().map(String::from).collect();
        let mut ranks = HashMap::new();
        while ranks.len() < merges {
            let (l, r) = (vocab[next(vocab.len())].clone(), vocab[next(vocab.len())].clone());
            // One merge per vocabulary entry, as a trained table has it.
            let merged = format!("{l}{r}");
            if !vocab.contains(&merged) && ranks.insert((l, r), ranks.len() as u32).is_none() {
                vocab.push(merged);
            }
        }
        ranks
    }

    #[test]
    fn the_heap_merge_equals_the_textbook_loop() {
        for seed in 0..40u64 {
            let ranks = trained_ranks(seed, 6 + (seed as usize % 20));
            let mut state = seed + 1000;
            for len in [2usize, 3, 5, 12, 40, 150] {
                let word: Vec<String> = (0..len)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ["a", "b", "c"][((state >> 33) % 3) as usize].to_string()
                    })
                    .collect();
                assert_eq!(bpe_merge(&ranks, word.clone()), reference_merge(&ranks, word.clone()), "seed {seed} word {word:?}");
            }
        }
    }

    /// A 100 000 symbol piece (no whitespace to split it) merges in well under a second: the textbook loop takes
    /// minutes on it.
    #[test]
    fn a_long_piece_without_whitespace_merges_in_linear_log_time() {
        let ranks = trained_ranks(7, 30);
        let word: Vec<String> = (0..100_000).map(|i| ["a", "b", "c"][(i * 7 + i / 3) % 3].to_string()).collect();
        let start = std::time::Instant::now();
        let merged = bpe_merge(&ranks, word);
        assert!(!merged.is_empty());
        assert!(start.elapsed() < std::time::Duration::from_secs(2), "took {:?}", start.elapsed());
    }


    fn tok() -> Gpt2Bpe {
        Gpt2Bpe::new()
    }

    #[test]
    fn vocab_size_is_50257() {
        assert_eq!(tok().vocab_size(), 50257);
    }

    #[test]
    fn known_vectors() {
        let t = tok();
        // These are exact GPT-2 encodings; they pin correctness.
        assert_eq!(t.encode("hello world"), vec![31373, 995]);
        assert_eq!(t.encode(" hello"), vec![23748]);
        assert_eq!(t.encode("Hello world"), vec![15496, 995]);
        assert_eq!(t.encode("\n"), vec![198]);
    }

    #[test]
    fn lossless_roundtrip() {
        let t = tok();
        let cases = [
            "hello world",
            " hello",
            "Hello world",
            "\n",
            "Hello, World! How are you?",
            "  leading and  internal   spaces  ",
            "line one\nline two\n\nline four",
            "MixedCASE and lower and UPPER",
            "Numbers 123 and 4567 mixed with words",
            "Punctuation?!... yes; no: maybe -- (parentheses) [brackets]",
            "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune.",
            "It's a test of contractions: I'll, you've, we're, they'd, he's, don't.",
            "",
            "a",
            " ",
            "   ",
            "tabs\tand\tspaces",
        ];
        for s in cases {
            let round = t.decode(&t.encode(s));
            assert_eq!(round, s, "round-trip failed for {s:?}");
        }
    }
}
