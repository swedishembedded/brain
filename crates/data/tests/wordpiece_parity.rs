// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WordPiece parity: `data::wordpiece::WordPiece` must produce **exactly** the
//! ids the real BERT-family `tokenizer.json` produces when loaded through the
//! `tokenizers` library (`BertNormalizer` + `BertPreTokenizer` + `WordPiece` +
//! `TemplateProcessing`).
//!
//! `CASES` is pinned output from `tools/goldens/wordpiece_dump_reference.py` run
//! against `sentence-transformers/all-MiniLM-L6-v2`'s tokenizer.
//! `TOKENIZER_STRINGS` in that script is the other copy of the input corpus and
//! the two must be edited together, or a drift in one side goes undetected.
//!
//! The corpus separates the three stages that can each be silently wrong alone:
//!
//! * **normalizer** - accent stripping (an uncased checkpoint leaves
//!   `strip_accents` null, which means "follow `lowercase`", i.e. ON, so the NFC
//!   and NFD spellings of `café` must collapse to the SAME ids), control- and
//!   format-character removal (`\u{0}`, `\u{7}`, and the zero-width space are
//!   DELETED, not replaced by a space - `null\u{0}and` is one word), and the
//!   space padding around CJK codepoints;
//! * **pre-tokenizer** - punctuation splitting, whitespace runs, and the
//!   one-word-per-ideograph CJK split;
//! * **WordPiece model** - `##` continuation, greedy longest-match-first, and
//!   the `max_input_chars_per_word` (100) cliff where a 100-character word is
//!   still piece-split but a 101-character one becomes a single `[UNK]`.
//!
//! A tokenizer that is right on plain ASCII and wrong on every one of those is
//! the failure this corpus exists to catch, because no downstream accuracy
//! metric would obviously show it.
//!
//! Padding and truncation are deliberately absent from both sides: the
//! checkpoint's own file declares padding to 128 and truncation to 256, but
//! those are CALLER policy, not tokenizer semantics - [`data::wordpiece`]
//! returns the bare sequence and the model pads.
//!
//! Skips itself when `$BRAIN_TESTDATA/decide/tokenizer/tokenizer.json` (default
//! `<repo>/testdata`) is absent; `make fetch/testdata` puts one there.

use std::path::PathBuf;

use data::tokenizer::Tokenizer;
use data::wordpiece::WordPiece;

fn testdata(rel: &str) -> PathBuf {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    PathBuf::from(root).join(rel)
}

// Pinned against the real `tokenizers` library on all-MiniLM-L6-v2's
// tokenizer.json (regenerate: tools/goldens/wordpiece_dump_reference.py).
const CASES: &[(&str, &[u32])] = &[
    ("I am still waiting on my card?", &[101, 1045, 2572, 2145, 3403, 2006, 2026, 4003, 1029, 102]),
    ("What can I do if my card still hasn't arrived after 2 weeks?", &[101, 2054, 2064, 1045, 2079, 2065, 2026, 4003, 2145, 8440, 1005, 1056, 3369, 2044, 1016, 3134, 1029, 102]),
    ("Hello world", &[101, 7592, 2088, 102]),
    ("hello", &[101, 7592, 102]),
    ("HELLO", &[101, 7592, 102]),
    ("Hello", &[101, 7592, 102]),
    ("", &[101, 102]),
    (" ", &[101, 102]),
    ("  ", &[101, 102]),
    ("a", &[101, 1037, 102]),
    ("  double  space", &[101, 3313, 2686, 102]),
    ("tabs\tand\nnewlines", &[101, 21628, 2015, 1998, 2047, 12735, 102]),
    ("trailing space ", &[101, 12542, 2686, 102]),
    (" leading space", &[101, 2877, 2686, 102]),
    ("null\u{0}and\u{7}bell", &[101, 19701, 5685, 17327, 102]),
    ("zero\u{200b}width\u{200b}space", &[101, 5717, 9148, 11927, 7898, 15327, 102]),
    ("café naïve", &[101, 7668, 15743, 102]),
    ("CAFÉ NAÏVE", &[101, 7668, 15743, 102]),
    ("café", &[101, 7668, 102]),
    ("cafe\u{301}", &[101, 7668, 102]),
    ("Ångström", &[101, 17076, 15687, 102]),
    ("A\u{30a}ngstro\u{308}m", &[101, 17076, 15687, 102]),
    ("Ελληνικά, русский, العربية, हिन\u{94d}दी", &[101, 1159, 29727, 29727, 24824, 16177, 18199, 29726, 14608, 1010, 1195, 29748, 29747, 29747, 23925, 15414, 1010, 1270, 23673, 29830, 17149, 29816, 14498, 19433, 1010, 1339, 29877, 29863, 29861, 29878, 102]),
    ("你好", &[101, 100, 100, 102]),
    ("你好世界", &[101, 100, 100, 1745, 100, 102]),
    ("中文english混合", &[101, 1746, 1861, 2394, 100, 1792, 102]),
    ("한국어", &[101, 1469, 30006, 30021, 29991, 30014, 30020, 29999, 30008, 102]),
    ("日本語のテキスト", &[101, 1864, 1876, 1950, 1671, 30239, 30227, 30233, 30240, 102]),
    ("don't you're I'll we've he'd it's", &[101, 2123, 1005, 1056, 2017, 1005, 2128, 1045, 1005, 2222, 2057, 1005, 2310, 2002, 1005, 1040, 2009, 1005, 1055, 102]),
    ("CamelCase snake_case kebab-case", &[101, 19130, 18382, 7488, 1035, 2553, 17710, 3676, 2497, 1011, 2553, 102]),
    ("((nested [brackets] {braces}))", &[101, 1006, 1006, 9089, 2098, 1031, 19719, 1033, 1063, 17180, 2015, 1065, 1007, 1007, 102]),
    ("v1.2.3-rc4", &[101, 1058, 2487, 1012, 1016, 1012, 1017, 1011, 22110, 2549, 102]),
    ("e.g. i.e. etc.", &[101, 1041, 1012, 1043, 1012, 1045, 1012, 1041, 1012, 4385, 1012, 102]),
    ("a,b;c:d!e?f", &[101, 1037, 1010, 1038, 1025, 1039, 1024, 1040, 999, 1041, 1029, 1042, 102]),
    ("100% of $5.00 @ #1", &[101, 2531, 1003, 1997, 1002, 1019, 1012, 4002, 1030, 1001, 1015, 102]),
    ("123456789", &[101, 13138, 19961, 2575, 2581, 2620, 2683, 102]),
    ("3.14159", &[101, 1017, 1012, 15471, 28154, 102]),
    ("2 weeks", &[101, 1016, 3134, 102]),
    ("unaffable", &[101, 14477, 20961, 3468, 102]),
    ("tokenization", &[101, 19204, 3989, 102]),
    ("antidisestablishmentarianism", &[101, 3424, 10521, 4355, 7875, 13602, 3672, 12199, 2964, 102]),
    ("pneumonoultramicroscopicsilicovolcanoconiosis", &[101, 1052, 2638, 2819, 17175, 11314, 6444, 2594, 7352, 26461, 27572, 11261, 6767, 15472, 6761, 8663, 10735, 2483, 102]),
    ("zzzzqqqqxxxx", &[101, 1062, 13213, 2480, 4160, 4160, 4160, 4160, 20348, 20348, 102]),
    ("🚀🐈🧪", &[101, 100, 102]),
    ("a rocket 🚀 and a cat 🐈", &[101, 1037, 7596, 100, 1998, 1037, 4937, 100, 102]),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", &[101, 13360, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 11057, 2050, 102]),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", &[101, 100, 102]),
    ("card arrival", &[101, 4003, 5508, 102]),
    ("top up by bank transfer charge", &[101, 2327, 2039, 2011, 2924, 4651, 3715, 102]),
    ("Which team should handle this [SEP] card arrival", &[101, 2029, 2136, 2323, 5047, 2023, 102, 4003, 5508, 102]),
    ("[CLS] already here [SEP]", &[101, 101, 2525, 2182, 102, 102]),
];

fn load() -> Option<WordPiece> {
    let path = testdata("decide/tokenizer/tokenizer.json");
    if !path.exists() {
        brain_testutil::skip(&format!(
            "{} missing - run `make fetch/testdata` (a ~470 kB tokenizer.json, not a checkpoint)",
            path.display()
        ));
        return None;
    }
    Some(WordPiece::from_file(path.to_str().unwrap()).expect("parse tokenizer.json"))
}

#[test]
fn wordpiece_matches_the_reference_tokenizer_exactly() {
    let Some(tok) = load() else { return };
    let mut failures = Vec::new();
    for (text, want) in CASES {
        let got = tok.encode(text);
        if got != *want {
            failures.push(format!("  {text:?}\n    want {want:?}\n    got  {got:?}"));
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), CASES.len(), failures.join("\n"));
}

/// The vocabulary the embedding table is sized against.
#[test]
fn vocab_size_is_the_checkpoints_own() {
    let Some(tok) = load() else { return };
    assert_eq!(tok.vocab_size(), 30522);
}

/// `[CLS]`/`[SEP]` written literally in the input are matched as ATOMIC added
/// tokens before WordPiece ever sees them - that is what lets a slot template
/// join an instruction to an option name with a real `[SEP]`. Without atomic
/// matching the text would tokenize as `[`, `cl`, `##s`, `]`.
#[test]
fn literal_special_tokens_are_atomic() {
    let Some(tok) = load() else { return };
    assert_eq!(tok.encode("[CLS] already here [SEP]"), vec![101, 101, 2525, 2182, 102, 102]);
}

/// Decoding is round-trippable for in-vocabulary ASCII: `##` continuations
/// rejoin without a space, ordinary pieces join with one.
#[test]
fn decode_rejoins_wordpiece_continuations() {
    let Some(tok) = load() else { return };
    let ids = tok.encode("tokenization of unaffable text");
    assert_eq!(tok.decode(&ids), "[CLS] tokenization of unaffable text [SEP]");
}
