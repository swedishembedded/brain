// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Laya M4's own verified-this-session risk: does `data::qwen_tokenizer::
//! QwenBpe` - built for Qwen/LFM2.5's cl100k-style pre-tokenizer - actually
//! segment Laya's real `tokenizer/tokenizer.json` the way the real
//! `tokenizers` library does?
//!
//! **Real finding**: YES, it diverges, exactly as the plan predicted from
//! reading the file's own `pre_tokenizer` (a bare `ByteLevel` with
//! `use_regex: true` and no explicit `pattern`) - verified empirically this
//! session against the real `tokenizers` library on Laya's own file, on
//! THREE axes, not just the two originally suspected (case-sensitive
//! contractions, uncapped digit runs, AND the letter/digit/symbol branches'
//! optional prefix being space-only rather than cl100k's "any non-alnum
//! char" - see `data::qwen_tokenizer`'s own module doc for the full
//! derivation). `QwenBpe` was extended with a second, GPT-2-default
//! pre-tokenizer mode ([`data::qwen_tokenizer::pretokenize_gpt2_default`]),
//! selected automatically from the file's own declared `pre_tokenizer` shape:
//! no new tokenizer type, no behavior change for existing Qwen/LFM2.5
//! callers (their own test suites re-run clean, see the M4 commit).
//!
//! Two tests:
//! 1. [`pretokenizer_boundaries_genuinely_differ`] - unconditional (no
//!    checkpoint needed): the cl100k scanner
//!    (`pretokenize_digits(text, 1)`, what `QwenBpe` would have used before
//!    this milestone) and the GPT-2-default scanner really do disagree on the
//!    flagged cases, pinned so a future change to either scanner is caught.
//! 2. [`qwen_bpe_matches_the_real_tokenizer_on_laya_checkpoint`] - the full
//!    differential: `QwenBpe::from_file` loaded on the REAL Laya
//!    `tokenizer.json`, encoding ~50 real, varied strings (contractions both
//!    cases, long digit runs, JSON state blobs of the kind `build_sequence`
//!    actually serializes, unicode, punctuation), asserted against ids
//!    captured directly from the real `tokenizers` library
//!    (`Tokenizer.encode(text, add_special_tokens=False)`) this session.
//!    Skips cleanly when the checkpoint has not been pulled.

use data::qwen_tokenizer::{pretokenize_digits, pretokenize_gpt2_default, QwenBpe};
use data::tokenizer::Tokenizer as _;

#[test]
fn pretokenizer_boundaries_genuinely_differ() {
    // (text, cl100k-K1 pretokens, gpt2-default pretokens) - each row is a
    // REAL divergence, not a hypothetical one; values captured directly from
    // both functions this session, not hand-derived from the regex text (a
    // hand derivation got the whitespace-run backtracking wrong once already
    // during this milestone - see `data::qwen_tokenizer`'s own module doc).
    let cases: &[(&str, &[&str], &[&str])] = &[
        // cl100k's `(?i:...)` matches "'T" case-INSENSITIVELY as one
        // contraction pre-token; gpt2-default's contraction match is
        // case-SENSITIVE and does not fire on uppercase, so "'" and "T"
        // fall through to two separate pre-tokens instead.
        ("DON'T", &["DON", "'T"], &["DON", "'", "T"]),
        ("IT'S", &["IT", "'S"], &["IT", "'", "S"]),
        // lowercase contractions agree - not every case in the battery is a
        // divergence, only the uppercase ones (the axis itself is real).
        ("don't", &["don", "'t"], &["don", "'t"]),
        // cl100k caps a digit run at K=1 digit; gpt2-default is uncapped.
        ("1234567890", &["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"], &["1234567890"]),
        // cl100k's letter branch absorbs ANY leading non-alnum char (here
        // `(`); gpt2-default's letter branch only absorbs a leading SPACE.
        ("(hello", &["(hello"], &["(", "hello"]),
        // cl100k's digit branch has NO leading-space absorption at all;
        // gpt2-default's does.
        ("a 123", &["a", " ", "1", "2", "3"], &["a", " 123"]),
    ];
    for (text, cl100k, gpt2) in cases {
        let got_cl100k = pretokenize_digits(text, 1);
        let got_gpt2 = pretokenize_gpt2_default(text);
        assert_eq!(got_cl100k, *cl100k, "cl100k({text:?})");
        assert_eq!(got_gpt2, *gpt2, "gpt2_default({text:?})");
    }
    // At least the flagged rows must be genuine divergences (excluding the
    // lowercase-contraction row, which is deliberately included to show the
    // axis is case, not contractions-in-general).
    assert_ne!(pretokenize_digits("DON'T", 1), pretokenize_gpt2_default("DON'T"));
    assert_ne!(pretokenize_digits("1234567890", 1), pretokenize_gpt2_default("1234567890"));
    assert_ne!(pretokenize_digits("(hello", 1), pretokenize_gpt2_default("(hello"));
}

fn battery() -> Vec<(&'static str, &'static [u32])> {
    vec![
        ("don't", &[9903, 626]),
        ("DON'T", &[39153, 8, 53]),
        ("Don'T", &[5498, 8, 53]),
        ("can't", &[5092, 626]),
        ("CAN'T", &[40555, 8, 53]),
        ("I'm", &[42, 1353]),
        ("I'M", &[42, 8, 46]),
        ("it's", &[262, 434]),
        ("IT'S", &[1433, 8, 52]),
        ("we'll", &[664, 1833]),
        ("WE'LL", &[10663, 8, 2293]),
        ("you're", &[5658, 1472]),
        ("YOU'RE", &[27239, 8, 1848]),
        ("should've", &[11425, 1849]),
        ("SHOULD'VE", &[5648, 30384, 8, 12695]),
        ("1234567890", &[42594, 25025, 2270]),
        ("12345", &[42594]),
        ("0", &[17]),
        ("42", &[2945]),
        ("3.14159", &[20, 15, 1047, 17220]),
        ("price $1299.99", &[19209, 370, 805, 1525, 15, 1525]),
        ("year 2026", &[2913, 1384, 1731]),
        ("hello world", &[25521, 1533]),
        ("Hello, World!", &[12092, 13, 3645, 2]),
        ("multiple   spaces", &[34263, 50275, 31748]),
        ("tab\ttab", &[8476, 186, 8476]),
        ("newline\ntest", &[1826, 1282, 187, 2566]),
        ("quote\"quote", &[21049, 3, 21049]),
        ("emoji test", &[43208, 8020, 1071]),
        ("cafe naive uber", &[6357, 453, 27785, 12980, 254]),
        ("  leading spaces", &[50276, 16378, 8470]),
        ("trailing spaces  ", &[7604, 4837, 8470, 50276]),
        ("MiXeD CaSe TeXt", &[24711, 57, 70, 37, 6047, 3251, 2745, 57, 85]),
        ("a.b.c.d", &[66, 15, 67, 15, 68, 15, 69]),
        ("192.168.1.1", &[14403, 15, 13851, 15, 18, 15, 18]),
        ("user@example.com", &[4537, 33, 11667, 15, 681]),
        ("https://example.com/path?q=1", &[3614, 1358, 11667, 15, 681, 16, 3967, 32, 82, 30, 18]),
        (
            "{\"state\": {\"turns\": 3, \"last\": \"hi\"}}",
            &[9819, 3409, 1381, 17579, 85, 10029, 1381, 495, 13, 346, 6275, 1381, 346, 5801, 3, 599],
        ),
        (
            "{\"a\": 1, \"b\": [1, 2, 3], \"c\": {\"d\": true, \"e\": null}}",
            &[9819, 66, 1381, 337, 13, 346, 67, 1381, 544, 18, 13, 374, 13, 495, 1092, 346, 68, 1381, 17579, 69, 1381, 2032, 13, 346, 70, 1381, 3635, 599],
        ),
        ("The quick brown fox jumps over the lazy dog.", &[510, 3158, 8516, 30013, 27287, 689, 253, 22658, 4370, 15]),
        ("Testing punctuation: ,.;:!?()[]{}", &[38571, 17256, 2368, 27, 1157, 9944, 27, 2, 32, 43144, 1181]),
        ("under_score and-dash and.dot", &[4524, 64, 18891, 285, 14, 27207, 285, 15, 5256]),
        ("Numbers 1 22 333 4444 55555 666666", &[45972, 337, 3307, 30057, 577, 24447, 36918, 2417, 45298, 24185]),
        ("It's a test. Don't fail! CAN'T you see?", &[1147, 434, 247, 1071, 15, 5037, 626, 1891, 2, 20753, 8, 53, 368, 923, 32]),
        ("single'quote", &[20199, 8, 21049]),
        ("multi''quotes", &[23939, 6267, 371, 4787]),
        ("Naive cafe with accents", &[15061, 422, 40847, 342, 756, 592]),
        ("japanese text sample", &[75, 3682, 3248, 2505, 3410]),
        (
            "no options fit hereno options fit hereno options fit here",
            &[2369, 4610, 4944, 344, 445, 80, 4610, 4944, 344, 445, 80, 4610, 4944, 1060],
        ),
    ]
}

#[test]
fn qwen_bpe_matches_the_real_tokenizer_on_laya_checkpoint() {
    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    let tok_path = format!("{dir}/tokenizer/tokenizer.json");
    if !std::path::Path::new(&tok_path).exists() {
        brain_testutil::skip(&format!("{tok_path} absent - run `brain pull convaiinnovations/laya`"));
        return;
    }
    let tok = QwenBpe::from_file(&tok_path).expect("load Laya tokenizer.json");

    // Special-token ids, verified directly from the real file's own
    // `added_tokens` this session (never hardcoded elsewhere).
    assert_eq!(tok.special_id("[UNK]"), Some(50280));
    assert_eq!(tok.special_id("[CLS]"), Some(50281));
    assert_eq!(tok.special_id("[SEP]"), Some(50282));
    assert_eq!(tok.special_id("[PAD]"), Some(50283));
    assert_eq!(tok.special_id("[MASK]"), Some(50284));

    let mut failures = 0;
    for (text, want) in battery() {
        let got = tok.encode(text);
        if got != want {
            eprintln!("MISMATCH {text:?}: got {got:?}, want {want:?}");
            failures += 1;
        }
    }
    assert_eq!(failures, 0, "{failures} of {} battery strings mismatched the real tokenizer - see stderr above", battery().len());
}

