// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T5-XXL SentencePiece-unigram parity, over a normalizer that includes
//! sentencepiece's **`Precompiled` charsmap**.
//!
//! `data::unigram::UnigramTokenizer` must produce **exactly** the ids
//! HuggingFace `tokenizers` produces from the same `tokenizer.json` - the real
//! `black-forest-labs/FLUX.1-dev` `tokenizer_2`, which is what `crates/flux1`
//! and `crates/t5encoder` condition on.
//!
//! This file exists because that tokenizer's normalizer is a three-step
//! `Sequence` the umT5 reader could not handle:
//!
//! 1. `Precompiled` - sentencepiece's own charsmap, a Darts double-array trie
//!    plus a blob of replacement strings, base64'd into one JSON field. It is
//!    the NFKC-ish folding (`U+FB01` -> `fi`, `U+FF21` -> `A`, `U+2160` -> `I`,
//!    NBSP/tab/newline -> space).
//! 2. `Strip { strip_right: true }`.
//! 3. `Replace " {2,}" -> "\u{2581}"` - note the replacement is the METASPACE
//!    character, not a space, so it interacts with the Metaspace pre-tokenizer
//!    that runs next. umT5 writes a space in that same slot.
//!
//! The corpus is chosen so each step, and the interactions between them, are
//! observable in the ids. In particular two cases pin the 6-byte grapheme
//! threshold inside `Precompiled::normalize_string`, which is the part of the
//! algorithm most likely to be "simplified" into something subtly wrong:
//!
//! * `"\u{1e9b}\u{323}"` is a 5-byte cluster, so the WHOLE cluster is looked
//!   up and it folds to the single character `\u{1e61}`.
//! * `"\u{ff76}\u{ff9e}"` is a 6-byte cluster, so each character is
//!   transformed on its own and it comes out DECOMPOSED as
//!   `\u{30ab}\u{3099}` rather than as the precomposed `\u{30ac}` a
//!   whole-cluster lookup would have given.
//!
//! A reader that segmented by `char` instead of by grapheme, or that got the
//! threshold wrong, produces different token ids on those two lines and on
//! nothing else in a plain English prompt - which is exactly why they are here
//! and not left to a real prompt to catch by luck.
//!
//! The golden ids are checked in rather than fetched: `/testdata` is
//! gitignored, so a fixture written there would not survive a clone, and a
//! parity gate nobody can run is not a gate. Regenerate with
//! `tools/goldens/t5xxl_tokenizer_dump_reference.py`.
//!
//! Skips itself when the tokenizer is absent.

use std::path::PathBuf;

use data::tokenizer::Tokenizer;
use data::unigram::UnigramTokenizer;

/// T5-XXL's window in the FLUX.1 pipeline.
const TEXT_LEN: usize = 512;

/// Mirrors `CASES` in tools/goldens/t5xxl_tokenizer_dump_reference.py.
#[rustfmt::skip]
const CASES: &[(&str, &[u32])] = &[
    // -> "a photo of a person standing in a park"
    ("a photo of a person standing in a park", &[3, 9, 1202, 13, 3, 9, 568, 4125, 16, 3, 9, 2447, 1]),
    // -> "a photo of a person standing in a park, highly detailed, 8k, cinematic lighting"
    ("a photo of a person standing in a park, highly detailed, 8k, cinematic lighting", &[3, 9, 1202, 13, 3, 9, 568, 4125, 16, 3, 9, 2447, 6, 1385, 3117, 6, 505, 157, 6, 10276, 1225, 3598, 1]),
    // -> "hello world"
    ("hello world", &[21820, 296, 1]),
    // -> ""
    ("", &[1]),
    // -> "double\u{2581}spaces\u{2581}here"
    ("double  spaces   here", &[1486, 4856, 270, 1]),
    // -> "one space only"
    ("one space only", &[80, 628, 163, 1]),
    // -> "a\u{2581}b\u{2581}c\u{2581}d"
    ("a  b   c    d", &[3, 9, 3, 115, 3, 75, 3, 26, 1]),
    // -> "\u{2581}leading and trailing"
    ("  leading and trailing  ", &[1374, 11, 5032, 53, 1]),
    // -> "tabs and newlines"
    ("tabs\tand\nnewlines", &[3808, 7, 11, 126, 6972, 1]),
    // -> "trailing whitespace"
    ("trailing whitespace   ", &[5032, 53, 872, 6633, 1]),
    // -> "nbsp and ideographic space"
    ("nbsp\u{a0}and\u{3000}ideographic space", &[3, 29, 115, 7, 102, 11, 3, 1599, 16587, 628, 1]),
    // -> "ligatures fi and fl and ffi"
    ("ligatures \u{fb01} and \u{fb02} and \u{fb03}", &[3, 17140, 10471, 361, 11, 3, 89, 40, 11, 3, 89, 89, 23, 1]),
    // -> "fullwidth ABC 123"
    ("fullwidth \u{ff21}\u{ff22}\u{ff23} \u{ff11}\u{ff12}\u{ff13}", &[423, 12018, 189, 14213, 3, 14574, 1]),
    // -> "roman numeral III and 1"
    ("roman numeral \u{2160}\u{2161} and \u{2460}", &[3408, 7507, 4900, 6289, 11, 209, 1]),
    // -> "1\u{2044}2 1\u{2044}4 1\u{2044}3 fractions"
    ("\u{bd} \u{bc} \u{2153} fractions", &[209, 2, 357, 209, 2, 591, 209, 2, 519, 12211, 7, 1]),
    // -> "superscript 23 and 5"
    ("superscript \u{b2}\u{b3} and \u{2075}", &[1355, 11815, 1902, 11, 305, 1]),
    // -> "circled AB parenthesized (1)"
    ("circled \u{24b6}\u{24b7} parenthesized \u{2474}", &[8196, 26, 3, 5359, 4208, 88, 5120, 5637, 1]),
    // -> "CJK compat \u{5e73}\u{6210} \u{30a2}\u{30d1}\u{30fc}\u{30c8}"
    ("CJK compat \u{337b} \u{3300}", &[205, 683, 439, 2890, 144, 3, 2, 3, 2, 1]),
    // -> "accents caf\u{e9} na\u{ef}ve r\u{e9}sum\u{e9}"
    ("accents caf\u{e9} na\u{ef}ve r\u{e9}sum\u{e9}", &[5820, 7, 11949, 3, 29, 9, 2, 162, 1417, 4078, 154, 1]),
    // -> "composed \u{e9} vs decomposed \u{e9}"
    ("composed \u{e9} vs decomposed e\u{301}", &[10431, 3, 154, 3, 208, 7, 20, 287, 12151, 3, 154, 1]),
    // -> "\u{130}stanbul T\u{fc}rkiye"
    ("\u{130}stanbul T\u{fc}rkiye", &[3, 2, 5627, 6724, 12087, 2168, 63, 15, 1]),
    // -> "\u{30ab}\u{3099} halfwidth katakana"
    ("\u{ff76}\u{ff9e} halfwidth katakana", &[3, 2, 985, 12018, 189, 3, 8682, 9, 3304, 9, 1]),
    // -> "\u{30ac} vs \u{30ab}\u{3099} katakana"
    ("\u{30ac} vs \u{30ab}\u{3099} katakana", &[3, 2, 3, 208, 7, 3, 2, 3, 8682, 9, 3304, 9, 1]),
    // -> "\u{1e61} dot above below"
    ("\u{1e9b}\u{323} dot above below", &[3, 2, 103, 17, 756, 666, 1]),
    // -> "\u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac}, \u{440}\u{443}\u{441}\u{441}\u{43a}\u{438}\u{439}, \u{4e2d}\u{6587}"
    ("\u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac}, \u{440}\u{443}\u{441}\u{441}\u{43a}\u{438}\u{439}, \u{4e2d}\u{6587}", &[3, 2, 6, 3, 23912, 5345, 30610, 2, 6, 3, 2, 1]),
    // -> "emoji \u{1f600} and family \u{1f468} \u{1f469} \u{1f466}"
    ("emoji \u{1f600} and family \u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f466}", &[3, 15, 51, 21892, 3, 2, 11, 384, 3, 2, 3, 2, 3, 2, 1]),
    // -> "mixed fine caf\u{e9}\u{2581}A\u{2581}I end"
    ("mixed \u{fb01}ne caf\u{e9}  \u{ff21}  \u{2160} end", &[4838, 1399, 11949, 71, 27, 414, 1]),
];

/// `$BRAIN_FLUX1_DIR/tokenizer_2` (the env var `flux1::caps` itself reads),
/// else the model store's copy of the released repo.
fn tokenizer() -> Option<UnigramTokenizer> {
    let dir = std::env::var("BRAIN_FLUX1_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| brain_testutil::model_dir("black-forest-labs/FLUX.1-dev").map(PathBuf::from))
        .map(|d| d.join("tokenizer_2"));
    let Some(dir) = dir.filter(|d| d.join("tokenizer.json").is_file()) else {
        brain_testutil::skip(
            "set BRAIN_FLUX1_DIR to a black-forest-labs/FLUX.1-dev checkout (needs tokenizer_2/tokenizer.json)",
        );
        return None;
    };
    Some(UnigramTokenizer::from_dir(&dir.to_string_lossy()).expect("load the T5-XXL tokenizer"))
}

#[test]
fn t5xxl_unigram_matches_the_reference_ids() {
    let Some(tok) = tokenizer() else { return };

    assert_eq!(tok.vocab_size(), 32100);
    assert_eq!(tok.pad_id(), 0);
    assert_eq!(tok.special_id("</s>"), Some(1));
    assert_eq!(tok.unk_id(), 2);

    let mut failed = 0usize;
    for (text, want) in CASES {
        let got = tok.encode(text);
        let ok = got.as_slice() == *want;
        if !ok {
            failed += 1;
            let at = got.iter().zip(want.iter()).position(|(a, b)| a != b);
            eprintln!("  MISMATCH {text:?}\n    at {at:?}\n    got  {got:?}\n    want {want:?}");
        }
    }
    eprintln!("  {} / {} cases match the reference", CASES.len() - failed, CASES.len());
    assert_eq!(failed, 0, "{failed} of {} cases disagree with HF tokenizers", CASES.len());
}

/// The charsmap is what makes these fold at all: without it every one of these
/// inputs would reach the lattice unchanged and tokenize as `unk`. Asserting on
/// the ids of the *folded* forms keeps the claim concrete - `\u{fb01}` really
/// does become the two pieces `f`/`i` and not one unknown.
#[test]
fn the_charsmap_folds_compatibility_characters_into_real_pieces() {
    let Some(tok) = tokenizer() else { return };
    let unk = tok.unk_id();

    // The roman-numeral row is not a typo: U+2160/1/2 are "I", "II" and "III",
    // so the three of them fold to SIX letters, not three.
    for (folded, raw) in [
        ("fi", "\u{fb01}"),
        ("ABC", "\u{ff21}\u{ff22}\u{ff23}"),
        ("IIIIII", "\u{2160}\u{2161}\u{2162}"),
    ] {
        let want = tok.encode(folded);
        let got = tok.encode(raw);
        assert_eq!(got, want, "{raw:?} must tokenize exactly like {folded:?}");
        assert!(!got.contains(&unk), "{raw:?} tokenized to unk - the charsmap did not run");
    }
}

/// What the FLUX.1 pipeline actually calls. Before the charsmap was
/// implemented this returned `Err("normalizer: \"Precompiled\" is not
/// implemented")` and no image could be generated at all.
#[test]
fn the_flux1_prompt_path_encodes_and_pads_to_the_window() {
    let Some(tok) = tokenizer() else { return };
    let prompt = "a photo of a person standing in a park";

    let (ids, mask) = tok.encode_padded(prompt, TEXT_LEN);
    assert_eq!(ids.len(), TEXT_LEN);
    assert_eq!(mask.len(), TEXT_LEN);
    let n = mask.iter().filter(|&&m| m == 1).count();
    assert_eq!(&ids[..n], CASES[0].1, "the padded path and the plain path must agree");
    assert_eq!(ids[n - 1], tok.special_id("</s>").unwrap());
    assert!(ids[n..].iter().all(|&i| i == tok.pad_id()));
    assert!(mask[n..].iter().all(|&m| m == 0));

    // Round-trips, which it could not if a piece had been dropped.
    assert_eq!(tok.decode(&ids[..n - 1]), prompt);
}
