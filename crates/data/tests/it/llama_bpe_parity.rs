// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaMA BPE parity: `data::llama_bpe::LlamaBpe` must produce **exactly** the
//! ids the real LLaMA-2/Vicuna-1.5 `tokenizer.json` (loaded through the
//! `tokenizers` library the same way `LlamaTokenizerFast` does) produces.
//!
//! `CASES` is pinned output from `tools/goldens/llava_tokenizer_dump_reference.py`
//! run against a real `tokenizer.json` (`NousResearch/Llama-2-13b-hf`, which
//! ships the identical byte-for-byte tokenizer Vicuna-1.5-13B - and hence
//! LLaVA-1.5's decoder half - uses). `TOKENIZER_STRINGS` in that script is the
//! other copy of the input corpus and the two must be edited together, or a
//! drift in one side goes undetected.
//!
//! Covers: SUPIR's own default caption prompt and the `vicuna_v1`-templated
//! `USER: <image>\n... ASSISTANT:` shape, ASCII casing/contractions/brackets,
//! whitespace runs (a single space, a double space, tabs, the empty string),
//! CJK/Greek/Cyrillic/Arabic/Devanagari/Hangul/Japanese, an astral-plane
//! emoji run, Latin-1 accents, and the literal `<image>` placeholder text
//! (which is NOT a special token here - see [`data::llama_bpe`]'s module
//! docs on the token splice).
//!
//! Skips itself when `$BRAIN_TESTDATA/llava/tokenizer/tokenizer.json` (default
//! `<repo>/testdata`) is absent (`tools/goldens/llava_tokenizer_dump_reference.py`
//! names how to fetch one - it is a small file, not the 13B checkpoint).

use std::path::PathBuf;

use data::llama_bpe::LlamaBpe;

fn testdata(rel: &str) -> PathBuf {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    PathBuf::from(root).join(rel)
}

// Pinned against the real `tokenizers` library on a LLaMA-2/Vicuna-1.5
// tokenizer.json (regenerate: tools/goldens/llava_tokenizer_dump_reference.py).
const CASES: &[(&str, &[u32])] = &[
    ("Describe this image and its style in a very detailed manner.", &[1, 20355, 915, 445, 1967, 322, 967, 3114, 297, 263, 1407, 13173, 8214, 29889]),
    ("Hello world", &[1, 15043, 3186]),
    (" Hello", &[1, 29871, 15043]),
    ("hello", &[1, 22172]),
    ("", &[1]),
    (" ", &[1, 259]),
    ("  ", &[1, 1678]),
    ("你好", &[1, 29871, 30919, 31076]),
    ("a", &[1, 263]),
    ("  double  space", &[1, 259, 3765, 29871, 2913]),
    ("tabs\tand\nnewlines", &[1, 18859, 12, 392, 13, 1482, 9012]),
    ("Ελληνικά, русский, العربية, हिन्दी, 한국어, 日本語のテキスト", &[1, 29871, 30311, 30142, 30142, 30183, 30133, 30136, 30173, 30216, 29892, 14251, 2542, 29892, 24508, 30218, 30156, 30177, 30163, 30242, 29892, 29871, 30714, 30436, 30424, 30296, 30694, 30580, 29892, 29871, 30877, 31293, 31129, 29892, 29871, 30325, 30346, 30968, 30199, 30572, 30454, 30255, 30279]),
    ("a rocket 🚀 and a cat 🐈 with a beaker 🧪", &[1, 263, 696, 3522, 29871, 243, 162, 157, 131, 322, 263, 6635, 29871, 243, 162, 147, 139, 411, 263, 367, 5790, 29871, 243, 162, 170, 173]),
    ("CamelCase snake_case kebab-case", &[1, 5500, 295, 8259, 269, 21040, 29918, 4878, 413, 774, 370, 29899, 4878]),
    ("123456789", &[1, 29871, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]),
    ("v1.2.3-rc4", &[1, 325, 29896, 29889, 29906, 29889, 29941, 29899, 2214, 29946]),
    ("<image>\nWhat is unusual about this image?", &[1, 529, 3027, 29958, 13, 5618, 338, 22910, 1048, 445, 1967, 29973]),
    ("USER: <image>\nDescribe this image and its style in a very detailed manner. ASSISTANT:", &[1, 3148, 1001, 29901, 529, 3027, 29958, 13, 4002, 29581, 445, 1967, 322, 967, 3114, 297, 263, 1407, 13173, 8214, 29889, 319, 1799, 9047, 13566, 29901]),
    ("café naïve", &[1, 274, 28059, 1055, 30085, 345]),
    ("don't you're I'll we've he'd it's", &[1, 1016, 29915, 29873, 366, 29915, 276, 306, 29915, 645, 591, 29915, 345, 540, 29915, 29881, 372, 29915, 29879]),
    ("((nested [brackets] {braces}))", &[1, 5135, 27420, 518, 2634, 9737, 29962, 426, 2634, 778, 20073]),
];

#[test]
fn llama_bpe_matches_the_reference_tokenizer_exactly() {
    let dir = testdata("llava/tokenizer");
    let path = dir.join("tokenizer.json");
    if !path.exists() {
        brain_testutil::skip(&format!(
            "{} missing - fetch a small LLaMA-2/Vicuna tokenizer.json (see tools/goldens/llava_tokenizer_dump_reference.py)",
            path.display()
        ));
        return;
    }
    let tok = LlamaBpe::from_dir(&dir).expect("load LLaMA tokenizer.json");
    assert_eq!(tok.vocab_size(), 32000);
    assert_eq!(tok.bos_id(), 1);
    assert_eq!(tok.eos_id(), 2);
    assert_eq!(tok.unk_id(), 0);

    for &(text, want) in CASES {
        let got = tok.encode(text);
        assert_eq!(got, want, "ids for {text:?}");
    }
    eprintln!("llama_bpe: {}/{} strings exact-id-match the reference tokenizer", CASES.len(), CASES.len());
}

/// Encoding is deterministic and content-only encoding never carries the BOS
/// the framed [`LlamaBpe::encode`] adds - this is the primitive the LLaVA
/// image-token splice needs (each `<image>`-split chunk after the first is
/// tokenized WITHOUT a fresh BOS).
#[test]
fn encode_raw_omits_bos_and_encode_is_encode_raw_plus_bos() {
    let dir = testdata("llava/tokenizer");
    if !dir.join("tokenizer.json").exists() {
        brain_testutil::skip(&format!("{} missing tokenizer.json", dir.display()));
        return;
    }
    let tok = LlamaBpe::from_dir(&dir).expect("load LLaMA tokenizer.json");
    for &(text, want) in CASES {
        let mut framed = vec![tok.bos_id()];
        framed.extend(tok.encode_raw(text));
        assert_eq!(framed, want, "encode_raw+bos must equal encode for {text:?}");
    }
}
