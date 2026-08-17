// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP BPE parity: `data::clip_bpe::ClipBpe` must produce **exactly** the ids
//! HuggingFace's `CLIPTokenizer` produces, for BOTH SDXL tokenizers.
//!
//! The golden is `testdata/clip/tokenizer/ids.safetensors`, written by
//! `tools/clip_dump_reference.py` from `transformers`' `CLIPTokenizer` at
//! `padding="max_length", max_length=77, truncation=True`:
//!
//! * `tok1_ids_padded` / `tok1_mask` / `tok1_len` — SDXL `tokenizer/` (CLIP-L,
//!   pad `<|endoftext|>`),
//! * `tok2_*` — SDXL `tokenizer_2/` (OpenCLIP-bigG, pad `!` = id 0).
//!
//! What this pinned down: the two tokenizers are **not** "same ids, different
//! padding". `tokenizer_2` registers its pad token `!` as an added token, and
//! HF splits on added tokens before the BPE — so a literal `!` in a prompt is
//! id 0 there and `!</w>` (256) in `tokenizer/`. Row 3 of the corpus (`World!`)
//! is the case that catches it.
//!
//! The strings are the tricky corpus the tokenizer has to survive: casing,
//! punctuation, digit runs, collapsed whitespace including a tab, non-ASCII
//! (`café naïve 你好`), an emoji outside the BMP, the empty prompt, and a
//! string long enough to truncate. They are duplicated here rather than read
//! from the fixture because a golden that carries its own inputs cannot detect
//! that the two sides drifted apart — `TOKENIZER_STRINGS` in the dump script is
//! the other copy and the two must be edited together.
//!
//! Skips itself when `$BRAIN_TESTDATA` (default `<repo>/testdata`) lacks the
//! fixture or the `vocab.json`/`merges.txt` pair (`make fetch/testdata`).

use std::path::PathBuf;

use data::clip_bpe::{ClipBpe, CONTEXT};

/// Mirrors `TOKENIZER_STRINGS` in `tools/clip_dump_reference.py`.
fn strings() -> Vec<String> {
    vec![
        "a red fox sitting on a mossy rock in a misty forest, morning light".to_string(),
        "a photo of a cat".to_string(),
        String::new(),
        "Hello, World!  MiXeD Case   and   collapsed\twhitespace".to_string(),
        "digits 0123456789 and symbols #@$%^&*()_+-=[]{}|;':\",./<>?".to_string(),
        "café naïve 你好 \u{1f98a} emoji".to_string(),
        format!("{}truncated tail", "a ".repeat(120)),
    ]
}

fn testdata(rel: &str) -> PathBuf {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    PathBuf::from(root).join(rel)
}

/// Minimal safetensors reader for the f32 tensors this golden stores — the
/// crate has no safetensors dependency and this test needs three arrays.
fn read_f32(path: &PathBuf, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let raw = std::fs::read(path).ok()?;
    let hdr_len = u64::from_le_bytes(raw.get(..8)?.try_into().ok()?) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(raw.get(8..8 + hdr_len)?).ok()?;
    let t = hdr.get(name)?;
    assert_eq!(t["dtype"], "F32", "{name}: golden is not F32");
    let shape: Vec<usize> = t["shape"].as_array()?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let off = t["data_offsets"].as_array()?;
    let (a, b) = (off[0].as_u64()? as usize, off[1].as_u64()? as usize);
    let bytes = &raw[8 + hdr_len + a..8 + hdr_len + b];
    let vals: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Some((shape, vals))
}

#[test]
fn clip_bpe_matches_the_hf_tokenizer_for_both_sdxl_towers() {
    let dir = testdata("clip/tokenizer");
    let golden = dir.join("ids.safetensors");
    if !golden.exists() || !dir.join("vocab.json").exists() || !dir.join("merges.txt").exists() {
        brain_testutil::skip(&format!("{} missing vocab.json/merges.txt/ids.safetensors (make fetch/testdata)", dir.display()));
        return;
    }

    // Both SDXL tokenizers ship the SAME vocab+merges; only `pad_token` differs.
    let tok1 = ClipBpe::from_dir(&dir).expect("load CLIP tokenizer assets");
    let tok2 = ClipBpe::from_dir(&dir).expect("load CLIP tokenizer assets").with_pad("!");
    assert_eq!(tok1.vocab_size(), 49408, "CLIP vocab size");
    assert_eq!(tok1.bos_id(), 49406);
    assert_eq!(tok1.eos_id(), 49407);
    assert_eq!(tok1.pad_id(), 49407, "tokenizer/ pads with <|endoftext|>");
    assert_eq!(tok2.pad_id(), 0, "tokenizer_2/ pads with '!' = 0");

    let (shape, ids1) = read_f32(&golden, "tok1_ids_padded").expect("tok1_ids_padded");
    let (_, ids2) = read_f32(&golden, "tok2_ids_padded").expect("tok2_ids_padded");
    let (_, mask1) = read_f32(&golden, "tok1_mask").expect("tok1_mask");
    let (_, len1) = read_f32(&golden, "tok1_len").expect("tok1_len");
    let texts = strings();
    assert_eq!(shape, vec![texts.len(), CONTEXT], "golden shape");

    let mut checked = 0usize;
    for (i, s) in texts.iter().enumerate() {
        let want1: Vec<u32> = ids1[i * CONTEXT..(i + 1) * CONTEXT].iter().map(|&v| v as u32).collect();
        let want2: Vec<u32> = ids2[i * CONTEXT..(i + 1) * CONTEXT].iter().map(|&v| v as u32).collect();
        let want_mask: Vec<u32> = mask1[i * CONTEXT..(i + 1) * CONTEXT].iter().map(|&v| v as u32).collect();

        let got1 = tok1.encode(s);
        let got2 = tok2.encode(s);
        assert_eq!(got1.ids, want1, "tok1 ids for {s:?}");
        assert_eq!(got2.ids, want2, "tok2 ids for {s:?}");
        assert_eq!(got1.mask, want_mask, "tok1 mask for {s:?}");
        assert_eq!(got1.eos_index + 1, len1[i] as usize, "tok1 eos_index for {s:?}");
        // Inside the content span the two towers agree EXCEPT where the text
        // contains a literal `!`: `tokenizer_2` registers its pad token as an
        // added token, so HF splits on it before the BPE and emits id 0 there.
        // (Row 3 of this corpus is the case that proves it — `World!`.)
        for (j, (&a, &b)) in got1.ids.iter().zip(&got2.ids).enumerate() {
            if want_mask[j] == 1 && b != tok2.pad_id() {
                assert_eq!(a, b, "tok1/tok2 disagree inside the content span for {s:?}");
            }
        }
        checked += 1;
    }
    assert_eq!(checked, texts.len());
    eprintln!("clip tokenizer parity: {checked}/{} strings exact on both towers", texts.len());
}

/// Truncation must keep BOS and EOS: the long string fills the whole context.
#[test]
fn truncation_keeps_the_frame() {
    let dir = testdata("clip/tokenizer");
    if !dir.join("vocab.json").exists() {
        brain_testutil::skip(&format!("{} missing vocab.json (make fetch/testdata)", dir.display()));
        return;
    }
    let tok = ClipBpe::from_dir(&dir).expect("load CLIP tokenizer assets");
    let e = tok.encode(&format!("{}truncated tail", "a ".repeat(120)));
    assert_eq!(e.ids.len(), CONTEXT);
    assert_eq!(e.ids[0], tok.bos_id());
    assert_eq!(*e.ids.last().unwrap(), tok.eos_id());
    assert_eq!(e.eos_index, CONTEXT - 1);
    assert!(e.mask.iter().all(|&m| m == 1), "a full context has no padding");
}

/// Decode is the inverse on the normalized text (CLIP's normalization is lossy
/// by construction: casing and whitespace runs do not survive it).
#[test]
fn decode_roundtrips_normalized_text() {
    let dir = testdata("clip/tokenizer");
    if !dir.join("vocab.json").exists() {
        brain_testutil::skip(&format!("{} missing vocab.json (make fetch/testdata)", dir.display()));
        return;
    }
    let tok = ClipBpe::from_dir(&dir).expect("load CLIP tokenizer assets");
    for (input, want) in [
        ("a photo of a cat", "a photo of a cat"),
        ("  Hello,   World!  ", "hello , world !"),
        ("café naïve", "café naïve"),
    ] {
        let got = tok.decode(&tok.encode_raw(input));
        assert_eq!(got, want, "decode of {input:?}");
    }
}

/// Pinned against `transformers` `CLIPTokenizer` (both SDXL tokenizer dirs),
/// `add_special_tokens=False`. `(text, tokenizer/ ids, tokenizer_2/ ids)`.
const CASES: &[(&str, &[u32], &[u32])] = &[
    ("hello world", &[3306, 1002], &[3306, 1002]),
    ("HELLO WORLD", &[3306, 1002], &[3306, 1002]),
    ("  \t\n leading   and\ttrailing   \n ", &[3833, 537, 37427], &[3833, 537, 37427]),
    ("don't you're I'll we've he'd it's", &[847, 713, 592, 982, 328, 1342, 649, 1200, 797, 1896, 585, 568], &[847, 713, 592, 982, 328, 1342, 649, 1200, 797, 1896, 585, 568]),
    ("supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis", &[1642, 2857, 13093, 2076, 5868, 26850, 835, 639, 38466, 28714, 749, 20253, 9800, 535, 532, 1065, 901, 1556, 13697, 9916, 78, 39031, 13903], &[1642, 2857, 13093, 2076, 5868, 26850, 835, 639, 38466, 28714, 749, 20253, 9800, 535, 532, 1065, 901, 1556, 13697, 9916, 78, 39031, 13903]),
    ("1234567890", &[272, 273, 274, 275, 276, 277, 278, 279, 280, 271], &[272, 273, 274, 275, 276, 277, 278, 279, 280, 271]),
    ("v1.2.3-rc4", &[341, 272, 269, 273, 269, 274, 268, 7457, 275], &[341, 272, 269, 273, 269, 274, 268, 7457, 275]),
    ("email@example.com https://a.b/c?d=e&f=g", &[4462, 287, 6228, 269, 2464, 30901, 12441, 320, 269, 321, 270, 322, 286, 323, 284, 324, 261, 325, 284, 326], &[4462, 287, 6228, 269, 2464, 30901, 12441, 320, 269, 321, 270, 322, 286, 323, 284, 324, 261, 325, 284, 326]),
    ("caf\u{e9} NA\u{cf}VE \u{dc}n\u{ef}c\u{f6}d\u{e9}", &[15304, 1097, 35689, 563, 6522, 77, 35689, 66, 7255, 67, 4166], &[15304, 1097, 35689, 563, 6522, 77, 35689, 66, 7255, 67, 4166]),
    ("\u{4f60}\u{597d}\u{4e16}\u{754c} \u{3053}\u{3093}\u{306b}\u{3061}\u{306f} \u{c548}\u{b155}\u{d558}\u{c138}\u{c694}", &[47466, 254, 29290, 121, 19759, 244, 163, 243, 490, 4813, 241, 3909, 241, 4813, 104, 4813, 94, 4813, 363, 16071, 230, 167, 227, 243, 33992, 15074, 116, 18541, 498], &[47466, 254, 29290, 121, 19759, 244, 163, 243, 490, 4813, 241, 3909, 241, 4813, 104, 4813, 94, 4813, 363, 16071, 230, 167, 227, 243, 33992, 15074, 116, 18541, 498]),
    ("\u{1f98a}\u{1f680}\u{1f3a8} emoji run", &[6040, 232, 22400, 13461, 16327, 1934], &[6040, 232, 22400, 13461, 16327, 1934]),
    ("mixed \u{1f600} emoji and text 42 times", &[6780, 7334, 16327, 537, 4160, 275, 273, 1661], &[6780, 7334, 16327, 537, 4160, 275, 273, 1661]),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", &[23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 19336], &[23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 23126, 19336]),
    ("!!! ??? ... ---", &[995, 3824, 678, 11079], &[0, 0, 0, 3824, 678, 11079]),
    ("<|startoftext|> literal <|endoftext|>", &[49406, 24269, 49407], &[49406, 24269, 49407]),
    ("tabs\tand\nnewlines\r\nand\u{a0}nbsp", &[29163, 537, 1218, 3418, 537, 6459, 4393], &[29163, 537, 1218, 3418, 537, 6459, 4393]),
    ("\u{dc}n\u{ef}c\u{f6}d\u{e9}  \u{2018}curly\u{2019} \u{201c}quotes\u{201d}", &[6522, 77, 35689, 66, 7255, 67, 4166, 728, 502, 20795, 728, 503, 728, 506, 5808, 728, 507], &[6522, 77, 35689, 66, 7255, 67, 4166, 728, 502, 20795, 728, 503, 728, 506, 5808, 728, 507]),
    ("((nested [brackets] {braces}))", &[13796, 557, 1356, 314, 36183, 316, 346, 19249, 92, 5167], &[13796, 557, 1356, 314, 36183, 316, 346, 19249, 92, 5167]),
    ("", &[], &[]),
    (" ", &[], &[]),
    // --- regressions, each a bug this file caught ------------------------------
    // A greedy `[^\s\p{L}\p{N}]+` run does NOT yield to the `'s`/`'re`/...
    // alternatives: alternation priority applies where a match STARTS, not
    // inside a run. `%'s` is `["%'", "s"]`, `''really''` is `["''", "really",
    // "''"]`, `>'rer` is `[">'", "rer"]`. Breaking out of the run instead gave
    // `["%", "'s"]` and was wrong on 351/3000 fuzzed strings.
    ("50%'s share", &[276, 271, 4, 262, 338, 1978], &[276, 271, 4, 262, 338, 1978]),
    ("he said ''really'' and 'tis so", &[797, 1946, 8445, 1414, 8445, 537, 713, 533, 706], &[797, 1946, 8445, 1414, 8445, 537, 713, 533, 706]),
    ("a >'rer b", &[320, 29, 262, 3407, 321], &[320, 29, 262, 3407, 321]),
    // `\p{L}` is the general category `L*`, NOT `char::is_alphabetic` (which is
    // the Alphabetic *property* = `L* | Nl | Other_Alphabetic`). Roman numerals
    // are `Nl`: they must take the `[\p{N}]` single-char branch, not glue onto
    // the following word and move the `</w>`.
    ("\u{2167}\u{2166} roman xyz", &[158, 227, 371, 158, 227, 370, 7370, 20023, 345], &[158, 227, 371, 158, 227, 370, 7370, 20023, 345]),
    ("\u{2460} circled \u{bd} half", &[158, 239, 510, 2459, 12827, 33613, 2349], &[158, 239, 510, 2459, 12827, 33613, 2349]),
];

/// A wider tricky-string gate than the 7-string safetensors golden: every id
/// below came from `transformers` on this box, so a drift in the pre-tokenizer,
/// the `</w>` placement, the digit rule or the added-token split shows up as an
/// exact-value mismatch, not as a tolerance.
///
/// Covers: casing, whitespace of every kind (tab/CR/LF/NBSP) and repeated runs,
/// contractions, a 100-character word, long compound words, digit runs, URLs,
/// Latin-1 accents, CJK, Hangul, astral-plane emoji, curly quotes, brackets,
/// literal `<|startoftext|>`/`<|endoftext|>`, and the empty / all-space strings.
#[test]
fn clip_bpe_matches_hf_on_a_wider_tricky_corpus() {
    let dir = testdata("clip/tokenizer");
    if !dir.join("vocab.json").exists() || !dir.join("merges.txt").exists() {
        brain_testutil::skip(&format!("{} missing vocab.json/merges.txt (make fetch/testdata)", dir.display()));
        return;
    }
    let tok1 = ClipBpe::from_dir(&dir).expect("load CLIP tokenizer assets");
    let tok2 = ClipBpe::from_dir(&dir).expect("load CLIP tokenizer assets").with_pad("!");
    let mut differ = 0usize;
    for &(text, want1, want2) in CASES {
        assert_eq!(tok1.encode_raw(text), want1, "tokenizer/ ids for {text:?}");
        assert_eq!(tok2.encode_raw(text), want2, "tokenizer_2/ ids for {text:?}");
        if want1 != want2 {
            differ += 1;
        }
    }
    assert!(differ > 0, "the corpus must contain at least one string where the two towers disagree");
    eprintln!("clip tokenizer: {} tricky strings exact on both towers ({differ} where they differ)", CASES.len());
}
