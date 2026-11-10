// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SentencePiece-BPE tokenizer: score-ordered merging, byte fallback,
//! user-defined markers, and ids identical to the reference implementation
//! on the released vocabulary. The golden ids come from
//! `tools/goldens/mirage_tokenizer_dump_reference.py`.

use mirage::cact::CactFile;
use mirage::cactmap::CactModel;
use mirage::tokenizer::{Tokenizer, TokenizerError};
use serde::Deserialize;

use crate::common::{released_or_skip, RELEASED_ARCHITECTURE_JSON};

const NORMAL: u8 = 0;
const UNKNOWN: u8 = 1;
const CONTROL: u8 = 2;
const USER_DEFINED: u8 = 3;
const BYTE: u8 = 4;

struct Vocab {
    pieces: Vec<(String, f32, u8)>,
    dummy_prefix: bool,
    byte_fallback: bool,
}

impl Vocab {
    /// `<pad> </s> <s> <unk>`, the 256 byte pieces, then the given normal and
    /// user-defined pieces; ids follow that order.
    fn new(extra: &[(&str, f32, u8)]) -> Vocab {
        let mut pieces: Vec<(String, f32, u8)> = vec![("<pad>".into(), 0.0, CONTROL), ("</s>".into(), 0.0, CONTROL), ("<s>".into(), 0.0, CONTROL), ("<unk>".into(), 0.0, UNKNOWN)];
        pieces.extend((0..256).map(|b| (format!("<0x{b:02X}>"), 0.0, BYTE)));
        pieces.extend(extra.iter().map(|(p, s, t)| (p.to_string(), *s, *t)));
        Vocab { pieces, dummy_prefix: false, byte_fallback: true }
    }

    fn id(&self, piece: &str) -> u32 {
        self.pieces.iter().position(|(p, _, _)| p == piece).unwrap_or_else(|| panic!("no piece {piece}")) as u32
    }

    fn blob(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for v in [self.pieces.len() as u32, 0, 1, 2, 3] {
            out.extend(v.to_le_bytes());
        }
        out.push(self.dummy_prefix as u8);
        out.push(self.byte_fallback as u8);
        out.extend(0u16.to_le_bytes());
        for (piece, score, kind) in &self.pieces {
            out.extend(score.to_le_bytes());
            out.push(*kind);
            out.extend((piece.len() as u16).to_le_bytes());
            out.extend(piece.as_bytes());
        }
        out
    }

    fn tokenizer(&self) -> Tokenizer {
        Tokenizer::from_blob(&self.blob()).unwrap()
    }
}

fn letters() -> Vec<(&'static str, f32, u8)> {
    vec![("▁", -9.0, NORMAL), ("a", -5.0, NORMAL), ("b", -5.0, NORMAL), ("c", -5.0, NORMAL), ("d", -5.0, NORMAL)]
}

#[test]
fn the_best_scoring_adjacent_pair_merges_first() {
    let mut extra = letters();
    extra.extend([("ab", -1.0, NORMAL), ("bc", -2.0, NORMAL), ("abc", -0.5, NORMAL), ("bcd", -0.1, NORMAL)]);
    let v = Vocab::new(&extra);
    let t = v.tokenizer();
    // "abc": ab (-1) beats bc (-2); then ab+c = abc exists.
    assert_eq!(t.encode("abc").unwrap(), vec![v.id("abc")]);
    // "abcd": bc (-2) and ab (-1) and cd(absent): ab first, then ab+c=abc(-0.5) vs c+d absent -> abc, then abc+d absent.
    assert_eq!(t.encode("abcd").unwrap(), vec![v.id("abc"), v.id("d")]);
    // "bcd" merges by the whole-word piece once bc forms.
    assert_eq!(t.encode("bcd").unwrap(), vec![v.id("bcd")]);
}

#[test]
fn equal_scores_merge_leftmost_first() {
    let mut extra = letters();
    extra.extend([("ab", -1.0, NORMAL), ("bc", -1.0, NORMAL)]);
    let v = Vocab::new(&extra);
    assert_eq!(v.tokenizer().encode("abc").unwrap(), vec![v.id("ab"), v.id("c")]);
}

#[test]
fn a_character_outside_the_vocabulary_falls_back_to_its_utf8_bytes() {
    let v = Vocab::new(&letters());
    assert_eq!(v.tokenizer().encode("aé").unwrap(), vec![v.id("a"), v.id("<0xC3>"), v.id("<0xA9>")]);
    assert_eq!(v.tokenizer().encode("😀").unwrap(), vec![v.id("<0xF0>"), v.id("<0x9F>"), v.id("<0x98>"), v.id("<0x80>")]);
    let mut no_fallback = Vocab::new(&letters());
    no_fallback.byte_fallback = false;
    assert_eq!(no_fallback.tokenizer().encode("aé").unwrap(), vec![no_fallback.id("a"), 3], "unknown pieces map to <unk>");
}

#[test]
fn user_defined_markers_are_atomic_and_split_the_text() {
    let mut extra = letters();
    extra.extend([("ab", -1.0, NORMAL), ("<|en|>", 0.0, USER_DEFINED), ("<|e", 0.0, USER_DEFINED)]);
    let v = Vocab::new(&extra);
    let t = v.tokenizer();
    assert_eq!(t.encode("ab<|en|>ab").unwrap(), vec![v.id("ab"), v.id("<|en|>"), v.id("ab")]);
    // The longer marker wins where two start at the same position.
    assert_eq!(t.encode("<|en|>").unwrap(), vec![v.id("<|en|>")]);
    // Merging never crosses a marker: a+<marker>+b stays three pieces.
    assert_eq!(t.encode("a<|en|>b").unwrap(), vec![v.id("a"), v.id("<|en|>"), v.id("b")]);
    assert_eq!(t.language_tag("en"), Some(v.id("<|en|>")));
    assert_eq!(t.language_tag("xx"), None);
}

#[test]
fn spaces_become_the_word_boundary_mark_and_the_dummy_prefix_is_optional() {
    let mut extra = letters();
    extra.extend([("▁a", -1.0, NORMAL)]);
    let mut v = Vocab::new(&extra);
    assert_eq!(v.tokenizer().encode("a a").unwrap(), vec![v.id("a"), v.id("▁a")]);
    v.dummy_prefix = true;
    let t = v.tokenizer();
    assert_eq!(t.encode("a").unwrap(), vec![v.id("▁a")]);
    assert_eq!(t.decode(&[v.id("▁a"), v.id("▁a")]).unwrap(), "a a", "the dummy prefix's space is dropped on decode");
    assert!(t.encode("").unwrap().is_empty());
}

#[test]
fn decode_assembles_bytes_and_skips_control_pieces() {
    let v = Vocab::new(&letters());
    let t = v.tokenizer();
    let ids = [2, v.id("a"), v.id("<0xC3>"), v.id("<0xA9>"), 3, v.id("▁"), v.id("b"), 1];
    assert_eq!(t.decode(&ids).unwrap(), "aé b");
    // A lone continuation byte decodes lossily rather than failing.
    assert_eq!(t.decode(&[v.id("<0xA9>")]).unwrap(), "\u{fffd}");
    assert!(matches!(t.decode(&[u32::MAX]), Err(TokenizerError::UnknownId(u32::MAX))));
}

#[test]
fn encoding_scales_to_long_inputs() {
    let mut extra = letters();
    extra.extend([("ab", -1.0, NORMAL), ("abab", -0.5, NORMAL)]);
    let v = Vocab::new(&extra);
    let text = "ab".repeat(100_000);
    let ids = v.tokenizer().encode(&text).unwrap();
    assert_eq!(ids.len(), 50_000);
    assert!(ids.iter().all(|&i| i == v.id("abab")));
    let big = "a".repeat(mirage::tokenizer::MAX_TEXT_BYTES + 1);
    assert!(matches!(v.tokenizer().encode(&big), Err(TokenizerError::TextTooLong { .. })));
}

#[test]
fn a_malformed_blob_is_refused_without_panicking() {
    let v = Vocab::new(&letters());
    let good = v.blob();
    assert!(Tokenizer::from_blob(&good).is_ok());
    for len in 0..good.len() {
        assert!(Tokenizer::from_blob(&good[..len]).is_err(), "a blob cut at {len} parsed");
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(Tokenizer::from_blob(&trailing).is_err(), "trailing bytes");

    let mut bad_kind = good.clone();
    bad_kind[24 + 4] = 9; // the first piece's type byte
    assert!(Tokenizer::from_blob(&bad_kind).unwrap_err().to_string().contains("type"));

    let mut huge = good.clone();
    huge[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Tokenizer::from_blob(&huge).is_err());

    let mut bad_special = good.clone();
    bad_special[4..8].copy_from_slice(&99_999u32.to_le_bytes());
    assert!(Tokenizer::from_blob(&bad_special).unwrap_err().to_string().contains("pad"));

    let mut non_utf8 = good.clone();
    let at = good.len() - 1;
    non_utf8[at] = 0xff; // the last byte of the last piece
    assert!(Tokenizer::from_blob(&non_utf8).is_err());

    let mut nan_score = good.clone();
    nan_score[24..28].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(Tokenizer::from_blob(&nan_score).unwrap_err().to_string().contains("score"));

    let mut dup = Vocab::new(&letters());
    dup.pieces.push(("a".into(), -1.0, NORMAL));
    assert!(Tokenizer::from_blob(&dup.blob()).unwrap_err().to_string().contains("duplicate"));

    let mut bad_byte = Vocab::new(&[]);
    bad_byte.pieces[10].0 = "<0xZZ>".into();
    assert!(Tokenizer::from_blob(&bad_byte.blob()).unwrap_err().to_string().contains("byte piece"));
}

// ---- the released vocabulary against the reference tokenizer ----

#[derive(Deserialize)]
struct Golden {
    cact_sha256: String,
    pieces: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

fn released_tokenizer(test: &str) -> Option<Tokenizer> {
    let dir = released_or_skip(test)?;
    let config = mirage::config::MirageConfig::from_json(RELEASED_ARCHITECTURE_JSON).unwrap();
    let model = CactModel::new(CactFile::open(&dir.join("mirage.cact")).unwrap(), config).unwrap();
    Some(Tokenizer::from_blob(model.tokenizer_blob().unwrap()).expect("the released tokenizer blob parses"))
}

#[test]
fn the_released_vocabulary_has_the_published_structure() {
    let Some(t) = released_tokenizer("released tokenizer structure") else { return };
    assert_eq!(t.vocab_size(), 8203);
    assert_eq!((t.pad_id(), t.eos_id(), t.bos_id(), t.unk_id()), (0, 1, 2, 3));
    assert!(!t.adds_dummy_prefix() && t.has_byte_fallback());
    let tags: Vec<_> = ["en", "es", "fr", "pt", "de", "it", "nl", "pl", "ru", "ar", "zh"].iter().map(|l| t.language_tag(l).unwrap()).collect();
    assert_eq!(tags, (8192..8203).collect::<Vec<u32>>());
}

#[test]
fn ids_match_the_reference_tokenizer_and_text_round_trips() {
    let Some(t) = released_tokenizer("released tokenizer golden") else { return };
    let golden: Golden = serde_json::from_str(include_str!("../golden/tokenizer.json")).unwrap();
    assert_eq!(golden.pieces, t.vocab_size(), "the golden was dumped from this vocabulary");
    assert_eq!(golden.cact_sha256.len(), 64);
    assert!(golden.cases.len() >= 20);
    let mut round_trips = 0;
    for case in &golden.cases {
        let ids = t.encode(&case.text).unwrap();
        assert_eq!(ids, case.ids, "ids for {:?}", case.text);
        let decoded = t.decode(&ids).unwrap();
        assert_eq!(decoded, case.decoded, "decode of {:?}", case.text);
        if case.decoded == case.text {
            round_trips += 1;
        }
    }
    assert!(round_trips >= 18, "most cases are exact round trips, got {round_trips}");
}
