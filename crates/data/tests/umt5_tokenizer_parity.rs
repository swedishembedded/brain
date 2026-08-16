// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! umT5 SentencePiece-unigram parity: `data::unigram::UnigramTokenizer` must
//! produce **exactly** the ids the real `google/umt5-xxl` tokenizer produces
//! for the prompts Wan2.1 is conditioned on.
//!
//! The golden is `testdata/golden/wan/t5/tokens.safetensors`, written by
//! `tools/goldens/wan_t5_dump_reference.py` at Wan's own contract
//! (`clean='whitespace'`, `padding='max_length'`, `max_length=512`,
//! `truncation=True`): `input_ids [P, 512]`, `attention_mask [P, 512]` and
//! `seq_lens [P]`.
//!
//! The prompt strings are duplicated here rather than read from the fixture:
//! a golden that carries its own inputs cannot notice that the two sides
//! drifted apart. `TOKENIZER_PROMPTS` in the dump script is the other copy and
//! the two must be edited together.
//!
//! The corpus is multilingual on purpose - a 256k vocabulary is the entire
//! reason umT5 replaces T5 v1.1 in this pipeline, and an ASCII-only fixture
//! would exercise none of it. It carries Wan's real Chinese negative prompt,
//! the empty prompt (the unconditional branch), six scripts, emoji outside the
//! BMP, a Deseret string no piece covers (the `unk` + `fuse_unk` path) and a
//! whitespace mess that only survives if `whitespace_clean` runs first.
//!
//! Skips itself when the fixture or `$BRAIN_WAN_TOKENIZER` is absent.

use std::path::Path;

use brain_testutil::testdata;
use data::tokenizer::Tokenizer;
use data::unigram::UnigramTokenizer;

/// `wan/configs/shared_config.py`: `text_len = 512`.
const TEXT_LEN: usize = 512;

/// Mirrors `TOKENIZER_PROMPTS` in `tools/goldens/wan_t5_dump_reference.py`.
fn prompts() -> Vec<String> {
    [
        "A belgian malinois running on a paved highway, cinematic lighting",
        "两只可爱的橘猫戴着墨镜,在阳光下的沙滩上散步。",
        "Un cafe au bord de la Seine a Paris, sous la pluie, ambiance cinematographique",
        concat!(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，",
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，",
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，",
            "杂乱的背景，三条腿，背景人很多，倒着走"
        ),
        "",
        "Ελληνικά, русский, العربية, हिन्दी, 한국어, 日本語のテキスト",
        "a rocket \u{1F680} and a cat \u{1F408} with a beaker \u{1F9EA}",
        "\u{10437}\u{10437} deseret, \u{FB01} ligature, e\u{301} combining",
        "   collapsed    whitespace   and\ttabs   ",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn tokenizer() -> Option<UnigramTokenizer> {
    let dir = std::env::var("BRAIN_WAN_TOKENIZER").ok().filter(|s| !s.is_empty());
    let Some(dir) = dir else {
        eprintln!("SKIP: set BRAIN_WAN_TOKENIZER to a google/umt5-xxl tokenizer directory");
        return None;
    };
    if !Path::new(&format!("{dir}/tokenizer.json")).exists() {
        eprintln!("SKIP: {dir}/tokenizer.json not found");
        return None;
    }
    Some(UnigramTokenizer::from_dir(&dir).expect("load umT5 tokenizer"))
}

#[test]
fn umt5_unigram_matches_the_reference_ids() {
    let fx = testdata("golden/wan/t5/tokens.safetensors");
    if !Path::new(&fx).exists() {
        eprintln!("SKIP: fixture {fx} absent - run tools/goldens/wan_t5_dump_reference.py");
        return;
    }
    let Some(tok) = tokenizer() else { return };

    let g = checkpoint::safetensors::read(&fx).expect("read golden");
    let find = |n: &str| g.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("golden {n}"));
    let ids = find("input_ids");
    let mask = find("attention_mask");
    let lens = find("seq_lens");
    let ps = prompts();
    assert_eq!(ids.shape, vec![ps.len(), TEXT_LEN], "fixture prompt count drifted from the test");

    // umT5's embedding table is padded to 256384 rows; the tokenizer itself
    // only ever emits ids below its own 256300-piece vocabulary.
    assert_eq!(tok.vocab_size(), 256300);
    assert_eq!(tok.pad_id(), 0);
    assert_eq!(tok.special_id("</s>"), Some(1));
    assert_eq!(tok.unk_id(), 3);

    let mut worst: Option<String> = None;
    for (p, text) in ps.iter().enumerate() {
        let want_ids: Vec<u32> = ids.data[p * TEXT_LEN..(p + 1) * TEXT_LEN].iter().map(|&x| x as u32).collect();
        let want_mask: Vec<u32> = mask.data[p * TEXT_LEN..(p + 1) * TEXT_LEN].iter().map(|&x| x as u32).collect();
        let n = lens.data[p] as usize;

        // Wan cleans every prompt (`clean='whitespace'`) before tokenizing.
        let cleaned = UnigramTokenizer::clean_whitespace(text);
        let (got_ids, got_mask) = tok.encode_padded(&cleaned, TEXT_LEN);
        let got_len = got_mask.iter().filter(|&&m| m == 1).count();
        eprintln!(
            "  prompt {p}: {got_len} tokens (reference {n}) {}",
            if got_ids == want_ids && got_mask == want_mask { "OK" } else { "MISMATCH" }
        );
        if got_ids != want_ids || got_mask != want_mask {
            let at = got_ids.iter().zip(&want_ids).position(|(a, b)| a != b);
            worst.get_or_insert(format!(
                "prompt {p} ({text:?}): first divergence at {at:?}\n  got  {:?}\n  want {:?}",
                &got_ids[..got_len.max(n).min(TEXT_LEN)],
                &want_ids[..got_len.max(n).min(TEXT_LEN)]
            ));
        }
        // Round-tripping the content ids must give the cleaned prompt back,
        // except where a piece was genuinely unknown (deseret, the ligature).
        if !got_ids[..got_len].contains(&tok.unk_id()) {
            let round = tok.decode(&got_ids[..got_len - 1]);
            assert_eq!(round, cleaned, "prompt {p} does not round-trip");
        }
    }
    assert!(worst.is_none(), "{}", worst.unwrap());
}

/// The one thing the fixture cannot pin: that `encode_padded` truncates a
/// prompt longer than the window while keeping the template's `</s>`, which is
/// what `truncation=True` does. Built from the real vocabulary so the piece
/// count is real.
#[test]
fn a_prompt_longer_than_the_window_truncates_and_keeps_the_eos() {
    let Some(tok) = tokenizer() else { return };
    let long = "a belgian malinois running on a paved highway, ".repeat(120);
    let (ids, mask) = tok.encode_padded(&long, TEXT_LEN);
    assert_eq!(ids.len(), TEXT_LEN);
    assert!(mask.iter().all(|&m| m == 1), "a truncated prompt has no padding");
    assert_eq!(ids[TEXT_LEN - 1], tok.special_id("</s>").unwrap(), "the eos survives truncation");
    assert!(tok.encode(&long).len() > TEXT_LEN, "the corpus is not long enough to truncate");
}
