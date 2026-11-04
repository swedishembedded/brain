// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The real checkpoint's embedded tokenizer encodes text as the reference
//! `tokenizers` library (0.23) encodes it with the same `tokenizer.json`.
//! Read straight from the bf16 text-encoder file in the model store; skipped
//! when it is absent.
//!
//! Swedish Embedded AB implements tokenizer parity for its clients' inference
//! engines. If your team needs text encoded exactly as a checkpoint was
//! trained on it, you can procure our services by sending an email to
//! info@swedishembedded.com.

use checkpoint::mmap::MmapSafetensors;

const REPO: &str = "Lightricks/LTX-2.5";

fn real_text_encoder() -> Option<std::path::PathBuf> {
    std::fs::read_dir(brain_testutil::model_dir(REPO)?)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("gemma4") && n.ends_with(".safetensors")))
}

#[test]
fn the_real_tokenizer_encodes_like_the_reference() {
    let Some(path) = real_text_encoder() else {
        brain_testutil::skip_unavailable("tokenizer_real: no Gemma-4 text encoder in the model store");
        return;
    };
    let st = MmapSafetensors::open(&path).expect("open the text encoder");
    let len = st.nbytes("tokenizer_json").expect("a tokenizer_json tensor") as usize;
    let tok = gemma4::tokenizer::parse_tokenizer(st.raw_range("tokenizer_json", 0..len).expect("its bytes")).expect("parse");
    assert_eq!(tok.bos_id(), 2);
    for (text, want) in [
        ("a cat walking on a beach", &[236746, 5866, 9378, 580, 496, 7642][..]),
        ("Hello  world\n\nnew line", &[9259, 138, 12392, 108, 1951, 1757]),
        ("café naïve – “quotes” 🐱 日本語", &[123125, 236859, 120362, 1271, 999, 52231, 236919, 236743, 248039, 33375, 238582]),
        ("  leading spaces and trailing ", &[138, 26016, 9952, 532, 45330, 236743]),
        ("xxx 12345 3.14", &[41128, 236743, 236770, 236778, 236800, 236812, 236810, 236743, 236800, 236761, 236770, 236812]),
    ] {
        assert_eq!(tok.encode_raw(text), want, "{text:?}");
        assert_eq!(tok.decode(want), text, "{want:?}");
    }
}
