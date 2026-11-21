// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `QwenBpe`, brain's generic HF byte-level BPE loader, on Inkling 2 Mini's own
//! `tokenizer.json` against the ids the `tokenizers` library produces from the same
//! file (`tools/goldens/inkling2_tokenizer_dump.py`).
//!
//! The file is byte-level BPE with no normalizer, an o200k-style `Split` regex,
//! `ByteLevel(use_regex: false)`, `ignore_merges` and 60 added special tokens after the
//! 199,998-entry vocabulary; the corpus spells every one of them inline beside hostile
//! text. The encode must match id for id and decode must reproduce `tokenizers`' text.

use data::qwen_tokenizer::QwenBpe;
use data::Tokenizer;

#[test]
fn the_generic_bpe_encodes_inkling2_as_the_tokenizers_library_does() {
    let golden = brain_testutil::testdata_path("inkling2/tokenizer.json");
    let Some(dir) = brain_testutil::model_dir("thinkingmachines/Inkling-2-Mini") else {
        brain_testutil::skip("Inkling-2-Mini: no models directory");
        return;
    };
    let file = std::path::Path::new(&dir).join("tokenizer.json");
    if !golden.exists() || !file.exists() {
        brain_testutil::skip(&format!("Inkling-2-Mini: needs {} and {}", golden.display(), file.display()));
        return;
    }
    let tok = QwenBpe::from_file(file.to_str().unwrap()).unwrap();
    let g: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&golden).unwrap()).unwrap();
    assert_eq!(tok.vocab_size(), g["vocab_size"].as_u64().unwrap() as usize, "vocab size");
    for (name, id) in g["added_tokens"].as_object().unwrap() {
        assert_eq!(tok.special_id(name), Some(id.as_u64().unwrap() as u32), "special token {name}");
    }
    for row in g["rows"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let want: Vec<u32> = row["file_plain"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(text), want, "encode of {text:?}");
        assert_eq!(tok.decode(&want), row["decode_plain"].as_str().unwrap(), "decode of {text:?}");
    }
}
