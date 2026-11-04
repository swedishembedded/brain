// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `QwenBpe` on every DeepSeek text checkpoint's own `tokenizer.json`, against
//! the ids the `tokenizers` library produces from the same file.
//!
//! The family ships four tokenizer shapes, and each is here:
//! NFC plus the Qwen2 split (R1-Distill-Qwen), the Llama 3 split with
//! `ignore_merges` (R1-Distill-Llama), a five-way split sequence
//! with `Digits` (deepseek-llm, deepseek-math, deepseek-coder-v1.5), and
//! coder v1's four-way split. The corpus
//! (`tools/goldens/deepseek_dump_reference.py`) is chosen to find where they
//! differ: digit runs, CJK and kana beside Latin, `\r\n`, trailing spaces,
//! emoji and ZWJ sequences, decomposed accents, fullwidth punctuation, code,
//! and every special token spelled inline.
//!
//! The ids compared are the file's own encoding without its post-processor
//! (`file_plain`); a BOS the post-processor or `tokenizer_config.json` adds
//! is not part of what this pins.

use data::qwen_tokenizer::QwenBpe;
use data::Tokenizer;

const CHECKPOINTS: &[&str] = &[
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "deepseek-coder-1.3b-base",
    "deepseek-coder-1.3b-instruct",
    "deepseek-coder-6.7b-base",
    "deepseek-coder-6.7b-instruct",
    "deepseek-coder-7b-base-v1.5",
    "deepseek-coder-7b-instruct-v1.5",
    "deepseek-llm-7b-base",
    "deepseek-llm-7b-chat",
    "deepseek-math-7b-base",
    "deepseek-math-7b-instruct",
];

#[test]
fn every_deepseek_tokenizer_encodes_as_the_tokenizers_library_does() {
    let mut compared = 0;
    for repo in CHECKPOINTS {
        let golden = brain_testutil::testdata_path(&format!("deepseek/{repo}/tokenizer.json"));
        let Some(dir) = brain_testutil::model_dir(&format!("deepseek-ai/{repo}")) else {
            brain_testutil::skip(&format!("{repo}: no models directory"));
            continue;
        };
        let file = std::path::Path::new(&dir).join("tokenizer.json");
        if !golden.exists() || !file.exists() {
            brain_testutil::skip(&format!("{repo}: needs {} and {}", golden.display(), file.display()));
            continue;
        }
        let tok = QwenBpe::from_file(file.to_str().unwrap()).unwrap_or_else(|e| panic!("{repo}: {e}"));
        let g: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&golden).unwrap()).unwrap();
        assert_eq!(tok.vocab_size(), g["vocab_size"].as_u64().unwrap() as usize, "{repo}: vocab size");
        for row in g["rows"].as_array().unwrap() {
            let text = row["text"].as_str().unwrap();
            let want: Vec<u32> = row["file_plain"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(tok.encode(text), want, "{repo}: {text:?}");
            assert_eq!(tok.decode(&want), row["decode_plain"].as_str().unwrap(), "{repo}: decode of {text:?}");
        }
        compared += 1;
    }
    println!("tokenizers compared: {compared}/{}", CHECKPOINTS.len());
}
