// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Each DeepSeek chat checkpoint's own Jinja template, rendered by brain,
//! against `transformers`' `apply_chat_template` on the same conversations.
//!
//! The conversations (`tools/goldens/deepseek_dump_reference.py`) cover what
//! the templates disagree on: a system prompt (coder-instruct substitutes its
//! own when there is none), R1's reasoning history (stripped from an earlier
//! assistant turn) and its `<think>\n` generation prefill, and whitespace
//! kept verbatim. A checkpoint without a template (the base models) must be
//! reported as having none, never rendered with another model's.

use std::collections::BTreeMap;

use data::chat_template::{parse_json_ordered, ChatTemplate};

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
fn every_deepseek_chat_template_renders_as_transformers_does() {
    let mut compared = 0;
    for repo in CHECKPOINTS {
        let golden = brain_testutil::testdata_path(&format!("deepseek/{repo}/chat.json"));
        let Some(dir) = brain_testutil::model_dir(&format!("deepseek-ai/{repo}")) else {
            brain_testutil::skip(&format!("{repo}: no models directory"));
            continue;
        };
        let dir = std::path::PathBuf::from(dir);
        if !golden.exists() || !dir.join("tokenizer_config.json").exists() {
            brain_testutil::skip(&format!("{repo}: needs {} and a checkpoint", golden.display()));
            continue;
        }
        let g: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&golden).unwrap()).unwrap();
        let tmpl = ChatTemplate::from_model_dir(&dir);
        if !g["has_template"].as_bool().unwrap() {
            assert!(tmpl.is_err(), "{repo} ships no chat template, and must not be given one");
            compared += 1;
            continue;
        }
        let tmpl = tmpl.unwrap_or_else(|e| panic!("{repo}: {e}"));
        for (name, r) in g["renders"].as_object().unwrap() {
            for (key, gen) in [("without_generation_prompt", false), ("with_generation_prompt", true)] {
                let Some(want) = r[key].as_str() else { continue };
                let messages = parse_json_ordered(&r["messages"].to_string()).unwrap();
                let got = tmpl.render(messages, None, gen, &BTreeMap::new()).unwrap_or_else(|e| panic!("{repo}/{name}/{key}: {e}"));
                assert_eq!(got, want, "{repo}/{name}/{key}");
            }
        }
        compared += 1;
    }
    println!("chat templates compared: {compared}/{}", CHECKPOINTS.len());
}
