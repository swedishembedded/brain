// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every DeepSeek text checkpoint in the model store, as downloaded (a
//! Hugging Face directory, safetensors or `pytorch_model*.bin`), resolves
//! through `Qwen3Spec` to its own `deepseek-ai/<repo>` id, its weights the
//! directory itself and its tokenizer the directory's own `tokenizer.json` -
//! including the checkpoints whose embedding table is padded past the
//! tokenizer's vocabulary (llm/math: 100015 tokens, 102400 rows).

use std::collections::BTreeMap;

use brain_modelstore::resolve::{resolve_under, ArchSpec, Resolution};

const TEXT: &[&str] = &[
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
fn every_deepseek_text_checkpoint_resolves_to_its_own_id() {
    let Some(root) = brain_modelstore::default_root() else {
        brain_testutil::skip("no model store");
        return;
    };
    let all = brain_modelstore::inventory::scan(&root);
    let spec = qwen3::spec::Qwen3Spec;
    let specs: Vec<&dyn ArchSpec> = vec![&spec];
    let mut resolved = 0;
    for repo in TEXT {
        let dir = root.join("deepseek-ai").join(repo);
        if !dir.join("config.json").exists() {
            brain_testutil::skip(&format!("{repo} not downloaded"));
            continue;
        }
        // This checkpoint's own records, resolved under the store root.
        let records: Vec<_> = all.iter().filter(|r| r.path.starts_with(&dir)).cloned().collect();
        match resolve_under(&root, "qwen3", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => {
                assert_eq!(a.id, format!("deepseek-ai/{repo}"));
                assert_eq!(a.roles["weights"], dir, "{repo}");
                assert_eq!(a.roles["tokenizer"], dir.join("tokenizer.json"), "{repo}");
            }
            other => panic!("{repo}: {other:?}"),
        }
        resolved += 1;
    }
    eprintln!("resolved {resolved}/{} DeepSeek text checkpoints", TEXT.len());
}
