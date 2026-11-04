// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::rope_scaling` against the rotary tables `transformers` itself built
//! for real checkpoints: every DeepSeek text decoder, one of each scaling
//! family - none (deepseek-llm / math / coder-v1.5, the R1 Qwen distills),
//! `linear` x4 (deepseek-coder 1.3b / 6.7b) and `llama3` x8 (R1-Distill-Llama).
//!
//! Each checkpoint's own `config.json` (from the model store) is parsed by
//! [`RopeScaling::from_config`] exactly as an import would, and the resulting
//! table is compared with `rope.safetensors` - the model's live
//! `rotary_emb.inv_freq` / `attention_scaling`, dumped by
//! `tools/goldens/deepseek_dump_reference.py` into
//! `testdata/deepseek/<repo>/`, which itself asserts that buffer against an
//! independent recomputation before writing it.

use model::rope_scaling::RopeScaling;

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
fn every_deepseek_rope_table_matches_the_one_transformers_built() {
    let mut compared = 0;
    for repo in CHECKPOINTS {
        let golden = brain_testutil::testdata_path(&format!("deepseek/{repo}/rope.safetensors"));
        let Some(config) = brain_testutil::model_dir(&format!("deepseek-ai/{repo}")).map(|d| std::path::Path::new(&d).join("config.json")) else {
            brain_testutil::skip(&format!("{repo}: no models directory"));
            continue;
        };
        if !golden.exists() || !config.exists() {
            brain_testutil::skip(&format!("{repo}: needs {} and {}", golden.display(), config.display()));
            continue;
        }
        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
        let theta = c["rope_theta"].as_f64().unwrap_or(10_000.0) as f32;
        let head_dim = c["head_dim"].as_u64().unwrap_or(c["hidden_size"].as_u64().unwrap() / c["num_attention_heads"].as_u64().unwrap()) as u32;
        let scaling = RopeScaling::from_config(&c["rope_scaling"]).unwrap_or_else(|e| panic!("{repo}: {e}"));
        let (got, af) = model::rope_scaling::inv_freq(head_dim, theta, scaling.as_ref());

        let tensors = checkpoint::safetensors::read(golden.to_str().unwrap()).unwrap();
        let get = |n: &str| tensors.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("{repo}: golden has no {n}")).data.clone();
        let want = get("inv_freq");
        let want_af = get("attention_scaling")[0];
        assert_eq!(got.len(), want.len(), "{repo}: table length");
        // Both sides are f32 reciprocals of f32 powers of the same inputs; two
        // units in the last place is rounding, anything wider is a formula.
        let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs() / w.abs()).fold(0.0f32, f32::max);
        assert!(worst <= 2.0 * f32::EPSILON, "{repo}: worst relative error {worst:e} ({scaling:?})");
        assert_eq!(af, want_af, "{repo}: attention factor");
        compared += 1;
    }
    println!("rope tables compared: {compared}/{}", CHECKPOINTS.len());
}
