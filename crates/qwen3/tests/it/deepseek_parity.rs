// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The DeepSeek dense text decoders, run by brain's qwen3 decoder straight
//! off their Hugging Face checkpoints, against what `transformers` computed.
//!
//! The goldens come from `tools/goldens/deepseek_dump_reference.py`, which
//! runs each model truncated to its first two decoder layers (final norm and
//! head still applied) in fp32 on the CPU. Each rung isolates one failure
//! the next would only blur:
//!
//! 1. the checkpoint's tensors map onto brain's parameters both ways (no
//!    tensor unused, none missing);
//! 2. the embedding rows are the checkpoint's, and layer 0's output agrees
//!    (bias, QK-norm switch, RoPE table, GQA grouping);
//! 3. the two-layer logits agree row by row.
//!
//! A checkpoint absent from the model store, or without a golden, is
//! skipped by name.

use qwen3::{Qwen, Shard};

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

/// Row-wise cosine floor for every compared activation.
const COS_FLOOR: f64 = 0.9999;

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// The smallest row-wise cosine between `got` and `want`, both `[rows, width]`.
fn min_row_cos(got: &[f32], want: &[f32], width: usize) -> f64 {
    assert_eq!(got.len(), want.len());
    got.chunks(width)
        .zip(want.chunks(width))
        .map(|(a, b)| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (na * nb)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn every_deepseek_text_decoder_matches_transformers_through_two_layers() {
    if gpu_disabled() {
        return;
    }
    let mut compared = 0;
    for repo in CHECKPOINTS {
        let golden_dir = brain_testutil::testdata_path(&format!("deepseek/{repo}"));
        let Some(model_dir) = brain_testutil::model_dir(&format!("deepseek-ai/{repo}")) else {
            brain_testutil::skip(&format!("{repo}: no models directory"));
            continue;
        };
        let model_dir = std::path::PathBuf::from(model_dir);
        if !golden_dir.join("forward.safetensors").exists() || !model_dir.join("config.json").exists() {
            brain_testutil::skip(&format!("{repo}: needs a golden in {} and a checkpoint in {}", golden_dir.display(), model_dir.display()));
            continue;
        }
        let reader = match checkpoint::weightio::WeightReader::open_hf_dir(&model_dir) {
            Ok(r) => r,
            Err(e) => {
                brain_testutil::skip(&format!("{repo}: checkpoint not readable as safetensors ({e})"));
                continue;
            }
        };
        let full = qwen3::hf::decoder_config(&std::fs::read_to_string(model_dir.join("config.json")).unwrap()).unwrap_or_else(|e| panic!("{repo}: {e}"));

        // Rung 1: every tensor maps both ways. `source` refuses a tensor it
        // cannot place and validates every parameter's presence and size.
        qwen3::import::source(&reader, &full).unwrap_or_else(|e| panic!("{repo}: coverage: {e}"));

        let gen: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(golden_dir.join("generate.json")).unwrap()).unwrap();
        let layers = gen["layers"].as_u64().unwrap() as usize;
        let ids: Vec<u32> = gen["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let shard = Shard { start: 0, end: layers, embed: true, head: true, gpu_index: Shard::whole(full.n_layers as usize).gpu_index };
        let src = qwen3::import::shard_source(&reader, &full, &shard).unwrap_or_else(|e| panic!("{repo}: {e}"));
        let cfg = qwen3::QwenConfig { n_layers: layers as u32, block_size: ids.len() as u32, ..full.clone() };
        let model = Qwen::new_shard_dt(cfg.clone(), 1, ids.len() as u32, &src, Shard::whole(layers), qwen3::Dtype::F32);

        let golden = checkpoint::safetensors::read(golden_dir.join("forward.safetensors").to_str().unwrap()).unwrap();
        let tap = |n: &str| golden.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("{repo}: golden has no {n}")).data.clone();
        let d = cfg.d_model as usize;

        // Rung 2: the embedding rows, then layer 0's output.
        let emb = model.encode_hidden(&ids, 0);
        assert_eq!(emb, tap("hidden_00"), "{repo}: embedding rows");
        let l0 = min_row_cos(&model.encode_hidden(&ids, 1), &tap("hidden_01"), d);
        assert!(l0 >= COS_FLOOR, "{repo}: layer 0 output row cosine {l0:.8}");

        // Rung 3: the logits of the two-layer model.
        let logits = min_row_cos(&model.logits_all(&ids), &tap("logits"), cfg.vocab as usize);
        assert!(logits >= COS_FLOOR, "{repo}: logits row cosine {logits:.8}");
        println!("{repo}: layer 0 cos {l0:.8}, logits cos {logits:.8}");
        compared += 1;
    }
    println!("checkpoints compared: {compared}/{}", CHECKPOINTS.len());
}
