// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning sparse
// mixture-of-experts language models for its clients. If your team needs
// expertise in training routed-expert models then you can procure our services
// by sending an email to info@swedishembedded.com.

// Fine-tuning is the `study` surface.
#![cfg(feature = "study")]

//! The generic full-parameter and preference fine-tuners on a `qwen35moe`
//! base (toy width, CPU backend): every weight trains in a full run, and a
//! preference run trains mixer adapters over frozen experts.
//!
//! The two training tests run on a card (`--ignored`, e.g. through the
//! reservation helper with a small reservation): this architecture's head
//! backward uses the naive weight-gradient kernel over the full logits, which
//! on the CPU backend grows steeply with the vocabulary (the tokenizer fixture
//! needs the real 151936), so a CPU run takes minutes per step; see the
//! knowledge entry on it. On a card both finish in seconds.

use std::path::{Path, PathBuf};

use brain::{ChatFullFineTune, FineTuneStatus, PreferenceFineTune};

#[path = "chat_fixture/common.rs"]
mod common;
use common::{cpu, write_tokenizer, TempDir};

/// A `Qwen35Config::tiny` mixture-of-experts checkpoint at the real vocabulary size.
fn qwen35moe_model_dir(tag: &str) -> TempDir {
    let dir = TempDir::new(tag);
    let cfg = qwen35moe::config::Qwen35Config { vocab: 151_936, block_size: 128, max_position_embeddings: 128, ..qwen35moe::config::Qwen35Config::tiny() };
    let init = qwen35moe::init::init_weights(&cfg, 11); // seed: any weights serve, the test reads the training, not the model
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, numel)| (name.clone(), vec![numel as u64], init[&name].clone())).collect();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
    write_tokenizer(&dir);
    dir
}

fn write_dataset(dir: &Path) -> PathBuf {
    let path = dir.join("train.jsonl");
    let rows: Vec<String> = [("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")]
        .iter()
        .map(|(u, a)| {
            serde_json::json!({
                "messages": [{"role": "user", "content": u, "train": false}, {"role": "assistant", "content": a, "train": true}],
                "tools": []
            })
            .to_string()
        })
        .collect();
    std::fs::write(&path, rows.join("\n")).unwrap();
    path
}

#[test]
#[ignore = "needs a card: the CPU backend takes minutes per step at this vocabulary"]
fn a_qwen35moe_full_fine_tune_trains_every_weight_and_writes_a_base() {
    let model = qwen35moe_model_dir("moe-full");
    let outcome = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("qwen35moe")
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .steps(2)
        .lr(1e-2)
        .seed(7) // seed: runs compared with each other need the same stream
        .run()
        .expect("the full fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let checkpoint = outcome.checkpoint.expect("a completed run writes its checkpoint directory");
    assert!(checkpoint.join("tokenizer.json").is_file());
    let weights = checkpoint.join("model.safetensors");
    let trained = checkpoint::weightio::WeightReader::open(weights.to_str().unwrap()).expect("the checkpoint opens");
    let cfg = qwen35moe::config::Qwen35Config::from_json(&trained.config());
    assert_eq!(trained.names().count(), cfg.param_list().len(), "every tensor of the architecture is written");
}

#[test]
#[ignore = "needs a card: the CPU backend takes minutes per step at this vocabulary"]
fn a_qwen35moe_preference_fine_tune_trains_mixer_adapters_over_frozen_experts() {
    let model = qwen35moe_model_dir("moe-dpo");
    let pairs = model.join("pairs.jsonl");
    let lines: Vec<String> = [("cab", "bad", "wig"), ("fed", "deaf", "vow"), ("ace", "face", "jug"), ("bag", "gab", "mop")]
        .iter()
        .map(|(u, c, r)| {
            serde_json::json!({
                "prompt": [{"role": "user", "content": u}],
                "chosen": {"role": "assistant", "content": c},
                "rejected": {"role": "assistant", "content": r},
                "metadata": {"source": "test"}
            })
            .to_string()
        })
        .collect();
    std::fs::write(&pairs, lines.join("\n")).unwrap();
    let outcome = PreferenceFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("qwen35moe")
        .dataset(&pairs)
        .out_dir(model.join("run"))
        .rank(2)
        .steps(2)
        .lr(1e-1)
        .seed(7) // seed: runs compared with each other need the same stream
        .run()
        .expect("the preference fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let initial = outcome.initial_loss.expect("measured");
    assert!((initial - std::f32::consts::LN_2).abs() < 1e-3, "the margin starts at zero against the starting model: {initial}");
    assert!(outcome.final_loss.is_some_and(|last| last < initial), "{outcome:?}");
    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let st = checkpoint::st::load_safetensors(adapter.to_str().unwrap()).expect("the adapter opens");
    assert!(st.tensors.keys().all(|n| model::adapter::device::is_adapter_param(n)), "an adapter file holds adapter tensors only, never experts");
    assert!(!st.tensors.is_empty());
    let score = outcome.train_score.expect("the resident model scores the training pairs");
    assert_eq!((score.pairs, score.skipped), (4, 0));
}

/// The generic fine-tuners refuse a run with no dataset by name before any
/// device is claimed, for this architecture as for the others.
#[test]
fn a_qwen35moe_full_fine_tune_without_a_dataset_is_refused_by_name() {
    let model = qwen35moe_model_dir("moe-nodata");
    let err = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).arch("qwen35moe").out_dir(model.join("run")).device(cpu()).run().expect_err("a run needs a dataset");
    assert!(err.to_string().contains("dataset"), "{err}");
}
