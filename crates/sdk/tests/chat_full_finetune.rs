// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full-parameter fine-tuning and adapter
// merging of open-weight chat models for its clients. If your team needs
// expertise in consolidating what a model learned into its own weights, you
// can procure our services by sending an email to info@swedishembedded.com.

#![cfg(all(feature = "study", feature = "text"))]

//! `brain::ChatFullFineTune` and `brain::merge_adapter` end to end on a
//! synthetic chat model, on the CPU backend.
//!
//! * A full fine-tune trains every weight and writes a checkpoint directory
//!   that is a base like any other: it scores, fine-tunes and serves.
//! * Merging folds an adapter into the weights it targets, so the merged
//!   checkpoint alone computes what the base with the adapter computed.
//! * A cancelled full fine-tune writes no checkpoint.

use std::path::{Path, PathBuf};

use brain::{merge_adapter, score_chat, ChatFineTune, ChatFullFineTune, FineTuneStatus};

mod chat_fixture;
use chat_fixture::{chat_model_dir, cpu};

fn record(user: &str, assistant: &str) -> String {
    serde_json::json!({
        "messages": [
            {"role": "user", "content": user, "train": false},
            {"role": "assistant", "content": assistant, "train": true}
        ],
        "tools": []
    })
    .to_string()
}

fn write_dataset(dir: &Path, name: &str, records: &[(&str, &str)]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, records.iter().map(|(u, a)| record(u, a)).collect::<Vec<_>>().join("\n")).unwrap();
    path
}

const TRAIN: [(&str, &str); 4] = [("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")];

fn loss(base: &Path, adapter: Option<&Path>, records: &Path) -> f32 {
    score_chat(base.to_str().unwrap(), adapter, records).expect("scored").loss.expect("a loss over the records")
}

#[test]
fn a_full_fine_tune_writes_a_checkpoint_directory_that_is_a_base_like_any_other() {
    let model = chat_model_dir("full");
    let train = write_dataset(&model, "train.jsonl", &TRAIN);
    let out = model.join("run");
    let outcome = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(&train)
        .out_dir(&out)
        .steps(40)
        .lr(5e-2)
        .seed(data::rng::random_seed())
        .device(cpu())
        .run()
        .expect("the full fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(outcome.steps_completed, 40);
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let checkpoint = outcome.checkpoint.clone().expect("a completed run writes its checkpoint directory");
    for file in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"] {
        assert!(checkpoint.join(file).is_file(), "{file} is part of the checkpoint directory");
    }
    let weights = outcome.weights.clone().expect("the weights file");
    assert_eq!(weights, checkpoint.join("model.safetensors"));
    assert!(outcome.weights_digest.as_deref().is_some_and(|d| d.starts_with("sha256:")));
    assert!(!out.join("tuned.brain.safetensors").exists(), "the training format is not left behind");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["max_position_embeddings"], 128, "the checkpoint states the base's context, not the training row's");

    // Every weight trained: the checkpoint predicts the training answers where
    // the base did not.
    let before = loss(&model.join("model.safetensors"), None, &train);
    let after = loss(&checkpoint, None, &train);
    assert!(after < before * 0.9, "the trained checkpoint fits its records: {before} -> {after}");

    // And it is a base: an adapter can be trained on it.
    let on_it = ChatFineTune::from_pretrained(checkpoint.to_str().unwrap()).dataset(&train).out_dir(model.join("on-it")).rank(2).steps(2).lr(1e-2).seed(data::rng::random_seed()).device(cpu()).run().expect("a checkpoint directory is a base");
    assert_eq!(on_it.status, FineTuneStatus::Completed);
}

/// The optimizer-state precision is one typed value on the builder: a full fine-tune with int8 AdamW state (held in
/// host memory) trains like the fp32 default, and a choice the trainer does not run is refused by parameter name.
#[test]
fn a_full_fine_tune_takes_its_optimizer_state_precision_as_one_typed_value() {
    let model = chat_model_dir("full-precision");
    let train = write_dataset(&model, "train.jsonl", &TRAIN);
    // Both precisions train on one stream so their losses are comparable.
    let seed = data::rng::random_seed();
    let run = |name: &str, spec: &str| {
        ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
            .dataset(&train)
            .out_dir(model.join(name))
            .steps(30)
            .lr(5e-2)
            .seed(seed)
            .train_precision(brain::TrainPrecision::parse(spec).unwrap())
            .device(cpu())
            .run()
    };
    let fp32 = run("fp32", "").expect("fp32 state");
    let int8 = run("int8", "state=int8:64,sr=on").expect("int8 state");
    let (f, q) = (fp32.final_loss.unwrap(), int8.final_loss.unwrap());
    assert!(q < int8.initial_loss.unwrap() && q < 1.5 * f + 0.1, "int8 state trains like fp32: {q} vs {f}");
    let err = run("master", "master=f32").unwrap_err().to_string();
    assert!(err.contains("master"), "{err}");
}

/// A full fine-tune given no seed draws one and records it, on the outcome
/// and in a training record beside the checkpoint; the same seed reproduces
/// the weights.
#[test]
fn a_full_fine_tune_records_the_seed_it_drew() {
    let model = chat_model_dir("full-seed");
    let train = write_dataset(&model, "train.jsonl", &TRAIN);
    let run = |name: &str, seed: Option<u64>| {
        let tune = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&train).out_dir(model.join(name)).steps(3).lr(5e-2).device(cpu());
        match seed {
            Some(seed) => tune.seed(seed),
            None => tune,
        }
        .run()
        .expect("the full fine-tune runs")
    };
    let first = run("first", None);
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(first.record.as_ref().expect("a training record")).unwrap()).unwrap();
    assert_eq!(record["seed"].as_u64(), Some(first.seed));
    assert_eq!(record["weights_digest"].as_str(), first.weights_digest.as_deref());
    assert_eq!(run("again", Some(first.seed)).weights_digest, first.weights_digest, "the recorded seed reproduces the weights");
}

#[test]
fn merging_an_adapter_gives_the_weights_the_adapter_computes() {
    let model = chat_model_dir("merge");
    let train = write_dataset(&model, "train.jsonl", &TRAIN);
    let base = model.join("model.safetensors");
    let tuned = ChatFineTune::from_pretrained(base.to_str().unwrap()).dataset(&train).out_dir(model.join("lora")).rank(2).steps(40).lr(3e-1).seed(data::rng::random_seed()).device(cpu()).run().expect("the adapter trains");
    let adapter = tuned.adapter.expect("an adapter");

    let merged = merge_adapter(base.to_str().unwrap(), &adapter, &model.join("merged")).expect("the adapter merges");
    let merged_dir = merged.dir.clone();
    assert!(merged_dir.join("config.json").is_file() && merged_dir.join("tokenizer.json").is_file(), "{merged:?}");
    assert_eq!(merged.weights, merged_dir.join("model.safetensors"));
    assert!(merged.weights.is_file() && merged.weights_digest.starts_with("sha256:"), "{merged:?}");

    let with_adapter = loss(&base, Some(&adapter), &train);
    let alone = loss(&merged_dir, None, &train);
    let unmerged = loss(&base, None, &train);
    assert!(unmerged - with_adapter > 1e-3, "the adapter moves the base: {unmerged} -> {with_adapter}");
    assert!((with_adapter - alone).abs() < 1e-4 * (unmerged - with_adapter), "the merged weights alone compute the adapter's loss: {with_adapter} vs {alone}");
}

#[test]
fn a_cancelled_full_fine_tune_writes_no_checkpoint() {
    let model = chat_model_dir("cancel");
    let train = write_dataset(&model, "train.jsonl", &TRAIN);
    let out = model.join("run");
    let cancel = brain::CancelToken::armed();
    let outcome = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(&train)
        .out_dir(&out)
        .steps(10)
        .lr(1e-2)
        .device(cpu())
        .run_with(&cancel, |p| {
            if p.step == 3 {
                cancel.cancel();
            }
        })
        .expect("a cancelled run is not an error");
    assert_eq!(outcome.status, FineTuneStatus::Cancelled);
    assert!(outcome.checkpoint.is_none() && outcome.weights.is_none());
    assert!(!out.join("checkpoint").exists());
}

#[test]
fn a_full_fine_tune_without_a_dataset_or_out_directory_is_refused_by_name() {
    let model = chat_model_dir("refuse");
    let base = model.join("model.safetensors");
    let no_dataset = ChatFullFineTune::from_pretrained(base.to_str().unwrap()).out_dir(model.join("a")).device(cpu()).run().unwrap_err().to_string();
    assert!(no_dataset.contains("dataset"), "{no_dataset}");
    let train = write_dataset(&model, "train.jsonl", &TRAIN);
    let no_out = ChatFullFineTune::from_pretrained(base.to_str().unwrap()).dataset(train).device(cpu()).run().unwrap_err().to_string();
    assert!(no_out.contains("out"), "{no_out}");
}
