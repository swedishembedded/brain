// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained models to
// the compute a deployment has, by removing what they need least, for its
// clients. If your team needs expertise in structured pruning and model
// compression then you can procure our services by sending an email to
// info@swedishembedded.com.

#![cfg(all(feature = "study", feature = "text"))]

//! `brain::ModelPruner` end to end on a synthetic chat model: the pruned
//! checkpoint is a directory any brain command opens, narrower where the
//! plan said, carrying the base's tokenizer; the serving engine measures it
//! as the trainable build's zeroed plan predicted; and a recovery adapter
//! trains over it. A run missing its dataset, or keeping a share of channels
//! outside (0, 1], is refused by name.

use std::path::Path;

use brain::ModelPruner;

mod chat_fixture;
use chat_fixture::{chat_model_dir, chat_model_dir_of, cpu};

fn write_dataset(dir: &Path, records: usize) -> std::path::PathBuf {
    let words = ["cab", "bad", "fed", "deaf", "ace", "face", "bag", "gab", "dab", "bead", "fade", "cafe"];
    let lines: Vec<String> = (0..records)
        .map(|i| {
            serde_json::json!({
                "messages": [
                    {"role": "user", "content": words[i % words.len()], "train": false},
                    {"role": "assistant", "content": words[(i * 5 + 3) % words.len()], "train": true}
                ],
                "tools": []
            })
            .to_string()
        })
        .collect();
    let path = dir.join("domain.jsonl");
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

#[test]
fn a_pruned_checkpoint_is_narrower_servable_and_recoverable() {
    let model = chat_model_dir_of("prune", qwen3::QwenConfig::tiny_i8());
    let dataset = write_dataset(&model, 12);
    let outcome = ModelPruner::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(&dataset)
        .out_dir(model.join("run"))
        .keep(0.5)
        .recover(2)
        .recover_lr(1e-3)
        .recover_rank(4)
        .run()
        .expect("the pruning runs");
    let r = &outcome.report;
    assert_eq!(r.widths, [("d_ff", 96, 48)]);
    assert!(r.params_after < r.params_before, "{r:?}");
    for file in ["model.safetensors", "config.json", "tokenizer.json"] {
        assert!(outcome.checkpoint.join(file).is_file(), "{file} is part of the pruned checkpoint");
    }
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(outcome.checkpoint.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["intermediate_size"], 48, "{config}");

    assert_eq!(outcome.measured_on, "serving engine");
    let pruned = outcome.deployed.iter().find(|d| d.label == "pruned").expect("the pruned checkpoint is measured").loss;
    assert!((pruned - r.pruned_loss).abs() <= 1e-3 * r.pruned_loss.abs(), "the engine on the narrower checkpoint computes what the zeroed plan did: {pruned} vs {}", r.pruned_loss);
    let rec = outcome.recovery.as_ref().expect("a recovery was asked for");
    assert!(rec.adapter.is_file() && rec.recovered_loss.is_finite(), "{rec:?}");
    assert!(outcome.report_path.is_file());
}

#[test]
fn a_pruning_without_a_dataset_or_with_an_impossible_share_is_refused_by_name() {
    let model = chat_model_dir("prune-refuse");
    let base = model.join("model.safetensors");
    let err = ModelPruner::from_pretrained(base.to_str().unwrap()).out_dir(model.join("run")).device(cpu()).run().expect_err("no dataset");
    assert!(err.to_string().contains("dataset"), "{err}");
    let dataset = write_dataset(&model, 4);
    let err = ModelPruner::from_pretrained(base.to_str().unwrap()).dataset(&dataset).out_dir(model.join("run")).keep(1.5).device(cpu()).run().expect_err("a share above one");
    assert!(err.to_string().contains("(0, 1]"), "{err}");
}
