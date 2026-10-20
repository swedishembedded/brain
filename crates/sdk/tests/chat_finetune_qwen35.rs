// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improvement loops that train a large
// hybrid-attention model on its own verified work and serve the result for its
// clients. If your team needs expertise in LoRA fine-tuning and serving of
// Qwen3.8-class models then you can procure our services by sending an email
// to info@swedishembedded.com.

// Fine-tuning is the `study` surface.
#![cfg(feature = "study")]

//! `brain::ChatFineTune` on a `qwen35` base (the 27B's architecture, at toy
//! width, CPU backend): the chat dataset trains a LoRA over every targetable
//! projection, the adapter file carries the card a server binds it by, and a
//! cancelled run resumes to the uninterrupted adapter.

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, FineTuneStatus};

#[path = "chat_fixture/common.rs"]
mod common;
use common::{cpu, write_tokenizer, TempDir};

/// The model directory of a `Qwen35Config::tiny` checkpoint (hybrid
/// Gated-DeltaNet and attention layers, random weights) at the real
/// vocabulary size, in `model.safetensors` form.
fn qwen35_chat_model_dir(tag: &str) -> TempDir {
    let dir = TempDir::new(tag);
    let cfg = qwen35::config::Qwen35Config { vocab: 151_936, block_size: 128, max_position_embeddings: 128, ..qwen35::config::Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, 11); // seed: any weights serve, the test reads the training, not the model
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

// seed: runs that are compared with each other need the same stream.
fn fine_tune(model: &Path, out: &Path) -> ChatFineTune {
    ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("qwen35")
        .dataset(write_dataset(model))
        .out_dir(out)
        .rank(2)
        .steps(6)
        .lr(1e-2)
        .seed(7)
        .device(cpu())
}

#[test]
fn a_qwen35_fine_tune_exports_a_servable_adapter_with_its_digests() {
    let model = qwen35_chat_model_dir("q35-export");
    let outcome = fine_tune(&model, &model.join("run")).adapter_id("local/tiny:lab:facts:v1").run().expect("the fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(outcome.steps_completed, 6);
    let (first, last) = (outcome.initial_loss.unwrap(), outcome.final_loss.unwrap());
    assert!(first.is_finite() && last.is_finite() && last < first, "the loss falls on its own training data: {first} -> {last}");

    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    assert_eq!(card.id, "local/tiny:lab:facts:v1");
    let spec = card.adapter.expect("an adapter card");
    assert_eq!((spec.kind.as_str(), spec.rank), ("lora", Some(2)));
    // All twelve targetable projections of the 27B architecture are adapted.
    assert_eq!(spec.targets.as_ref().map(Vec::len), Some(12), "{:?}", spec.targets);
    let training = card.training.expect("the card records its training run");
    assert_eq!((training.regime.as_str(), training.seed), ("sft_lora", 7));
    let base_digest = brain_modelstore::fetch::file_digest(&model.join("model.safetensors")).unwrap();
    assert_eq!(training.base_digest.as_deref(), Some(base_digest.as_str()), "the base is named by content");
    assert_eq!(outcome.adapter_digest.as_deref(), Some(brain_modelstore::fetch::file_digest(&adapter).unwrap().as_str()));
    assert!(outcome.record.as_ref().is_some_and(|p| p.exists()), "the training record is written");
}

/// A Hugging Face checkpoint of the 27B names its architecture in its
/// config; the caller leaves it out and the qwen35 trainer is the one that runs.
#[test]
fn a_base_whose_config_names_qwen35_trains_without_the_architecture_being_named() {
    let model = qwen35_chat_model_dir("q35-detect");
    std::fs::write(model.join("config.json"), r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#).unwrap();
    let run = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .adapter_id("local/tiny:lab:facts:v1")
        .rank(2)
        .steps(2)
        .lr(1e-2)
        .seed(7)
        .device(cpu());
    let outcome = run.run().expect("the fine-tune runs on the qwen35 trainer");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let card = checkpoint::st::read_card(outcome.adapter.unwrap().to_str().unwrap()).unwrap().unwrap();
    assert_eq!(card.family, "qwen35");
}

#[test]
fn a_cancelled_qwen35_run_resumes_to_the_uninterrupted_adapter() {
    let model = qwen35_chat_model_dir("q35-resume");
    let straight = fine_tune(&model, &model.join("straight")).run().unwrap();

    let out = model.join("interrupted");
    let cancel = brain::CancelToken::armed();
    let stopped = fine_tune(&model, &out)
        .run_with(&cancel, |p| {
            if p.step == 3 {
                cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!((stopped.status, stopped.steps_completed), (FineTuneStatus::Cancelled, 3));
    assert!(stopped.resume_state.as_ref().is_some_and(|p| p.exists()));

    let resumed = fine_tune(&model, &out).run().unwrap();
    assert_eq!((resumed.status, resumed.resumed_at, resumed.steps_completed), (FineTuneStatus::Completed, Some(3), 6));
    assert_eq!(resumed.adapter_digest, straight.adapter_digest, "the same adapter file, byte for byte");
}

/// The monitoring pass, the held-out score and the per-evaluation adapters a 27B self-distillation run
/// selects on: scored on the training process's own model (a second copy of the base does not fit
/// beside it), over the positions training supervises.
#[test]
fn a_qwen35_fine_tune_scores_held_out_records_and_keeps_every_evaluation() {
    let model = qwen35_chat_model_dir("q35-heldout");
    let records = write_dataset(&model);
    let evaluations = model.join("evaluations");
    let outcome = fine_tune(&model, &model.join("run"))
        .held_out(&records)
        .held_out_text(&records)
        .monitor(&records)
        .eval_every(2)
        .keep_evaluations(&evaluations)
        .run()
        .expect("the fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let (base, tuned) = (outcome.base_score.expect("the base is scored on the held-out set"), outcome.tuned_score.expect("the tuned model is too"));
    let (base_loss, tuned_loss) = (base.loss.unwrap(), tuned.loss.unwrap());
    assert!(base_loss.is_finite() && tuned_loss < base_loss, "records it trained on score better after training: {base_loss} -> {tuned_loss}");
    assert_eq!((base.records, tuned.records), (4, 4));
    assert!(base.positions > 0 && base.positions == tuned.positions, "the same supervised positions: {base:?} {tuned:?}");
    assert!(outcome.base_text_score.is_some() && outcome.tuned_text_score.is_some(), "the second held-out set is scored apart");
    // The untouched base on the monitoring set is what the loop measured before step 0.
    assert!((outcome.initial_loss.unwrap() - base_loss).abs() < 1e-3 * base_loss.abs().max(1.0), "{:?} vs {base_loss}", outcome.initial_loss);

    for step in [2, 4, 6] {
        let file = evaluations.join(format!("step-{step}.safetensors"));
        let card = checkpoint::st::read_card(file.to_str().unwrap()).unwrap().unwrap_or_else(|| panic!("{} has no card", file.display()));
        assert_eq!(card.family, "qwen35", "step {step}");
    }
    assert_eq!(outcome.curve.iter().map(|p| p.step).collect::<Vec<_>>(), vec![2, 4, 6]);
}

#[test]
fn what_qwen35_cannot_hold_is_refused_before_anything_is_read() {
    let model = qwen35_chat_model_dir("q35-refuse");
    let err = fine_tune(&model, &model.join("run")).int8_base(true).run().unwrap_err().to_string();
    assert!(err.contains("int8"), "{err}");
}

/// Continuing an adapter trains ITS factors at its own shape and records it
/// as the new adapter's parent, on the outcome and on the card: the chain a
/// multi-generation campaign (adapter on adapter) is audited by.
#[test]
fn continuing_a_qwen35_adapter_records_its_parent() {
    let model = qwen35_chat_model_dir("q35-continue");
    let first = fine_tune(&model, &model.join("first")).run().unwrap();
    let parent = first.adapter.clone().unwrap();

    let second = fine_tune(&model, &model.join("second")).continue_from(&parent).run().unwrap();
    assert_eq!(second.status, FineTuneStatus::Completed);
    assert_eq!((second.rank, second.alpha), (first.rank, first.alpha));
    assert_eq!(second.trained_from, first.adapter_digest);
    assert_ne!(second.adapter_digest, first.adapter_digest, "it trained on top of the parent");
    let card = checkpoint::st::read_card(second.adapter.as_ref().unwrap().to_str().unwrap()).unwrap().unwrap();
    assert_eq!(card.training.unwrap().trained_from, first.adapter_digest);
}
