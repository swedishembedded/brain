// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Fine-tuning is the `study` surface; serving the result is `text`.
#![cfg(all(feature = "study", feature = "text"))]

//! `brain::ChatFineTune` end to end on a synthetic chat model, on the CPU
//! backend: train, export, score, serve the adapter; continue an adapter;
//! cancel and resume exactly.
//!
//! The fixture model is `chat_fixture`'s synthetic chat checkpoint.

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, FineTuneStatus};

mod chat_fixture;
use chat_fixture::{chat_model_dir, cpu};

/// One JSONL chat record: a user turn and a supervised assistant answer.
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

/// A fine-tune of the fixture model: its weights, a small dataset and a
/// held-out set, on the CPU.
fn fine_tune(model: &Path, out: &Path) -> ChatFineTune {
    let train = write_dataset(model, "train.jsonl", &[("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")]);
    let held_out = write_dataset(model, "held_out.jsonl", &[("cad", "bead")]);
    ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(train).held_out(held_out).out_dir(out).rank(2).steps(6).lr(1e-2).seed(7).device(cpu())
}

/// Train, export the adapter and its record, score base and tuned on the
/// held-out set, and serve the adapter: the served identity names the
/// exported adapter by the digest the outcome reports.
#[test]
fn a_fine_tune_exports_an_adapter_that_scores_and_serves() {
    let model = chat_model_dir("export");
    let out = model.join("run");
    let mut steps_seen = Vec::new();
    let outcome = fine_tune(&model, &out).run_with(&brain::CancelToken::armed(), |p| steps_seen.push(p.step)).expect("the fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(steps_seen, (1..=6).collect::<Vec<_>>(), "one progress report per step");
    assert_eq!(outcome.steps_completed, 6);
    assert!(outcome.initial_loss.is_some_and(f32::is_finite) && outcome.final_loss.is_some_and(f32::is_finite), "{outcome:?}");
    assert_eq!(outcome.resumed_at, None);
    assert_eq!(outcome.trained_from, None, "a fresh adapter has no parent");
    assert!(outcome.resume_state.is_none());

    let adapter = outcome.adapter.clone().expect("a completed run exports its adapter");
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    let training = card.training.expect("the card records its training run");
    assert_eq!(training.regime, "sft_lora");
    assert_eq!(training.seed, 7);
    assert_eq!(training.trained_from, None);
    assert_eq!(training.environment, "cpu", "these tests run on the CPU backend");
    // The base is named by content, so a server can refuse to fold this
    // adapter into any other base.
    assert_eq!(training.base_digest.as_deref(), Some(brain_modelstore::fetch::file_digest(&model.join("model.safetensors")).unwrap().as_str()));
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.as_ref().expect("a training record")).unwrap()).unwrap();
    assert_eq!(record["adapter_digest"].as_str(), outcome.adapter_digest.as_deref());

    let (base, tuned) = (outcome.base_score.expect("scored before"), outcome.tuned_score.expect("scored after"));
    assert!(base.loss.is_some_and(f32::is_finite) && tuned.loss.is_some_and(f32::is_finite), "{base:?} {tuned:?}");
    assert!(base.positions > 0 && base.positions == tuned.positions, "{base:?} {tuned:?}");
    let scored = brain::score_chat(model.join("model.safetensors").to_str().unwrap(), Some(&adapter), &model.join("held_out.jsonl")).unwrap();
    assert_eq!(scored.loss, tuned.loss, "the public scorer is the one the run used");

    let chat = brain::ChatPipeline::from(
        brain::TextGenerationPipeline::builder(model.join("model.safetensors").to_str().unwrap())
            .tokenizer(model.join("tokenizer.json").to_str().unwrap())
            .adapter(adapter.to_str().unwrap())
            .capacity(256)
            .device(cpu())
            .load()
            .expect("the exported adapter serves"),
    );
    assert_eq!(chat.identity().adapter.as_ref().map(|a| a.digest.as_str()), outcome.adapter_digest.as_deref());
}

/// Continuing an adapter trains ITS factors, at its own rank, and records
/// it as the new adapter's parent - on the outcome and on the card.
#[test]
fn continuing_an_adapter_records_its_parent() {
    let model = chat_model_dir("continue");
    let first = fine_tune(&model, &model.join("first")).run().unwrap();
    let parent = first.adapter.clone().unwrap();

    let err = fine_tune(&model, &model.join("second")).rank(5).continue_from(&parent).run().unwrap_err();
    assert!(err.to_string().contains("rank"), "an adapter is continued at its own rank: {err}");

    let second = fine_tune(&model, &model.join("second")).continue_from(&parent).run().unwrap();
    assert_eq!(second.status, FineTuneStatus::Completed);
    assert_eq!((second.rank, second.alpha), (first.rank, first.alpha), "the continued adapter's shape");
    assert_eq!(second.trained_from, first.adapter_digest);
    let card = checkpoint::st::read_card(second.adapter.as_ref().unwrap().to_str().unwrap()).unwrap().unwrap();
    assert_eq!(card.training.unwrap().trained_from, first.adapter_digest);
    assert_ne!(second.adapter_digest, first.adapter_digest);
}

/// A cancelled run stops at the next step boundary, exports nothing and
/// leaves its state; a state from different options is refused; the same
/// fine-tune run again resumes it and ends at the bit-identical adapter an
/// uninterrupted run produces.
#[test]
fn a_cancelled_run_resumes_to_the_uninterrupted_result() {
    let model = chat_model_dir("resume");
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
    assert_eq!(stopped.status, FineTuneStatus::Cancelled);
    assert_eq!(stopped.steps_completed, 3);
    assert!(stopped.adapter.is_none() && stopped.tuned_score.is_none(), "{stopped:?}");
    assert!(stopped.resume_state.as_ref().is_some_and(|p| p.exists()));

    let err = fine_tune(&model, &out).lr(5e-3).run().unwrap_err();
    assert!(err.to_string().contains("different training options"), "{err}");

    let resumed = fine_tune(&model, &out).run().unwrap();
    assert_eq!(resumed.status, FineTuneStatus::Completed);
    assert_eq!(resumed.resumed_at, Some(3));
    assert_eq!(resumed.steps_completed, 6);
    assert!(!stopped.resume_state.unwrap().exists(), "an exported run leaves no state behind");

    let tensors = |p: &Option<PathBuf>| checkpoint::st::load_safetensors(p.as_ref().unwrap().to_str().unwrap()).unwrap().tensors;
    let (a, b) = (tensors(&straight.adapter), tensors(&resumed.adapter));
    assert_eq!(a.len(), b.len());
    for (name, values) in &a {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(values), bits(&b[name]), "{name} differs between the uninterrupted and the resumed run");
    }
    assert_eq!(resumed.adapter_digest, straight.adapter_digest, "the same adapter file, byte for byte");
}
