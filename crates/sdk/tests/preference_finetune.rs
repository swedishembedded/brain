// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "study")]

//! `brain::PreferenceFineTune` end to end on `chat_fixture`'s synthetic chat
//! model, on the CPU backend: validate a preference dataset, train a LoRA
//! adapter by DPO, export it with its record, score it; continue an adapter;
//! cancel and resume exactly.

use std::path::{Path, PathBuf};

use brain::{FineTuneStatus, PreferenceFineTune};

mod chat_fixture;
use chat_fixture::{chat_model_dir, cpu};

/// One `generic-preference-v1` line: a user turn, the preferred answer and
/// the answer to prefer it over.
fn pair(user: &str, chosen: &str, rejected: &str) -> String {
    serde_json::json!({
        "prompt": [{"role": "user", "content": user}],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
        "metadata": {"source": "test"}
    })
    .to_string()
}

fn write_pairs(dir: &Path, name: &str, pairs: &[(&str, &str, &str)]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, pairs.iter().map(|(u, c, r)| pair(u, c, r)).collect::<Vec<_>>().join("\n")).unwrap();
    path
}

const TRAIN: &[(&str, &str, &str)] = &[("cab", "bad", "wig"), ("fed", "deaf", "vow"), ("ace", "face", "jug"), ("bag", "gab", "mop")];

/// A preference fine-tune of the fixture model on a small pair set and a
/// held-out set, on the CPU.
fn fine_tune(model: &Path, out: &Path) -> PreferenceFineTune {
    let train = write_pairs(model, "train.jsonl", TRAIN);
    let held_out = write_pairs(model, "held_out.jsonl", &[("cad", "bead", "kiwi")]);
    PreferenceFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(train)
        .held_out(held_out)
        .out_dir(out)
        .rank(2)
        .steps(8)
        .lr(1e-1)
        .seed(7)
        .device(cpu())
}

/// The dataset is checked against the base's own tokenizer and template,
/// and a pair that cannot fit the row is refused by record before any
/// device is claimed.
#[test]
fn a_preference_dataset_is_validated_against_the_base() {
    let model = chat_model_dir("pref-validate");
    let path = write_pairs(&model, "pairs.jsonl", TRAIN);
    let summary = brain::validate_preference_dataset_for(&path, &*model, None).expect("the pairs encode");
    assert_eq!(summary.pairs, 4);
    assert_eq!(summary.prompt_messages, 4);
    assert!(summary.longest_tokens.is_some_and(|n| n > 0), "{summary:?}");
    assert_eq!(brain::validate_preference_dataset(&path).expect("parses").longest_tokens, None, "not encoded, not measured");

    let err = brain::validate_preference_dataset_for(&path, &*model, Some(4)).unwrap_err();
    assert!(err.contains("record 1") && err.contains("max_block"), "{err}");

    let err = fine_tune(&model, &model.join("run")).max_block(4).run().unwrap_err();
    assert!(err.to_string().contains("max_block"), "{err}");
}

/// DPO from a fresh adapter: the reference is the starting model, so the
/// first loss is exactly `-log sigma(0) = ln 2`; after training, the policy
/// prefers the chosen answers of the TRAINING pairs over the rejected ones
/// by a positive margin relative to that reference. The adapter's card and
/// record carry the regime, beta and the reference.
#[test]
fn dpo_raises_the_training_preference_margin_and_exports_its_record() {
    let model = chat_model_dir("pref-export");
    let base = model.join("model.safetensors");
    let mut steps_seen = Vec::new();
    let outcome = fine_tune(&model, &model.join("run")).run_with(&brain::CancelToken::armed(), |p| steps_seen.push(p.step)).expect("the fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(steps_seen, (1..=8).collect::<Vec<_>>(), "one progress report per step");
    assert_eq!(outcome.beta, 0.1, "the documented default");
    let initial = outcome.initial_loss.expect("measured");
    assert!((initial - std::f32::consts::LN_2).abs() < 1e-4, "the margin starts at zero against the starting model: {initial}");

    let train = outcome.train_score.expect("a completed run scores its training pairs");
    let (margin, accuracy) = (train.mean_margin.expect("measured"), train.accuracy.expect("measured"));
    let last = outcome.final_loss.expect("measured");
    println!("training pairs after {} steps: DPO loss {initial:.4} -> {last:.4}, mean margin {margin:.4} nats, accuracy {accuracy:.2} ({} pairs)", outcome.steps_completed, train.pairs);
    assert!(last < initial, "the last step's DPO loss must be below the starting ln 2: {last}");
    assert_eq!((train.pairs, train.skipped), (4, 0));
    assert!(margin > 0.0, "the policy must prefer chosen over rejected more than the reference does: {train:?}");
    assert!(accuracy > 0.5, "{train:?}");
    let held_out = outcome.held_out_score.expect("a held-out set was given");
    assert_eq!(held_out.pairs, 1);
    assert!(held_out.mean_margin.is_some_and(f32::is_finite), "{held_out:?}");
    println!("held-out pair: {held_out:?}");

    let adapter = outcome.adapter.clone().expect("a completed run exports its adapter");
    let scored = brain::score_preference(base.to_str().unwrap(), &adapter, &model.join("train.jsonl")).unwrap();
    assert_eq!(scored, train, "the public scorer is the one the run used, and a fresh run's reference is the base");

    let base_digest = brain_modelstore::fetch::file_digest(&base).unwrap();
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    let training = card.training.expect("the card records its training run");
    assert_eq!(training.regime, "dpo");
    assert_eq!(training.base_digest.as_deref(), Some(base_digest.as_str()));
    assert_eq!(outcome.base_digest, training.base_digest, "the outcome reports the base digest the card records");
    assert_eq!(training.hyperparams["beta"].as_f64(), Some(0.1f32 as f64));
    assert_eq!(training.hyperparams["reference"]["base_digest"].as_str(), Some(base_digest.as_str()));
    assert!(training.hyperparams["reference"]["adapter_digest"].is_null(), "a fresh run's reference is the base alone");

    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.as_ref().expect("a training record")).unwrap()).unwrap();
    assert_eq!(record["regime"], "dpo");
    assert_eq!(record["adapter_digest"].as_str(), outcome.adapter_digest.as_deref());
    assert_eq!(record["train_score"]["mean_margin"].as_f64().map(|v| v as f32), Some(margin));
}

/// Continuing an adapter trains it further against itself: its digest is the
/// new adapter's parent and the reference's adapter.
#[test]
fn continuing_an_adapter_makes_it_the_reference() {
    let model = chat_model_dir("pref-continue");
    let first = fine_tune(&model, &model.join("first")).run().unwrap();
    let parent = first.adapter.clone().unwrap();

    let second = fine_tune(&model, &model.join("second")).continue_from(&parent).run().unwrap();
    assert_eq!(second.status, FineTuneStatus::Completed);
    assert_eq!(second.trained_from, first.adapter_digest);
    let initial = second.initial_loss.unwrap();
    assert!((initial - std::f32::consts::LN_2).abs() < 1e-4, "the continued adapter is the reference, so the margin starts at zero: {initial}");
    let card = checkpoint::st::read_card(second.adapter.as_ref().unwrap().to_str().unwrap()).unwrap().unwrap();
    let training = card.training.unwrap();
    assert_eq!(training.trained_from, first.adapter_digest);
    assert_eq!(training.hyperparams["reference"]["adapter_digest"].as_str(), first.adapter_digest.as_deref());
}

/// A cancelled run stops at the step boundary, exports nothing and leaves its
/// state; a state from a different beta is refused; running the same
/// fine-tune again resumes it and ends at the byte-identical adapter an
/// uninterrupted run produces.
#[test]
fn a_cancelled_preference_run_resumes_to_the_uninterrupted_result() {
    let model = chat_model_dir("pref-resume");
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
    assert!(stopped.adapter.is_none() && stopped.train_score.is_none() && stopped.held_out_score.is_none(), "{stopped:?}");
    assert!(stopped.resume_state.as_ref().is_some_and(|p| p.exists()));

    let err = fine_tune(&model, &out).beta(0.2).run().unwrap_err();
    assert!(err.to_string().contains("different data or a different starting point"), "{err}");

    let resumed = fine_tune(&model, &out).run().unwrap();
    assert_eq!(resumed.status, FineTuneStatus::Completed);
    assert_eq!(resumed.resumed_at, Some(3));
    assert_eq!(resumed.steps_completed, 8);
    assert!(!stopped.resume_state.unwrap().exists(), "an exported run leaves no state behind");
    assert_eq!(resumed.adapter_digest, straight.adapter_digest, "the same adapter file, byte for byte");
}
