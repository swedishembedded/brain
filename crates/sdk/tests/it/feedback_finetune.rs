// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "study")]

//! `brain::FeedbackFineTune` end to end on `chat_fixture`'s synthetic chat model, on the CPU backend: KTO on desirable
//! and undesirable completions raises the desirable ones against the starting policy and lowers the undesirable ones,
//! lowers its own loss, exports an adapter with its record, and refuses a dataset it cannot train by record.

use std::path::{Path, PathBuf};

use brain::{FeedbackFineTune, FineTuneStatus};

use crate::chat_fixture::{chat_model_dir, cpu};

fn example(user: &str, answer: &str, desirable: bool) -> String {
    serde_json::json!({"prompt": [{"role": "user", "content": user}], "completion": {"role": "assistant", "content": answer}, "desirable": desirable}).to_string()
}

fn write(dir: &Path, name: &str, lines: &[String]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

const TRAIN: &[(&str, &str, bool)] = &[("cab", "bad", true), ("cab", "wig", false), ("fed", "deaf", true), ("fed", "vow", false), ("ace", "face", true), ("ace", "jug", false)];

/// The judgement, not the text, decides the direction: against the starting policy the desirable completions end
/// above the undesirable ones (KTO pushes a desirable log-ratio above the reference point and an undesirable one under
/// it, and the point itself rises as the policy moves), and swapping every judgement swaps the order.
#[test]
fn kto_raises_the_desirable_completions_over_the_undesirable_ones_and_exports_an_adapter() {
    let model = chat_model_dir("feedback-train");
    let base = model.join("model.safetensors");
    let run = |flip: bool| {
        let lines: Vec<String> = TRAIN.iter().map(|(u, a, d)| example(u, a, *d != flip)).collect();
        let train = write(&model, if flip { "flipped.jsonl" } else { "train.jsonl" }, &lines);
        // seed: the two runs are compared with each other.
        FeedbackFineTune::from_pretrained(base.to_str().unwrap()).dataset(&train).out_dir(model.join(if flip { "flipped" } else { "run" })).rank(2).steps(40).grad_accum(3).lr(5e-2).seed(7).device(cpu()).run().expect("the run")
    };
    let outcome = run(false);
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!((outcome.desirable_examples, outcome.undesirable_examples), (3, 3));
    let (up, down) = (outcome.desirable_logratio.expect("measured"), outcome.undesirable_logratio.expect("measured"));
    assert!(up > down, "desirable {up}, undesirable {down}");
    let (first, last) = (outcome.initial_loss.unwrap(), outcome.final_loss.unwrap());
    assert!((first - 0.5).abs() < 1e-3, "at the reference every term is lambda * (1 - sigmoid(0)): {first}");
    assert!(last < first, "the loss falls: {first} -> {last}");
    assert!(outcome.adapter.as_ref().is_some_and(|p| p.exists()));
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.as_ref().unwrap()).unwrap()).unwrap();
    assert_eq!((record["regime"].as_str(), record["desirable_examples"].as_u64()), (Some("feedback"), Some(3)));

    let flipped = run(true);
    let (up, down) = (flipped.desirable_logratio.unwrap(), flipped.undesirable_logratio.unwrap());
    assert!(up > down, "with every judgement swapped, the newly desirable completions end on top: desirable {up}, undesirable {down}");
}

#[test]
fn a_feedback_dataset_it_cannot_train_is_refused_by_record() {
    let model = chat_model_dir("feedback-refused");
    let base = model.join("model.safetensors");
    let run = |lines: &[String]| FeedbackFineTune::from_pretrained(base.to_str().unwrap()).dataset(write(&model, "data.jsonl", lines)).out_dir(model.join("run")).steps(1).seed(7).device(cpu()).run().unwrap_err().to_string();
    let e = run(&[example("cab", "bad", true)]);
    assert!(e.contains("at least two"), "{e}");
    let e = run(&[example("cab", "bad", true), "{\"prompt\": [], \"completion\": {\"role\": \"assistant\", \"content\": \"x\"}, \"desirable\": true}".to_string()]);
    assert!(e.contains(":2:") && e.contains("empty"), "{e}");
    let e = FeedbackFineTune::from_pretrained(base.to_str().unwrap()).dataset(write(&model, "ok.jsonl", &TRAIN.iter().map(|(u, a, d)| example(u, a, *d)).collect::<Vec<_>>())).out_dir(model.join("run")).beta(0.0).device(cpu()).run().unwrap_err().to_string();
    assert!(e.contains("beta"), "{e}");
}

/// A base whose weights are a repo directory is named by the directory's content digest, not read as one file ("Is a
/// directory": the served feedback jobs failed on it before training a step).
#[test]
fn a_base_whose_weights_are_a_directory_is_named_by_digest_not_read_as_a_file() {
    let model = chat_model_dir("kto-base-dir");
    let train = write(&model, "train.jsonl", &TRAIN.iter().map(|(u, a, d)| example(u, a, *d)).collect::<Vec<_>>());
    let (store, base) = crate::chat_fixture::chat_model_as_repo_dir(&model);
    let run = FeedbackFineTune::from_pretrained(base).models_dir(store.to_str().unwrap()).dataset(train).out_dir(model.join("by-dir")).rank(2).steps(2).lr(5e-2).seed(7).device(cpu()).run();
    // The fixture is not a loadable `transformers` checkout, so the run ends where the trainer opens the weights; what
    // is gated here is the step before it, which named the base by reading the directory as a file.
    if let Err(e) = run {
        assert!(!e.to_string().contains("Is a directory"), "the base directory was read as a file: {e}");
    }
}
