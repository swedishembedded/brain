// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "study")]

//! `brain::DistillFineTune` end to end on the toy chat checkpoint (`brain::toy`), on the CPU backend: a fresh adapter
//! distilled from the base with a trained adapter applied closes the divergence it starts at; a continued adapter
//! distilled from the base alone moves back toward it; a teacher that needs an adapter it was not given is refused.

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, DistillFineTune, DistillTeacher, FineTuneStatus};

use crate::chat_fixture::common::{cpu, TempDir};

/// The toy checkpoint in a fresh directory.
fn chat_model_dir(name: &str) -> TempDir {
    let dir = TempDir::new(name);
    // seed: a fixture checkpoint.
    brain::toy::write_toy_chat_checkpoint(&dir, 5).unwrap();
    dir
}

fn write(dir: &Path, name: &str, records: &[(&str, &str)]) -> PathBuf {
    let path = dir.join(name);
    let lines: Vec<String> = records.iter().map(|(u, a)| serde_json::json!({"messages": [{"role": "user", "content": u, "train": false}, {"role": "assistant", "content": a, "train": true}], "tools": []}).to_string()).collect();
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

const RECORDS: &[(&str, &str)] = &[("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")];

/// An adapter trained on the records: the specialist a distillation copies.
fn specialist(model: &Path) -> PathBuf {
    let train = write(model, "train.jsonl", RECORDS);
    // seed: the specialist is a fixture.
    let outcome = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(train).out_dir(model.join("specialist")).rank(8).steps(40).lr(2e-2).seed(3).device(cpu()).run().expect("the specialist");
    outcome.adapter.expect("an adapter")
}

#[test]
fn a_fresh_adapter_distilled_from_a_trained_one_closes_the_divergence_it_starts_at() {
    let model = chat_model_dir("distill-adapter");
    let teacher = specialist(&model);
    let base = model.join("model.safetensors");
    let dataset = write(&model, "prompts.jsonl", RECORDS);
    // seed: the assertions are about this run.
    let outcome = DistillFineTune::from_pretrained(base.to_str().unwrap()).dataset(&dataset).teacher(DistillTeacher::Adapter(teacher.clone())).top_k(4).out_dir(model.join("distill")).rank(8).steps(40).lr(2e-2).seed(7).device(cpu()).run().expect("the distillation");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!((outcome.teacher, outcome.examples, outcome.top_k), ("adapter", 4, 4));
    assert!(outcome.teacher_adapter.as_deref().is_some_and(|d| d.starts_with("sha256:")), "{outcome:?}");
    let (first, last) = (outcome.initial_loss.unwrap(), outcome.final_loss.unwrap());
    assert!(first > 1e-3, "a fresh adapter is the base, which the trained teacher differs from: {first}");
    assert!(last < first - 0.5, "the divergence closes: {first} -> {last}");
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.unwrap()).unwrap()).unwrap();
    assert_eq!((record["regime"].as_str(), record["teacher"].as_str()), (Some("distill"), Some("adapter")));

    // The same adapter continued and taught by the base alone moves back toward the base.
    let back = DistillFineTune::from_pretrained(base.to_str().unwrap()).dataset(&dataset).teacher(DistillTeacher::Base).continue_from(&teacher).top_k(4).out_dir(model.join("back")).steps(40).lr(2e-2).seed(7).device(cpu()).run().expect("the distillation");
    assert!(back.initial_loss.unwrap() > 1e-3 && back.final_loss.unwrap() < back.initial_loss.unwrap(), "{back:?}");
    assert_eq!(back.trained_from, outcome.teacher_adapter, "the continued adapter is the student's parent");
}

#[test]
fn a_teacher_that_needs_an_adapter_it_was_not_given_is_refused() {
    let model = chat_model_dir("distill-refused");
    let dataset = write(&model, "prompts.jsonl", RECORDS);
    let e = DistillFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&dataset).teacher(DistillTeacher::Start).out_dir(model.join("run")).device(cpu()).run().unwrap_err().to_string();
    assert!(e.contains("continue_from"), "{e}");
    let e = DistillFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&dataset).top_k(0).out_dir(model.join("run")).device(cpu()).run().unwrap_err().to_string();
    assert!(e.contains("top_k"), "{e}");
}
