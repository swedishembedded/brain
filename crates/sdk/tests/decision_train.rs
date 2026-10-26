// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "decision")]

//! `brain decide train` end to end on a tiny random-weight `decide` model:
//! examples in, a checkpoint with its lineage out.
//!
//! The base is the fixture checkpoint of `decisions::fixtures`, so these tests
//! run everywhere and need no downloaded weights. They pin what a training run
//! promises: the manifest records where the weights came from and never claims
//! validation, the same seed gives the same bytes, and a bad request is refused
//! before anything is written.
//!
//! Swedish Embedded AB implements reproducible training runs whose artifacts
//! name the data, code and base they came from. If your team needs expertise
//! in training provenance, you can procure our services by sending an email to
//! info@swedishembedded.com.

use brain::decision_run::{train, TrainRun};
use decisions::engine::Backend;
use decisions::fixtures::{synthetic_jsonl, write_decide, Scratch, SyntheticTask};
use decisions::manifest::{DecisionManifest, Validation, FILE};

/// A base, a data file and an output location inside one scratch directory.
struct Setup {
    root: Scratch,
    /// The training seed every run of this setup shares.
    seed: u64,
}

impl Setup {
    fn new(tag: &str, task: &SyntheticTask) -> Setup {
        let root = Scratch::new(tag);
        write_decide(&root.join("base"), 11);
        std::fs::write(root.join("data.jsonl"), synthetic_jsonl(task)).unwrap();
        Setup { root, seed: brain::random_seed() }
    }

    fn run(&self, out: &str) -> TrainRun {
        let mut run = TrainRun::new(self.root.join("data.jsonl"), self.root.join("base").to_str().unwrap(), self.root.join(out));
        run.steps = 6;
        run.batch = 4;
        run.seed = self.seed;
        run.brain_commit = "abc1234".into();
        run
    }
}

#[test]
fn a_run_writes_a_checkpoint_whose_manifest_records_its_lineage_and_is_unvalidated() {
    let s = Setup::new("train-lineage", &SyntheticTask::new(12, 4, 4));
    let run = s.run("out");
    let outcome = train(&run, &mut |_, _| {}).unwrap();
    assert!(outcome.final_loss.is_finite());

    let m = DecisionManifest::read(&run.out, Backend::Decide).unwrap();
    assert_eq!(m, outcome.manifest);
    assert_eq!(m.validation, Validation::Unvalidated, "training never claims validation");
    assert!(run.out.join(FILE).exists());
    let l = &m.lineage;
    assert_eq!(l.base.as_deref(), Some("local/base".to_string().as_str()));
    assert_eq!(l.base_sha256, Some(brain_modelstore::fetch::base_digest(&s.root.join("base")).unwrap()));
    assert_eq!(l.data_sha256, Some(brain_modelstore::fetch::file_digest(&s.root.join("data.jsonl")).unwrap()));
    let counts = l.split_counts.as_ref().unwrap();
    assert_eq!((counts["train"], counts["validation"], counts["test"]), (12, 4, 4));
    assert_eq!((l.seed, l.brain_commit.as_deref()), (Some(s.seed), Some("abc1234")));
    let h = l.hyperparameters.as_ref().unwrap();
    assert_eq!((h["steps"].as_u64(), h["batch"].as_u64(), h["freeze_encoder"].as_bool()), (Some(6), Some(4), Some(true)));
    // The declared hashes bind the manifest to the bytes beside it.
    assert_eq!(m.weights_sha256, Some(brain_modelstore::fetch::file_digest(&run.out.join("model.safetensors")).unwrap()));
}

#[test]
fn the_same_seed_gives_the_same_weights_and_another_seed_does_not() {
    let s = Setup::new("train-determinism", &SyntheticTask::new(12, 4, 4));
    let a = train(&s.run("a"), &mut |_, _| {}).unwrap();
    let b = train(&s.run("b"), &mut |_, _| {}).unwrap();
    assert_eq!(a.manifest.weights_sha256, b.manifest.weights_sha256);
    let mut other = s.run("c");
    other.seed = s.seed ^ 1;
    let c = train(&other, &mut |_, _| {}).unwrap();
    assert_ne!(a.manifest.weights_sha256, c.manifest.weights_sha256);
}

#[test]
fn a_bad_request_is_refused_before_anything_is_written() {
    let s = Setup::new("train-refusals", &SyntheticTask::new(12, 4, 4));
    let mut no_steps = s.run("x");
    no_steps.steps = 0;
    assert!(train(&no_steps, &mut |_, _| {}).unwrap_err().to_string().contains("--steps"));

    let mut missing = s.run("x");
    missing.data = s.root.join("absent.jsonl");
    assert!(train(&missing, &mut |_, _| {}).is_err());

    // No train split: nothing to learn from.
    std::fs::write(s.root.join("test-only.jsonl"), synthetic_jsonl(&SyntheticTask::new(0, 0, 4))).unwrap();
    let mut test_only = s.run("x");
    test_only.data = s.root.join("test-only.jsonl");
    assert!(train(&test_only, &mut |_, _| {}).unwrap_err().to_string().contains("train"));

    // A leaking file is refused with the leak report.
    let leaky = synthetic_jsonl(&SyntheticTask::new(2, 0, 0)).replace("\"split\":\"train\"", "\"split\":\"test\"");
    let doubled = format!("{}{}", synthetic_jsonl(&SyntheticTask::new(2, 0, 0)), leaky.replace("\"run\":\"run-1\"", "\"run\":\"other\"").replace("\"run\":\"run-2\"", "\"run\":\"other2\""));
    std::fs::write(s.root.join("leaky.jsonl"), doubled).unwrap();
    let mut leaky_run = s.run("x");
    leaky_run.data = s.root.join("leaky.jsonl");
    assert!(train(&leaky_run, &mut |_, _| {}).unwrap_err().to_string().contains("leak"));

    // An existing output is never overwritten.
    std::fs::create_dir_all(s.root.join("taken")).unwrap();
    assert!(train(&s.run("taken"), &mut |_, _| {}).unwrap_err().to_string().contains("already exists"));

    assert!(!s.root.join("x").exists(), "a refused run leaves no output");
}
