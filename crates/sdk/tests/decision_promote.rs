// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "decision")]

//! `brain decide promote`: a trained checkpoint becomes validated only when a
//! fresh measurement on the test split meets the policy, and a failed gate
//! changes nothing.
//!
//! The candidates are real trainings of a tiny random-weight `decide` model on
//! a synthetic task whose answer is known: one model trained on the truth, one
//! on shuffled labels (the negative control). Both are measured by the
//! promotion itself; nothing is taken from a manifest.
//!
//! Swedish Embedded AB implements release gates that put a model into
//! production only on evidence measured at the gate. If your team needs
//! expertise in model promotion and rollback, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::PathBuf;

use brain::decision_run::{promote_model, rollback_model, train, PromoteRun, TrainRun};
use decisions::engine::Backend;
use decisions::fixtures::{synthetic_jsonl, write_decide, Scratch, SyntheticTask};
use decisions::manifest::{DecisionManifest, Validation};
use decisions::promotion::{Pointer, EVALUATION_FILE};

struct World {
    root: Scratch,
}

impl World {
    fn dir(&self, name: &str) -> PathBuf {
        self.root.join("models").join(name)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Train a model named `name` on `data`.
    fn train(&self, name: &str, data: &str, seed: u64) {
        let mut run = TrainRun::new(self.file(data), self.file("base").to_str().unwrap(), self.dir(name));
        // A head on a frozen random encoder converges slowly and at a rate that
        // depends on the seed: at 600 steps seed 3 fit the task and seed 4 had
        // not (loss 0.75, lift 0.13), so the second promotion below failed its
        // gate. 1500 steps fit the task from either seed.
        run.steps = 1500;
        run.batch = 8;
        run.seed = seed;
        train(&run, &mut |_, _| {}).unwrap();
    }

    fn promote(&self, name: &str, data: &str, policy: &str) -> brain::decision_run::PromoteOutcome {
        promote_model(&PromoteRun { model: self.dir(name).to_str().unwrap().to_string(), data: self.file(data), policy: self.file(policy) }).unwrap()
    }

    fn manifest(&self, name: &str) -> DecisionManifest {
        DecisionManifest::read(&self.dir(name), Backend::Decide).unwrap()
    }

    fn pointer(&self) -> Option<Pointer> {
        Pointer::read(&self.root.join("models"), "desks").unwrap()
    }
}

fn policy(extra: &str) -> String {
    format!(r#"{{"schema": 1, "profile": "desks", "min_test_examples": 50, "min_accuracy_lift": 0.2, "max_order_instability": 0.1{extra}}}"#)
}

fn world() -> World {
    let root = Scratch::new("promote");
    std::fs::create_dir_all(root.join("models")).unwrap();
    write_decide(&root.join("base"), 11);
    let mut control = SyntheticTask::new(320, 0, 60);
    control.shuffle_train_labels = true;
    std::fs::write(root.join("truth.jsonl"), synthetic_jsonl(&SyntheticTask::new(320, 0, 60))).unwrap();
    std::fs::write(root.join("control.jsonl"), synthetic_jsonl(&control)).unwrap();
    // The same desks test split plus a second family whose desks are rotated by
    // one place: a model that learned the first mapping is wrong on it.
    let mut other = SyntheticTask::new(0, 0, 60);
    other.family = "other";
    other.label_shift = 1;
    other.first_id = 10_000;
    other.seed = brain::random_seed();
    std::fs::write(root.join("mixed.jsonl"), synthetic_jsonl(&SyntheticTask::new(320, 0, 60)) + &synthetic_jsonl(&other)).unwrap();
    std::fs::write(root.join("policy.json"), policy(r#", "required_families": ["desks"]"#)).unwrap();
    std::fs::write(root.join("policy-other.json"), policy(r#", "required_families": ["other"]"#)).unwrap();
    std::fs::write(root.join("policy-both.json"), policy(r#", "required_families": ["desks", "other"]"#)).unwrap();
    World { root }
}

fn assert_untouched(w: &World, name: &str) {
    assert_eq!(w.manifest(name).validation, Validation::Unvalidated);
    assert!(!w.dir(name).join(EVALUATION_FILE).exists(), "a failed promotion writes no evidence");
    assert!(w.pointer().is_none());
}

#[test]
fn only_a_model_that_beats_its_baseline_is_promoted_and_a_failed_gate_changes_nothing() {
    let w = world();
    w.train("good", "truth.jsonl", 3);
    w.train("control", "control.jsonl", 3);

    // The negative control: trained on shuffled labels, it fails the lift gate.
    let rejected = w.promote("control", "control.jsonl", "policy.json");
    assert!(rejected.pointer.is_none() && !rejected.verdict.passed());
    let failed: Vec<&str> = rejected.verdict.failures().map(|g| g.name.as_str()).collect();
    assert!(failed.contains(&"min_accuracy_lift"), "{failed:?}\n{}", rejected.verdict);
    assert_untouched(&w, "control");

    // A policy requiring a family the checkpoint was never trained on, or tested on, fails.
    let absent = w.promote("good", "truth.jsonl", "policy-other.json");
    assert_eq!(absent.verdict.failures().map(|g| g.name.as_str()).collect::<Vec<_>>(), ["required_family:other"]);
    assert_untouched(&w, "good");

    // With the second family in the test split the model is measured on it, and fails there.
    let mixed = w.promote("good", "mixed.jsonl", "policy-both.json");
    let failed: Vec<&str> = mixed.verdict.failures().map(|g| g.name.as_str()).collect();
    assert!(failed.contains(&"required_family:other:min_accuracy_lift"), "{failed:?}");
    assert_untouched(&w, "good");

    // The same checkpoint meets the policy it was trained for.
    let promoted = w.promote("good", "truth.jsonl", "policy.json");
    assert!(promoted.verdict.passed(), "{}", promoted.verdict);
    let weights = brain_modelstore::fetch::file_digest(&w.dir("good").join("model.safetensors")).unwrap();
    assert_eq!(promoted.evaluation.weights_sha256, weights);
    let m = w.manifest("good");
    assert_eq!(m.validation, Validation::Validated { profile: "desks".into(), evidence: Some(EVALUATION_FILE.into()) });
    assert_eq!(m.weights_sha256, Some(weights));
    let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(w.dir("good").join(EVALUATION_FILE)).unwrap()).unwrap();
    assert_eq!(record["data"]["split"], "test");
    assert_eq!(record["verdict"]["passed"], true);
    assert_eq!(w.pointer().unwrap().current.dir, "good");

    // The control, promoted after, still fails and the good model stays validated.
    assert!(w.promote("control", "control.jsonl", "policy.json").pointer.is_none());
    assert!(w.manifest("good").validation.is_validated());

    // Promoting a second checkpoint of the profile keeps the first as last-known-good.
    w.train("good2", "truth.jsonl", 4);
    let second = w.promote("good2", "truth.jsonl", "policy.json");
    assert!(second.verdict.passed(), "{}", second.verdict);
    assert_eq!(second.pointer.as_ref().unwrap().last_known_good.as_ref().map(|e| e.dir.as_str()), Some("good"));
    assert!(!w.manifest("good").validation.is_validated() && w.manifest("good2").validation.is_validated());
    rollback_model(w.dir("good2").to_str().unwrap()).unwrap();
    assert!(w.manifest("good").validation.is_validated() && !w.manifest("good2").validation.is_validated());
}

#[test]
fn a_bad_policy_or_dataset_is_an_error_not_a_verdict() {
    let w = world();
    std::fs::write(w.file("bad-policy.json"), r#"{"schema": 1, "profile": "desks", "min_test_examples": 5}"#).unwrap();
    let run = |data: &str, policy: &str| promote_model(&PromoteRun { model: w.dir("m").to_str().unwrap().into(), data: w.file(data), policy: w.file(policy) });
    assert!(run("truth.jsonl", "bad-policy.json").unwrap_err().to_string().contains("at least one of"));
    assert!(run("absent.jsonl", "policy.json").is_err());
    assert!(w.pointer().is_none());
}
