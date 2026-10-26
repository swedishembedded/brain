// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "decision")]

//! The lifecycle of a trained Laya head: measured, promoted under a policy,
//! chosen by `auto` only afterwards, and rolled back.
//!
//! The base is the tiny Laya fixture whose trunk mixes information between
//! tokens (`write_laya_mixing`: a frozen random-weight trunk cannot tell the
//! options apart, so no head could learn through it), the task is the
//! synthetic one whose answer is known, and the negative control is a head
//! trained on shuffled labels. Evaluation, promotion, selection and rollback
//! are the architecture-agnostic ones: nothing here is Laya-specific except
//! the directory they are pointed at, which is the claim. Nothing is claimed
//! about real weights.
//!
//! Swedish Embedded AB implements release gates that treat every model
//! architecture the same way: measured at the gate, promoted on evidence,
//! reversible. If your team needs expertise in model promotion and rollback,
//! you can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use brain::decision_run::{evaluate_model, promote_model, rollback_model, train, EvalRun, PromoteRun, TrainRun};
use decisions::engine::Backend;
use decisions::examples::Split;
use decisions::fixtures::{synthetic_jsonl, write_laya_mixing, Scratch, SyntheticTask};
use decisions::manifest::{weights_identity, DecisionManifest, Validation};
use decisions::promotion::{Pointer, EVALUATION_FILE};
use decisions::select::Rule;
use decisions::store::Catalog;

struct World {
    root: Scratch,
}

impl World {
    fn store(&self) -> PathBuf {
        self.root.join("store")
    }

    fn dir(&self, name: &str) -> PathBuf {
        self.store().join("acme").join(name)
    }

    /// Train the head `name` in the store on `data`, over the store's base.
    fn train(&self, name: &str, data: &str) {
        let mut run = TrainRun::new(self.root.join(data), self.store().join("convai/laya").to_str().unwrap(), self.dir(name));
        run.steps = 150;
        run.batch = 8;
        // seed: the accuracy-lift gates of the promotion were tuned on this training draw
        run.seed = 3;
        train(&run, &mut |_, _| {}).unwrap();
    }

    fn promote(&self, name: &str) -> brain::decision_run::PromoteOutcome {
        promote_model(&PromoteRun { model: self.dir(name).to_str().unwrap().to_string(), data: self.root.join("truth.jsonl"), policy: self.root.join("policy.json") }).unwrap()
    }

    fn manifest(&self, name: &str) -> DecisionManifest {
        DecisionManifest::read(&self.dir(name), Backend::Laya).unwrap()
    }

    fn pointer(&self) -> Option<Pointer> {
        Pointer::read(&self.store().join("acme"), "desks").unwrap()
    }

    fn auto(&self) -> decisions::select::Selection {
        Catalog::new(Some(self.store()), vec![], None).select("auto")
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn a_trained_laya_head_is_promoted_only_on_a_measured_lift_and_auto_follows_the_promotion() {
    let root = Scratch::new("promote-laya");
    write_laya_mixing(&root.join("store/convai/laya"), 128);
    let mut control = SyntheticTask::new(320, 0, 120);
    control.shuffle_train_labels = true;
    std::fs::write(root.join("truth.jsonl"), synthetic_jsonl(&SyntheticTask::new(320, 0, 120))).unwrap();
    std::fs::write(root.join("control.jsonl"), synthetic_jsonl(&control)).unwrap();
    std::fs::write(root.join("policy.json"), r#"{"schema": 1, "profile": "desks", "min_test_examples": 100, "min_accuracy_lift": 0.15}"#).unwrap();
    let w = World { root };
    w.train("good", "truth.jsonl");
    w.train("control", "control.jsonl");

    // Installed, unvalidated, and therefore not chosen: auto says why.
    let before = w.auto();
    assert_eq!((before.chosen, before.rule), (None, Rule::None));
    assert!(before.reason.contains("acme/good") && before.reason.contains("no recorded validation"), "{}", before.reason);

    // `decide eval` measures the directory like any checkpoint.
    let measured = evaluate_model(&EvalRun::new(w.dir("good").to_str().unwrap(), w.root.join("truth.jsonl"), Split::Test)).unwrap();
    eprintln!("good: acc {:.3} lift {:+.3}", measured.report.accuracy, measured.report.lift);
    assert_eq!(measured.weights_sha256, weights_identity(&w.dir("good"), Backend::Laya).unwrap());
    assert_eq!(measured.report.examples, 120);

    // The negative control: a head trained on shuffled labels fails the lift gate and nothing changes.
    let rejected = w.promote("control");
    eprintln!("control: acc {:.3} lift {:+.3}", rejected.evaluation.report.accuracy, rejected.evaluation.report.lift);
    assert!(rejected.pointer.is_none() && !rejected.verdict.passed());
    let failed: Vec<&str> = rejected.verdict.failures().map(|g| g.name.as_str()).collect();
    assert_eq!(failed, ["min_accuracy_lift"], "{}", rejected.verdict);
    assert_eq!(w.manifest("control").validation, Validation::Unvalidated);
    assert!(!w.dir("control").join(EVALUATION_FILE).exists() && w.pointer().is_none());
    assert_eq!(w.auto().chosen, None);

    // A second checkpoint of the same bytes, to hand the profile over and back.
    copy_dir(&w.dir("good"), &w.dir("good2"));

    // The head that learned the task meets the policy, and is then (only then) chosen.
    let promoted = w.promote("good");
    assert!(promoted.verdict.passed(), "{}", promoted.verdict);
    let identity = weights_identity(&w.dir("good"), Backend::Laya).unwrap();
    assert_eq!(promoted.evaluation.weights_sha256, identity);
    let m = w.manifest("good");
    assert_eq!(m.validation, Validation::Validated { profile: "desks".into(), evidence: Some(EVALUATION_FILE.into()) });
    assert_eq!(m.weights_sha256, Some(identity));
    assert_eq!(w.pointer().unwrap().current.dir, "good");
    let chosen = w.auto();
    assert_eq!((chosen.chosen.as_deref(), chosen.rule), (Some("acme/good"), Rule::ValidatedLaya));

    // Promoting the next checkpoint hands the choice over, and a rollback hands it back.
    let second = w.promote("good2");
    assert!(second.verdict.passed(), "{}", second.verdict);
    assert_eq!(second.pointer.as_ref().unwrap().last_known_good.as_ref().map(|e| e.dir.as_str()), Some("good"));
    assert!(!w.manifest("good").validation.is_validated());
    assert_eq!(w.auto().chosen.as_deref(), Some("acme/good2"));
    rollback_model(w.dir("good2").to_str().unwrap()).unwrap();
    assert!(w.manifest("good").validation.is_validated() && !w.manifest("good2").validation.is_validated());
    assert_eq!(w.auto().chosen.as_deref(), Some("acme/good"));
}
