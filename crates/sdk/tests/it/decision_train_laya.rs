// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "decision")]

//! `brain decide train` on a Laya base: the frozen backbone is referenced, not
//! copied, and the trained head comes out as a directory the loader serves.
//!
//! The base is the tiny random-weight Laya fixture of `decisions::fixtures`, so
//! these tests run everywhere and need no downloaded weights. They pin what a
//! Laya run promises beyond a `decide` one: no weights are duplicated, the
//! manifest binds the head AND the base it was trained on, the same seed gives
//! the same bytes, and the one thing that would need a full copy (training the
//! trunk) is refused before anything is written.
//!
//! Swedish Embedded AB implements fine-tuning pipelines that store only what a
//! run changed and name what it did not. If your team needs expertise in
//! adapter-style model artifacts, you can procure our services by sending an
//! email to info@swedishembedded.com.

use brain::decision_run::{load_model, train, TrainRun};
use decisions::engine::Backend;
use decisions::fixtures::{synthetic_jsonl, write_laya, Scratch, SyntheticTask};
use decisions::laya_head::{BaseRef, BASE_FILE, HEAD_FILE};
use decisions::manifest::{weights_identity, DecisionManifest, Validation};

struct Setup {
    root: Scratch,
    /// The training seed every run of this setup shares.
    seed: u64,
}

impl Setup {
    fn new(tag: &str) -> Setup {
        let root = Scratch::new(tag);
        write_laya(&root.join("base"), 48);
        std::fs::write(root.join("data.jsonl"), synthetic_jsonl(&SyntheticTask::new(12, 4, 4))).unwrap();
        Setup { root, seed: brain::random_seed() }
    }

    fn run(&self, out: &str) -> TrainRun {
        let mut run = TrainRun::new(self.root.join("data.jsonl"), self.root.join("base").to_str().unwrap(), self.root.join(out));
        run.steps = 4;
        run.batch = 2;
        run.seed = self.seed;
        run.brain_commit = "abc1234".into();
        run
    }
}

#[test]
fn a_laya_run_writes_the_head_and_a_reference_to_the_backbone_with_its_lineage() {
    let s = Setup::new("train-laya-lineage");
    let run = s.run("out");
    let outcome = train(&run, &mut |_, _| {}).unwrap();
    assert!(outcome.final_loss.is_finite());

    assert!(!run.out.join("model.safetensors").exists(), "the frozen backbone must not be duplicated");
    assert!(run.out.join(HEAD_FILE).is_file() && run.out.join(BASE_FILE).is_file());
    let leftovers: Vec<_> = std::fs::read_dir(&*s.root).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n.contains("partial")).collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    let m = DecisionManifest::read(&run.out, Backend::Laya).unwrap();
    assert_eq!(m, outcome.manifest);
    assert_eq!(m.validation, Validation::Unvalidated, "training never claims validation");
    let base_digest = brain_modelstore::fetch::base_digest(&s.root.join("base")).unwrap();
    let l = &m.lineage;
    assert_eq!((l.base.as_deref(), l.base_sha256.as_deref()), (Some("local/base"), Some(base_digest.as_str())));
    assert_eq!(l.data_sha256, Some(brain_modelstore::fetch::file_digest(&s.root.join("data.jsonl")).unwrap()));
    let counts = l.split_counts.as_ref().unwrap();
    assert_eq!((counts["train"], counts["validation"], counts["test"]), (12, 4, 4));
    assert_eq!((l.seed, l.brain_commit.as_deref()), (Some(s.seed), Some("abc1234")));
    let h = l.hyperparameters.as_ref().unwrap();
    assert_eq!((h["steps"].as_u64(), h["batch"].as_u64(), h["freeze_encoder"].as_bool()), (Some(4), Some(2), Some(true)));

    // The declared hashes bind the manifest to the head AND the base it names.
    assert_eq!(m.weights_sha256, Some(weights_identity(&run.out, Backend::Laya).unwrap()));
    assert_eq!(BaseRef::read(&run.out).unwrap().unwrap().sha256, base_digest);
    assert_eq!(m.tokenizer_sha256, Some(brain_modelstore::fetch::file_digest(&run.out.join("tokenizer/tokenizer.json")).unwrap()));

    // And the result loads through the same path a stock checkpoint does.
    let loaded = load_model(run.out.to_str().unwrap()).unwrap();
    assert_eq!(loaded.weights_sha256, weights_identity(&run.out, Backend::Laya).unwrap());
}

#[test]
fn the_same_seed_gives_the_same_weights_and_another_seed_does_not() {
    let s = Setup::new("train-laya-determinism");
    let a = train(&s.run("a"), &mut |_, _| {}).unwrap();
    let b = train(&s.run("b"), &mut |_, _| {}).unwrap();
    assert_eq!(a.manifest.weights_sha256, b.manifest.weights_sha256);
    assert_eq!(std::fs::read(s.root.join("a").join(HEAD_FILE)).unwrap(), std::fs::read(s.root.join("b").join(HEAD_FILE)).unwrap());
    let mut other = s.run("c");
    other.seed = s.seed ^ 1;
    let c = train(&other, &mut |_, _| {}).unwrap();
    assert_ne!(a.manifest.weights_sha256, c.manifest.weights_sha256);
}

#[test]
fn training_the_trunk_and_training_from_a_trained_directory_are_refused_before_anything_is_written() {
    let s = Setup::new("train-laya-refusals");
    let mut finetune = s.run("x");
    finetune.freeze_encoder = false;
    let e = train(&finetune, &mut |_, _| {}).unwrap_err().to_string();
    assert!(e.contains("--finetune-encoder") && e.contains("trunk"), "{e}");

    train(&s.run("trained"), &mut |_, _| {}).unwrap();
    let mut chained = s.run("x");
    chained.base = s.root.join("trained").to_str().unwrap().to_string();
    let e = train(&chained, &mut |_, _| {}).unwrap_err().to_string();
    assert!(e.contains("itself a trained head"), "{e}");

    assert!(!s.root.join("x").exists(), "a refused run leaves no output");
    let leftovers: Vec<_> = std::fs::read_dir(&*s.root).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n.contains("partial")).collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}
