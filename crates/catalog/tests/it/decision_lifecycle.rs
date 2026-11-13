// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the tooling that turns recorded outcomes into
// governed decision models, from the first training run to a promotion gate.
// If your team needs expertise in decision-model lifecycle, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `brain decide train|eval|promote|rollback|inspect` as actions of the
//! `brain/decision` provider, on tiny random-weight checkpoints so they run
//! with no downloaded weights.
//!
//! What is pinned is the contract of the command line: the declared manifest
//! validates, a missing dataset is a clean error before any model loads, a
//! training run echoes the seed it drew and honours cancellation, and a server
//! registers none of these host-only actions.

use std::sync::Arc;

use capability::{CancelToken, Invocation, Manifest, Progress, Provider};
use decisions::fixtures::{synthetic_jsonl, write_decide, Scratch, SyntheticTask};
use decisions::provider::{served_manifest, DecisionProvider};
use decisions::store::Catalog;
use serde_json::json;

fn provider(root: &Scratch) -> Arc<dyn Provider> {
    let catalog = Catalog::new(Some(root.to_path_buf()), Vec::new(), None);
    Arc::new(DecisionProvider::new(catalog).with_lifecycle(brain::decision_actions::lifecycle_provider()))
}

fn run(p: &dyn Provider, action: &str, params: serde_json::Value, cancel: CancelToken, progress: &mut dyn FnMut(Progress)) -> Result<capability::Outcome, String> {
    let act = p.action(action).unwrap_or_else(|| panic!("no action {action}"));
    let mut inv = Invocation::new();
    inv.params = params;
    inv.cancel = cancel;
    let inv = act.spec().validate(inv)?;
    act.run(&inv, progress)
}

fn call(p: &dyn Provider, action: &str, params: serde_json::Value) -> Result<capability::Outcome, String> {
    run(p, action, params, CancelToken::default(), &mut |_| {})
}

fn world(tag: &str) -> Scratch {
    let root = Scratch::new(tag);
    write_decide(&root.join("base"), 11);
    std::fs::write(root.join("data.jsonl"), synthetic_jsonl(&SyntheticTask::new(12, 4, 4))).unwrap();
    root
}

fn text(outcome: &capability::Outcome, blob: &str) -> String {
    String::from_utf8(outcome.blobs[blob].bytes.clone()).unwrap()
}

#[test]
fn the_manifest_validates_and_the_lifecycle_is_host_only() {
    let root = world("lifecycle-manifest");
    let manifest: Manifest = provider(&root).manifest();
    for name in decisions::lifecycle::ACTIONS {
        let spec = manifest.actions.iter().find(|a| a.name == name).unwrap_or_else(|| panic!("{name} is declared"));
        let files: Vec<&str> = spec.params.iter().filter(|p| ["data", "base", "out", "model", "costs", "policy"].contains(&p.name.as_str())).filter(|p| !p.host_resolved).map(|p| p.name.as_str()).collect();
        assert!(files.is_empty(), "{name}: a file or checkpoint the action reads or writes is the host's to name: {files:?}");
    }
    let served: Vec<String> = served_manifest().actions.into_iter().map(|a| a.name).collect();
    assert_eq!(served, ["models", "select"], "a server registers neither the lifecycle nor the per-checkpoint decide");
    assert!(provider(&root).action("train").unwrap().spec().training);
}

#[test]
fn without_a_lifecycle_provider_the_actions_are_absent() {
    let root = world("lifecycle-absent");
    let bare = DecisionProvider::new(Catalog::new(Some(root.to_path_buf()), Vec::new(), None));
    assert!(bare.action("train").is_none());
    assert!(bare.action("models").is_some());
}

#[test]
fn a_missing_dataset_is_a_clean_error_before_any_model_loads() {
    let root = world("lifecycle-missing");
    let p = provider(&root);
    let missing = root.join("absent.jsonl").display().to_string();
    let base = root.join("base").display().to_string();
    for (action, params) in [
        ("inspect", json!({"data": missing})),
        ("train", json!({"data": missing, "base": base, "out": root.join("out").display().to_string()})),
        ("eval", json!({"model": base, "data": missing, "split": "test"})),
    ] {
        let e = call(&*p, action, params).unwrap_err();
        assert!(e.contains("absent.jsonl"), "{action}: {e}");
    }
    assert!(!root.join("out").exists());
}

#[test]
fn inspect_summarizes_a_dataset_and_binds_it_by_digest() {
    let root = world("lifecycle-inspect");
    let out = call(&*provider(&root), "inspect", json!({"data": root.join("data.jsonl").display().to_string()})).unwrap();
    assert!(out.outputs["data_sha256"].as_str().is_some_and(|d| d.starts_with("sha256:")), "{}", out.outputs);
    assert_eq!(out.outputs["summary"]["total"], 20, "{}", out.outputs);
    assert!(text(&out, "report").contains("train"));
}

#[test]
fn train_echoes_the_seed_it_drew_and_the_same_seed_reproduces_the_weights() {
    let root = world("lifecycle-train");
    let p = provider(&root);
    let args = |out: &str, seed: Option<u64>| {
        let mut v = json!({"data": root.join("data.jsonl").display().to_string(), "base": root.join("base").display().to_string(), "out": root.join(out).display().to_string(), "steps": 6, "batch": 4});
        if let Some(seed) = seed {
            v["seed"] = json!(seed);
        }
        v
    };
    let mut steps = Vec::new();
    let first = run(&*p, "train", args("a", None), CancelToken::default(), &mut |p| steps.push(p.step)).unwrap();
    assert_eq!(steps, [1, 2, 3, 4, 5, 6], "one progress report per optimizer step");
    let seed = first.outputs["seed"].as_u64().expect("an absent seed is drawn and echoed");
    assert_eq!(first.outputs["validation"], "unvalidated");
    assert!(text(&first, "report").contains("unvalidated"));

    let again = call(&*p, "train", args("b", Some(seed))).unwrap();
    assert_eq!(again.outputs["seed"], seed);
    let weights = |dir: &str| std::fs::read(root.join(dir).join("model.safetensors")).unwrap();
    assert_eq!(weights("a"), weights("b"));

    let taken = call(&*p, "train", args("a", Some(seed))).unwrap_err();
    assert!(taken.contains("already exists"), "{taken}");
}

#[test]
fn train_stops_between_steps_on_cancellation_and_writes_nothing() {
    let root = world("lifecycle-cancel");
    let cancel = CancelToken::armed();
    let handle = cancel.clone();
    let params = json!({"data": root.join("data.jsonl").display().to_string(), "base": root.join("base").display().to_string(), "out": root.join("out").display().to_string(), "steps": 50, "batch": 2});
    let e = run(&*provider(&root), "train", params, cancel, &mut |p| {
        if p.step == 2 {
            handle.cancel();
        }
    })
    .unwrap_err();
    assert_eq!(e, "cancelled after step 2");
    assert!(!root.join("out").exists());
}

#[test]
fn eval_reports_the_record_and_echoes_the_order_seed_and_a_train_split_is_refused() {
    let root = world("lifecycle-eval");
    let p = provider(&root);
    let model = root.join("base").display().to_string();
    let data = root.join("data.jsonl").display().to_string();
    call(&*p, "train", json!({"data": data, "base": model, "out": root.join("tuned").display().to_string(), "steps": 4, "batch": 2})).unwrap();
    let tuned = root.join("tuned").display().to_string();
    let drawn = call(&*p, "eval", json!({"model": tuned, "data": data, "split": "test"})).unwrap();
    let seed = drawn.outputs["seed"].as_u64().expect("the drawn order seed is echoed");
    assert_eq!(drawn.outputs["record"]["data"]["split"], "test");
    assert!(text(&drawn, "report").contains("weights"));
    let fixed = call(&*p, "eval", json!({"model": tuned, "data": data, "split": "test", "seed": seed})).unwrap();
    assert_eq!(fixed.outputs["record"], drawn.outputs["record"], "the same order seed gives the same record");

    assert!(call(&*p, "eval", json!({"model": tuned, "data": data, "split": "train"})).is_err(), "the train split is not held out");
}

#[test]
fn promote_with_an_unmet_gate_fails_with_the_margin_and_changes_nothing() {
    let root = world("lifecycle-promote");
    let p = provider(&root);
    let data = root.join("data.jsonl").display().to_string();
    let tuned = root.join("tuned").display().to_string();
    call(&*p, "train", json!({"data": data, "base": root.join("base").display().to_string(), "out": tuned, "steps": 4, "batch": 2})).unwrap();
    std::fs::write(root.join("policy.json"), r#"{"schema": 1, "profile": "desks", "min_test_examples": 1, "min_accuracy_lift": 0.9}"#).unwrap();
    let e = call(&*p, "promote", json!({"model": tuned, "data": data, "policy": root.join("policy.json").display().to_string()})).unwrap_err();
    assert!(e.contains("not promoted") && e.contains("nothing was changed"), "{e}");
    let manifest = std::fs::read_to_string(root.join("tuned").join(decisions::manifest::FILE)).unwrap();
    assert!(manifest.contains("unvalidated"), "{manifest}");

    let rollback = call(&*p, "rollback", json!({"model": tuned})).unwrap_err();
    assert!(!rollback.is_empty());
}
