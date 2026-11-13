// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the service refuses before it claims a device, and what it refuses when it reads an input: the
//! security surface of the workload (names, paths, digests, sizes) and its cheap failure modes.

use serde_json::json;

use crate::support::{data_digest, plain, rig, rig_with, run, Frames, BASE};

fn data_of(digest: &trainjob::Digest) -> String {
    json!({"train": {"digest": digest.hex()}}).to_string()
}

#[test]
fn a_run_for_what_cannot_exist_is_refused_before_any_work_and_names_why() {
    let rig = rig("door-refused", 8 << 30, true);
    let train = data_digest(&rig, &plain());
    let frames = Frames::default();
    let ask = |params: serde_json::Value| run(&rig, params, &[], &frames).unwrap_err();

    assert!(ask(json!({"base": "nobody/nothing", "data": data_of(&train), "return_adapter": true})).contains("not in the model store"));
    // A name is a model reference, never a path: none of these reaches the filesystem as given.
    for base in ["/etc", "../../etc/passwd", "local/../../etc", "local/toy/extra", "local/toy:acme:chat:v1", "", ".", ".."] {
        let e = ask(json!({"base": base, "data": data_of(&train)}));
        assert!(e.starts_with("train:"), "{base:?}: {e}");
        assert!(!e.contains(rig.dir.to_str().unwrap()), "{base:?}: the refusal names a path of this machine: {e}");
    }
    assert!(frames.events().is_empty(), "a refused job reports nothing: it never started");
    assert!(crate::support::work_is_clean(&rig.work_root), "and made nothing");
    let placed = rig.exec.residency();
    assert!(placed.placements.is_empty() && placed.budgets.iter().all(|b| b.used == 0), "and reserved nothing");
}

#[test]
fn a_digest_that_is_not_a_digest_is_refused_whatever_it_looks_like() {
    let rig = rig("door-digest", 8 << 30, true);
    let frames = Frames::default();
    for bad in ["../../../etc/passwd", "/etc/passwd", "local/toy", &"A".repeat(64), &"g".repeat(64), &"a".repeat(63), &"a".repeat(65), "blake3:", ""] {
        let data = json!({"train": {"digest": bad}}).to_string();
        let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": data}), &[], &frames).unwrap_err();
        assert!(e.contains("64 lowercase hexadecimal"), "{bad:?}: {e}");
    }
}

#[test]
fn a_digest_the_store_does_not_hold_fails_the_run_by_name_and_leaves_nothing() {
    let rig = rig("door-missing", 8 << 30, true);
    let frames = Frames::default();
    let absent = trainjob::store::digest_of(b"never published");
    let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": data_of(&absent)}), &[], &frames).unwrap_err();
    assert!(e.contains(&absent.hex()) && e.contains("not in the artifact store"), "{e}");
    assert!(!e.contains(rig.dir.to_str().unwrap()), "{e}");
    assert!(crate::support::work_is_clean(&rig.work_root));
}

#[test]
fn a_stored_dataset_that_was_swapped_is_refused_by_its_hash() {
    let rig = rig("door-swapped", 8 << 30, true);
    let frames = Frames::default();
    let digest = data_digest(&rig, &plain());
    std::fs::write(rig.store.root().join(digest.hex()), b"{\"messages\": []}\n").unwrap();
    let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": data_of(&digest)}), &[], &frames).unwrap_err();
    assert!(e.contains("does not hash to its digest"), "{e}");
    assert!(crate::support::work_is_clean(&rig.work_root), "the unverified copy is not left for anything to read");
}

#[test]
fn a_server_without_an_artifact_store_takes_no_digest_and_must_return_the_adapter() {
    let rig = rig("door-no-store", 8 << 30, false);
    let frames = Frames::default();
    let digest = trainjob::store::digest_of(b"x");
    let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": data_of(&digest), "return_adapter": true}), &[], &frames).unwrap_err();
    assert!(e.contains("no artifact store") && e.contains("digest"), "{e}");
    let by_blob = json!({"train": {"blob": "dataset"}}).to_string();
    let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": by_blob}), &[("dataset", plain())], &frames).unwrap_err();
    assert!(e.contains("return_adapter"), "an adapter nobody could collect is not trained: {e}");
    let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": by_blob, "return_adapter": true, "keep_evaluations": true, "eval_every": 1}), &[("dataset", plain())], &frames).unwrap_err();
    assert!(e.contains("keep_evaluations"), "{e}");
    assert!(frames.events().is_empty());
}

#[test]
fn a_run_that_cannot_fit_the_largest_card_is_refused_with_what_it_needs() {
    // A card of 1 GiB: the toy base is a few MB, but every run reserves its fixed overhead too.
    let rig = rig_with("door-card", 8 << 30, true, Some(1 << 30), Vec::new());
    let train = data_digest(&rig, &plain());
    let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": data_of(&train)}), &[], &Frames::default()).unwrap_err();
    assert!(e.contains("GiB") && e.contains("largest card") && e.contains("max_block"), "{e}");
}

#[test]
fn what_a_caller_may_not_set_is_refused_as_unknown() {
    let rig = rig("door-unknown", 8 << 30, true);
    let train = data_digest(&rig, &plain());
    for name in ["weights", "out_dir", "models_dir", "dataset", "path", "device", "save", "tokenizer"] {
        let e = run(&rig, json!({"base": BASE, "arch": "qwen35", "data": data_of(&train), name: "../x"}), &[], &Frames::default()).unwrap_err();
        assert!(e.contains("unknown param"), "{name}: {e}");
    }
    // Nor does the model list offer it to anyone as a way to name one.
    let manifest = rig.exec.manifests().iter().find(|m| m.model == trainjob::MODEL).cloned().unwrap().for_serving();
    let names: Vec<String> = manifest.actions[0].params.iter().map(|p| p.name.clone()).collect();
    for path_like in ["weights", "out_dir", "models_dir", "path", "save", "device"] {
        assert!(!names.contains(&path_like.to_string()), "{path_like} is offered");
    }
}

#[test]
fn the_service_has_exactly_the_train_action() {
    let rig = rig("door-action", 8 << 30, true);
    let manifest = rig.exec.manifests().iter().find(|m| m.model == trainjob::MODEL).cloned().unwrap();
    assert_eq!(manifest.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["train"]);
    let mut inv = capability::Invocation::new();
    inv.params = json!({});
    let e = rig.exec.run_blocking(trainjob::MODEL, "shell", inv, |_| {}).unwrap_err();
    assert!(e.contains("no action 'shell'"), "{e}");
}

/// A preference job is held to the same door as every other objective: a dataset is a digest or a blob, never a path;
/// settings of the other objectives are refused by name; and nothing starts or is reserved by a refusal.
#[test]
fn a_dpo_job_is_refused_at_the_door_for_paths_and_for_settings_that_are_not_its_own() {
    let rig = rig("door-dpo", 8 << 30, true);
    let train = data_digest(&rig, &crate::support::pairs());
    let frames = Frames::default();
    let ask = |params: serde_json::Value| run(&rig, params, &[], &frames).unwrap_err();
    for held_out in [json!({"path": "/etc/passwd"}), json!({"digest": "../../etc/passwd"})] {
        let data = json!({"train": {"digest": train.hex()}, "held_out": held_out}).to_string();
        let e = ask(json!({"base": BASE, "objective": "dpo", "data": data}));
        assert!(e.contains("data"), "{e}");
    }
    let e = ask(json!({"base": BASE, "objective": "dpo", "data": data_of(&train), "replay_share": 0.3}));
    assert!(e.contains("replay_share") && e.contains("dpo"), "{e}");
    let e = ask(json!({"base": BASE, "objective": "dpo", "data": data_of(&train), "reference_model": "/etc/passwd"}));
    assert!(e.contains("unknown param"), "a reference is the run's start, never a path a caller names: {e}");
    assert!(frames.events().is_empty() && crate::support::work_is_clean(&rig.work_root));
    assert!(rig.exec.residency().placements.is_empty());
}
