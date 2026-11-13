// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The jobs that produce a checkpoint, run through the executor on the toy `qwen3` base: every weight trains and the
//! checkpoint leaves as a manifest of artifacts and as a model of the model store; an adapter is folded into its
//! base the same way; and what is not a plain model-store name, a digest or an attached blob is refused.

use serde_json::{json, Value};
use trainjob::consolidate::{full_spec, merge_spec, FULL_ACTION, MERGE_ACTION};

use crate::support::{assert_reservation_returned, data_digest, plain, rig_qwen3, run, run_action, work_is_clean, Frames, BASE};

/// The files of a published checkpoint, each verified against the digest the manifest names.
fn published_files(rig: &crate::support::Rig, report: &Value) -> std::collections::BTreeMap<String, Vec<u8>> {
    let manifest = trainjob::Digest::parse(report["checkpoint_digest"].as_str().expect("a checkpoint digest")).unwrap();
    let listed: Value = serde_json::from_slice(&std::fs::read(rig.store.root().join(manifest.hex())).unwrap()).unwrap();
    assert_eq!(listed["format"], trainjob::consolidate::MANIFEST_FORMAT);
    assert_eq!(listed["files"], report["files"]);
    listed["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let bytes = std::fs::read(rig.store.root().join(f["digest"].as_str().unwrap())).unwrap();
            assert_eq!(trainjob::store::digest_of(&bytes).hex(), f["digest"].as_str().unwrap());
            assert_eq!(bytes.len() as u64, f["bytes"].as_u64().unwrap());
            (f["name"].as_str().unwrap().to_string(), bytes)
        })
        .collect()
}

#[test]
fn a_full_fine_tune_publishes_its_checkpoint_by_manifest_installs_it_and_a_later_job_trains_on_it() {
    let rig = rig_qwen3("full", 16 << 30);
    let train = data_digest(&rig, &plain());
    let params = json!({"base": BASE, "steps": 3, "lr": 1e-3, "seed": 7, "max_block": 128, "train_precision": "state=int8,sr=on", "install_as": "tuned-1", "data": json!({"train": {"digest": train.hex()}}).to_string()});
    let frames = Frames::default();
    let outcome = run_action(&rig, FULL_ACTION, &full_spec(), params.clone(), &[], &frames).expect("the run");

    assert_eq!(frames.steps().len(), 3, "{:?}", frames.events());
    let report = &outcome.outputs;
    assert_eq!((report["status"].as_str(), report["objective"].as_str(), report["steps_completed"].as_u64(), report["records"].as_u64()), (Some("completed"), Some("full_sft"), Some(3), Some(4)));
    assert!(report["final_loss"].is_number() && report["weights_sha256"].as_str().is_some_and(|s| s.starts_with("sha256:")), "{report}");
    let files = published_files(&rig, report);
    assert!(files.contains_key("model.safetensors") && files.contains_key("tokenizer.json"), "{:?}", files.keys());
    assert_eq!(report["installed_as"], "local/tuned-1");
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);

    // The installed checkpoint is a base like any other: an adapter trains on it.
    let sft = json!({"base": "local/tuned-1", "steps": 1, "rank": 2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string(), "return_adapter": true});
    assert_eq!(run(&rig, sft, &[], &Frames::default()).expect("a run on the installed base").outputs["status"], "completed");

    // A name that is taken is refused, and the model store keeps what it had.
    let again = run_action(&rig, FULL_ACTION, &full_spec(), params, &[], &Frames::default()).expect_err("the name is taken");
    assert!(again.contains("already in the model store"), "{again}");
    assert!(work_is_clean(&rig.work_root));
}

#[test]
fn a_merge_folds_an_adapter_into_its_base_and_the_merged_checkpoint_differs_from_the_base() {
    let rig = rig_qwen3("merge", 16 << 30);
    let train = data_digest(&rig, &plain());
    let sft = json!({"base": BASE, "steps": 4, "rank": 2, "lr": 5e-2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string()});
    let adapter = run(&rig, sft, &[], &Frames::default()).expect("the adapter");
    let adapter_digest = adapter.outputs["adapter_digest"].as_str().unwrap().to_string();

    let by_digest = run_action(&rig, MERGE_ACTION, &merge_spec(), json!({"base": BASE, "adapter_digest": adapter_digest, "install_as": "merged-1"}), &[], &Frames::default()).expect("the merge");
    assert_eq!((by_digest.outputs["status"].as_str(), by_digest.outputs["objective"].as_str(), by_digest.outputs["installed_as"].as_str()), (Some("completed"), Some("merge"), Some("local/merged-1")));
    let merged = published_files(&rig, &by_digest.outputs);
    let base_weights = std::fs::read(rig.dir.join("models").join("local").join("toy").join("model.brain.safetensors")).unwrap();
    assert_ne!(merged["model.safetensors"], base_weights, "the adapter changed the weights");

    // The same adapter attached as a blob folds to the same checkpoint.
    let bytes = std::fs::read(rig.store.root().join(&adapter_digest)).unwrap();
    let by_blob = run_action(&rig, MERGE_ACTION, &merge_spec(), json!({"base": BASE}), &[("adapter", bytes)], &Frames::default()).expect("the merge by blob");
    assert_eq!(by_blob.outputs["weights_sha256"], by_digest.outputs["weights_sha256"], "the fold is deterministic");
    assert!(by_blob.outputs["installed_as"].is_null());
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

#[test]
fn checkpoint_jobs_refuse_what_is_not_a_digest_a_blob_or_a_plain_name_before_any_work() {
    let rig = rig_qwen3("consolidate-door", 16 << 30);
    let train = data_digest(&rig, &plain());
    let data = json!({"train": {"digest": train.hex()}}).to_string();
    let frames = Frames::default();
    let merge = |params: Value, blobs: &[(&str, Vec<u8>)]| run_action(&rig, MERGE_ACTION, &merge_spec(), params, blobs, &frames).unwrap_err();
    let full = |params: Value| run_action(&rig, FULL_ACTION, &full_spec(), params, &[], &frames).unwrap_err();

    for name in ["../escape", "a/b", "..", ".hidden", "", "x y", "local/other"] {
        let e = full(json!({"base": BASE, "data": data, "install_as": name}));
        assert!(e.contains("install_as"), "{name:?}: {e}");
        let e = merge(json!({"base": BASE, "adapter_digest": train.hex(), "install_as": name}), &[]);
        assert!(e.contains("install_as"), "{name:?}: {e}");
    }
    assert!(merge(json!({"base": BASE}), &[]).contains("exactly once"));
    assert!(merge(json!({"base": BASE, "adapter_digest": "/etc/passwd"}), &[]).contains("64 lowercase hexadecimal"));
    let absent = trainjob::store::digest_of(b"never published");
    assert!(merge(json!({"base": BASE, "adapter_digest": absent.hex()}), &[]).contains("not in the artifact store"));
    assert!(merge(json!({"base": "nobody/nothing", "adapter_digest": train.hex()}), &[]).contains("not in the model store"));
    let e = full(json!({"base": BASE, "data": data, "train_precision": "state=nonsense"}));
    assert!(e.contains("train_precision"), "{e}");
    for path in ["path", "out", "save", "checkpoint"] {
        let e = merge(json!({"base": BASE, "adapter_digest": train.hex(), path: "elsewhere"}), &[]);
        assert!(e.contains("unknown param"), "{path}: {e}");
    }
    let held = json!({"train": {"digest": train.hex()}, "held_out": {"digest": train.hex()}}).to_string();
    assert!(full(json!({"base": BASE, "data": held})).contains("held_out"));
    assert!(frames.events().is_empty() && work_is_clean(&rig.work_root));
    assert!(!rig.dir.join("models").join("local").join("x y").exists());
    assert!(rig.exec.residency().placements.is_empty());
}

/// A server with no artifact store has nowhere to publish a checkpoint: it must be told to install it, and a merge
/// cannot take its adapter by digest.
#[test]
fn a_server_without_an_artifact_store_must_install_what_it_trains() {
    let rig = crate::support::rig_with_store_flag("consolidate-no-store", false);
    let data = json!({"train": {"blob": "dataset"}}).to_string();
    let frames = Frames::default();
    let e = run_action(&rig, FULL_ACTION, &full_spec(), json!({"base": BASE, "data": data}), &[("dataset", plain())], &frames).unwrap_err();
    assert!(e.contains("install_as"), "{e}");
    let e = run_action(&rig, MERGE_ACTION, &merge_spec(), json!({"base": BASE, "adapter_digest": trainjob::store::digest_of(b"x").hex()}), &[], &frames).unwrap_err();
    assert!(e.contains("no artifact store"), "{e}");
}

/// A dry run reports the whole file without training: counts, token lengths in the base's own tokenizer, supervised
/// tokens, the examples a row of `max_block` would refuse, and every bad line (not only the first), with an example that supervises nothing counted.
#[test]
fn a_dataset_dry_run_reports_the_file_and_names_every_bad_line_without_claiming_a_card() {
    use trainjob::dryrun::{spec, ACTION};
    let rig = rig_qwen3("dryrun", 16 << 30);
    let long = crate::support::record(&"c".repeat(150), "bad", None);
    let file = [crate::support::record("cab", "bad", None), long, "{\"messages\": 5}".to_string(), String::new(), crate::support::record("ace", "cab", None), r#"{"messages":[{"role":"user","content":"x","train":false}]}"#.to_string()].join("\n");
    let digest = data_digest(&rig, file.as_bytes());
    let frames = Frames::default();
    let ask = |params: Value, blobs: &[(&str, Vec<u8>)]| run_action(&rig, ACTION, &spec(), params, blobs, &frames);

    let out = ask(json!({"base": BASE, "dataset_digest": digest.hex(), "max_block": 64}), &[]).expect("the report").outputs;
    assert_eq!((out["records"].as_u64(), out["valid_records"].as_u64(), out["examples"].as_u64()), (Some(5), Some(4), Some(4)), "{out}");
    assert_eq!((out["over_max_block"].as_u64(), out["over_max_block_lines"].clone()), (Some(1), json!([2])), "only the long record is past a 64-token row");
    assert_eq!(out["invalid_total"], 1, "a record that does not parse is invalid; one that parses and supervises nothing is counted, not refused");
    assert_eq!(out["invalid"][0]["line"], 3);
    assert!(out["invalid"][0]["error"].as_str().is_some_and(|e| !e.is_empty()));
    assert!(out["tokens"]["max"].as_u64().unwrap() > 150 && out["tokens"]["min"].as_u64().unwrap() < 64, "{out}");
    assert!(out["supervised_tokens"]["max"].as_u64().unwrap() < out["tokens"]["max"].as_u64().unwrap());
    assert_eq!(out["examples_without_supervision"], 1, "the user-only record on line 6 would train on nothing");
    assert_eq!(out["supervised_tokens"]["min"], 0);

    // The same file attached as a blob reads the same; preference pairs are their own kind.
    let by_blob = ask(json!({"base": BASE, "max_block": 64}), &[("dataset", file.clone().into_bytes())]).expect("the report").outputs;
    assert_eq!(by_blob, out);
    let pairs = ask(json!({"base": BASE, "kind": "preference"}), &[("dataset", crate::support::pairs())]).expect("the report").outputs;
    assert_eq!((pairs["records"].as_u64(), pairs["examples"].as_u64(), pairs["invalid_total"].as_u64()), (Some(4), Some(8), Some(0)), "two candidates per pair: {pairs}");

    // Nothing was staged for later, trained, or reserved; a file that is not named once, or by a path, is refused.
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
    assert!(ask(json!({"base": BASE}), &[]).unwrap_err().contains("exactly once"));
    assert!(ask(json!({"base": BASE, "dataset_digest": "/etc/passwd"}), &[]).unwrap_err().contains("64 lowercase hexadecimal"));
    assert!(ask(json!({"base": BASE, "dataset_digest": digest.hex(), "path": "/etc/passwd"}), &[]).unwrap_err().contains("unknown param"));
    assert!(ask(json!({"base": "nobody/nothing", "dataset_digest": digest.hex()}), &[]).unwrap_err().contains("not in the model store"));
    assert!(ask(json!({"base": BASE, "dataset_digest": trainjob::store::digest_of(b"absent").hex()}), &[]).unwrap_err().contains("not in the artifact store"));
}

/// A timeline model trains as a job of the service, leaves by manifest digest and is installed in the model store with
/// a manifest of family `horizon`, which is how the model-dir scan serves it as a model of its own.
#[test]
fn a_timeline_job_publishes_its_model_and_installs_it_as_a_horizon_model_of_the_store() {
    use horizon::synthetic::population;
    use trainjob::timeline::{spec, ACTION};

    let lines = |subjects: &[horizon::timeline::Subject]| -> Vec<u8> { subjects.iter().map(|s| serde_json::to_string(s).unwrap() + "\n").collect::<String>().into_bytes() };
    let (train, _) = population(120, 1);
    let (held, _) = population(40, 2);
    let rig = rig_qwen3("timeline", 16 << 30);
    let held_digest = data_digest(&rig, &lines(&held));
    let params = json!({"absorbing": "death:a,death:b", "knots": "0,2,5,10", "steps": 10, "batch": 32, "eval_interval": 5, "seed": 3, "held_out_digest": held_digest.hex(), "install_as": "risk-1"});
    let frames = Frames::default();
    let outcome = run_action(&rig, ACTION, &spec(), params.clone(), &[("dataset", lines(&train))], &frames).expect("the run");

    let report = &outcome.outputs;
    assert_eq!((report["status"].as_str(), report["objective"].as_str(), report["kind"].as_str()), (Some("completed"), Some("timeline"), Some("single")), "{report}");
    assert_eq!(report["train_subjects"], 120);
    assert!(!frames.0.lock().unwrap().is_empty(), "progress per evaluation interval");
    let files = published_files(&rig, report);
    for name in ["model.safetensors", "vocab.json", "brain.manifest.json"] {
        assert!(files.contains_key(name), "{name} in {:?}", files.keys());
    }
    assert_eq!(report["installed_as"], "local/risk-1");

    // The store finds it by name, as a model of family `horizon` served from its directory.
    let models = rig.dir.join("models");
    let local = brain_modelstore::Store::new(&models).local(&brain_modelref::ModelRef::parse("local/risk-1").unwrap()).expect("the installed model is in the store");
    assert_eq!(local.card.as_ref().map(|c| (c.id.as_str(), c.family.as_str())), Some(("local/risk-1", "horizon")));
    let dir = local.roles.as_ref().and_then(|r| r.get("weights")).expect("the weights role is the model directory");
    let served = catalog::resident_horizon::HorizonResident::new("local/risk-1", dir).expect("a saved timeline model");
    assert_eq!(residency::ResidentModel::manifest(&served).actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["predict", "eval", "calibrate"]);

    // The name is taken; a dataset named twice, a path and a digest the store lacks are refused before any work.
    assert!(run_action(&rig, ACTION, &spec(), params.clone(), &[("dataset", lines(&train))], &Frames::default()).unwrap_err().contains("already in the model store"));
    let ask = |p: Value, blobs: &[(&str, Vec<u8>)]| run_action(&rig, ACTION, &spec(), p, blobs, &Frames::default()).unwrap_err();
    assert!(ask(json!({"held_out_digest": held_digest.hex()}), &[]).contains("exactly once"));
    assert!(ask(json!({"held_out_digest": held_digest.hex(), "dataset_digest": held_digest.hex()}), &[("dataset", lines(&train))]).contains("exactly once"));
    assert!(ask(json!({"held_out_digest": "/etc/passwd", "dataset_digest": held_digest.hex()}), &[]).contains("64 lowercase hexadecimal"));
    assert!(ask(json!({"held_out_digest": trainjob::store::digest_of(b"absent").hex(), "dataset_digest": held_digest.hex()}), &[]).contains("not in the artifact store"));
    assert!(ask(json!({"held_out_digest": held_digest.hex(), "dataset_digest": held_digest.hex(), "out": "elsewhere"}), &[]).contains("unknown param"), "the output directory is never a caller's");
    assert!(ask(json!({"held_out_digest": held_digest.hex(), "dataset_digest": held_digest.hex(), "install_as": "../x"}), &[]).contains("install_as"));
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}
