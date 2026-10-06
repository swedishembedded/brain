// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the whole path from a client's records to a
// checked, calibrated risk model, for its clients. If your team needs
// expertise in training and validating time-to-event models you can procure
// our services by sending an email to info@swedishembedded.com.

//! The model's life cycle through the capability layer: train a model on a
//! small synthetic population, judge it, calibrate it, predict with it. A
//! cancelled or failed training leaves no model directory, and the schemas of
//! the four actions are what a caller discovers.

use std::path::{Path, PathBuf};

use capability::{Blob, CancelToken, Invocation, Media, Outcome, Provider};
use horizon::caps::{manifest, manifest_resident, HorizonProvider};
use horizon::synthetic::population;
use horizon::timeline::Subject;
use serde_json::{json, Value};

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("horizon-life-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn jsonl(subjects: &[Subject]) -> Blob {
    let text: String = subjects.iter().map(|s| serde_json::to_string(s).unwrap() + "\n").collect();
    Blob::new(Media::Text, text.into_bytes())
}

fn run(action: &str, inv: &Invocation) -> Result<Outcome, String> {
    HorizonProvider::new()
        .action(action)
        .unwrap_or_else(|| panic!("no action {action}"))
        .run(inv, &mut |_| {})
}

fn train_request(out: &Path, train: &[Subject], held: &[Subject], steps: i64) -> Invocation {
    Invocation::new()
        .set("out", json!(out.to_string_lossy()))
        .set("absorbing", json!("death:a,death:b"))
        .set("knots", json!("0,2,5,10"))
        .set("steps", json!(steps))
        .set("batch", json!(32))
        .set("eval_interval", json!(10))
        .blob("dataset", jsonl(train))
        .blob("held_out", jsonl(held))
}

fn json_lines(outcome: &Outcome) -> Vec<Value> {
    std::str::from_utf8(&outcome.blobs["predictions"].bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn train_eval_calibrate_predict_through_the_capability_layer() {
    if skip_gpu() {
        return;
    }
    let (train, _) = population(400, 1);
    let (held, _) = population(100, 2);
    let (validation, _) = population(500, 3);
    let (test, _) = population(500, 4);
    let work = scratch("e2e");
    let model = work.join("model");

    // Train: the model appears, with a report that names what was trained.
    let mut seen = Vec::new();
    let trained = HorizonProvider::new()
        .action("train")
        .unwrap()
        .run(&train_request(&model, &train, &held, 30), &mut |p| seen.push(p))
        .unwrap();
    assert!(model.join("model.safetensors").is_file() && model.join("vocab.json").is_file());
    assert!(!seen.is_empty() && seen.iter().all(|p| p.total == 30), "progress per evaluation interval");
    assert_eq!(seen.last().unwrap().step, 30);
    assert_eq!(trained.outputs["kind"], json!("single"));
    assert_eq!(trained.outputs["codes"], json!(["death:a", "death:b", "onset"]));
    let nll = trained.outputs["members"][0]["held_out_event_nll"].as_f64().unwrap();
    assert!(nll.is_finite());

    // The directory is refused, before any training, once it exists.
    let again = run("train", &train_request(&model, &train, &held, 30)).unwrap_err();
    assert!(again.contains("already exists"), "{again}");

    // Eval: per code and horizon, absent where events are too few.
    let eval = Invocation::new()
        .set("weights", json!(model.to_string_lossy()))
        .set("horizons", json!("2,5"))
        .set("min_events", json!(20))
        .blob("dataset", jsonl(&test));
    let evaluation = run("eval", &eval).unwrap();
    let results = evaluation.outputs["results"].as_array().unwrap();
    assert!(!results.is_empty());
    for r in results {
        assert!(r["events"].as_u64().unwrap() >= 20);
        assert!(r["uno_c"].is_number() && r["auc"].is_number() && r["brier"].is_number());
        assert!(r["integrated_brier"].is_number() && r["calibration"]["slope"].is_number());
    }
    assert!(evaluation.outputs["event_nll"].as_f64().unwrap().is_finite());
    let too_strict = run("eval", &eval.clone().set("min_events", json!(100_000))).unwrap();
    assert_eq!(too_strict.outputs["results"], json!([]), "horizons with too few events are absent");
    assert!(!too_strict.outputs["absent"].as_array().unwrap().is_empty());
    let past = run("eval", &eval.clone().set("horizons", json!("11"))).unwrap_err();
    assert!(past.contains("outside the model's range"), "{past}");

    // Calibrate in place: calibration.json appears, bound to the weights, and
    // is refused a second time without force.
    let calibrate = Invocation::new()
        .set("weights", json!(model.to_string_lossy()))
        .set("horizons", json!("2,5"))
        .set("min_events", json!(10))
        .blob("validation", jsonl(&validation));
    let calibrated = run("calibrate", &calibrate).unwrap();
    assert!(calibrated.outputs["calibrated"].as_u64().unwrap() > 0);
    assert_eq!(calibrated.outputs["kind"], json!("logistic"), "the default kind, recorded in the file");
    let bad = run("calibrate", &calibrate.clone().set("kind", json!("platt")).set("force", json!(true))).unwrap_err();
    assert!(bad.contains("unknown kind"), "{bad}");
    assert!(model.join("calibration.json").is_file());
    assert_eq!(
        std::fs::read(model.join("calibration.json")).unwrap(),
        calibrated.blobs["calibration"].bytes
    );
    let refused = run("calibrate", &calibrate).unwrap_err();
    assert!(refused.contains("already holds"), "{refused}");
    assert!(run("calibrate", &calibrate.clone().set("force", json!(true))).is_ok());

    // Calibrate into a copy: the original is untouched.
    let copy = work.join("copy");
    let before = std::fs::read(model.join("calibration.json")).unwrap();
    run("calibrate", &calibrate.clone().set("out", json!(copy.to_string_lossy())).set("force", json!(true))).unwrap();
    assert!(copy.join("model.safetensors").is_file() && copy.join("calibration.json").is_file());
    assert_eq!(std::fs::read(model.join("calibration.json")).unwrap(), before);
    let blocked = run("calibrate", &calibrate.clone().set("out", json!(copy.to_string_lossy()))).unwrap_err();
    assert!(blocked.contains("already exists"), "{blocked}");

    // Predict from the trained, calibrated model.
    let predict = Invocation::new()
        .set("weights", json!(model.to_string_lossy()))
        .set("times", json!("2,5"))
        .blob("subjects", jsonl(&test[..5]));
    let lines = json_lines(&run("predict", &predict).unwrap());
    assert_eq!(lines.len(), 5);
    for l in &lines {
        assert_eq!(l["cif"]["death:a"].as_array().unwrap().len(), 2);
        assert!(l.get("cif_calibrated").is_some(), "{l}");
    }
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn a_cancelled_training_leaves_no_model_directory() {
    if skip_gpu() {
        return;
    }
    let (train, _) = population(200, 1);
    let (held, _) = population(50, 2);
    let work = scratch("cancel");
    let model = work.join("model");
    let cancel = CancelToken::armed();
    let mut inv = train_request(&model, &train, &held, 500);
    inv.cancel = cancel.clone();
    let mut reports = 0;
    let err = HorizonProvider::new()
        .action("train")
        .unwrap()
        .run(&inv, &mut |_| {
            reports += 1;
            cancel.cancel();
        })
        .unwrap_err();
    assert_eq!(err, "cancelled");
    assert_eq!(reports, 1, "it stopped at the step after the first report");
    assert!(!model.exists());
    assert_eq!(std::fs::read_dir(&work).unwrap().count(), 0, "nothing left beside it either");

    // An ensemble stops the same way, between and inside members.
    let cancel = CancelToken::armed();
    let mut inv = train_request(&model, &train, &held, 500).set("members", json!(3));
    inv.cancel = cancel.clone();
    let err = HorizonProvider::new()
        .action("train")
        .unwrap()
        .run(&inv, &mut |_| cancel.cancel())
        .unwrap_err();
    assert_eq!(err, "cancelled");
    assert_eq!(std::fs::read_dir(&work).unwrap().count(), 0);
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn a_bad_dataset_is_named_and_trains_nothing() {
    let work = scratch("bad");
    let (train, _) = population(20, 1);
    let good = serde_json::to_string(&train[0]).unwrap();
    let broken = Blob::new(Media::Text, format!("{good}\n\n{{\"subject_id\":\"b\"}}\n").into_bytes());
    let inv = train_request(&work.join("model"), &train, &train, 5).blob("dataset", broken);
    let err = run("train", &inv).unwrap_err();
    assert!(err.contains("dataset") && err.contains("line 3"), "{err}");
    let no_held_out = Invocation::new()
        .set("out", json!(work.join("model").to_string_lossy()))
        .blob("dataset", jsonl(&train));
    let err = run("train", &no_held_out).unwrap_err();
    assert!(err.contains("held_out"), "{err}");
    let err = run("eval", &Invocation::new().set("weights", json!("/nonexistent/model")).set("horizons", json!("5"))).unwrap_err();
    assert!(err.contains("no saved timeline model"), "{err}");
    assert!(!work.join("model").exists());
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn the_action_schemas_are_discoverable() {
    let m = manifest();
    let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["predict", "train", "eval", "calibrate"]);
    let action = |name: &str| m.actions.iter().find(|a| a.name == name).unwrap();
    let params = |name: &str| action(name).params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>();
    for p in [
        "out", "codes", "absorbing", "knots", "steps", "batch", "seed", "next_events", "next_weight", "mixer",
        "blocks", "forecasts", "members", "ensemble",
    ] {
        assert!(params("train").contains(&p), "train has no param {p}");
    }
    assert!(action("train").streaming, "train reports progress");
    let inputs = |name: &str| action(name).inputs.iter().map(|b| b.name.as_str()).collect::<Vec<_>>();
    assert_eq!(inputs("train"), ["dataset", "held_out"]);
    assert_eq!(inputs("eval"), ["dataset"]);
    assert_eq!(inputs("calibrate"), ["validation"]);
    assert!(params("eval").contains(&"horizons") && params("calibrate").contains(&"horizons"));

    // A served surface names no host path: the weights, the training output
    // and the calibration's destination are the host's.
    let served = manifest_resident();
    for a in &served.actions {
        for p in &a.params {
            assert!(!["weights", "out", "force"].contains(&p.name.as_str()), "{} serves {}", a.name, p.name);
        }
    }
    assert_eq!(served.actions.len(), 4);
}
