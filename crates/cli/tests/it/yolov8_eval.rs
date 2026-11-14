// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain yolov8 eval` end to end on a tiny generated dataset: split
//! selection, every metric the action reports, a low confidence floor, and the
//! prediction dump.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const N_IMAGES: usize = 20;

fn brain(args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_brain"))
        .args(args)
        .args(["--device", "cpu"])
        .output()
        .expect("run brain");
    assert!(out.status.success(), "brain {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
    out
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-yolov8-eval-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A dataset plus a barely trained checkpoint: the eval plumbing is under
/// test here, not the detector's quality.
fn fixture(dir: &Path) -> (String, String) {
    let data = dir.join("data").to_string_lossy().into_owned();
    let weights = dir.join("yolo.safetensors").to_string_lossy().into_owned();
    brain(&["data", "gen", "detect", "--out", &data, "--n", &N_IMAGES.to_string(), "--seed", "7"]);
    brain(&["yolov8", "train", "--data", &data, "--save", &weights, "--steps", "2", "--batch", "2", "--seed", "7"]);
    (data, weights)
}

/// The eval action's scalar outputs, as `--json` prints them.
fn report(args: &[&str]) -> serde_json::Value {
    let mut argv = args.to_vec();
    argv.push("--json");
    serde_json::from_slice(&brain(&argv).stdout).expect("eval prints one JSON object")
}

#[test]
fn eval_reports_both_maps_per_class_ap_and_dumps_predictions() {
    let dir = scratch("full");
    let (data, weights) = fixture(&dir);
    let dump = dir.join("preds.jsonl");

    let all = report(&[
        "yolov8", "eval", "--weights", &weights, "--data", &data, "--split", "all", "--conf", "0.001",
        "--dump-preds", dump.to_str().unwrap(),
    ]);
    for key in ["map50", "map50_95", "precision50", "recall50"] {
        assert!(all[key].is_number(), "missing {key} in {all}");
    }
    let per_class = all["per_class"].as_array().expect("per_class");
    assert!(!per_class.is_empty() && per_class.iter().all(|c| c["ap50"].is_number() && c["ap50_95"].is_number()), "{all}");
    assert_eq!(all["images"], N_IMAGES);

    let records = eval::detection_report::read_jsonl(std::io::BufReader::new(std::fs::File::open(&dump).unwrap())).unwrap();
    assert_eq!(records.len(), N_IMAGES);
    assert_eq!(records.iter().map(|r| r.index).collect::<Vec<_>>(), (0..N_IMAGES).collect::<Vec<_>>());
    assert!(records.iter().all(|r| !r.gts.is_empty()), "the generator labels every image");

    // The default split is the held-out last 10%.
    let val = report(&["yolov8", "eval", "--weights", &weights, "--data", &data]);
    assert_eq!(val["images"], N_IMAGES / 10);
}

#[test]
fn eval_rejects_an_unknown_split() {
    let out = Command::new(env!("CARGO_BIN_EXE_brain"))
        .args(["yolov8", "eval", "--weights", "w", "--data", "d", "--split", "train"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--split"));
}
