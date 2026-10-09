// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain yolov8 eval` end to end on a tiny generated dataset: split
//! selection, the full metric table, a low confidence floor, and the
//! prediction dump.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const N_IMAGES: usize = 20;

fn brain(args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_brain"))
        .args(args)
        .env("BRAIN_DEVICE", "cpu")
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
    brain(&["yolov8", "train", &data, "--out", &weights, "--steps", "2", "--batch", "2", "--seed", "7"]);
    (data, weights)
}

fn images_scored(stdout: &str) -> usize {
    let tail = stdout.split("(images ").nth(1).expect("summary line");
    tail.trim_end_matches(|c: char| c == ')' || c.is_whitespace()).parse().expect("image count")
}

#[test]
fn eval_reports_both_maps_per_class_ap_and_dumps_predictions() {
    let dir = scratch("full");
    let (data, weights) = fixture(&dir);
    let dump = dir.join("preds.jsonl");

    let all = brain(&[
        "yolov8", "eval", "--weights", &weights, "--data", &data, "--split", "all", "--conf", "0.001",
        "--dump-preds", dump.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8(all.stdout).unwrap();
    for needle in ["mAP@0.5 ", "mAP@0.5:0.95", "class  AP@0.5  AP@0.5:0.95"] {
        assert!(stdout.contains(needle), "missing {needle:?} in:\n{stdout}");
    }
    assert_eq!(images_scored(&stdout), N_IMAGES);

    let records = eval::detection_report::read_jsonl(std::io::BufReader::new(std::fs::File::open(&dump).unwrap())).unwrap();
    assert_eq!(records.len(), N_IMAGES);
    assert_eq!(records.iter().map(|r| r.index).collect::<Vec<_>>(), (0..N_IMAGES).collect::<Vec<_>>());
    assert!(records.iter().all(|r| !r.gts.is_empty()), "the generator labels every image");

    // The default split is the held-out last 10%.
    let val = brain(&["yolov8", "eval", "--weights", &weights, "--data", &data]);
    assert_eq!(images_scored(&String::from_utf8(val.stdout).unwrap()), N_IMAGES / 10);
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
