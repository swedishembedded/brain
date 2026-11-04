// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P8 name-map sanity gate (no torch / no GPU required).
//!
//! The offline export path (`tools/yolo_export/export_yolov8.py`) maps each
//! Ultralytics `yolov8n.pt` tensor name onto the brain native name registered by
//! [`YoloConfig::full_param_list`]. That Python map is unauditable from Rust, so
//! this test pins down the brain SIDE of the contract: it asserts that
//! `full_param_list()` produces EXACTLY the set of names + per-tensor element
//! counts the export script targets, and that every name follows the documented
//! scheme (`<prefix>.conv.weight` + the 4 BN tensors per Conv, `cv1/cv2/m.{i}`
//! inside C2f/SPPF, `head.{s}.{cls,reg}.{0,1}` Convs + `.2.weight` bias-free 1x1).
//!
//! If the export script's checked-in brain-name dump (regenerated from this
//! test's output) ever drifts from this list, the parity test will load the
//! wrong tensors — so this is the cheap guard that runs in CI.

use std::collections::HashSet;

use yolov8::YoloConfig;

/// Every brain tensor name belongs to one of these documented namespaces, and
/// every Conv contributes exactly the 5-tensor `conv.weight`+4-BN group (or a
/// bare `.2.weight` for the head's final bias-free 1x1).
fn classify(name: &str) -> &'static str {
    if name.ends_with(".conv.weight") {
        "conv_weight"
    } else if name.ends_with(".bn.gamma") {
        "bn_gamma"
    } else if name.ends_with(".bn.beta") {
        "bn_beta"
    } else if name.ends_with(".bn.run_mean") {
        "bn_run_mean"
    } else if name.ends_with(".bn.run_var") {
        "bn_run_var"
    } else if name.ends_with(".2.weight") {
        "head_proj"
    } else if name.ends_with(".2.bias") {
        "head_bias"
    } else {
        "UNKNOWN"
    }
}

#[test]
fn every_name_matches_documented_scheme() {
    let cfg = YoloConfig::yolov8n();
    let plist = cfg.full_param_list();

    // No unknown namespaces.
    let bad: Vec<&String> =
        plist.iter().filter(|(n, _)| classify(n) == "UNKNOWN").map(|(n, _)| n).collect();
    assert!(bad.is_empty(), "tensor names outside the documented scheme: {bad:?}");

    // Names are unique.
    let mut seen = HashSet::new();
    for (n, _) in &plist {
        assert!(seen.insert(n.clone()), "duplicate tensor name: {n}");
    }

    // Every namespace prefix is one of backbone./neck./head.
    for (n, _) in &plist {
        assert!(
            n.starts_with("backbone.") || n.starts_with("neck.") || n.starts_with("head."),
            "unexpected top-level prefix: {n}"
        );
    }
}

#[test]
fn bn_groups_are_complete() {
    // Every `<p>.conv.weight` must have its full set of 4 BN tensors at the same
    // prefix `<p>`. (The export fold rule: brain Conv is bias-free; the 4 BN
    // tensors come straight from Ultralytics `bn.{weight,bias,running_mean,
    // running_var}` 1:1 — no value arithmetic.)
    let cfg = YoloConfig::yolov8n();
    let names: HashSet<String> = cfg.full_param_list().into_iter().map(|(n, _)| n).collect();
    for n in &names {
        if let Some(p) = n.strip_suffix(".conv.weight") {
            for suffix in [".bn.gamma", ".bn.beta", ".bn.run_mean", ".bn.run_var"] {
                let want = format!("{p}{suffix}");
                assert!(names.contains(&want), "Conv {p} missing BN tensor {want}");
            }
        }
    }
}

#[test]
fn yolov8n_tensor_and_param_counts() {
    let cfg = YoloConfig::yolov8n();
    let plist = cfg.full_param_list();

    let n_tensors = plist.len();
    let n_params: usize = plist.iter().map(|(_, c)| *c).sum();

    // Count by class, for a precise structural fingerprint.
    let mut convs = 0usize;
    let mut head_projs = 0usize;
    let mut head_biases = 0usize;
    for (n, _) in &plist {
        match classify(n) {
            "conv_weight" => convs += 1,
            "head_proj" => head_projs += 1,
            "head_bias" => head_biases += 1,
            _ => {}
        }
    }
    // Each full Conv = 5 tensors (1 conv.weight + 4 BN). Each head branch adds a
    // bias-free 1x1 weight + a per-channel bias (P12: the head is biased).
    assert_eq!(
        n_tensors,
        convs * 5 + head_projs + head_biases,
        "tensor count = 5*convs + head_proj_weights + head_biases"
    );

    eprintln!(
        "yolov8n full_param_list: tensors={n_tensors} params={n_params} \
         (full_convs={convs}, head_proj_1x1={head_projs}, head_bias={head_biases})"
    );

    // Pin the fingerprint so accidental graph changes are caught. These numbers
    // are the CANONICAL Ultralytics yolov8n graph (input 640, nc 80, reg_max 16,
    // backbone channels [16,32,32,64,64,128,128,256,256,256], C2f depths
    // [1,2,2,1], neck widths [128,64,64,128,128,256], biased head cls/reg
    // hidden 80/64). 57 full Convs * 5 + 6 head weights + 6 head biases = 297.
    assert_eq!(convs, EXPECTED_FULL_CONVS, "full Conv count drifted");
    assert_eq!(head_projs, 6, "head has 3 scales * (cls+reg) = 6 final 1x1 weight projections");
    assert_eq!(head_biases, 6, "head has 3 scales * (cls+reg) = 6 per-channel biases");
    assert_eq!(n_tensors, EXPECTED_TENSORS, "tensor count drifted");
    assert_eq!(n_params, EXPECTED_PARAMS, "total scalar param count drifted");
}

/// Dump the full ordered name list + element counts. Run with
/// `cargo test -p brain-yolo --test p8_names dump_brain_names -- --nocapture`
/// to regenerate the checked-in `brain_names.txt` the export script verifies
/// against.
#[test]
fn dump_brain_names() {
    let cfg = YoloConfig::yolov8n();
    for (n, c) in cfg.full_param_list() {
        println!("{n} {c}");
    }
}

// --- pinned fingerprint (see test output `eprintln!` to regenerate) ---
// CANONICAL Ultralytics yolov8n: 57 full Convs (each = conv.weight + 4 BN) + 6
// head final-1x1 weights + 6 head per-channel biases
// => 57*5 + 6 + 6 = 297 tensors, 3_167_776 scalar params total (~3.17M, matching
// the official yolov8n parameter count).
const EXPECTED_FULL_CONVS: usize = 57;
const EXPECTED_TENSORS: usize = 297;
const EXPECTED_PARAMS: usize = 3_167_776;
