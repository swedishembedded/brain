// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P6 detection-evaluator GOLDEN tests (model-free).
//!
//! The user's spec: the evaluator itself must be golden-tested before any mAP is
//! trusted. Each case is constructed input through the pure-Rust metrics
//! (IoU / matching / precision / recall / AP@0.5 / mAP).

use eval::detection::{ap_for_class, map50, match_dets, pairwise_iou, precision_recall, GtBox, Match};

fn det(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32, cls: f32) -> [f32; 6] {
    [x1, y1, x2, y2, conf, cls]
}
fn gt(class: u32, b: [f32; 4]) -> GtBox {
    GtBox { class, bbox: b }
}

#[test]
fn iou_one_seventh_via_evaluator() {
    let a = [0.0, 0.0, 10.0, 10.0];
    let b = [5.0, 5.0, 15.0, 15.0];
    assert!((pairwise_iou(a, b) - 1.0 / 7.0).abs() < 1e-6);
}

#[test]
fn perfect_predictions_give_p_r_ap_one() {
    // predictions == labels (conf 1, exact boxes, correct classes).
    let gts = vec![gt(0, [0.0, 0.0, 10.0, 10.0]), gt(1, [20.0, 20.0, 30.0, 30.0])];
    let preds = vec![det(0.0, 0.0, 10.0, 10.0, 1.0, 0.0), det(20.0, 20.0, 30.0, 30.0, 1.0, 1.0)];
    let (p, r) = precision_recall(&preds, &gts, 0.5);
    assert!((p - 1.0).abs() < 1e-6, "precision {p}");
    assert!((r - 1.0).abs() < 1e-6, "recall {r}");
    assert!((map50(&preds, &gts, 2) - 1.0).abs() < 1e-6, "AP50 must be 1");
}

#[test]
fn empty_predictions_with_labels_recall_zero_ap_zero() {
    let gts = vec![gt(0, [0.0, 0.0, 10.0, 10.0])];
    let preds: Vec<[f32; 6]> = vec![];
    let (p, r) = precision_recall(&preds, &gts, 0.5);
    assert_eq!(r, 0.0, "no predictions -> recall 0");
    assert_eq!(p, 0.0, "no predictions -> precision defined 0");
    // class 0 has GT but no preds -> AP 0 (included in the mean).
    assert_eq!(ap_for_class(&preds, &gts, 0, 0.5), Some(0.0));
    assert_eq!(map50(&preds, &gts, 2), 0.0, "no crash, AP 0");
}

#[test]
fn predictions_on_empty_image_all_fp() {
    let gts: Vec<GtBox> = vec![];
    let preds = vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0.0), det(5.0, 5.0, 9.0, 9.0, 0.8, 0.0)];
    let matches = match_dets(&preds, &gts, 0.5);
    assert!(matches.iter().all(|(_, m)| *m == Match::Fp), "all predictions are FP on an empty image");
    let (p, r) = precision_recall(&preds, &gts, 0.5);
    assert_eq!(p, 0.0, "all FP -> precision 0");
    assert_eq!(r, 0.0, "no GT -> recall 0");
    // No class has GT -> mAP averaging excludes all classes -> 0.
    assert_eq!(map50(&preds, &gts, 2), 0.0);
}

#[test]
fn wrong_class_correct_box_no_tp() {
    // Box overlaps the GT perfectly but the predicted CLASS is wrong.
    let gts = vec![gt(0, [0.0, 0.0, 10.0, 10.0])];
    let preds = vec![det(0.0, 0.0, 10.0, 10.0, 0.95, 1.0)]; // class 1, GT is class 0
    let matches = match_dets(&preds, &gts, 0.5);
    assert_eq!(matches[0].1, Match::Fp, "wrong-class prediction is not a TP");
    // class 0 (has GT, no correct pred) -> AP 0; class 1 (no GT) -> excluded.
    assert_eq!(ap_for_class(&preds, &gts, 0, 0.5), Some(0.0));
    assert_eq!(ap_for_class(&preds, &gts, 1, 0.5), None, "class with no GT is excluded");
    assert_eq!(map50(&preds, &gts, 2), 0.0);
}

#[test]
fn confidence_order_changes_ap() {
    // One GT. Two predictions on it: one true positive (well-overlapping) and one
    // false positive (disjoint). AP must be HIGHER when the TP outranks the FP.
    let gts = vec![gt(0, [0.0, 0.0, 10.0, 10.0])];

    // TP ranked first (higher conf) -> AP = 1.0 (recall reaches 1 at precision 1).
    let tp_first = vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0.0), det(100.0, 100.0, 110.0, 110.0, 0.8, 0.0)];
    let ap_good = ap_for_class(&tp_first, &gts, 0, 0.5).unwrap();

    // FP ranked first -> precision at the recall=1 point drops to 0.5 -> AP = 0.5.
    let fp_first = vec![det(100.0, 100.0, 110.0, 110.0, 0.9, 0.0), det(0.0, 0.0, 10.0, 10.0, 0.8, 0.0)];
    let ap_bad = ap_for_class(&fp_first, &gts, 0, 0.5).unwrap();

    assert!((ap_good - 1.0).abs() < 1e-6, "TP-first AP should be 1, got {ap_good}");
    assert!((ap_bad - 0.5).abs() < 1e-6, "FP-first AP should be 0.5, got {ap_bad}");
    assert!(ap_good > ap_bad, "ranking a FP before the TP must lower AP");
}

#[test]
fn duplicate_detection_is_fp() {
    // Two predictions both hitting the single GT: highest-conf is TP, the second
    // is a FP (GT already matched).
    let gts = vec![gt(0, [0.0, 0.0, 10.0, 10.0])];
    let preds = vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0.0), det(1.0, 1.0, 11.0, 11.0, 0.8, 0.0)];
    let matches = match_dets(&preds, &gts, 0.5);
    let tps = matches.iter().filter(|(_, m)| matches!(m, Match::Tp(_))).count();
    assert_eq!(tps, 1, "only one of the duplicates is a TP");
    let (p, _) = precision_recall(&preds, &gts, 0.5);
    assert!((p - 0.5).abs() < 1e-6, "1 TP / 2 preds -> precision 0.5");
}

#[test]
fn map_averages_only_classes_with_gt() {
    // class 0: perfect (AP 1). class 1: has GT but no pred (AP 0). class 2: no GT
    // (excluded). mean over {0,1} = 0.5.
    let gts = vec![gt(0, [0.0, 0.0, 10.0, 10.0]), gt(1, [20.0, 20.0, 30.0, 30.0])];
    let preds = vec![det(0.0, 0.0, 10.0, 10.0, 1.0, 0.0)];
    let m = map50(&preds, &gts, 3);
    assert!((m - 0.5).abs() < 1e-6, "mAP over present classes {{0,1}} = 0.5, got {m}");
}
