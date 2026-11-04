// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec for the dataset-level detection report (mAP@0.5, mAP@0.5:0.95,
//! per-class AP, JSONL prediction dump) on a tiny dataset with KNOWN boxes.

use eval::detection::GtBox;
use eval::detection_report::{read_jsonl, score, write_jsonl, ImageRecord};

fn gt(class: u32, bbox: [f32; 4]) -> GtBox {
    GtBox { class, bbox }
}

/// Two images, two classes. Every box is 100x100; `shift_x` moves each
/// prediction horizontally so IoU = (100 - s) / (100 + s).
fn dataset(shift_x: f32) -> Vec<ImageRecord> {
    let boxes = [(0, [10.0, 10.0, 110.0, 110.0]), (1, [20.0, 30.0, 120.0, 130.0])];
    boxes
        .iter()
        .enumerate()
        .map(|(index, &(class, b))| ImageRecord {
            index,
            gts: vec![gt(class, b)],
            preds: vec![[b[0] + shift_x, b[1], b[2] + shift_x, b[3], 0.9, class as f32]],
        })
        .collect()
}

#[test]
fn perfect_predictions_score_one_on_both_metrics() {
    let r = score(&dataset(0.0), 2);
    assert!((r.map50 - 1.0).abs() < 1e-6, "{r:?}");
    assert!((r.map50_95 - 1.0).abs() < 1e-6, "{r:?}");
    assert_eq!(r.per_class.len(), 2);
    for c in &r.per_class {
        assert!((c.ap50 - 1.0).abs() < 1e-6 && (c.ap50_95 - 1.0).abs() < 1e-6, "{c:?}");
    }
    assert_eq!((r.n_images, r.n_preds, r.n_gts), (2, 2, 2));
}

#[test]
fn shifted_predictions_keep_map50_and_lose_the_strict_thresholds() {
    // s = 15 -> IoU = 85/115 = 0.7391: a true positive at thresholds
    // 0.50, 0.55, 0.60, 0.65, 0.70 (5 of the 10), a miss at 0.75..0.95.
    let r = score(&dataset(15.0), 2);
    assert!((r.map50 - 1.0).abs() < 1e-6, "{r:?}");
    assert!((r.map50_95 - 0.5).abs() < 1e-6, "{r:?}");
    for c in &r.per_class {
        assert!((c.ap50_95 - 0.5).abs() < 1e-6, "{c:?}");
    }
}

#[test]
fn images_never_match_each_other_even_with_overlapping_coordinates() {
    // Same box in both images; only image 1 has a prediction. Scored jointly,
    // it must not satisfy image 0's ground truth.
    let b = [0.0, 0.0, 50.0, 50.0];
    let imgs = vec![
        ImageRecord { index: 0, gts: vec![gt(0, b)], preds: vec![] },
        ImageRecord { index: 1, gts: vec![gt(0, b)], preds: vec![[0.0, 0.0, 50.0, 50.0, 0.8, 0.0]] },
    ];
    let r = score(&imgs, 1);
    assert!((r.map50 - 0.5).abs() < 1e-6, "{r:?}");
    assert!((r.recall50 - 0.5).abs() < 1e-6, "{r:?}");
}

#[test]
fn jsonl_round_trips_one_line_per_image() {
    let imgs = dataset(15.0);
    let mut buf = Vec::new();
    write_jsonl(&mut buf, &imgs).unwrap();
    let text = String::from_utf8(buf.clone()).unwrap();
    assert_eq!(text.lines().count(), imgs.len());
    let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(first["image"], 0);
    assert_eq!(first["gts"][0]["class"], 0);
    assert!(first["preds"][0]["score"].is_number());
    assert_eq!(first["preds"][0]["xyxy"].as_array().unwrap().len(), 4);

    let back = read_jsonl(buf.as_slice()).unwrap();
    assert_eq!(back.len(), imgs.len());
    for (a, b) in imgs.iter().zip(&back) {
        assert_eq!(a.index, b.index);
        assert_eq!(a.preds, b.preds);
        assert_eq!(a.gts.len(), b.gts.len());
        assert_eq!((a.gts[0].class, a.gts[0].bbox), (b.gts[0].class, b.gts[0].bbox));
    }
    // Offline rescoring from the dump reproduces the in-process report.
    assert_eq!(score(&back, 2).map50_95, score(&imgs, 2).map50_95);
}

#[test]
fn malformed_jsonl_is_an_error_not_a_silent_skip() {
    assert!(read_jsonl(&b"{\"image\":0,\"gts\":[],\"preds\":[}\n"[..]).is_err());
}
