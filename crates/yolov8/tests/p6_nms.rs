// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P6 NMS + box-math golden tests (model-free).
//!
//! These exercise the user's exact NMS spec cases plus the IoU / DFL-distance /
//! letterbox golden cases that the inference path relies on. No trained model is
//! needed: everything is constructed input through the pure-Rust box math + the
//! `dfl_decode` kernel (CPU backend).

use yolov8::boxmath::{self, dist_to_xyxy, iou, xyxy_to_dist, Letterbox};
use yolov8::infer::dfl_decode_dist;
use yolov8::{nms, nms_agnostic};

// ---------------------------------------------------------------- IoU golden

#[test]
fn iou_identical_disjoint_and_one_seventh() {
    let b = [10.0, 20.0, 50.0, 80.0];
    assert!((iou(b, b) - 1.0).abs() < 1e-6, "identical IoU must be 1");

    let disjoint_a = [0.0, 0.0, 10.0, 10.0];
    let disjoint_b = [100.0, 100.0, 110.0, 110.0];
    assert!(iou(disjoint_a, disjoint_b).abs() < 1e-9, "disjoint IoU must be 0");

    // [0,0,10,10] vs [5,5,15,15]: inter 25, union 175 -> 1/7.
    let a = [0.0, 0.0, 10.0, 10.0];
    let c = [5.0, 5.0, 15.0, 15.0];
    assert!((iou(a, c) - 1.0 / 7.0).abs() < 1e-6, "expected 1/7, got {}", iou(a, c));
}

// -------------------------------------------------- DFL decode round trip

#[test]
fn dfl_decode_target_3_25() {
    // target 3.25 expressed as bins 3:0.75, 4:0.25 (a two-spike distribution
    // whose expectation E = 3*0.75 + 4*0.25 = 3.25). We feed LOGITS whose softmax
    // is exactly that distribution: pick logits so exp(l3)/Z = .75, exp(l4)/Z=.25,
    // all else 0 prob. Use l3 = ln 0.75, l4 = ln 0.25, others = -inf-ish.
    let reg_max = 8usize;
    let mut logits = vec![-1e4f32; reg_max]; // one (anchor,side)
    logits[3] = 0.75f32.ln();
    logits[4] = 0.25f32.ln();
    let gpu = gpu_core::Gpu::new_cpu(yolov8::net::PIPELINES);
    // one decode "anchor" with 4 sides -> replicate the same dist on all 4 sides.
    let mut buf = Vec::new();
    for _ in 0..4 {
        buf.extend_from_slice(&logits);
    }
    let dist = dfl_decode_dist(&gpu, &buf, 1, reg_max);
    for side in 0..4 {
        assert!((dist[side] - 3.25).abs() < 1e-4, "side {side}: {} != 3.25", dist[side]);
    }
}

#[test]
fn dist_box_round_trip_known() {
    // anchor (ax,ay) = (50,60) in feature units, stride 1 -> distances are pixels.
    // ltrb [10,20,30,40] -> box [40,40,80,100] and back.
    let (ax, ay, s) = (50.0f32, 60.0f32, 1.0f32);
    let dist = [10.0f32, 20.0, 30.0, 40.0];
    let b = dist_to_xyxy(dist, ax, ay, s);
    assert_eq!(b, [40.0, 40.0, 80.0, 100.0]);
    let back = xyxy_to_dist(b, ax, ay, s);
    assert_eq!(back, dist);
}

// ---------------------------------------------------- Letterbox golden

#[test]
fn letterbox_recovers_rect_within_1px() {
    // square, wide, tall — recover a rectangle within <= 1px.
    for &(w0, h0) in &[(100u32, 100u32), (200, 100), (100, 200)] {
        let lb = Letterbox::compute(w0, h0, 128);
        let rect = [10.0, 20.0, (w0 as f32) * 0.6, (h0 as f32) * 0.7];
        let back = lb.invert_box(lb.apply_box(rect), w0, h0);
        for k in 0..4 {
            assert!(
                (back[k] - rect[k]).abs() <= 1.0,
                "({w0}x{h0}) side {k}: {} vs {}",
                back[k],
                rect[k]
            );
        }
    }
}

#[test]
fn letterbox_rgb_layout() {
    // 4x2 RGB -> 8x8 input; finite, correct length, pad value present.
    let src: Vec<f32> = (0..4 * 2 * 3).map(|i| (i as f32) / 100.0).collect();
    let (chw, lb) = boxmath::letterbox_rgb(&src, 4, 2, 8, 0.5);
    assert_eq!(chw.len(), 3 * 8 * 8);
    assert!(chw.iter().all(|v| v.is_finite()));
    assert!((lb.scale - 2.0).abs() < 1e-6); // width 4 -> 8 fills
}

// ----------------------------------------------------------- NMS golden

fn d(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32, cls: f32) -> [f32; 6] {
    [x1, y1, x2, y2, conf, cls]
}

#[test]
fn nms_two_identical_same_class_keep_highest() {
    let dets = vec![d(0.0, 0.0, 10.0, 10.0, 0.9, 0.0), d(0.0, 0.0, 10.0, 10.0, 0.8, 0.0)];
    let out = nms(&dets, 0.5, 100);
    assert_eq!(out.len(), 1);
    assert!((out[0][4] - 0.9).abs() < 1e-6, "must keep the 0.9 box");
}

#[test]
fn nms_overlapping_different_class_aware_keeps_both() {
    let dets = vec![d(0.0, 0.0, 10.0, 10.0, 0.9, 0.0), d(0.0, 0.0, 10.0, 10.0, 0.8, 1.0)];
    assert_eq!(nms(&dets, 0.5, 100).len(), 2, "class-aware keeps both classes");
}

#[test]
fn nms_overlapping_agnostic_suppresses_lower() {
    let dets = vec![d(0.0, 0.0, 10.0, 10.0, 0.9, 0.0), d(0.0, 0.0, 10.0, 10.0, 0.8, 1.0)];
    let out = nms_agnostic(&dets, 0.5, 100);
    assert_eq!(out.len(), 1);
    assert!((out[0][4] - 0.9).abs() < 1e-6);
}

#[test]
fn nms_non_overlapping_keeps_both() {
    let dets = vec![d(0.0, 0.0, 10.0, 10.0, 0.9, 0.0), d(100.0, 100.0, 110.0, 110.0, 0.8, 0.0)];
    assert_eq!(nms(&dets, 0.5, 100).len(), 2);
}

#[test]
fn nms_threshold_boundary_strict_gt() {
    // Two boxes with IoU exactly 1/7. With iou_thresh = 1/7 the comparison is
    // STRICT `>`, so 1/7 is NOT > 1/7 -> both survive. Just above (1/7 - eps)
    // -> suppressed.
    let a = d(0.0, 0.0, 10.0, 10.0, 0.9, 0.0);
    let b = d(5.0, 5.0, 15.0, 15.0, 0.8, 0.0); // IoU(a,b) = 1/7
    let exactly = nms(&[a, b], 1.0 / 7.0, 100);
    assert_eq!(exactly.len(), 2, "IoU exactly at threshold survives (strict >)");
    let below = nms(&[a, b], 1.0 / 7.0 - 0.01, 100);
    assert_eq!(below.len(), 1, "threshold just below IoU suppresses the lower box");
}

#[test]
fn nms_respects_max_det() {
    let dets: Vec<[f32; 6]> = (0..10)
        .map(|i| d(i as f32 * 100.0, 0.0, i as f32 * 100.0 + 10.0, 10.0, 1.0 - i as f32 * 0.05, 0.0))
        .collect();
    let out = nms(&dets, 0.5, 3);
    assert_eq!(out.len(), 3, "max_det caps the kept count");
}
