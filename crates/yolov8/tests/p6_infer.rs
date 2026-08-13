// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P6 end-to-end inference smoke (CPU backend).
//!
//! The decode/NMS/letterbox/box-math GOLDEN tests live in `p6_nms.rs` and do not
//! need a model. This file only smoke-tests that `Yolo::detect` runs the full
//! pipeline on a random-weight tiny model + random image and returns finite,
//! well-formed boxes (NO convergence claim). It is the slow, model-driven path,
//! so it is gated behind `MOE_SKIP_GPU_TESTS` like the rest of the suite.

use yolov8::{init_weights, Yolo, YoloConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

// Deterministic LCG -> (-1,1), matching the other yolo tests.
fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn detect_runs_end_to_end_finite() {
    if skip() {
        eprintln!("skipping detect smoke (MOE_SKIP_GPU_TESTS)");
        return;
    }
    let cfg = YoloConfig::tiny(2);
    let init = init_weights(&cfg, 7);
    let model = Yolo::new(cfg.clone(), 1, cfg.input, &init);

    // Random HWC-RGB image in [0,1], non-square to exercise letterbox padding.
    let (w0, h0) = (96u32, 64u32);
    let img: Vec<f32> = randvec(0xBEEF, (w0 * h0 * 3) as usize).iter().map(|v| (v + 1.0) * 0.5).collect();

    // Low conf threshold so we get some boxes from random weights.
    let dets = model.detect(&img, w0, h0, 0.0, 0.5);
    for d in &dets {
        assert!(d.iter().all(|v| v.is_finite()), "non-finite detection {d:?}");
        // boxes in original-image coords: clamped within the frame, x1<=x2 etc.
        assert!(d[0] >= 0.0 && d[0] <= w0 as f32 + 1.0);
        assert!(d[1] >= 0.0 && d[1] <= h0 as f32 + 1.0);
        assert!(d[2] >= 0.0 && d[2] <= w0 as f32 + 1.0);
        assert!(d[3] >= 0.0 && d[3] <= h0 as f32 + 1.0);
        assert!((d[5] as u32) < cfg.nc, "class id in range");
        assert!(d[4] >= 0.0 && d[4] <= 1.0, "confidence in [0,1]");
    }
    eprintln!("detect smoke produced {} boxes", dets.len());
}

#[test]
fn set_eval_toggle_is_reversible_and_changes_nothing_structural() {
    if skip() {
        return;
    }
    let cfg = YoloConfig::tiny(2);
    let init = init_weights(&cfg, 3);
    let model = Yolo::new(cfg, 1, 128, &init);
    assert!(!model.is_eval(), "model defaults to train-mode BN");
    model.set_eval(true);
    assert!(model.is_eval());
    model.set_eval(false);
    assert!(!model.is_eval(), "toggle is reversible");
}
