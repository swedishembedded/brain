// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The viewer estimator: steadier than its input when still, close behind a
//! moving head once predicted, and no frame ever jumps - not on loss, not on
//! reacquisition, not on an outlier.

use data::rng::Lcg;
use headpose::{EstimatorConfig, Rejected, TrackingStatus, ViewerEstimator};
use sfm::linalg::{norm, sub};

const CAMERA_HZ: f64 = 30.0;
const DISPLAY_HZ: f64 = 60.0;
const HALF_IPD: f64 = 0.0315;

fn eyes(c: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    ([c[0] - HALF_IPD, c[1], c[2]], [c[0] + HALF_IPD, c[1], c[2]])
}

fn feed(est: &mut ViewerEstimator, t: f64, c: [f64; 3]) {
    let (r, l) = eyes(c);
    est.observe(t, r, l, 0.9, 1.0).unwrap();
}

/// The steepest per-frame step a crossfade of `distance` metres may take:
/// smoothstep's peak slope is 1.5 over the blend.
fn blend_step(distance: f64) -> f64 {
    1.05 * 1.5 * distance / (EstimatorConfig::default().blend_s * DISPLAY_HZ)
}

/// The largest move of the rendered eye between consecutive display frames
/// over `[t0, t1)`.
fn max_step(est: &ViewerEstimator, t0: f64, t1: f64) -> f64 {
    let mut prev = est.predict(t0).cyclopean();
    let mut worst: f64 = 0.0;
    let mut t = t0;
    while t < t1 {
        t += 1.0 / DISPLAY_HZ;
        let p = est.predict(t).cyclopean();
        worst = worst.max(norm(sub(p, prev)));
        prev = p;
    }
    worst
}

#[test]
fn a_still_head_shimmers_far_less_than_its_measurements() {
    let mut est = ViewerEstimator::new(EstimatorConfig::default());
    let mut rng = Lcg::new(11);
    let truth = [0.02, 0.05, 0.55];
    let sigma = 0.003;
    let mut se = 0.0;
    let mut n = 0.0;
    for k in 0..300 {
        let t = k as f64 / CAMERA_HZ;
        let noisy = [truth[0] + sigma * rng.signed() as f64 * 1.732, truth[1] + sigma * rng.signed() as f64 * 1.732, truth[2] + sigma * rng.signed() as f64 * 1.732];
        feed(&mut est, t, noisy);
        if k > 60 {
            let e = sub(est.predict(t + 0.03).cyclopean(), truth);
            se += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            n += 3.0;
        }
    }
    let rms = (se / n).sqrt();
    // a single-pole smoother at the rest cutoff alone reaches ~0.3 sigma at
    // this rate; prediction and the speed-dependent cutoff may cost a little
    assert!(rms < 0.5 * sigma, "output RMS {:.2} mm against input {:.2} mm", rms * 1e3, sigma * 1e3);
}

#[test]
fn prediction_keeps_a_moving_head_within_a_few_millimetres() {
    // perf-number: a synthetic scenario's latency, not a measurement
    // 0.3 m/s sideways, measured at 30 Hz, shown 30 ms after each capture:
    // without prediction the eye trails by 9 mm plus filter lag.
    let mut est = ViewerEstimator::new(EstimatorConfig::default());
    let v = 0.3;
    let mut worst: f64 = 0.0;
    for k in 0..90 {
        let t = k as f64 / CAMERA_HZ;
        feed(&mut est, t, [-0.15 + v * t, 0.0, 0.6]);
        if k > 30 {
            let shown = t + 0.03;
            let e = (est.predict(shown).cyclopean()[0] - (-0.15 + v * shown)).abs();
            worst = worst.max(e);
        }
    }
    assert!(worst < 0.006, "trails by {:.1} mm", worst * 1e3);
}

#[test]
fn losing_the_viewer_holds_then_eases_to_rest_without_a_jump() {
    let cfg = EstimatorConfig::default();
    let mut est = ViewerEstimator::new(cfg);
    for k in 0..30 {
        feed(&mut est, k as f64 / CAMERA_HZ, [0.12, -0.04, 0.5]);
    }
    let last = 29.0 / CAMERA_HZ;
    assert_eq!(est.predict(last + 0.01).status, TrackingStatus::Tracking);
    assert_eq!(est.predict(last + cfg.stale_s + 0.01).status, TrackingStatus::Coasting);
    assert_eq!(est.predict(last + cfg.lost_s + 0.01).status, TrackingStatus::Lost);
    let far = est.predict(last + cfg.lost_s + 6.0 * cfg.return_tau_s).cyclopean();
    assert!(norm(sub(far, cfg.rest_eye)) < 0.002, "eases to rest, got {far:?}");
    assert!(max_step(&est, last, last + 5.0) < 0.004, "no frame jumps while losing the viewer");
}

#[test]
fn reacquiring_crossfades_from_what_was_on_screen() {
    let cfg = EstimatorConfig::default();
    let mut est = ViewerEstimator::new(cfg);
    for k in 0..30 {
        feed(&mut est, k as f64 / CAMERA_HZ, [-0.1, 0.0, 0.6]);
    }
    // gone for two seconds, back somewhere else
    let t = 29.0 / CAMERA_HZ + 2.0;
    let before = est.predict(t).cyclopean();
    feed(&mut est, t, [0.15, 0.05, 0.45]);
    let at = est.predict(t).cyclopean();
    assert!(norm(sub(at, before)) < 1e-9, "the first frame after reacquisition is the frame before it");
    for k in 1..12 {
        feed(&mut est, t + k as f64 / CAMERA_HZ, [0.15, 0.05, 0.45]);
    }
    let settled = est.predict(t + cfg.blend_s + 0.2).cyclopean();
    assert!(norm(sub(settled, [0.15, 0.05, 0.45])) < 0.002, "and then arrives: {settled:?}");
    assert!(max_step(&est, t - 0.1, t + 0.5) < blend_step(norm(sub([0.15, 0.05, 0.45], before))), "spread over the whole blend");
}

#[test]
fn a_one_frame_outlier_never_reaches_the_screen_and_a_confirmed_move_is_blended() {
    let mut est = ViewerEstimator::new(EstimatorConfig::default());
    for k in 0..30 {
        feed(&mut est, k as f64 / CAMERA_HZ, [0.0, 0.0, 0.6]);
    }
    let t = 30.0 / CAMERA_HZ;
    let (r, l) = eyes([0.3, 0.0, 0.6]);
    assert_eq!(est.observe(t, r, l, 0.9, 1.0), Err(Rejected::Unconfirmed));
    feed(&mut est, t + 1.0 / CAMERA_HZ, [0.0, 0.0, 0.6]);
    assert!(norm(sub(est.predict(t + 0.1).cyclopean(), [0.0, 0.0, 0.6])) < 1e-3, "the glitch left no trace");

    // a viewer who really moved is confirmed by the next frame
    let t = t + 2.0 / CAMERA_HZ;
    let before = est.predict(t + 1.0 / CAMERA_HZ).cyclopean();
    let (r, l) = eyes([0.3, 0.0, 0.6]);
    assert_eq!(est.observe(t, r, l, 0.9, 1.0), Err(Rejected::Unconfirmed));
    feed(&mut est, t + 1.0 / CAMERA_HZ, [0.3, 0.0, 0.6]);
    assert!(norm(sub(est.predict(t + 1.0 / CAMERA_HZ).cyclopean(), before)) < 1e-6, "accepted without a jump");
    for k in 2..15 {
        feed(&mut est, t + k as f64 / CAMERA_HZ, [0.3, 0.0, 0.6]);
    }
    assert!(max_step(&est, t, t + 0.4) < blend_step(0.3));
    assert!(norm(sub(est.predict(t + 0.45).cyclopean(), [0.3, 0.0, 0.6])) < 0.002);
}

#[test]
fn a_head_that_stops_is_not_overshot_by_more_than_a_few_millimetres() {
    let mut est = ViewerEstimator::new(EstimatorConfig::default());
    let v = 0.3;
    let stop = 1.0;
    let mut worst: f64 = 0.0;
    for k in 0..90 {
        let t = k as f64 / CAMERA_HZ;
        let x = -0.15 + v * t.min(stop);
        feed(&mut est, t, [x, 0.0, 0.6]);
        if t >= stop {
            worst = worst.max(est.predict(t + 0.03).cyclopean()[0] - (-0.15 + v * stop));
        }
    }
    // an ideal extrapolator over the same 30 ms overshoots v * lead = 9 mm
    // until it learns the head stopped; the estimator must beat that
    assert!(worst < 0.008, "overshoots by {:.1} mm", worst * 1e3);
}

#[test]
fn out_of_order_unconfident_and_non_finite_measurements_are_refused() {
    let mut est = ViewerEstimator::new(EstimatorConfig::default());
    let (r, l) = eyes([0.0, 0.0, 0.6]);
    assert_eq!(est.predict(0.0).status, TrackingStatus::Searching);
    est.observe(1.0, r, l, 0.9, 1.0).unwrap();
    assert_eq!(est.observe(1.0, r, l, 0.9, 1.0), Err(Rejected::OutOfOrder));
    assert_eq!(est.observe(0.5, r, l, 0.9, 1.0), Err(Rejected::OutOfOrder));
    assert_eq!(est.observe(2.0, r, l, 0.1, 1.0), Err(Rejected::LowConfidence));
    assert_eq!(est.observe(2.0, [f64::NAN, 0.0, 0.6], l, 0.9, 1.0), Err(Rejected::NotFinite));
}
