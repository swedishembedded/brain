// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rigid head fit on synthetic faces seen through a known lens: exact
//! without noise, within the eye-position budget with landmark noise, and
//! rotation is not mistaken for translation.

use data::rng::Lcg;
use headpose::{fit_head, FaceModel, FitError};
use sfm::camera::Pose;
use sfm::linalg::{norm, sub};

use crate::{rotation, webcam};

const N: usize = 66;

/// 66 landmarks of `model` posed at `pose`, plus pixel noise of `sigma`. The
/// indices the model does not use get a finite placeholder and confidence 0.
fn landmarks(model: &FaceModel, pose: &Pose, sigma: f64, rng: &mut Lcg) -> (Vec<[f64; 2]>, Vec<f64>) {
    let k = webcam();
    let mut pts = vec![[320.0, 240.0]; N];
    let mut conf = vec![0.0; N];
    for &(i, p) in &model.points {
        let q = k.project(pose.to_cam(p)).expect("synthetic face is in view");
        pts[i] = [q[0] + sigma * rng.signed() as f64 * 1.7320508, q[1] + sigma * rng.signed() as f64 * 1.7320508];
        conf[i] = 0.9;
    }
    (pts, conf)
}

fn poses() -> Vec<Pose> {
    let mut out = Vec::new();
    for &(x, y, z) in &[(0.0, 0.0, 0.6), (0.15, -0.05, 0.45), (-0.12, 0.08, 0.9), (0.05, 0.1, 0.35)] {
        for &(yaw, pitch, roll) in &[(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (-35.0, 10.0, 5.0), (10.0, -20.0, -12.0)] {
            out.push(Pose { r: rotation(yaw, pitch, roll), t: [x, y, z] });
        }
    }
    out
}

#[test]
fn noise_free_landmarks_give_the_exact_eyes_from_a_cold_start() {
    let model = FaceModel::canonical(0.063);
    let mut rng = Lcg::new(1);
    for pose in poses() {
        let (pts, conf) = landmarks(&model, &pose, 0.0, &mut rng);
        let fit = fit_head(&pts, &conf, &webcam(), &model, None, 0.5, 2.0).unwrap();
        let err = norm(sub(fit.right_eye, pose.to_cam(model.right_eye))).max(norm(sub(fit.left_eye, pose.to_cam(model.left_eye))));
        assert!(err < 1e-6, "pose {:?}: eye error {err:e} m", pose.t);
        assert!(fit.rmse_px < 1e-4);
    }
}

#[test]
fn half_pixel_landmark_noise_stays_inside_the_eye_position_budget() {
    // The budget: lateral error is what the viewer sees as the world
    // swimming sideways; depth error mostly rescales the parallax. Measured
    // over many noisy frames, so the bound is on the error distribution, not
    // on one lucky draw.
    let model = FaceModel::canonical(0.063);
    let mut rng = Lcg::new(7);
    let (mut lat2, mut dep2, mut n) = (0.0, 0.0, 0.0);
    for pose in poses().into_iter().filter(|p| p.t[2] <= 0.6) {
        for _ in 0..25 {
            let (pts, conf) = landmarks(&model, &pose, 0.5, &mut rng);
            let fit = fit_head(&pts, &conf, &webcam(), &model, None, 0.5, 4.0).unwrap();
            let e = sub(fit.cyclopean_eye(), pose.t);
            lat2 += e[0] * e[0] + e[1] * e[1];
            dep2 += e[2] * e[2];
            n += 1.0;
        }
    }
    let (lat, dep) = ((lat2 / n).sqrt(), (dep2 / n).sqrt());
    assert!(lat < 0.005, "lateral RMS {:.1} mm", lat * 1e3);
    assert!(dep < 0.020, "depth RMS {:.1} mm", dep * 1e3);
}

#[test]
fn turning_the_head_in_place_does_not_move_the_eyes_sideways() {
    // The head pivots about its own origin here, so the true cyclopean eye
    // never moves: anything the fit reports is false translation.
    let model = FaceModel::canonical(0.063);
    let mut rng = Lcg::new(3);
    let mut prior = None;
    for yaw in (-40..=40).step_by(5) {
        let pose = Pose { r: rotation(yaw as f64, 0.0, 0.0), t: [0.02, 0.01, 0.55] };
        let (pts, conf) = landmarks(&model, &pose, 0.0, &mut rng);
        let fit = fit_head(&pts, &conf, &webcam(), &model, prior.as_ref(), 0.5, 2.0).unwrap();
        let drift = norm(sub(fit.cyclopean_eye(), pose.t));
        assert!(drift < 1e-6, "yaw {yaw}: eye drifted {drift:e} m");
        prior = Some(fit.head_to_camera);
    }
}

#[test]
fn unusable_landmarks_are_refused_not_fitted() {
    let model = FaceModel::canonical(0.063);
    let mut rng = Lcg::new(5);
    let pose = Pose { r: rotation(0.0, 0.0, 0.0), t: [0.0, 0.0, 0.6] };
    let (pts, mut conf) = landmarks(&model, &pose, 0.0, &mut rng);
    // too short
    assert!(matches!(fit_head(&pts[..30], &conf[..30], &webcam(), &model, None, 0.5, 2.0), Err(FitError::BadLandmarks(_))));
    // NaN
    let mut bad = pts.clone();
    bad[30] = [f64::NAN, 0.0];
    assert!(matches!(fit_head(&bad, &conf, &webcam(), &model, None, 0.5, 2.0), Err(FitError::BadLandmarks(_))));
    // scrambled: a pose cannot explain it
    let mut scrambled = pts.clone();
    for &(i, _) in &model.points {
        scrambled[i] = [100.0 + 37.0 * (i % 7) as f64, 400.0 - 23.0 * (i % 5) as f64];
    }
    assert!(fit_head(&scrambled, &conf, &webcam(), &model, None, 0.5, 2.0).is_err());
    // unconfident
    for c in conf.iter_mut() {
        *c = 0.1;
    }
    assert!(matches!(fit_head(&pts, &conf, &webcam(), &model, None, 0.5, 2.0), Err(FitError::TooFewPoints { .. })));
}
