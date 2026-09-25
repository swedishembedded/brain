// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The geometric core of structure from motion, on exact synthetic
//! correspondences with outliers: relative pose from an essential matrix,
//! absolute pose from 2D-3D matches, and bundle adjustment recovering a
//! focal length and lens distortion it was not told.
//!
//! Swedish Embedded AB implements camera calibration and multi-view
//! reconstruction for its clients. If your team needs photographs turned into
//! calibrated cameras and geometry, you can procure our services by sending
//! an email to info@swedishembedded.com.

use data::rng::Lcg;
use sfm::ba::{bundle_adjust, rms, BaCfg, Observation};
use sfm::camera::{project, Intrinsics, Pose};
use sfm::linalg::{exp_so3, log_so3, mm, normalize, sub, transpose, V3};
use sfm::pnp::{p3p, ransac_pnp};
use sfm::twoview::{ransac_essential, relative_pose};

fn cloud(rng: &mut Lcg, n: usize) -> Vec<V3> {
    (0..n).map(|_| [2.0 * rng.signed() as f64, 1.5 * rng.signed() as f64, 5.0 + 1.5 * rng.signed() as f64]).collect()
}

fn pose(w: V3, t: V3) -> Pose {
    Pose { r: exp_so3(w), t }
}

fn angle_between(a: &Pose, b: &Pose) -> f64 {
    let d = mm(&a.r, &transpose(&b.r));
    let l = log_so3(&d);
    (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt()
}

/// 30% of the correspondences are garbage; the relative pose still comes
/// back to within a tenth of a degree (a few garbage points fall inside the
/// epipolar band by chance and join the refit) and the translation direction
/// to within a few tenths.
#[test]
fn relative_pose_survives_outliers() {
    let mut rng = Lcg::new(3);
    let x = cloud(&mut rng, 300);
    let second = pose([0.05, -0.2, 0.03], [-1.0, 0.1, 0.2]);
    let first = Pose::identity();
    let proj = |p: &Pose, x: V3| {
        let c = p.to_cam(x);
        [c[0] / c[2], c[1] / c[2]]
    };
    let mut a: Vec<[f64; 2]> = x.iter().map(|v| proj(&first, *v)).collect();
    let b: Vec<[f64; 2]> = x.iter().map(|v| proj(&second, *v)).collect();
    for p in a.iter_mut().take(90) {
        *p = [0.5 * rng.signed() as f64, 0.4 * rng.signed() as f64];
    }
    let (e, mask) = ransac_essential(&a, &b, 1e-3, 500, 1).unwrap();
    let inl = mask.iter().filter(|&&v| v).count();
    assert!((205..=215).contains(&inl), "{inl} inliers of 210 true ones");
    let (p, _) = relative_pose(&e, &a, &b, &mask);
    assert!(angle_between(&p, &second).to_degrees() < 0.1, "rotation off by {} deg", angle_between(&p, &second).to_degrees());
    let dt = normalize(p.t);
    let want = normalize(second.t);
    let cosang = dt[0] * want[0] + dt[1] * want[1] + dt[2] * want[2];
    assert!(cosang.acos().to_degrees() < 0.3, "translation direction off by {} deg", cosang.acos().to_degrees());
}

/// Absolute pose from 2D-3D matches, 30% of them wrong.
#[test]
fn absolute_pose_survives_outliers() {
    let mut rng = Lcg::new(5);
    let x = cloud(&mut rng, 200);
    let truth = pose([0.1, 0.3, -0.05], [0.4, -0.2, 0.5]);
    let mut u: Vec<[f64; 2]> = x
        .iter()
        .map(|v| {
            let c = truth.to_cam(*v);
            [c[0] / c[2], c[1] / c[2]]
        })
        .collect();
    for p in u.iter_mut().take(60) {
        *p = [0.5 * rng.signed() as f64, 0.4 * rng.signed() as f64];
    }
    let (p, mask) = ransac_pnp(&x, &u, 1e-3, 300, 2).unwrap();
    assert!(mask.iter().filter(|&&v| v).count() >= 138);
    assert!(angle_between(&p, &truth).to_degrees() < 0.01);
    let d = sub(p.centre(), truth.centre());
    assert!((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < 1e-3);
}

/// Six cameras round a cloud, observed through a lens with barrel
/// distortion and a focal length 15% off the starting guess, with half a
/// pixel of noise. Bundle adjustment recovers the calibration and brings the
/// error down to the noise.
#[test]
fn bundle_adjustment_self_calibrates() {
    let mut rng = Lcg::new(9);
    let x = cloud(&mut rng, 400);
    let truth_k = Intrinsics { f: 800.0, cx: 480.0, cy: 360.0, k1: -0.12, k2: 0.02, width: 960, height: 720 };
    let poses: Vec<Pose> = (0..6)
        .map(|i| {
            let a = (i as f64 - 2.5) * 0.12;
            let r = exp_so3([0.02 * i as f64, a, 0.0]);
            // orbit: centre on a circle round the cloud's centre
            let c = [-5.0 * a.sin(), 0.1 * i as f64, 5.0 - 5.0 * a.cos()];
            let t = sfm::linalg::scale(sfm::linalg::mv(&r, c), -1.0);
            Pose { r, t }
        })
        .collect();
    let mut obs = Vec::new();
    for (ci, p) in poses.iter().enumerate() {
        for (pi, v) in x.iter().enumerate() {
            if let Some(uv) = project(&truth_k, p, *v) {
                if uv[0] > 0.0 && uv[1] > 0.0 && uv[0] < 960.0 && uv[1] < 720.0 {
                    obs.push(Observation { cam: ci, point: pi, px: [uv[0] + 0.5 * rng.signed() as f64, uv[1] + 0.5 * rng.signed() as f64] });
                }
            }
        }
    }
    let mut k = Intrinsics { f: 920.0, k1: 0.0, k2: 0.0, ..truth_k };
    let mut est_poses: Vec<Pose> = poses
        .iter()
        .enumerate()
        .map(|(i, p)| if i == 0 { *p } else { Pose { r: mm(&exp_so3([0.01, -0.01, 0.005]), &p.r), t: [p.t[0] + 0.05, p.t[1] - 0.03, p.t[2]] } })
        .collect();
    let mut pts: Vec<V3> = x.iter().map(|v| [v[0] * 1.02, v[1] * 0.98, v[2] * 1.03]).collect();
    // the gauge: one camera and the scale (a second camera's distance) are
    // fixed by holding two cameras
    let cfg = BaCfg { iters: 100, fixed: vec![0, 5], ..BaCfg::default() };
    est_poses[5] = poses[5];
    let rep = bundle_adjust(&mut k, &mut est_poses, &mut pts, &obs, &cfg);
    assert!(rep.rms_after < 0.45, "rms {:.3} px after, {:.3} before", rep.rms_after, rep.rms_before);
    // a narrow orbit separates focal length from depth only weakly
    assert!((k.f - 800.0).abs() < 12.0, "focal {:.1} against 800", k.f);
    assert!((k.k1 + 0.12).abs() < 0.01, "k1 {:.4} against -0.12", k.k1);
    assert!(rms(&k, &est_poses, &pts, &obs) < 0.45);
}

/// P3P returns the true pose among its solutions, and absolute pose works on
/// a PLANE - where a linear 3x4 estimate is degenerate, and which is most of
/// what a capture of an object on a floor contains.
#[test]
fn absolute_pose_on_a_plane() {
    let mut rng = Lcg::new(21);
    let truth = pose([0.3, -0.2, 0.1], [0.2, 0.4, 4.0]);
    let x: Vec<V3> = (0..150).map(|_| [2.0 * rng.signed() as f64, 2.0 * rng.signed() as f64, 0.0]).collect();
    let mut u: Vec<[f64; 2]> = x
        .iter()
        .map(|v| {
            let c = truth.to_cam(*v);
            [c[0] / c[2], c[1] / c[2]]
        })
        .collect();
    let sols = p3p([x[0], x[1], x[2]], [u[0], u[1], u[2]]);
    assert!(sols.iter().any(|p| angle_between(p, &truth) < 1e-6), "{} solutions, none the truth", sols.len());
    for p in u.iter_mut().take(60) {
        *p = [0.5 * rng.signed() as f64, 0.4 * rng.signed() as f64];
    }
    let (p, mask) = ransac_pnp(&x, &u, 1e-3, 10_000, 4).unwrap();
    assert!(mask.iter().filter(|&&v| v).count() >= 88);
    assert!(angle_between(&p, &truth).to_degrees() < 0.01);
}
