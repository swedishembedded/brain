// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The geometric core of structure from motion, on exact synthetic
//! correspondences with outliers: the minimal five-point essential solver,
//! relative pose from an essential matrix, absolute pose from 2D-3D matches,
//! and bundle adjustment recovering a focal length and lens distortion it was
//! not told, inside an explicit gauge.
//!
//! Every image measurement here is a BEARING - a unit ray in the camera's
//! frame - so nothing assumes the point lies in front of an image plane.
//!
//! Swedish Embedded AB implements camera calibration and multi-view
//! reconstruction for its clients. If your team needs photographs turned into
//! calibrated cameras and geometry, you can procure our services by sending
//! an email to info@swedishembedded.com.

use camera::{Intrinsics, Lens};
use data::rng::Lcg;
use sfm::ba::{bundle_adjust, rms, BaCfg, Observation, Param};
use sfm::camera::{project, Pose};
use sfm::linalg::{exp_so3, log_so3, mm, norm, normalize, sub, transpose, M3, V3};
use sfm::pnp::{p3p, ransac_pnp};
use sfm::twoview::{essential_5pt, essential_of, ransac_essential, refine_essential, relative_pose, sampson};

fn cloud(rng: &mut Lcg, n: usize) -> Vec<V3> {
    (0..n).map(|_| [2.0 * rng.signed() as f64, 1.5 * rng.signed() as f64, 5.0 + 1.5 * rng.signed() as f64]).collect()
}

fn pose(w: V3, t: V3) -> Pose {
    Pose { r: exp_so3(w), t }
}

fn angle_between(a: &Pose, b: &Pose) -> f64 {
    let d = mm(&a.r, &transpose(&b.r));
    norm(log_so3(&d))
}

fn bearing(p: &Pose, x: V3) -> V3 {
    normalize(p.to_cam(x))
}

/// A random unit direction.
fn direction(rng: &mut Lcg) -> V3 {
    normalize([rng.signed() as f64, rng.signed() as f64, rng.signed() as f64 + 1e-3])
}

/// `b` tipped by a random angle of up to `sigma` radians.
fn jitter(rng: &mut Lcg, b: V3, sigma: f64) -> V3 {
    normalize([b[0] + sigma * rng.signed() as f64, b[1] + sigma * rng.signed() as f64, b[2] + sigma * rng.signed() as f64])
}

/// Distance between two essential matrices up to scale and sign.
fn e_distance(a: &M3, b: &M3) -> f64 {
    let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let plus: f64 = (0..9).map(|i| (a[i] / na - b[i] / nb).powi(2)).sum::<f64>().sqrt();
    let minus: f64 = (0..9).map(|i| (a[i] / na + b[i] / nb).powi(2)).sum::<f64>().sqrt();
    plus.min(minus)
}

/// Five exact correspondences determine the essential matrix up to at most
/// ten solutions, one of which is the truth - also when some of the rays
/// point more than 90 degrees off the optical axis, which a normalized
/// `(x/z, y/z)` formulation cannot even represent.
#[test]
fn five_points_give_the_exact_essential_matrix() {
    let mut rng = Lcg::new(17);
    for trial in 0..100 {
        let second = pose([0.3 * rng.signed() as f64, 0.3 * rng.signed() as f64, 0.3 * rng.signed() as f64], normalize([rng.signed() as f64, rng.signed() as f64, rng.signed() as f64]));
        let truth = essential_of(&second);
        // points all round the first camera, at 2..6 units
        let x: Vec<V3> = (0..5).map(|_| sfm::linalg::scale(direction(&mut rng), 2.0 + 4.0 * rng.unit() as f64)).collect();
        let a: Vec<V3> = x.iter().map(|v| normalize(*v)).collect();
        let b: Vec<V3> = x.iter().map(|v| bearing(&second, *v)).collect();
        let sols = essential_5pt(&a, &b);
        assert!(!sols.is_empty() && sols.len() <= 10, "trial {trial}: {} solutions", sols.len());
        let best = sols.iter().map(|e| e_distance(e, &truth)).fold(f64::INFINITY, f64::min);
        assert!(best < 1e-8, "trial {trial}: nearest of {} solutions is {best:.2e} from the truth", sols.len());
    }
}

/// 30% of the correspondences are garbage and the rest carry about 0.007
/// degrees of noise per axis. LO-RANSAC over the five-point solver with a
/// final nonlinear refinement lands on the optimum of the data: the same
/// cost as refining from the TRUE pose, and lower than the truth's own. On
/// this narrow cloud that optimum is 0.013 degrees of rotation from the
/// truth - the noise floor, not an error of the estimator.
#[test]
fn relative_pose_survives_outliers_and_noise() {
    let mut rng = Lcg::new(3);
    let x = cloud(&mut rng, 300);
    let second = pose([0.05, -0.2, 0.03], [-1.0, 0.1, 0.2]);
    let first = Pose::identity();
    let noise = 2e-4;
    let mut a: Vec<V3> = x.iter().map(|v| jitter(&mut rng, bearing(&first, *v), noise)).collect();
    let b: Vec<V3> = x.iter().map(|v| jitter(&mut rng, bearing(&second, *v), noise)).collect();
    for p in a.iter_mut().take(90) {
        *p = normalize([0.5 * rng.signed() as f64, 0.4 * rng.signed() as f64, 1.0]);
    }
    let (e, mask) = ransac_essential(&a, &b, 1e-3, 5_000, 1).unwrap();
    let inl = mask.iter().filter(|&&v| v).count();
    assert!((205..=215).contains(&inl), "{inl} inliers of 210 true ones");
    let (ia, ib): (Vec<V3>, Vec<V3>) = (90..300).map(|i| (a[i], b[i])).unzip();
    let cost = |e: &M3| (0..ia.len()).map(|i| sampson(e, ia[i], ib[i])).sum::<f64>();
    let oracle = refine_essential(&essential_of(&second), &ia, &ib, 50);
    assert!(cost(&e) <= cost(&oracle) * (1.0 + 1e-6), "cost {:.6e} against {:.6e} refined from the truth", cost(&e), cost(&oracle));
    assert!(cost(&e) < cost(&essential_of(&second)), "the estimate explains the data worse than the truth");
    let (p, _) = relative_pose(&e, &a, &b, &mask);
    let rot = angle_between(&p, &second).to_degrees();
    assert!(rot < 0.02, "rotation off by {rot} deg");
    let cosang = sfm::linalg::dot(normalize(p.t), normalize(second.t));
    assert!(cosang.acos().to_degrees() < 0.1, "translation direction off by {} deg", cosang.acos().to_degrees());
}

/// Absolute pose from 2D-3D matches, 30% of them wrong.
#[test]
fn absolute_pose_survives_outliers() {
    let mut rng = Lcg::new(5);
    let x = cloud(&mut rng, 200);
    let truth = pose([0.1, 0.3, -0.05], [0.4, -0.2, 0.5]);
    let mut u: Vec<V3> = x.iter().map(|v| bearing(&truth, *v)).collect();
    for p in u.iter_mut().take(60) {
        *p = normalize([0.5 * rng.signed() as f64, 0.4 * rng.signed() as f64, 1.0]);
    }
    let (p, mask) = ransac_pnp(&x, &u, 1e-3, 300, 2).unwrap();
    assert!(mask.iter().filter(|&&v| v).count() >= 138);
    assert!(angle_between(&p, &truth).to_degrees() < 0.01);
    assert!(norm(sub(p.centre(), truth.centre())) < 1e-3);
}

/// The same with noise on every bearing: local optimization and the final
/// Gauss-Newton on the inliers average it down well below the noise of any
/// three-point sample.
#[test]
fn absolute_pose_refines_through_noise() {
    let mut rng = Lcg::new(6);
    let x = cloud(&mut rng, 300);
    let truth = pose([-0.2, 0.1, 0.05], [0.3, 0.1, -0.4]);
    let mut u: Vec<V3> = x.iter().map(|v| jitter(&mut rng, bearing(&truth, *v), 3e-4)).collect();
    for p in u.iter_mut().take(90) {
        *p = normalize([0.5 * rng.signed() as f64, 0.4 * rng.signed() as f64, 1.0]);
    }
    let (p, mask) = ransac_pnp(&x, &u, 1.5e-3, 5_000, 3).unwrap();
    assert!(mask.iter().filter(|&&v| v).count() >= 205);
    let rot = angle_between(&p, &truth).to_degrees();
    assert!(rot < 0.005, "rotation off by {rot} deg");
    let dc = norm(sub(p.centre(), truth.centre()));
    assert!(dc < 2e-3, "centre off by {dc}");
}

/// P3P returns the true pose among its solutions, and absolute pose works on
/// a PLANE - where a linear 3x4 estimate is degenerate, and which is most of
/// what a capture of an object on a floor contains.
#[test]
fn absolute_pose_on_a_plane() {
    let mut rng = Lcg::new(21);
    let truth = pose([0.3, -0.2, 0.1], [0.2, 0.4, 4.0]);
    let x: Vec<V3> = (0..150).map(|_| [2.0 * rng.signed() as f64, 2.0 * rng.signed() as f64, 0.0]).collect();
    let mut u: Vec<V3> = x.iter().map(|v| bearing(&truth, *v)).collect();
    let sols = p3p([x[0], x[1], x[2]], [u[0], u[1], u[2]]);
    assert!(sols.iter().any(|p| angle_between(p, &truth) < 1e-6), "{} solutions, none the truth", sols.len());
    for p in u.iter_mut().take(60) {
        *p = normalize([0.5 * rng.signed() as f64, 0.4 * rng.signed() as f64, 1.0]);
    }
    let (p, mask) = ransac_pnp(&x, &u, 1e-3, 10_000, 4).unwrap();
    assert!(mask.iter().filter(|&&v| v).count() >= 88);
    assert!(angle_between(&p, &truth).to_degrees() < 0.01);
}

/// Six cameras on an orbit round a cloud.
fn orbit(n: usize) -> Vec<Pose> {
    (0..n)
        .map(|i| {
            let a = (i as f64 - (n as f64 - 1.0) / 2.0) * 0.12;
            let r = exp_so3([0.02 * i as f64, a, 0.0]);
            let c = [-5.0 * a.sin(), 0.1 * i as f64, 5.0 - 5.0 * a.cos()];
            Pose { r, t: sfm::linalg::scale(sfm::linalg::mv(&r, c), -1.0) }
        })
        .collect()
}

fn observe(k: &Intrinsics, poses: &[Pose], x: &[V3], rng: &mut Lcg, noise: f64) -> Vec<Observation> {
    let mut obs = Vec::new();
    for (ci, p) in poses.iter().enumerate() {
        for (pi, v) in x.iter().enumerate() {
            if let Some(uv) = project(k, p, *v) {
                if uv[0] > 0.0 && uv[1] > 0.0 && uv[0] < k.width as f64 && uv[1] < k.height as f64 {
                    obs.push(Observation { cam: ci, point: pi, px: [uv[0] + noise * rng.signed() as f64, uv[1] + noise * rng.signed() as f64] });
                }
            }
        }
    }
    obs
}

/// Observed through a lens with barrel distortion and a focal length 15% off
/// the starting guess, with half a pixel of noise, bundle adjustment recovers
/// the calibration and brings the error down to the noise.
#[test]
fn bundle_adjustment_self_calibrates() {
    let mut rng = Lcg::new(9);
    let x = cloud(&mut rng, 400);
    let truth_k = Intrinsics { lens: Lens::radial(-0.12, 0.02), ..Intrinsics::pinhole(800.0, 960, 720) };
    let poses = orbit(6);
    let obs = observe(&truth_k, &poses, &x, &mut rng, 0.5);
    let mut ks = vec![Intrinsics { fx: 920.0, fy: 920.0, lens: Lens::radial(0.0, 0.0), ..truth_k }];
    let mut est: Vec<Pose> = poses
        .iter()
        .enumerate()
        .map(|(i, p)| if i == 0 { *p } else { Pose { r: mm(&exp_so3([0.01, -0.01, 0.005]), &p.r), t: [p.t[0] + 0.05, p.t[1] - 0.03, p.t[2]] } })
        .collect();
    // the scale: camera 5 at its true distance from camera 0
    est[5] = poses[5];
    let mut pts: Vec<V3> = x.iter().map(|v| [v[0] * 1.02, v[1] * 0.98, v[2] * 1.03]).collect();
    let sensor = vec![0; 6];
    let cfg = BaCfg { iters: 100, free: vec![vec![Param::Focal, Param::Coeff(0), Param::Coeff(1)]], anchor: Some(0), scale: Some(5), ..BaCfg::default() };
    let rep = bundle_adjust(&mut ks, &sensor, &mut est, &mut pts, &obs, &cfg);
    assert!(rep.rms_after < 0.45, "rms {:.3} px after, {:.3} before", rep.rms_after, rep.rms_before);
    // a narrow orbit separates focal length from depth only weakly
    let k = ks[0];
    assert!((k.fx - 800.0).abs() < 12.0, "focal {:.1} against 800", k.fx);
    assert_eq!(k.fx, k.fy, "square pixels stay square");
    assert!((k.lens.coeffs()[0] + 0.12).abs() < 0.01, "k1 {:.4} against -0.12", k.lens.coeffs()[0]);
    assert!(rms(&ks, &sensor, &est, &pts, &obs) < 0.45);
}

/// The gauge is explicit: the anchor camera does not move AT ALL, and the
/// distance from it to the scale camera is exactly what it was, however far
/// everything else had to travel.
#[test]
fn the_gauge_holds_the_anchor_and_the_scale() {
    let mut rng = Lcg::new(11);
    let x = cloud(&mut rng, 300);
    let k = Intrinsics::pinhole(800.0, 960, 720);
    let poses = orbit(6);
    let obs = observe(&k, &poses, &x, &mut rng, 0.3);
    // every camera but the anchor badly off, scale included
    let mut est: Vec<Pose> = poses
        .iter()
        .enumerate()
        .map(|(i, p)| if i == 0 { *p } else { Pose { r: mm(&exp_so3([0.02, 0.01, -0.01]), &p.r), t: [p.t[0] * 1.1 + 0.1, p.t[1] - 0.05, p.t[2] * 0.9] } })
        .collect();
    let mut pts: Vec<V3> = x.iter().map(|v| [v[0] * 1.05, v[1] * 0.95, v[2] * 1.1]).collect();
    let anchor = est[0];
    let d0 = norm(sub(est[3].centre(), est[0].centre()));
    let sensor = vec![0; 6];
    let cfg = BaCfg { iters: 60, anchor: Some(0), scale: Some(3), ..BaCfg::default() };
    let rep = bundle_adjust(&mut [k], &sensor, &mut est, &mut pts, &obs, &cfg);
    assert!(rep.rms_after < 0.35, "rms {:.3} px after, {:.3} before", rep.rms_after, rep.rms_before);
    assert_eq!(est[0], anchor, "the anchor camera moved");
    let d1 = norm(sub(est[3].centre(), est[0].centre()));
    assert!((d1 - d0).abs() < 1e-9 * d0, "scale moved: {d0} -> {d1}");
}
