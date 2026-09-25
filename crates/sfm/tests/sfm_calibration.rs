// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Self-calibration of real lenses by bundle adjustment: a full Brown-Conrady
//! lens (three radial terms, decentering, an off-centre principal point) and
//! a Kannala-Brandt fisheye looking past 90 degrees are recovered from noisy
//! synthetic observations, and model selection names the model that
//! generated the data - no more, no less.
//!
//! Recovery is judged on the LENS MAP - where each direction lands in the
//! image - as well as on the coefficients: high-order coefficients trade off
//! against one another, and it is the map a renderer uses.
//!
//! Swedish Embedded AB implements camera calibration and lens modelling for
//! photogrammetry and embedded vision. If your team needs cameras calibrated
//! from ordinary photographs, you can procure our services by sending an
//! email to info@swedishembedded.com.

use camera::{Intrinsics, Lens};
use data::rng::Lcg;
use sfm::ba::{bundle_adjust, BaCfg, Observation, Param};
use sfm::camera::{project, Pose};
use sfm::lens::{fit_lenses, LensCfg, LensChoice};
use sfm::linalg::{cross, exp_so3, mm, mv, normalize, scale, sub, V3};

fn look_at(eye: V3, target: V3, roll: f64) -> Pose {
    let f = normalize(sub(target, eye));
    let r = normalize(cross(f, [0.0, -1.0, 0.0]));
    let d = normalize(cross(f, r));
    let rot = [r[0], r[1], r[2], d[0], d[1], d[2], f[0], f[1], f[2]];
    // roll about the optical axis, so the principal point is not the only
    // thing that looks the same from every view
    let rot = mm(&exp_so3([0.0, 0.0, roll]), &rot);
    Pose { r: rot, t: scale(mv(&rot, eye), -1.0) }
}

/// Fourteen cameras on two rings round a box of points, rolled differently.
fn orbit_scene(rng: &mut Lcg) -> (Vec<Pose>, Vec<V3>) {
    let x: Vec<V3> = (0..1500).map(|_| [3.0 * rng.signed() as f64, 2.0 * rng.signed() as f64, 1.5 * rng.signed() as f64]).collect();
    let poses = (0..14)
        .map(|i| {
            let a = i as f64 * 0.35 - 2.3;
            let h = if i % 2 == 0 { -1.5 } else { 1.2 };
            look_at([7.0 * a.sin(), h, -7.0 * a.cos()], [0.2 * rng.signed() as f64, 0.0, 0.0], 0.5 * rng.signed() as f64)
        })
        .collect();
    (poses, x)
}

/// Ten cameras near the middle of a shell of points, looking every way: a
/// fisheye sees points up to and past 90 degrees off its axis.
fn room_scene(rng: &mut Lcg) -> (Vec<Pose>, Vec<V3>) {
    let x: Vec<V3> = (0..3000)
        .map(|_| {
            let d = normalize([rng.signed() as f64, rng.signed() as f64, rng.signed() as f64]);
            scale(d, 4.0 + 4.0 * rng.unit() as f64)
        })
        .collect();
    let poses = (0..10)
        .map(|i| {
            let a = i as f64 * 0.6;
            let eye = [0.8 * a.cos(), 0.3 * rng.signed() as f64, 0.8 * a.sin()];
            let target = [eye[0] + (a * 1.7).cos(), eye[1] + 0.4 * rng.signed() as f64, eye[2] + (a * 1.7).sin()];
            look_at(eye, target, 0.6 * rng.signed() as f64)
        })
        .collect();
    (poses, x)
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

/// Poses and points knocked off the truth, as an incremental solve leaves
/// them; camera 0 is the anchor and keeps its pose.
fn perturb(rng: &mut Lcg, poses: &[Pose], x: &[V3]) -> (Vec<Pose>, Vec<V3>) {
    let p = poses
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i == 0 {
                return *p;
            }
            let w = [0.005 * rng.signed() as f64, 0.005 * rng.signed() as f64, 0.005 * rng.signed() as f64];
            Pose { r: mm(&exp_so3(w), &p.r), t: [p.t[0] + 0.03 * rng.signed() as f64, p.t[1] + 0.03 * rng.signed() as f64, p.t[2] + 0.03 * rng.signed() as f64] }
        })
        .collect();
    let x = x.iter().map(|v| [v[0] + 0.02 * rng.signed() as f64, v[1] + 0.02 * rng.signed() as f64, v[2] + 0.02 * rng.signed() as f64]).collect();
    (p, x)
}

/// Largest distance in pixels between where the two calibrations put the
/// same camera-frame direction, over the pixels of the frame no further from
/// its centre than the furthest observation: beyond that the calibration is
/// extrapolation, and the high-order coefficients trade off against each
/// other freely.
fn map_error(truth: &Intrinsics, est: &Intrinsics, obs: &[Observation]) -> f64 {
    let (cx, cy) = (truth.width as f64 / 2.0, truth.height as f64 / 2.0);
    let reach = obs.iter().map(|o| (o.px[0] - cx).hypot(o.px[1] - cy)).fold(0.0, f64::max);
    let mut worst = 0.0f64;
    for j in 0..=24 {
        for i in 0..=32 {
            let px = [(i as f64 + 0.5) / 33.5 * truth.width as f64, (j as f64 + 0.5) / 25.5 * truth.height as f64];
            if (px[0] - cx).hypot(px[1] - cy) > reach {
                continue;
            }
            let Some(d) = truth.unproject(px) else { continue };
            let Some(q) = est.project(d) else { return f64::INFINITY };
            worst = worst.max((q[0] - px[0]).hypot(q[1] - px[1]));
        }
    }
    worst
}

fn full(lens: &Lens) -> Vec<Param> {
    let mut v = vec![Param::Focal, Param::Cx, Param::Cy];
    match lens {
        Lens::Brown { .. } => v.extend([Param::Coeff(0), Param::Coeff(1), Param::Coeff(2), Param::Coeff(6), Param::Coeff(7)]),
        Lens::Fisheye { .. } => v.extend((0..4).map(Param::Coeff)),
        _ => {}
    }
    v
}

/// A phone-like lens: three radial terms, decentering and a principal point
/// 12 px off centre. From a centred, undistorted start 5% off in focal
/// length, bundle adjustment lands on the truth.
#[test]
fn bundle_adjustment_recovers_a_brown_lens() {
    let mut rng = Lcg::new(31);
    let (poses, x) = orbit_scene(&mut rng);
    let truth = Intrinsics {
        fx: 1000.0,
        fy: 1000.0,
        cx: 652.5,
        cy: 472.0,
        lens: Lens::Brown { k: [-0.18, 0.06, -0.01, 0.0, 0.0, 0.0], p: [6e-4, -4e-4], s: [0.0; 4] },
        width: 1280,
        height: 960,
    };
    let obs = observe(&truth, &poses, &x, &mut rng, 0.3);
    let (mut est, mut pts) = perturb(&mut rng, &poses, &x);
    est[1] = poses[1]; // the scale camera at its true distance
    let start = Intrinsics { fx: 1050.0, fy: 1050.0, cx: 640.0, cy: 480.0, lens: Lens::Brown { k: [0.0; 6], p: [0.0; 2], s: [0.0; 4] }, ..truth };
    let mut ks = vec![start];
    let sensor = vec![0; poses.len()];
    let cfg = BaCfg { iters: 200, free: vec![full(&start.lens)], anchor: Some(0), scale: Some(1), ..BaCfg::default() };
    let rep = bundle_adjust(&mut ks, &sensor, &mut est, &mut pts, &obs, &cfg);
    let k = ks[0];
    let c = k.lens.coeffs();
    eprintln!("brown: rms {:.3} -> {:.3} px, {k:?}", rep.rms_before, rep.rms_after);
    // uniform noise of 0.3 px per axis is 0.245 px rms in 2D
    assert!(rep.rms_after < 0.26, "rms {:.3} px", rep.rms_after);
    assert!((k.fx - 1000.0).abs() < 1.0, "fx {:.2}", k.fx);
    assert!((k.cx - 652.5).abs() < 1.0 && (k.cy - 472.0).abs() < 1.0, "principal point ({:.2}, {:.2})", k.cx, k.cy);
    assert!((c[0] + 0.18).abs() < 0.005, "k1 {:.4}", c[0]);
    assert!((c[6] - 6e-4).abs() < 1e-4 && (c[7] + 4e-4).abs() < 1e-4, "p ({:.2e}, {:.2e})", c[6], c[7]);
    let m = map_error(&truth, &k, &obs);
    assert!(m < 0.3, "lens map off by up to {m:.3} px");
}

/// A Kannala-Brandt fisheye seeing points past 90 degrees off axis, from an
/// equidistant start: the same accuracy, through the same code.
#[test]
fn bundle_adjustment_recovers_a_fisheye_lens() {
    let mut rng = Lcg::new(37);
    let (poses, x) = room_scene(&mut rng);
    let truth = Intrinsics { fx: 380.0, fy: 380.0, cx: 604.0, cy: 597.0, lens: Lens::Fisheye { k: [0.05, -0.02, 0.004, -0.0005] }, width: 1200, height: 1200 };
    let obs = observe(&truth, &poses, &x, &mut rng, 0.3);
    let beyond = obs
        .iter()
        .filter(|o| truth.unproject(o.px).is_some_and(|d| d[2] < 0.0))
        .count();
    assert!(beyond > 500, "only {beyond} observations past 90 degrees");
    let (mut est, mut pts) = perturb(&mut rng, &poses, &x);
    est[1] = poses[1];
    let start = Intrinsics { fx: 400.0, fy: 400.0, cx: 600.0, cy: 600.0, lens: Lens::Fisheye { k: [0.0; 4] }, ..truth };
    let mut ks = vec![start];
    let sensor = vec![0; poses.len()];
    let cfg = BaCfg { iters: 200, free: vec![full(&start.lens)], anchor: Some(0), scale: Some(1), ..BaCfg::default() };
    let rep = bundle_adjust(&mut ks, &sensor, &mut est, &mut pts, &obs, &cfg);
    let k = ks[0];
    eprintln!("fisheye: rms {:.3} -> {:.3} px, {k:?}", rep.rms_before, rep.rms_after);
    assert!(rep.rms_after < 0.26, "rms {:.3} px", rep.rms_after);
    assert!((k.fx - 380.0).abs() < 0.5, "fx {:.2}", k.fx);
    assert!((k.cx - 604.0).abs() < 0.5 && (k.cy - 597.0).abs() < 0.5, "principal point ({:.2}, {:.2})", k.cx, k.cy);
    let m = map_error(&truth, &k, &obs);
    assert!(m < 0.3, "lens map off by up to {m:.3} px");
}

/// Refined from the same incremental result (a plain radial camera), each
/// candidate model is fitted and the information criterion names the one
/// that generated the observations: extra coefficients that only fit noise
/// do not win, and a model that cannot describe the lens does not either.
#[test]
fn model_selection_names_the_generating_lens() {
    let cases: Vec<(LensChoice, Intrinsics, bool)> = vec![
        (LensChoice::Pinhole, Intrinsics::pinhole(1000.0, 1280, 960), false),
        (LensChoice::Radial, Intrinsics { lens: Lens::radial(-0.15, 0.04), ..Intrinsics::pinhole(1000.0, 1280, 960) }, false),
        (
            LensChoice::Brown,
            Intrinsics { lens: Lens::Brown { k: [-0.15, 0.04, -0.012, 0.0, 0.0, 0.0], p: [1.5e-3, -1e-3], s: [0.0; 4] }, ..Intrinsics::pinhole(1000.0, 1280, 960) },
            false,
        ),
        (LensChoice::Fisheye, Intrinsics { fx: 380.0, fy: 380.0, cx: 604.0, cy: 597.0, lens: Lens::Fisheye { k: [0.05, -0.02, 0.004, -0.0005] }, width: 1200, height: 1200 }, true),
    ];
    for (want, truth, wide) in cases {
        let mut rng = Lcg::new(41);
        let (poses, x) = if wide { room_scene(&mut rng) } else { orbit_scene(&mut rng) };
        let obs = observe(&truth, &poses, &x, &mut rng, 0.3);
        let (mut est, pts) = perturb(&mut rng, &poses, &x);
        est[1] = poses[1];
        // what an incremental solve hands over: a radial camera for a
        // perspective lens, an equidistant one for a fisheye
        let start = if wide {
            Intrinsics { fx: 390.0, fy: 390.0, cx: 600.0, cy: 600.0, lens: Lens::Fisheye { k: [0.0; 4] }, ..truth }
        } else {
            Intrinsics { fx: 1020.0, fy: 1020.0, lens: Lens::radial(0.0, 0.0), ..truth }
        };
        let sensor = vec![0; poses.len()];
        let cfg = LensCfg { choice: LensChoice::Auto, ba: BaCfg { anchor: Some(0), scale: Some(1), ..BaCfg::default() }, ..LensCfg::default() };
        let fits = fit_lenses(&[start], &sensor, &est, &pts, &obs, &cfg);
        let best = fits.iter().min_by(|a, b| a.fit.bic.total_cmp(&b.fit.bic)).unwrap();
        for c in &fits {
            eprintln!("{want:?} data: {:?} rms {:.3} px bic {:.1}", c.fit.choice, c.fit.rms_px, c.fit.bic);
        }
        assert_eq!(best.fit.choice, want, "picked {:?} for a {:?} lens", best.fit.choice, want);
        assert!(best.fit.rms_px < 0.26, "{want:?}: rms {:.3} px", best.fit.rms_px);
        let m = map_error(&truth, &best.fit.intrinsics[0], &obs);
        assert!(m < 0.3, "{want:?}: lens map off by up to {m:.3} px: {:?}", best.fit.intrinsics[0]);
    }
}
