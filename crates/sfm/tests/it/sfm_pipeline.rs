// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Photographs in, cameras out: the whole pipeline on images ray-cast from a
//! known scene through a known lens (barrel distortion included), so every
//! recovered quantity can be checked against the truth - up to the
//! similarity no set of photographs can pin down.
//!
//! Swedish Embedded AB implements camera calibration and multi-view
//! reconstruction for its clients. If your team needs photographs turned into
//! calibrated cameras and geometry, you can procure our services by sending
//! an email to info@swedishembedded.com.

use camera::{Intrinsics, Lens};
use sfm::camera::Pose;
use sfm::georef::{enu, Wgs84};
use sfm::incremental::{reconstruct, Gauge, Initializer, Photo, SfmCfg, SfmError};
use sfm::linalg::{dot, mtv, mv, normalize, scale, sub, V3};
use sfm::retrieval::PairSelection;

/// Multi-octave value noise in [0,1] - a texture with detail at every scale,
/// like a wooden deck, so features exist at every octave.
fn noise(x: f64, y: f64) -> f64 {
    fn hash(i: i64, j: i64) -> f64 {
        let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (j as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 29)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z >> 11) as f64) / (1u64 << 53) as f64
    }
    let mut v = 0.0;
    let mut amp = 0.5;
    let mut f = 4.0;
    for _ in 0..5 {
        let (fx, fy) = (x * f, y * f);
        let (i, j) = (fx.floor() as i64, fy.floor() as i64);
        let (u, w) = (fx - fx.floor(), fy - fy.floor());
        let (u, w) = (u * u * (3.0 - 2.0 * u), w * w * (3.0 - 2.0 * w));
        let a = hash(i, j) * (1.0 - u) + hash(i + 1, j) * u;
        let b = hash(i, j + 1) * (1.0 - u) + hash(i + 1, j + 1) * u;
        v += amp * (a * (1.0 - w) + b * w);
        amp *= 0.5;
        f *= 2.0;
    }
    v
}

/// Ray-cast one image: a textured ground plane y = 1 and a textured box on
/// it, through `k` (distortion included) from `pose`.
fn shoot(k: &Intrinsics, pose: &Pose) -> Vec<u8> {
    let (w, h) = (k.width as usize, k.height as usize);
    let c = pose.centre();
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let Some(n) = k.unproject([x as f64 + 0.5, y as f64 + 0.5]) else { continue };
            let d = sfm::linalg::mtv(&pose.r, n);
            let mut best = f64::INFINITY;
            let mut tex = 0.3;
            // ground
            if d[1] > 1e-6 {
                let t = (1.0 - c[1]) / d[1];
                if t > 0.0 {
                    best = t;
                    let p = [c[0] + t * d[0], 1.0, c[2] + t * d[2]];
                    tex = noise(p[0] * 0.5, p[2] * 0.5);
                }
            }
            // box [-0.6,0.6] x [0.0,1.0] x [-0.6,0.6], slab test
            let (lo, hi) = ([-0.6, 0.0, -0.6], [0.6, 1.0, 0.6]);
            let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
            let mut axis = 0;
            for a in 0..3 {
                if d[a].abs() < 1e-12 {
                    if c[a] < lo[a] || c[a] > hi[a] {
                        t0 = f64::INFINITY;
                    }
                    continue;
                }
                let (mut ta, mut tb) = ((lo[a] - c[a]) / d[a], (hi[a] - c[a]) / d[a]);
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                if ta > t0 {
                    t0 = ta;
                    axis = a;
                }
                t1 = t1.min(tb);
            }
            if t0 < t1 && t0 < best {
                let p = [c[0] + t0 * d[0], c[1] + t0 * d[1], c[2] + t0 * d[2]];
                let (u, v) = match axis {
                    0 => (p[1], p[2]),
                    1 => (p[0], p[2]),
                    _ => (p[0], p[1]),
                };
                tex = 0.15 + 0.85 * noise(u * 1.3 + 17.0 * axis as f64, v * 1.3);
            }
            let o = (y * w + x) * 3;
            let g = (tex * 255.0).clamp(0.0, 255.0) as u8;
            out[o] = g;
            out[o + 1] = (g as f64 * 0.9) as u8;
            out[o + 2] = (g as f64 * 0.8) as u8;
        }
    }
    out
}

fn look_at(eye: V3, target: V3) -> Pose {
    let f = normalize(sub(target, eye));
    let r = normalize(sfm::linalg::cross(f, [0.0, -1.0, 0.0]));
    let d = normalize(sfm::linalg::cross(f, r));
    let rot = [r[0], r[1], r[2], d[0], d[1], d[2], f[0], f[1], f[2]];
    Pose { r: rot, t: scale(mv(&rot, eye), -1.0) }
}

/// Cameras on half an orbit round the scene, `phase` steps along it.
fn orbit(views: usize, phase: f64) -> Vec<Pose> {
    (0..views)
        .map(|i| {
            let a = (i as f64 + phase) * std::f64::consts::TAU / views as f64 * 0.5;
            look_at([3.2 * a.sin(), -1.6, -3.2 * a.cos()], [0.0, 0.6, 0.0])
        })
        .collect()
}

/// The worst camera-centre error, as a fraction of the rig's size, after
/// the best similarity from the reconstruction to the truth.
fn centre_error(est_poses: &[Option<Pose>], truth: &[Pose]) -> f64 {
    let est: Vec<V3> = est_poses.iter().map(|p| p.unwrap().centre()).collect();
    let tru: Vec<V3> = truth.iter().map(|p| p.centre()).collect();
    let (ce, ct) = (centroid(&est), centroid(&tru));
    let se = est.iter().map(|p| norm2(sub(*p, ce))).sum::<f64>().sqrt();
    let st = tru.iter().map(|p| norm2(sub(*p, ct))).sum::<f64>().sqrt();
    // rotation between the two centred, normalized sets via the camera
    // orientations: R_align = R_true^T R_est for any view
    let r0 = sfm::linalg::mm(&sfm::linalg::transpose(&truth[0].r), &est_poses[0].unwrap().r);
    let mut worst = 0.0f64;
    for i in 0..truth.len() {
        let a = scale(sub(est[i], ce), 1.0 / se);
        let b = scale(sub(tru[i], ct), 1.0 / st);
        let e = sub(mv(&r0, a), b);
        worst = worst.max(dot(e, e).sqrt());
    }
    worst
}

#[test]
fn photographs_of_a_known_scene_give_back_its_cameras() {
    let k = Intrinsics { lens: Lens::radial(-0.08, 0.01), ..Intrinsics::pinhole(420.0, 640, 480) };
    let truth = orbit(8, 0.0);
    let images: Vec<Vec<u8>> = truth.iter().map(|p| shoot(&k, p)).collect();
    let photos: Vec<Photo> = images.iter().map(|rgb| Photo { width: 640, height: 480, rgb, sensor: 0, focal_px: None, gps: None }).collect();
    let cfg = SfmCfg { verbose: true, ..SfmCfg::default() };
    let rec = reconstruct(&photos, &cfg).expect("reconstruction");

    assert!(rec.poses.iter().all(|p| p.is_some()), "unregistered views: {:?}", rec.poses.iter().map(|p| p.is_some()).collect::<Vec<_>>());
    assert!(rec.rms_px < 1.0, "final reprojection rms {:.3} px", rec.rms_px);
    let est = rec.intrinsics[0];
    assert!((est.fx - k.fx).abs() < 0.03 * k.fx, "focal {:.1} against {}", est.fx, k.fx);
    let k1 = match est.lens {
        Lens::Brown { k, .. } => k[0],
        other => panic!("a perspective lens was described as {}", other.name()),
    };
    assert!((k1 + 0.08).abs() < 0.03, "k1 {k1:.4} against -0.08");
    let worst = centre_error(&rec.poses, &truth);
    assert!(worst < 0.02, "worst camera-centre error {worst:.4} of the rig's size");
}

/// Up in the scene is -Y. The angle between each camera's own up axis and
/// the reconstruction's up must be the angle it really was: a statement
/// about gravity that holds in whatever frame the reconstruction is in.
fn up_errors_deg(est_poses: &[Option<Pose>], est_up: V3, truth: &[Pose]) -> f64 {
    let mut worst = 0.0f64;
    for (e, t) in est_poses.iter().zip(truth) {
        let e = e.unwrap();
        let cam_up_est = mtv(&e.r, [0.0, -1.0, 0.0]);
        let cam_up_true = mtv(&t.r, [0.0, -1.0, 0.0]);
        let a = dot(cam_up_est, est_up).clamp(-1.0, 1.0).acos();
        let b = dot(cam_up_true, [0.0, -1.0, 0.0]).clamp(-1.0, 1.0).acos();
        worst = worst.max((a - b).abs().to_degrees());
    }
    worst
}

/// The default path on a capture matched by retrieval instead of every
/// pair: cameras initialized globally (rotation averaging, then positions
/// of cameras and points together) and refined by the same bundle
/// adjustment, with the direction of gravity read off how the cameras were
/// held.
#[test]
fn a_capture_is_solved_globally_from_retrieved_pairs_and_knows_up() {
    let k = Intrinsics { lens: Lens::radial(-0.08, 0.01), ..Intrinsics::pinhole(420.0, 640, 480) };
    let truth = orbit(8, 0.0);
    let images: Vec<Vec<u8>> = truth.iter().map(|p| shoot(&k, p)).collect();
    let photos: Vec<Photo> = images.iter().map(|rgb| Photo { width: 640, height: 480, rgb, sensor: 0, focal_px: None, gps: None }).collect();
    let pairs = PairSelection { exhaustive_up_to: 0, top_k: 3, sequential: 1, ..PairSelection::default() };
    let cfg = SfmCfg { pairs, verbose: true, ..SfmCfg::default() };
    let rec = reconstruct(&photos, &cfg).expect("reconstruction");

    assert!(rec.report.pairs_matched < 8 * 7 / 2, "{} pairs matched, as many as every pair", rec.report.pairs_matched);
    assert_eq!(rec.report.initializer, Initializer::Global);
    assert!(rec.poses.iter().all(|p| p.is_some()), "unregistered views: {:?}", rec.poses.iter().map(|p| p.is_some()).collect::<Vec<_>>());
    assert!(rec.rms_px < 1.0, "final reprojection rms {:.3} px", rec.rms_px);
    let worst = centre_error(&rec.poses, &truth);
    assert!(worst < 0.02, "worst camera-centre error {worst:.4} of the rig's size");
    assert!(matches!(rec.gauge, Gauge::Seed { .. }), "no fixes, so the gauge is the seed pair's: {:?}", rec.gauge);
    let up = rec.up.expect("the cameras say which way is up");
    let e = up_errors_deg(&rec.poses, up, &truth);
    assert!(e < 2.0, "up is off by up to {e:.2} deg");
}

/// Photographs carrying satellite fixes come back in metres, east-north-up
/// about the fixes' origin: every camera where it was, gravity along +Z.
#[test]
fn photographs_with_fixes_come_out_in_metres_east_north_up() {
    let k = Intrinsics { lens: Lens::radial(-0.08, 0.01), ..Intrinsics::pinhole(420.0, 640, 480) };
    let truth = orbit(8, 0.0);
    let images: Vec<Vec<u8>> = truth.iter().map(|p| shoot(&k, p)).collect();
    // the scene's x is east, its z north and its -y up; one degree of
    // latitude is 111.4 km here and one of longitude 56.6 km
    let (lat, lon) = (59.43, 17.99);
    let (mn, me) = (111_402.5, 56_757.0);
    let at = |c: V3| Wgs84 { latitude_deg: lat + c[2] / mn, longitude_deg: lon + c[0] / me, altitude_m: Some(20.0 - c[1]) };
    let photos: Vec<Photo> = images
        .iter()
        .zip(&truth)
        .map(|(rgb, p)| Photo { width: 640, height: 480, rgb, sensor: 0, focal_px: None, gps: Some(at(p.centre())) })
        .collect();
    let rec = reconstruct(&photos, &SfmCfg { verbose: true, ..SfmCfg::default() }).expect("reconstruction");
    let Gauge::Enu { origin, .. } = rec.gauge else { panic!("fixes were not used: {:?}", rec.gauge) };
    assert_eq!(rec.up, Some([0.0, 0.0, 1.0]));
    for (i, (p, t)) in rec.poses.iter().zip(&truth).enumerate() {
        let got = p.expect("registered").centre();
        let want = enu(&origin, &at(t.centre()));
        let e = sfm::linalg::norm(sub(got, want));
        assert!(e < 0.05, "camera {i} is {e:.3} m from where it was");
    }
}

/// Two physical cameras of different image size and focal length in one
/// capture, interleaved round the scene: each is calibrated on its own, and
/// the poses of both land in one frame.
#[test]
fn two_cameras_of_different_sizes_are_calibrated_separately() {
    let ka = Intrinsics { lens: Lens::radial(-0.08, 0.01), ..Intrinsics::pinhole(420.0, 640, 480) };
    let kb = Intrinsics { lens: Lens::radial(-0.03, 0.0), ..Intrinsics::pinhole(330.0, 560, 420) };
    let (ta, tb) = (orbit(8, 0.0), orbit(8, 0.5));
    let mut truth = Vec::new();
    let mut images = Vec::new();
    let mut sensor = Vec::new();
    for i in 0..8 {
        for (s, (k, t)) in [(&ka, &ta), (&kb, &tb)].into_iter().enumerate() {
            truth.push(t[i]);
            images.push(shoot(k, &t[i]));
            sensor.push(s);
        }
    }
    let photos: Vec<Photo> = images
        .iter()
        .zip(&sensor)
        .map(|(rgb, &s)| {
            let k = if s == 0 { &ka } else { &kb };
            Photo { width: k.width, height: k.height, rgb, sensor: s, focal_px: None, gps: None }
        })
        .collect();
    let rec = reconstruct(&photos, &SfmCfg { verbose: true, ..SfmCfg::default() }).expect("reconstruction");
    assert_eq!(rec.sensor, sensor);
    assert_eq!(rec.intrinsics.len(), 2);
    assert!(rec.poses.iter().all(|p| p.is_some()), "unregistered views: {:?}", rec.poses.iter().map(|p| p.is_some()).collect::<Vec<_>>());
    for (k, est) in [ka, kb].iter().zip(&rec.intrinsics) {
        assert_eq!((est.width, est.height), (k.width, k.height));
        assert!((est.fx - k.fx).abs() < 0.03 * k.fx, "focal {:.1} against {}", est.fx, k.fx);
    }
    let worst = centre_error(&rec.poses, &truth);
    assert!(worst < 0.02, "worst camera-centre error {worst:.4} of the rig's size");
}

/// One sensor is one camera: photographs claiming the same sensor at
/// different sizes are refused before any work is done.
#[test]
fn one_sensor_at_two_sizes_is_refused() {
    let a = vec![0u8; 64 * 48 * 3];
    let b = vec![0u8; 48 * 64 * 3];
    let photos = [
        Photo { width: 64, height: 48, rgb: &a, sensor: 0, focal_px: None, gps: None },
        Photo { width: 48, height: 64, rgb: &b, sensor: 0, focal_px: None, gps: None },
    ];
    assert!(matches!(reconstruct(&photos, &SfmCfg::default()), Err(SfmError::MixedSizes { sensor: 0 })));
}

fn centroid(p: &[V3]) -> V3 {
    let n = p.len() as f64;
    [p.iter().map(|v| v[0]).sum::<f64>() / n, p.iter().map(|v| v[1]).sum::<f64>() / n, p.iter().map(|v| v[2]).sum::<f64>() / n]
}

fn norm2(v: V3) -> f64 {
    dot(v, v)
}
