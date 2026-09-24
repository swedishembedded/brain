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

use sfm::camera::{Intrinsics, Pose};
use sfm::incremental::{reconstruct, Photo, SfmCfg};
use sfm::linalg::{dot, mv, normalize, scale, sub, V3};

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
            let n = k.to_normalized([x as f64 + 0.5, y as f64 + 0.5]);
            let d = normalize(sfm::linalg::mtv(&pose.r, [n[0], n[1], 1.0]));
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

#[test]
fn photographs_of_a_known_scene_give_back_its_cameras() {
    let k = Intrinsics { f: 420.0, cx: 320.0, cy: 240.0, k1: -0.08, k2: 0.01, width: 640, height: 480 };
    let views = 8;
    let truth: Vec<Pose> = (0..views)
        .map(|i| {
            let a = i as f64 * std::f64::consts::TAU / views as f64 * 0.5;
            look_at([3.2 * a.sin(), -1.6, -3.2 * a.cos()], [0.0, 0.6, 0.0])
        })
        .collect();
    let images: Vec<Vec<u8>> = truth.iter().map(|p| shoot(&k, p)).collect();
    let photos: Vec<Photo> = images.iter().map(|rgb| Photo { width: 640, height: 480, rgb }).collect();
    let cfg = SfmCfg { focal_guess: 0.8, verbose: true, ..SfmCfg::default() };
    let rec = reconstruct(&photos, &cfg).expect("reconstruction");

    assert!(rec.poses.iter().all(|p| p.is_some()), "unregistered views: {:?}", rec.poses.iter().map(|p| p.is_some()).collect::<Vec<_>>());
    assert!(rec.rms_px < 1.0, "final reprojection rms {:.3} px", rec.rms_px);
    assert!((rec.intrinsics.f - k.f).abs() < 0.03 * k.f, "focal {:.1} against {}", rec.intrinsics.f, k.f);
    assert!((rec.intrinsics.k1 - k.k1).abs() < 0.03, "k1 {:.4} against {}", rec.intrinsics.k1, k.k1);

    // camera centres agree with the truth after the best similarity
    let est: Vec<V3> = rec.poses.iter().map(|p| p.unwrap().centre()).collect();
    let tru: Vec<V3> = truth.iter().map(|p| p.centre()).collect();
    let (ce, ct) = (centroid(&est), centroid(&tru));
    let se = est.iter().map(|p| norm2(sub(*p, ce))).sum::<f64>().sqrt();
    let st = tru.iter().map(|p| norm2(sub(*p, ct))).sum::<f64>().sqrt();
    // rotation between the two centred, normalized sets via the camera
    // orientations: R_align = R_true^T R_est for any view
    let r0 = sfm::linalg::mm(&sfm::linalg::transpose(&truth[0].r), &rec.poses[0].unwrap().r);
    let mut worst = 0.0f64;
    for i in 0..views {
        let a = scale(sub(est[i], ce), 1.0 / se);
        let b = scale(sub(tru[i], ct), 1.0 / st);
        let a_in_truth = mv(&r0, a);
        let e = sub(a_in_truth, b);
        worst = worst.max(dot(e, e).sqrt());
    }
    assert!(worst < 0.02, "worst camera-centre error {worst:.4} of the rig's size");
}

fn centroid(p: &[V3]) -> V3 {
    let n = p.len() as f64;
    [p.iter().map(|v| v[0]).sum::<f64>() / n, p.iter().map(|v| v[1]).sum::<f64>() / n, p.iter().map(|v| v[2]).sum::<f64>() / n]
}

fn norm2(v: V3) -> f64 {
    dot(v, v)
}
