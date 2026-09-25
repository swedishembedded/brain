// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end validation of photographs -> splat scene on a scene whose truth
//! is KNOWN: a textured ground, a sphere and a box, all made of small flat
//! gaussians. Training images are rendered from two camera rings; the result
//! is judged on held-out cameras at azimuths, heights and distances no
//! training camera had, against the truth rendered from the same cameras.
//!
//! Two reconstructions are scored:
//!
//! * **A - trainer only**: the true cameras, started from every 10th true
//!   point. Isolates the fit from camera recovery. `--bare-sphere` leaves the
//!   sphere without a single starting point, as structure from motion leaves
//!   a textureless object: the fit has to grow it.
//! * **B - the whole pipeline**: the rendered images alone -> structure from
//!   motion -> training set -> fit, exactly what `brain splat train` runs; the
//!   result is compared in the truth's frame through the similarity the
//!   recovered cameras imply.
//!
//! ```text
//! synthetic_e2e <out dir> [iters] [width] [--only a|b|sfm] [--bare-sphere]
//!               [--contrast c] [--min-inliers n] <fit options>
//! ```
//!
//! Writes `heldout_<i>.png` (truth | A | B) for every held-out camera.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines and the evaluation that keeps them honest. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod common;

use common::{fit_cfg, fitted_opts, montage, psnr_masked, render, Flags, FIT_USAGE};
use gpu_core::Gpu;
use imaging::Rgb8;
use splat::opt::{fit_full, FitCfg, TargetView};
use splat::quality::sharpness_ratio;
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn hash(i: i64, j: i64, k: i64) -> f32 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (j as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9) ^ (k as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    z = (z ^ (z >> 29)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z >> 40) as f32) / (1u64 << 24) as f32
}

/// Multi-scale value noise in [0,1]: texture with detail at every scale.
fn noise(u: f32, v: f32, salt: i64) -> f32 {
    let (mut sum, mut amp, mut f) = (0.0, 0.5, 3.0f32);
    for o in 0..5 {
        let (x, y) = (u * f, v * f);
        let (i, j) = (x.floor() as i64, y.floor() as i64);
        let (a, b) = (x - x.floor(), y - y.floor());
        let (a, b) = (a * a * (3.0 - 2.0 * a), b * b * (3.0 - 2.0 * b));
        let h = |di: i64, dj: i64| hash(i + di, j + dj, salt * 7 + o);
        sum += amp * ((h(0, 0) * (1.0 - a) + h(1, 0) * a) * (1.0 - b) + (h(0, 1) * (1.0 - a) + h(1, 1) * a) * b);
        amp *= 0.5;
        f *= 2.0;
    }
    sum / 0.96875
}

/// A flat gaussian disc at `p` facing `n`, radius `r`, colour `c`.
fn disc(s: &mut Splats, p: [f32; 3], n: [f32; 3], r: f32, c: [f32; 3]) {
    // the quaternion taking +Z to n
    let (x, y, z) = (n[0], n[1], n[2]);
    let q = if z < -0.9999 { [0.0, 1.0, 0.0, 0.0] } else {
        let w = 1.0 + z;
        let l = (w * w + y * y + x * x).sqrt();
        [w / l, -y / l, x / l, 0.0]
    };
    s.means.extend_from_slice(&p);
    s.quats.extend_from_slice(&q);
    s.scales.extend_from_slice(&[r, r, r * 0.1]);
    s.opacities.push(0.97);
    s.colors.extend_from_slice(&c);
}

/// Where [`truth`] puts the sphere's gaussians: after the 160x160 ground.
const SPHERE: std::ops::Range<usize> = 160 * 160..160 * 160 + 5000;

/// The truth: +Y is DOWN, the ground is y = 0.5.
fn truth() -> Splats {
    let mut s = Splats::default();
    let n = 160;
    for i in 0..n {
        for j in 0..n {
            let (x, z) = (-4.0 + 8.0 * (i as f32 + 0.5) / n as f32, -4.0 + 8.0 * (j as f32 + 0.5) / n as f32);
            let t = noise(x * 0.8, z * 0.8, 1);
            let plank = if ((x * 2.0).floor() as i64) % 2 == 0 { 0.08 } else { -0.05 };
            // per-disc speckle: gravel-like detail at the truth's own
            // resolution, which is what gives feature matching corners
            let v = (0.3 + 0.45 * t + plank + 0.35 * (hash(i as i64, j as i64, 11) - 0.5)).clamp(0.0, 1.0);
            disc(&mut s, [x, 0.5, z], [0.0, -1.0, 0.0], 0.04, [v, 0.9 * v, 0.75 * v]);
        }
    }
    // sphere centre (0, -0.1, 0), radius 0.6: Fibonacci points
    let m = 5000;
    for k in 0..m {
        let yy = 1.0 - 2.0 * (k as f32 + 0.5) / m as f32;
        let rr = (1.0 - yy * yy).sqrt();
        let th = k as f32 * 2.399_963;
        let nrm = [rr * th.cos(), yy, rr * th.sin()];
        let p = [0.6 * nrm[0], -0.1 + 0.6 * nrm[1], 0.6 * nrm[2]];
        let stripe = if ((th.rem_euclid(std::f32::consts::TAU) * 3.0) as i64) % 2 == 0 { 1.0 } else { 0.4 };
        let t = noise(p[0] * 3.0 + 5.0, p[1] * 3.0 + p[2], 2) + 0.4 * (hash(k as i64, 0, 12) - 0.5);
        disc(&mut s, p, nrm, 0.022, [0.15 + 0.3 * t, (0.4 + 0.5 * t) * stripe, 0.2]);
    }
    // box [1.0, 1.8] x [-0.3, 0.5] x [-1.4, -0.6]
    let (lo, hi) = ([1.0f32, -0.3, -1.4], [1.8f32, 0.5, -0.6]);
    let steps = 40;
    for face in 0..5 {
        for a in 0..steps {
            for b in 0..steps {
                let (u, v) = ((a as f32 + 0.5) / steps as f32, (b as f32 + 0.5) / steps as f32);
                let (p, nrm) = match face {
                    0 => ([lo[0], lo[1] + u * (hi[1] - lo[1]), lo[2] + v * (hi[2] - lo[2])], [-1.0, 0.0, 0.0]),
                    1 => ([hi[0], lo[1] + u * (hi[1] - lo[1]), lo[2] + v * (hi[2] - lo[2])], [1.0, 0.0, 0.0]),
                    2 => ([lo[0] + u * (hi[0] - lo[0]), lo[1] + v * (hi[1] - lo[1]), lo[2]], [0.0, 0.0, -1.0]),
                    3 => ([lo[0] + u * (hi[0] - lo[0]), lo[1] + v * (hi[1] - lo[1]), hi[2]], [0.0, 0.0, 1.0]),
                    _ => ([lo[0] + u * (hi[0] - lo[0]), lo[1], lo[2] + v * (hi[2] - lo[2])], [0.0, -1.0, 0.0]),
                };
                let t = noise(u * 2.0 + face as f32, v * 2.0, 3) + 0.5 * (hash(a as i64, b as i64, 13 + face as i64) - 0.5);
                let check = if (a / 8 + b / 8) % 2 == 0 { 0.9 } else { 0.3 };
                disc(&mut s, p, nrm, 0.012, [check * (0.5 + 0.5 * t), 0.25 + 0.3 * t, 0.2 + 0.6 * check]);
            }
        }
    }
    s
}

fn ring(n: usize, radius: f32, height: f32, phase: f32, w: u32, h: u32) -> Vec<Camera> {
    (0..n)
        .map(|i| {
            let a = phase + i as f32 * std::f32::consts::TAU / n as f32;
            let eye = [radius * a.sin(), height, -radius * a.cos()];
            let mut c = Camera::look_at(eye, [0.3, 0.0, -0.3], [0.0, -1.0, 0.0], 55.0, w, h);
            c.fx = c.fy;
            c
        })
        .collect()
}

fn to_rgb8(rgb: &[f32], w: u32, h: u32) -> Rgb8 {
    Rgb8 { w, h, px: rgb.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect() }
}

fn main() {
    let flags = Flags::from_env();
    if flags.get("help").is_some() {
        println!("usage: synthetic_e2e <out dir> [iters] [width] [--only a|b|sfm] [--bare-sphere] [--contrast c] [--min-inliers n] {FIT_USAGE}");
        return;
    }
    let only = flags.get("only");
    let run_a = only.is_none() || only == Some("a");
    let run_b = only != Some("a");
    let sfm_only = only == Some("sfm");
    let out: String = flags.arg(0, "synthetic_e2e".to_string());
    let iters: usize = flags.arg(1, 3000);
    let width: u32 = flags.arg(2, 512);
    let height = width * 3 / 4;
    std::fs::create_dir_all(&out).expect("out dir");
    let g = Gpu::new(splat::PIPELINES);
    let ks = Kernels::at(0);
    let t0 = truth();
    let see = RenderOpts::default();

    let mut train = ring(16, 3.2, -1.4, 0.0, width, height);
    train.extend(ring(8, 2.6, -2.4, 0.2, width, height));
    // held out: between the training azimuths, at heights and distances no
    // training camera had
    let mut held = ring(4, 2.9, -1.9, 0.19, width, height);
    held.extend(ring(4, 3.6, -0.9, 0.61, width, height));
    let photos: Vec<Vec<f32>> = train.iter().map(|c| render(&g, &t0, c, &see)).collect();
    let truth_held: Vec<Vec<f32>> = held.iter().map(|c| render(&g, &t0, c, &see)).collect();
    for (i, p) in photos.iter().enumerate().step_by(8) {
        imaging::save(format!("{out}/train_{i}.png"), &to_rgb8(p, width, height)).expect("png");
    }
    println!("truth: {} gaussians, {} training views, {} held-out views at {width}x{height}", t0.len(), train.len(), held.len());

    let score = |s: &Splats, cams: &[Camera], want: &[Vec<f32>], fitted: &FitCfg| -> (f64, f64, Vec<Vec<f32>>) {
        let o = fitted_opts(fitted);
        let imgs: Vec<Vec<f32>> = cams.iter().map(|c| render(&g, s, c, &o)).collect();
        let each: Vec<f64> = imgs.iter().zip(want).map(|(i, w)| psnr_masked(i, w, None)).collect();
        let p = each.iter().sum::<f64>() / each.len() as f64;
        if cams.len() == held.len() {
            println!("  per held-out view: {}", each.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(" "));
        }
        let d = imgs.iter().zip(want).map(|(i, w)| sharpness_ratio(i, w, width as usize, height as usize)).sum::<f64>() / imgs.len() as f64;
        (p, d, imgs)
    };

    let cfg = fit_cfg(&flags, iters, train.len());
    let mut columns: Vec<Vec<Vec<f32>>> = vec![truth_held.clone()];

    // ---- A: the trainer alone, true cameras, a sparse start ----
    if run_a {
        let (mut xyz, mut rgb) = (Vec::new(), Vec::new());
        let bare = flags.get("bare-sphere").is_some();
        for i in (0..t0.len()).step_by(10) {
            if bare && SPHERE.contains(&i) {
                continue;
            }
            xyz.extend_from_slice(&t0.means[i * 3..i * 3 + 3]);
            rgb.extend_from_slice(&t0.colors[i * 3..i * 3 + 3]);
        }
        let mut init = splat::init::from_points(&xyz, &rgb, 0.5);
        splat::init::floor_to_pixels(&mut init, &train, 1.0);
        let targets: Vec<TargetView> = train.iter().zip(&photos).map(|(c, p)| TargetView::new(*c, p.clone())).collect();
        let t = std::time::Instant::now();
        let out_a = fit_full(&g, ks, &init, &targets, &cfg, &mut |_, _| true);
        let secs = t.elapsed().as_secs_f64();
        let (on_train, _, _) = score(&out_a.scene, &train, &photos, &cfg);
        let (on_held, detail, imgs) = score(&out_a.scene, &held, &truth_held, &cfg);
        println!(
            "A trainer only: {} gaussians in {secs:.0} s; training views {on_train:.2} dB; HELD-OUT {on_held:.2} dB, detail {detail:.3}",
            out_a.scene.len()
        );
        splat::ply::write(&format!("{out}/a.ply"), &out_a.scene).expect("ply");
        columns.push(imgs);
    }

    // ---- B: the whole pipeline from the images alone ----
    if run_b {
        let rgb8: Vec<Rgb8> = photos.iter().map(|p| to_rgb8(p, width, height)).collect();
        let mut sfm_cfg = sfm::incremental::SfmCfg { verbose: true, ..Default::default() };
        if let Some(c) = flags.parse("contrast") {
            sfm_cfg.sift.contrast = c;
        }
        if let Some(m) = flags.parse("min-inliers") {
            sfm_cfg.min_inliers = m;
        }
        let set = recon::photogrammetry::training_set(&rgb8, width, 0.5, &sfm_cfg).expect("structure from motion");
        let k = set.sfm.intrinsics[0];
        let true_f = train[0].fx as f64;
        println!(
            "B structure from motion: {}/{} registered, {} points, rms {:.2} px, focal {:.1} (true {true_f:.1}, {:+.2}%), {}",
            set.targets.len(),
            train.len(),
            set.sfm.points.len(),
            set.sfm.rms_px,
            k.fx,
            100.0 * (k.fx - true_f) / true_f,
            sfm::lens::describe(&k)
        );
        let tru: Vec<[f64; 16]> = set.source.iter().map(|&i| train[i].c2w.map(|v| v as f64)).collect();
        let rig = {
            let c: Vec<[f64; 3]> = tru.iter().map(|m| [m[3], m[7], m[11]]).collect();
            let m: [f64; 3] = std::array::from_fn(|k| c.iter().map(|p| p[k]).sum::<f64>() / c.len() as f64);
            (c.iter().map(|p| (0..3).map(|k| (p[k] - m[k]).powi(2)).sum::<f64>()).sum::<f64>() / c.len() as f64).sqrt()
        };
        // the similarity from the truth's frame into the reconstruction's,
        // and the worst camera centre left over, as a fraction of the rig
        let align = |cams: &[Camera]| {
            let est: Vec<[f64; 16]> = cams.iter().map(|c| c.c2w.map(|v| v as f64)).collect();
            let sim = splat::align::sim3_from_cameras(&est, &tru).expect("similarity");
            let worst = est
                .iter()
                .zip(&tru)
                .map(|(e, t)| {
                    let m = splat::align::transform_c2w_sim3(t, &sim);
                    ((m[3] - e[3]).powi(2) + (m[7] - e[7]).powi(2) + (m[11] - e[11]).powi(2)).sqrt() / sim.s
                })
                .fold(0.0f64, f64::max);
            (sim, worst / rig)
        };
        let sfm_cams: Vec<Camera> = set.targets.iter().map(|t| t.cam).collect();
        println!("B camera centres: worst {:.3}% of the rig radius after the similarity", 100.0 * align(&sfm_cams).1);
        if sfm_only {
            return;
        }
        let t = std::time::Instant::now();
        let out_b = fit_full(&g, ks, &set.init, &set.targets, &cfg, &mut |_, _| true);
        let secs = t.elapsed().as_secs_f64();
        let (sim, err) = align(&out_b.cams);
        if cfg.pose_lr > 0.0 {
            println!("B camera centres after the fit refined them: worst {:.3}% of the rig radius", 100.0 * err);
        }
        // the held-out cameras, in the reconstruction's frame, through its own
        // recovered calibration
        let f_scaled = (set.targets[0].cam.fx / set.targets[0].cam.width as f32) * width as f32;
        let held_b: Vec<Camera> = held
            .iter()
            .map(|c| {
                let m = splat::align::transform_c2w_sim3(&c.c2w.map(|v| v as f64), &sim);
                Camera { c2w: m.map(|v| v as f32), fx: f_scaled, fy: f_scaled, ..*c }
            })
            .collect();
        let train_cams = &out_b.cams;
        let train_imgs: Vec<Vec<f32>> = set.targets.iter().map(|t| t.rgb.clone()).collect();
        let (on_train, _, _) = score(&out_b.scene, train_cams, &train_imgs, &cfg);
        let (on_held, detail, imgs) = score(&out_b.scene, &held_b, &truth_held, &cfg);
        println!(
            "B whole pipeline: {} gaussians in {secs:.0} s; training views {on_train:.2} dB; HELD-OUT {on_held:.2} dB, detail {detail:.3}",
            out_b.scene.len()
        );
        splat::ply::write(&format!("{out}/b.ply"), &out_b.scene).expect("ply");
        columns.push(imgs);
    }

    for i in 0..held.len() {
        let row: Vec<&[f32]> = columns.iter().map(|c| c[i].as_slice()).collect();
        imaging::save(format!("{out}/heldout_{i}.png"), &montage(&row, width, height)).expect("png");
    }
    println!("held-out comparisons (truth | A | B) -> {out}/heldout_*.png");
}
