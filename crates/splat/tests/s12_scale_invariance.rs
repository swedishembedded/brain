// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The same scene, at three sizes, must come out the same.
//!
//! A reconstruction has no intrinsic units. Photograph a model railway or a
//! mountain and the pipeline sees the same rays; only the numbers differ. So
//! every result here should be invariant to scaling the whole world, and any
//! constant expressed in world units breaks that invariance silently.
//!
//! This is not a hypothetical class of bug. Three shipped at once:
//!
//! - the largest a fitted gaussian could be was 0.3 world units, which is
//!   about 6 px on a scene three units away and about 140 px on a capture
//!   whose camera orbit has radius 0.5. At the second scale the optimizer
//!   could cover any gap it could not explain with a splat spanning a fifth
//!   of the frame, and the result looked correct from the views it was fitted
//!   to and like hair from every other;
//! - duplicate gaussians were fused at 0.002 world units, which on that same
//!   capture is 0.94 px, so it merged NEIGHBOURING PIXELS of one view rather
//!   than duplicates across views, destroying the pixel alignment the whole
//!   back-projection depends on;
//! - the smallest a gaussian could be was 1e-4 world units, meaningless at
//!   either end.
//!
//! Every test in this suite ran at exactly one scale, so not one of them could
//! see any of it. Testing at several is what makes the class visible at all,
//! and it is cheap: the same fit, three times, on a small scene.
//!
//! Swedish Embedded AB implements reconstruction pipelines whose behaviour is
//! a property of the geometry rather than of the units it arrived in. If your
//! team needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit, FitCfg, TargetView};
use splat::quality::{psnr, sharpness_ratio};
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

/// A textured slab at unit scale, to be multiplied by `k`.
fn scene(k: f32) -> Splats {
    let mut r = Lcg(0x5ca1e);
    let mut s = Splats::default();
    for iy in 0..20 {
        for ix in 0..20 {
            s.means.extend_from_slice(&[
                (ix as f32 * 0.13 - 1.235) * k,
                (iy as f32 * 0.13 - 1.235) * k,
                (3.0 + 0.3 * (ix as f32 * 0.9).sin()) * k,
            ]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.075 * k, 0.075 * k, 0.075 * k]);
            s.opacities.push(0.93);
            let sh = 0.55 + 0.45 * ((ix / 2 + iy / 2) % 2) as f32;
            s.colors.extend_from_slice(&[r.next() * sh, r.next() * sh, r.next() * sh]);
        }
    }
    s
}

fn cams(k: f32, w: u32, h: u32) -> Vec<Camera> {
    [[0.0f32, 0.0, 0.0], [0.8, -0.3, 0.3], [-0.8, 0.3, 0.3], [0.1, 0.9, 0.2]]
        .iter()
        .map(|e| {
            Camera::look_at(
                [e[0] * k, e[1] * k, e[2] * k],
                [0.0, 0.0, 3.0 * k],
                [0.0, -1.0, 0.0],
                55.0,
                w,
                h,
            )
        })
        .collect()
}

/// Longest axis of the biggest gaussian, in PIXELS as its own cameras see it.
/// Expressed this way the answer cannot depend on the scene's units, so if it
/// moves with `k` something in the pipeline is measuring the world in metres.
fn biggest_px(s: &Splats, cams: &[Camera]) -> f32 {
    let unit = splat::mip::smoothing_sigma(s, cams, 1.0);
    let mut worst = 0.0f32;
    for i in 0..s.len() {
        if unit[i] <= 0.0 {
            continue;
        }
        let lng = s.scales[i * 3].max(s.scales[i * 3 + 1]).max(s.scales[i * 3 + 2]);
        worst = worst.max(lng / unit[i]);
    }
    worst
}

struct Outcome {
    train_db: f64,
    novel_db: f64,
    sharp: f64,
    biggest_px: f32,
}

fn run(g: &Gpu, ks: Kernels, k: f32) -> Outcome {
    let (w, h) = (56u32, 56u32);
    let truth = scene(k);
    let all = cams(k, w, h);
    let o = RenderOpts::default();
    let mut ren = Renderer::new(g, ks, truth.len() * 2, w, h, 0);
    let gs = GpuSplats::upload(g, &truth);
    let shots: Vec<Vec<f32>> = all
        .iter()
        .map(|c| {
            ren.render(g, &gs, c, &o);
            ren.read_rgba(g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
        })
        .collect();

    // fit against the first three, hold the fourth back
    let targets: Vec<TargetView> = all[..3]
        .iter()
        .zip(&shots)
        .map(|(c, rgb)| TargetView::new(*c, rgb.clone()))
        .collect();

    // start from a subsample, so the fit has to grow the scene and is tempted
    // to do it by inflating what it already has
    let mut init = Splats::default();
    for i in (0..truth.len()).step_by(3) {
        init.means.extend_from_slice(&truth.means[i * 3..i * 3 + 3]);
        init.quats.extend_from_slice(&truth.quats[i * 4..i * 4 + 4]);
        init.scales.extend_from_slice(&truth.scales[i * 3..i * 3 + 3]);
        init.opacities.push(truth.opacities[i]);
        init.colors.extend_from_slice(&truth.colors[i * 3..i * 3 + 3]);
    }

    let cfg = FitCfg { iters: 140, lr: 8e-3, log_every: 0, ..Default::default() };
    let (fitted, _) = fit(g, ks, &init, &targets, &cfg, &mut |_, _| true);

    let mut ren2 = Renderer::new(g, ks, fitted.len(), w, h, 0);
    let fgs = GpuSplats::upload(g, &fitted);
    let score = |ren: &mut Renderer, c: &Camera, want: &[f32]| -> (f64, f64) {
        ren.render(g, &fgs, c, &o);
        let got: Vec<f32> =
            ren.read_rgba(g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        (psnr(&got, want), sharpness_ratio(&got, want, w as usize, h as usize))
    };
    let mut train = 0.0;
    let mut sharp = 0.0;
    for (c, want) in all[..3].iter().zip(&shots) {
        let (d, s) = score(&mut ren2, c, want);
        train += d / 3.0;
        sharp += s / 3.0;
    }
    let (novel_db, _) = score(&mut ren2, &all[3], &shots[3]);
    Outcome { train_db: train, novel_db, sharp, biggest_px: biggest_px(&fitted, &all) }
}

/// Shrink the world by a hundred and grow it by a hundred. Nothing should move.
#[test]
fn a_fit_gives_the_same_answer_whatever_the_scene_is_measured_in() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let scales = [0.01f32, 1.0, 100.0];
    let r: Vec<Outcome> = scales.iter().map(|&k| run(&g, ks, k)).collect();

    for (k, o) in scales.iter().zip(&r) {
        println!(
            "  scale x{k:<7} train {:.1} dB  novel {:.1} dB  sharpness {:.3}  biggest splat {:.1} px",
            o.train_db, o.novel_db, o.sharp, o.biggest_px
        );
    }

    let span = |f: &dyn Fn(&Outcome) -> f64| -> f64 {
        let v: Vec<f64> = r.iter().map(f).collect();
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    assert!(
        span(&|o| o.train_db) < 1.5,
        "training quality varies by {:.1} dB across scene scale - something in the fit is \
         measuring the world in units rather than in what the cameras resolve",
        span(&|o| o.train_db)
    );
    assert!(
        span(&|o| o.novel_db) < 1.5,
        "novel-view quality varies by {:.1} dB across scene scale",
        span(&|o| o.novel_db)
    );
    assert!(
        span(&|o| o.biggest_px as f64) < 6.0,
        "the biggest splat a fit produces varies by {:.1} px across scene scale. That is the \
         shape of the bug this test exists for: a bound in world units is no bound at all, and \
         at the scale where it stops binding the scene fills with streaks",
        span(&|o| o.biggest_px as f64)
    );
}

/// Scale invariance is worth nothing if the fit is bad at every scale, so pin
/// the quality too - and pin it on a view the fit never saw.
#[test]
fn and_the_answer_it_gives_is_a_good_one() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let o = run(&g, ks, 1.0);
    println!(
        "  train {:.1} dB  novel {:.1} dB  sharpness {:.3}  biggest splat {:.1} px",
        o.train_db, o.novel_db, o.sharp, o.biggest_px
    );
    assert!(o.novel_db > 20.0, "a view the fit never saw scores only {:.1} dB", o.novel_db);
    assert!(
        o.train_db - o.novel_db < 8.0,
        "the fit is {:.1} dB better on the views it trained against than on one it did not, which \
         is memorisation rather than reconstruction",
        o.train_db - o.novel_db
    );
    assert!(
        (0.5..1.6).contains(&o.sharp),
        "fitted sharpness {:.3}: below the band is blur, above it is the speckle of a scene made \
         of splats too small to cover what they are supposed to",
        o.sharp
    );
}
