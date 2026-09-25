// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The whole sparse-start objective (`FitCfg::from_sparse_points`) end to
//! end: a textured scene started from a sparse subset of its own points - what
//! structure from motion hands the fit - must come out SHARPER, not as fog.
//!
//! The preset composes every trainer subsystem (L1 + D-SSIM, camera model,
//! credit-assigned density control, Mip filter, SH schedule, surface terms,
//! minibatches, the coarse phase), and its first version tested each of those
//! in isolation and none together. With raw linear learning rates it turned a
//! real 16-photo capture into uniform grey fog within 60 iterations; this is
//! the test that would have said so.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines for its clients. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit_full, FitCfg, TargetView};
use splat::quality::{psnr, sharpness_ratio};
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// A textured slab of small flat gaussians: detail everywhere, like a deck.
fn truth() -> Splats {
    let mut s = Splats::default();
    let n = 28;
    let step = 2.4 / n as f32;
    for iy in 0..n {
        for ix in 0..n {
            s.means.extend_from_slice(&[-1.2 + step * (ix as f32 + 0.5), -1.2 + step * (iy as f32 + 0.5), 3.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.6, step * 0.6, step * 0.1]);
            s.opacities.push(0.98);
            let v = 0.5 + 0.4 * ((ix * 7 + iy * 3) % 5) as f32 / 4.0 - 0.2 * ((ix / 3 + iy / 2) % 2) as f32;
            s.colors.extend_from_slice(&[v, 0.8 * v + 0.1, 0.6 * v]);
        }
    }
    // Distant background, as every real capture has (a horizon, a far
    // wall): it is what sets the scene's extent, so the subject's gaussians
    // are SMALL in the fit's normalized units - which is the regime where a
    // raw learning rate is enormous.
    for (x, y) in [(-12.0f32, -9.0f32), (12.0, -9.0), (-12.0, 9.0), (12.0, 9.0), (0.0, 0.0)] {
        s.means.extend_from_slice(&[x, y, 40.0]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[8.0, 8.0, 1.0]);
        s.opacities.push(0.98);
        s.colors.extend_from_slice(&[0.3, 0.4, 0.6]);
    }
    s
}

fn render(g: &Gpu, s: &Splats, c: &Camera, aa: bool) -> Vec<f32> {
    let mut r = Renderer::new(g, Kernels::at(0), s.len().max(1), c.width, c.height, 0).growable();
    let o = RenderOpts { antialiased: aa, ..Default::default() };
    r.render(g, &GpuSplats::upload(g, s), c, &o);
    rgba_to_rgb(&r.read_rgba(g, c.width, c.height))
}

#[test]
fn a_sparse_start_becomes_sharper_not_fog() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (48u32, 48u32);
    let t0 = truth();
    let targets: Vec<TargetView> = [[0.0, 0.0, 0.0], [0.6, -0.2, 0.2], [-0.6, 0.25, 0.25], [0.2, 0.5, 0.1], [-0.3, -0.5, 0.0], [0.5, 0.4, 0.3]]
        .iter()
        .map(|e| {
            let c = Camera::look_at(*e, [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 50.0, w, h);
            TargetView::new(c, render(&g, &t0, &c, false))
        })
        .collect();
    // every 12th point of the truth, as structure from motion would sample it
    let (mut xyz, mut rgb) = (Vec::new(), Vec::new());
    for i in (0..t0.len() - 5).step_by(12).chain(t0.len() - 5..t0.len()) {
        xyz.extend_from_slice(&t0.means[i * 3..i * 3 + 3]);
        rgb.extend_from_slice(&t0.colors[i * 3..i * 3 + 3]);
    }
    let init = splat::init::from_points(&xyz, &rgb, 0.5);
    let cfg = FitCfg { log_every: 0, ..FitCfg::from_sparse_points(160, 1200) };
    let out = fit_full(&g, Kernels::at(0), &init, &targets, &cfg, &mut |_, _| true);

    let score = |s: &Splats| -> (f64, f64) {
        let (mut p, mut sharp) = (0.0, 0.0);
        for t in &targets {
            let img = render(&g, s, &t.cam, true);
            p += psnr(&img, &t.rgb);
            sharp += sharpness_ratio(&img, &t.rgb, w as usize, h as usize);
        }
        (p / targets.len() as f64, sharp / targets.len() as f64)
    };
    let ((p0, s0), (p1, s1)) = (score(&init), score(&out.scene));
    // Measured: 16.61 dB and 0.001 of the target's detail at the start; the
    // preset reaches 25.97 dB and 0.234. With raw linear learning rates (the
    // first version) it reached 20.70 dB and 0.042 - fog.
    assert!(p1 > p0 + 6.0, "the fit took the sparse start from {p0:.2} dB to {p1:.2} dB");
    assert!(s1 > 0.15, "detail went from {s0:.3} to {s1:.3} of the target's; fog has none");
}

/// A region structure from motion left EMPTY is grown in from its
/// neighbours: a textureless object gives no keypoints, so the fit is handed
/// no points on it at all and has to reach it by travel and growth - here a
/// step raised in front of the slab, so reaching it means moving in depth
/// and not only along the surface the neighbours already sit on.
///
/// The step each gaussian takes is a budget in radii of the scene's median
/// gaussian. Measured on that median as density control left it, the step
/// shrank with every round that subdivided the scene, and a synthetic sphere
/// started with no points came back with its top half missing (29.8 dB on
/// its training views against 40.2 once the radius is the one the fit was
/// given) - which is how a real capture's watering can came apart.
#[test]
fn an_unseeded_region_is_grown_in() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (48u32, 48u32);
    // the slab's right third raised half a unit toward the cameras
    let mut t0 = truth();
    let raised = |x: f32, z: f32| x > 0.4 && z < 10.0;
    for i in 0..t0.len() {
        if raised(t0.means[i * 3], t0.means[i * 3 + 2]) {
            t0.means[i * 3 + 2] = 2.5;
        }
    }
    let targets: Vec<TargetView> = [[0.0, 0.0, 0.0], [0.6, -0.2, 0.2], [-0.6, 0.25, 0.25], [0.2, 0.5, 0.1], [-0.3, -0.5, 0.0], [0.5, 0.4, 0.3]]
        .iter()
        .map(|e| {
            let c = Camera::look_at(*e, [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 50.0, w, h);
            TargetView::new(c, render(&g, &t0, &c, false))
        })
        .collect();
    // every 12th point, except none at all on the slab's right third
    let hole = |i: usize| raised(t0.means[i * 3], t0.means[i * 3 + 2]);
    let (mut xyz, mut rgb) = (Vec::new(), Vec::new());
    for i in (0..t0.len() - 5).step_by(12).chain(t0.len() - 5..t0.len()).filter(|&i| !hole(i)) {
        xyz.extend_from_slice(&t0.means[i * 3..i * 3 + 3]);
        rgb.extend_from_slice(&t0.colors[i * 3..i * 3 + 3]);
    }
    let init = splat::init::from_points(&xyz, &rgb, 0.5);
    let cfg = FitCfg { log_every: 0, ..FitCfg::from_sparse_points(160, 1200) };
    let out = fit_full(&g, Kernels::at(0), &init, &targets, &cfg, &mut |_, _| true);

    // PSNR over the pixels where the head-on view sees the unseeded patch
    let c = targets[0].cam;
    let img = render(&g, &out.scene, &c, true);
    let (mut se, mut k) = (0.0f64, 0usize);
    for y in 8..40usize {
        for x in 0..w as usize {
            // the step spans x in (0.4, 1.2] at depth 2.5, 50 degrees across
            let wx = (x as f32 + 0.5 - 24.0) / 24.0 * 2.5 * (25.0f32.to_radians()).tan();
            if wx <= 0.45 || wx > 1.15 {
                continue;
            }
            for ch in 0..3 {
                let p = (y * w as usize + x) * 3 + ch;
                se += ((img[p] - targets[0].rgb[p]) as f64).powi(2);
                k += 1;
            }
        }
    }
    let patch = 10.0 * (k as f64 / se).log10();
    println!("unseeded patch: {patch:.2} dB over {} values", k);
    assert!(patch > 22.0, "the unseeded patch reached only {patch:.2} dB; the fit did not grow into it");
}
