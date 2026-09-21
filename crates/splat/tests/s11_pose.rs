// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fitting the cameras, not just the scene.
//!
//! A reconstruction pipeline that predicts its own poses hands the optimizer
//! cameras that are wrong, and `fit` has been treating them as ground truth.
//! It cannot work: gaussian positions are depth unprojected through a camera,
//! so a pose error is a position error for every pixel of that frame, and no
//! amount of moving gaussians reconciles two frames that disagree about where
//! they were taken from. Measured on a real capture, one view's gaussians
//! rendered from another view's camera score 7.9 dB against 23.9 dB from their
//! own - individually good, mutually inconsistent.
//!
//! The optimizer already has everything it needs. Perturbing a camera is
//! exactly equivalent to applying the inverse rigid motion to the scene, and
//! the gradients with respect to gaussian positions and orientations are
//! already computed exactly, so a pose gradient is a reduction over numbers
//! the backward pass has already produced - no new rasterizer path, and no
//! approximation of the chain rule.
//!
//! Swedish Embedded AB implements bundle-adjusted 3D reconstruction, including
//! the pose refinement that decides whether multiple views can agree at all.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit_bundle, FitCfg, TargetView};
use splat::quality::psnr;
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

/// Textured slab: enough structure that a pose error is visible in the loss.
fn slab() -> Splats {
    let mut r = Lcg(0xb00c);
    let mut s = Splats::default();
    for iy in 0..24 {
        for ix in 0..24 {
            s.means.extend_from_slice(&[
                ix as f32 * 0.11 - 1.265,
                iy as f32 * 0.11 - 1.265,
                4.0 + 0.35 * (ix as f32 * 0.7).sin(),
            ]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.07, 0.07, 0.07]);
            s.opacities.push(0.95);
            s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
        }
    }
    s
}

fn cams(w: u32, h: u32) -> Vec<Camera> {
    [[-1.1f32, -0.3, 0.2], [0.0, 0.0, 0.0], [1.1, 0.3, 0.2], [0.2, -1.0, 0.1]]
        .iter()
        .map(|e| Camera::look_at(*e, [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h))
        .collect()
}

/// Rotate a camera slightly about its own centre and nudge it sideways: the
/// shape of error a predicted pose actually has.
fn disturb(c: &Camera, ang: f32, shift: [f32; 3]) -> Camera {
    let (s, co) = (ang.sin(), ang.cos());
    let mut out = *c;
    // rotate about world Y, then translate
    for col in 0..4 {
        let (x, z) = (c.c2w[col], c.c2w[8 + col]);
        out.c2w[col] = co * x + s * z;
        out.c2w[8 + col] = -s * x + co * z;
    }
    for k in 0..3 {
        out.c2w[k * 4 + 3] += shift[k];
    }
    out
}

fn pose_err(a: &Camera, b: &Camera) -> (f32, f32) {
    let t = ((a.c2w[3] - b.c2w[3]).powi(2)
        + (a.c2w[7] - b.c2w[7]).powi(2)
        + (a.c2w[11] - b.c2w[11]).powi(2))
    .sqrt();
    // angle between the two rotations, via trace of R_a^T R_b
    let mut tr = 0.0f32;
    for i in 0..3 {
        for k in 0..3 {
            tr += a.c2w[i * 4 + k] * b.c2w[i * 4 + k];
        }
    }
    (((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos().to_degrees(), t)
}

#[test]
fn a_fit_can_recover_cameras_that_were_handed_to_it_wrong() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = slab();
    let truth_cams = cams(w, h);
    let o = RenderOpts::default();

    let mut ren = Renderer::new(&g, ks, truth.len(), w, h, 0);
    let gs = GpuSplats::upload(&g, &truth);
    let shots: Vec<Vec<f32>> = truth_cams
        .iter()
        .map(|c| {
            ren.render(&g, &gs, c, &o);
            ren.read_rgba(&g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
        })
        .collect();

    // hand the optimizer poses that are wrong, in different ways per view
    let wrong: Vec<Camera> = truth_cams
        .iter()
        .enumerate()
        .map(|(i, c)| match i {
            0 => disturb(c, 0.030, [0.04, -0.02, 0.03]),
            2 => disturb(c, -0.025, [-0.03, 0.03, -0.02]),
            3 => disturb(c, 0.020, [0.02, 0.02, 0.02]),
            // All four move: with the scene held fixed there is no gauge
            // freedom to anchor, and leaving one exact would make its PSNR
            // infinite and the mean meaningless.
            _ => disturb(c, -0.018, [0.03, -0.03, 0.01]),
        })
        .collect();

    let targets: Vec<TargetView> =
        wrong.iter().zip(&shots).map(|(c, rgb)| TargetView { cam: *c, rgb: rgb.clone() }).collect();

    let before: f64 = {
        let mut acc = 0.0;
        for (c, want) in wrong.iter().zip(&shots) {
            ren.render(&g, &gs, c, &o);
            let got: Vec<f32> =
                ren.read_rgba(&g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            acc += psnr(&got, want);
        }
        acc / shots.len() as f64
    };

    // The scene is already correct, so anything gained here is the cameras.
    let cfg = FitCfg { iters: 200, lr: 0.0, pose_lr: 3e-3, log_every: 0, ..Default::default() };
    let (fitted, refined, _) = fit_bundle(&g, ks, &truth, &targets, &cfg, &mut |_, _| true);

    let after: f64 = {
        let gs2 = GpuSplats::upload(&g, &fitted);
        let mut acc = 0.0;
        for (c, want) in refined.iter().zip(&shots) {
            ren.render(&g, &gs2, c, &o);
            let got: Vec<f32> =
                ren.read_rgba(&g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            acc += psnr(&got, want);
        }
        acc / shots.len() as f64
    };

    let mut worst_before = (0.0f32, 0.0f32);
    let mut worst_after = (0.0f32, 0.0f32);
    for i in 0..truth_cams.len() {
        let (a0, t0) = pose_err(&wrong[i], &truth_cams[i]);
        let (a1, t1) = pose_err(&refined[i], &truth_cams[i]);
        worst_before = (worst_before.0.max(a0), worst_before.1.max(t0));
        worst_after = (worst_after.0.max(a1), worst_after.1.max(t1));
    }
    println!(
        "  pose error: {:.3} deg / {:.4} before, {:.3} deg / {:.4} after; {before:.1} -> {after:.1} dB",
        worst_before.0, worst_before.1, worst_after.0, worst_after.1
    );

    assert!(
        after > before + 3.0,
        "refining the cameras took the views from {before:.1} dB to {after:.1} dB. The scene it \
         was given is exact, so everything here is the poses."
    );
    assert!(
        worst_after.0 < worst_before.0 * 0.6 && worst_after.1 < worst_before.1 * 0.6,
        "pose error went from {:.3} deg / {:.4} to {:.3} deg / {:.4} - the render may be improving \
         for some other reason, so the poses themselves have to move toward the truth",
        worst_before.0, worst_before.1, worst_after.0, worst_after.1
    );
}
