// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adaptive density control: the fit may ADD gaussians where the scene needs
//! detail, instead of only moving and growing the ones it started with.
//!
//! Without it, a fit has exactly one way to cover an under-represented
//! region - make the gaussians it already has bigger. Measured on a real
//! six-photograph reconstruction, that is visible in the size distribution:
//! fitting drove the median projected splat DOWN (0.61 to 0.37 px) while
//! dragging the 99th percentile UP (3.34 to 8.72 px). The scene grows a tail
//! of blobs precisely where it should have grown detail.
//!
//! The reference pipeline has this and brain did not. Upstream 3DGS densifies
//! on the positional gradient, splitting gaussians that are large and cloning
//! those that are small, and prunes the near-transparent ones.
//!
//! Swedish Embedded AB implements 3D Gaussian Splatting optimizers, including
//! the density control that decides where detail can appear at all. If your
//! team needs reconstruction quality that is not capped by its initialization,
//! you can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit, FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// A checkerboard slab of `cells`x`cells` gaussians spanning the same area
/// whatever `cells` is - so a coarse one is the SAME scene under-sampled, not
/// a smaller scene.
fn board(cells: usize, flat: bool) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[
                -1.2 + step * (ix as f32 + 0.5),
                -1.2 + step * (iy as f32 + 0.5),
                3.0,
            ]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.45, step * 0.45, step * 0.45]);
            s.opacities.push(0.99);
            let v = if flat { 0.5 } else if (ix + iy) % 2 == 0 { 0.88 } else { 0.12 };
            s.colors.extend_from_slice(&[v, v, v * 0.92 + 0.04]);
        }
    }
    s
}

fn targets(g: &Gpu, ks: Kernels, truth: &Splats, w: u32, h: u32) -> Vec<TargetView> {
    let cams = [
        Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([0.7, -0.25, 0.3], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([-0.7, 0.25, 0.3], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
    ];
    let o = RenderOpts::default();
    let mut ren = Renderer::new(g, ks, truth.len(), w, h, 0);
    let gs = GpuSplats::upload(g, truth);
    cams.iter()
        .map(|c| {
            ren.render(g, &gs, c, &o);
            let img = ren.read_rgba(g, c.width, c.height);
            TargetView { cam: *c, rgb: img.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect() }
        })
        .collect()
}

/// The scene to recover has detail at a spatial frequency the starting set of
/// gaussians cannot represent - eight times fewer of them, spread over the
/// same area. No amount of moving, recolouring or resizing 64 gaussians makes
/// a 16x16 checkerboard, so a fit that cannot add any is capped by its
/// initialization and a fit that can is not.
#[test]
fn a_fit_that_can_add_gaussians_beats_one_that_cannot() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = board(16, false);
    let t = targets(&g, ks, &truth, w, h);
    let coarse = board(8, true);

    let fixed = FitCfg { iters: 160, lr: 1e-2, log_every: 0, densify_every: 0, ..Default::default() };
    let (a, mse_fixed) = fit(&g, ks, &coarse, &t, &fixed, &mut |_, _| true);

    let grown = FitCfg { iters: 160, lr: 1e-2, log_every: 0, densify_every: 20, densify_after: 10, densify_frac: 0.30, ..Default::default() };
    let (b, mse_grown) = fit(&g, ks, &coarse, &t, &grown, &mut |_, _| true);

    // Staging alone costs ~3.5% of final loss (Adam momentum restarts at each
    // boundary), so density control has to earn that back before it pays at
    // all - which is why the margin below is demanded rather than any
    // improvement being accepted.
    assert_eq!(a.len(), coarse.len(), "the fixed-set fit must not change the gaussian count");
    assert!(
        b.len() > coarse.len(),
        "density control added nothing: still {} gaussians from {}",
        b.len(), coarse.len()
    );
    assert!(
        mse_grown < mse_fixed * 0.9,
        "density control did not pay for itself: {mse_grown:.6} against {mse_fixed:.6} for a fit \
         that cannot add gaussians, on a target the initial set cannot represent"
    );

    // and the improvement is visible in the rendered frame, not only in the loss
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, b.len().max(a.len()), w, h, 0);
    let mut render = |s: &Splats| -> Vec<f32> {
        let gs = GpuSplats::upload(&g, s);
        ren.render(&g, &gs, &t[0].cam, &o);
        ren.read_rgba(&g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
    };
    let (ra, rb) = (render(&a), render(&b));
    let (da, db) = (psnr(&ra, &t[0].rgb), psnr(&rb, &t[0].rgb));
    assert!(db > da + 1.0, "densified fit renders at {db:.1} dB against {da:.1} dB for the fixed one");
}

/// Density control must not run away: a scene that ALREADY represents its
/// targets has no under-reconstructed region to subdivide, so the count must
/// stay in the same order of magnitude rather than doubling every interval.
#[test]
fn density_control_leaves_an_adequate_scene_alone() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = board(12, false);
    let t = targets(&g, ks, &truth, w, h);

    let cfg = FitCfg { iters: 120, lr: 5e-3, log_every: 0, densify_every: 30, densify_after: 10, ..Default::default() };
    let (out, _) = fit(&g, ks, &truth, &t, &cfg, &mut |_, _| true);
    assert!(
        out.len() < truth.len() * 3,
        "density control grew an already-adequate scene from {} to {} gaussians",
        truth.len(), out.len()
    );
}

/// A fitted gaussian has to stay a blob rather than becoming a hair.
///
/// `fit` clamped each axis to `[min_scale, 0.3]` INDEPENDENTLY, which bounds
/// how big a gaussian gets and says nothing about its shape. The optimizer
/// exploited that: a needle aligned with a training view's ray lowers that
/// view's loss and is invisible in it, while from any other angle it is a
/// streak across the frame. Measured on a real 23-view reconstruction, the
/// model's own output was 0.1% needles (worst axis ratio 54:1) and the same
/// scene after 400 iterations of fitting was 40.7% needles at up to 3000:1 -
/// entirely manufactured by the fit, and entirely invisible to a metric that
/// only renders the views it was fitted to.
#[test]
fn fitting_does_not_stretch_gaussians_into_needles() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = board(16, false);
    let t = targets(&g, ks, &truth, w, h);

    // start from a scene that must move a long way, which is when the
    // optimizer reaches for degenerate shapes
    let mut init = board(8, true);
    for v in init.means.iter_mut() {
        *v *= 1.04;
    }

    let aspect = |s: &Splats| -> f32 {
        (0..s.len())
            .map(|i| {
                let a = &s.scales[i * 3..i * 3 + 3];
                a.iter().fold(0.0f32, |m, &v| m.max(v)) / a.iter().fold(f32::MAX, |m, &v| m.min(v)).max(1e-12)
            })
            .fold(0.0f32, f32::max)
    };

    let cfg = FitCfg { iters: 200, lr: 2e-2, log_every: 0, max_aspect: 6.0, ..Default::default() };
    let (fitted, _) = fit(&g, ks, &init, &t, &cfg, &mut |_, _| true);
    let worst = aspect(&fitted);
    assert!(
        worst <= 6.0 + 1e-3,
        "a fitted gaussian reached {worst:.1}:1 against a {}:1 cap; the clamp is not binding",
        cfg.max_aspect
    );

    // and the cap has to be the thing doing it - without one, the same fit
    // reaches for far more extreme shapes
    let loose = FitCfg { iters: 200, lr: 2e-2, log_every: 0, max_aspect: 0.0, ..Default::default() };
    let (unclamped, _) = fit(&g, ks, &init, &t, &loose, &mut |_, _| true);
    assert!(
        aspect(&unclamped) > worst * 1.5,
        "the unclamped fit only reached {:.1}:1, so this test is not exercising the clamp",
        aspect(&unclamped)
    );
}

/// Density control must not be blind to the gaussians it exists to find.
///
/// A gaussian big enough to straddle a detail is pushed one way by the pixels
/// on one side and the other way by the pixels on the other. Those pushes
/// cancel in the summed gradient, so the standard criterion - the magnitude of
/// the summed 2D position gradient - reports the blurriest gaussian in the
/// scene as perfectly converged and never splits it. That is the whole
/// mechanism behind reconstructions that stay soft no matter how long they
/// train, and it is why AbsGS sums magnitudes instead.
///
/// Constructed so the cancellation is exact rather than approximate: one
/// gaussian centred in frame with a uniform image-space gradient. The pull
/// from every pixel left of centre is the exact negative of its mirror on the
/// right, so the summed criterion is zero by symmetry while every pixel is
/// telling the gaussian something.
#[test]
fn the_split_criterion_sees_a_gaussian_that_straddles_a_detail() {
    use splat::renderer::{BwdScratch, SplatGrads};

    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h);

    let s = Splats {
        means: vec![0.0, 0.0, 4.0],
        quats: vec![1.0, 0.0, 0.0, 0.0],
        scales: vec![0.5, 0.5, 0.5],
        opacities: vec![0.8],
        colors: vec![0.9, 0.9, 0.9],
        sh_rest: None,
    };
    let px = (w * h) as usize;
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, 1, w, h, 0);
    let gs = GpuSplats::upload(&g, &s);
    ren.render(&g, &gs, &cam, &o);

    // uniform dL/dimg: symmetric about the gaussian, so the SUM cancels
    let wimg: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { 1.0 }).collect();
    let dimg = g.storage_init("dimg", &wimg);
    let grads = SplatGrads::new(&g, 1);
    g.submit(
        &[&grads.d_gauss, &grads.d_opac, &grads.d_colors, &grads.d_absgrad, &grads.d_sumgrad],
        &[],
    );
    let mut scr = BwdScratch::new(&g, 1, px, 0);
    ren.render_bwd(&g, &gs, &cam, &o, &dimg, &mut scr, &grads).expect("fits");

    let sg = g.read(&grads.d_sumgrad, 2);
    let summed = (sg[0] * sg[0] + sg[1] * sg[1]).sqrt();
    let homodirectional = g.read(&grads.d_absgrad, 1)[0];

    assert!(
        homodirectional > 1e-3,
        "the homodirectional criterion is {homodirectional:.3e}: every pixel is pulling on this \
         gaussian, so it must not read as zero"
    );
    assert!(
        summed < 0.02 * homodirectional,
        "the summed criterion reads {summed:.3e} against a homodirectional {homodirectional:.3e}. \
         These are supposed to disagree by a lot here - if they do not, the cancellation this \
         test is built on is not happening and it is checking nothing."
    );
}
