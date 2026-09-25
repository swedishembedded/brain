// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Credit-assigned, budgeted density control (`Densify::Hybrid`).
//!
//! The controller rests on one identity: the rasterizer's colour gradient is
//! `Σ_p Tα · dL/dC_p`, so a backward pass fed a per-pixel QUANTITY instead of
//! a loss gradient returns that quantity summed by each gaussian's own
//! compositing weights. That turns "which gaussians are responsible for the
//! remaining error" from a guess read off a positional gradient into a
//! measurement. These tests hold the identity, the split it drives, and what
//! it buys at an equal primitive budget.
//!
//! Swedish Embedded AB implements 3D reconstruction optimizers whose density
//! control spends a primitive budget where the image error is. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;
use splat::density::{self, Evidence, Policy};
use splat::opt::{fit, Densify, FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{rgba_to_rgb, BwdScratch, GpuSplats, Renderer, SplatGrads};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn cam(w: u32, h: u32) -> Camera {
    Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h)
}

fn render(g: &Gpu, s: &Splats, c: &Camera) -> Vec<f32> {
    let mut r = Renderer::new(g, Kernels::at(0), s.len().max(1), c.width, c.height, 0);
    r.render(g, &GpuSplats::upload(g, s), c, &RenderOpts { eps2d: 0.0, ..Default::default() });
    rgba_to_rgb(&r.read_rgba(g, c.width, c.height))
}

fn gaussian(x: f32, y: f32, s: [f32; 3], q: [f32; 4], rgb: [f32; 3]) -> Splats {
    Splats { means: vec![x, y, 3.0], quats: q.to_vec(), scales: s.to_vec(), opacities: vec![0.8], colors: rgb.to_vec(), sh_rest: None }
}

fn join(parts: &[Splats]) -> Splats {
    let mut s = Splats::default();
    for p in parts {
        s.means.extend_from_slice(&p.means);
        s.quats.extend_from_slice(&p.quats);
        s.scales.extend_from_slice(&p.scales);
        s.opacities.extend_from_slice(&p.opacities);
        s.colors.extend_from_slice(&p.colors);
    }
    s
}

/// The credit pass is exact bookkeeping: contributions sum to the frame's
/// rendered alpha, and the residual lands on the gaussian that caused it.
#[test]
fn credit_assignment_attributes_the_residual_to_its_cause() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let c = cam(48, 48);
    let scene = join(&[
        gaussian(-0.5, 0.0, [0.25, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.2]),
        gaussian(0.5, 0.0, [0.25, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.8]),
    ]);
    // the target differs from the render only in the LEFT gaussian's colour
    let mut wrong = scene.clone();
    wrong.colors[0] = 0.3;
    let target = render(&g, &wrong, &c);

    let o = RenderOpts { eps2d: 0.0, ..Default::default() };
    let mut r = Renderer::new(&g, Kernels::at(0), 2, c.width, c.height, 0);
    let gs = GpuSplats::upload(&g, &scene);
    r.render(&g, &gs, &c, &o);
    let rgba = r.read_rgba(&g, c.width, c.height);
    let pred = rgba_to_rgb(&rgba);
    let edges = density::edge_map(&target, 48, 48);
    let up = density::credit_upstream(&pred, &target, &edges, None);
    let dimg = g.storage_init("dimg", &up);
    let grads = SplatGrads::new(&g, 2);
    let mut scr = BwdScratch::new(&g, 2, 48 * 48, 0);
    r.render_bwd(&g, &gs, &c, &o, &dimg, None, &mut scr, &grads).unwrap();
    let mut ev = Evidence::new(2);
    ev.add_view(&g.read(&grads.d_colors, 6));

    let alpha: f32 = rgba.chunks_exact(4).map(|p| p[3]).sum();
    let total: f32 = ev.contribution.iter().sum();
    assert!((total - alpha).abs() < 1e-3 * alpha, "contributions sum to {total}, the frame's alpha to {alpha}");
    assert!(
        ev.residual[0] > 20.0 * ev.residual[1],
        "residual attributed {:.4} to the gaussian whose colour is wrong and {:.4} to the one that is right",
        ev.residual[0], ev.residual[1]
    );
}

/// Splitting along the long axis with the parent's mean and covariance
/// preserved changes the render far less than the reference split, which
/// shrinks every axis by 1.6 and moves the children half a sigma.
#[test]
fn the_long_axis_split_preserves_the_image() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let c = cam(64, 64);
    let q = [0.92, 0.0, 0.0, 0.39]; // rotated in the image plane
    let parent = gaussian(0.0, 0.0, [0.6, 0.12, 0.12], q, [0.7, 0.5, 0.3]);
    let before = render(&g, &parent, &c);

    let mut ev = Evidence::new(1);
    ev.contribution[0] = 1.0;
    ev.views[0] = 1;
    let mut ours = parent.clone();
    let mut marks = Vec::new();
    let policy = Policy { refine_frac: 1.0, ..Default::default() };
    let r = density::round(&mut ours, &ev, &[10.0], &mut marks, 2, &policy, 1);
    assert_eq!((r.split, ours.len()), (1, 2));

    let mut reference = parent.clone();
    let cfg = FitCfg { densify_frac: 1.0, ..Default::default() };
    splat::opt::densify_for_test(&mut reference, &[1.0], &cfg);
    assert_eq!(reference.len(), 2);

    let (a, b) = (psnr(&render(&g, &ours, &c), &before), psnr(&render(&g, &reference, &c), &before));
    assert!(a > 30.0 && a > b + 5.0, "long-axis split renders at {a:.1} dB against its parent, the reference split at {b:.1} dB");
}

fn board(cells: usize, flat: bool) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[-1.2 + step * (ix as f32 + 0.5), -1.2 + step * (iy as f32 + 0.5), 3.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.45, step * 0.45, step * 0.45]);
            s.opacities.push(0.99);
            let v = if flat { 0.5 } else if (ix + iy) % 2 == 0 { 0.88 } else { 0.12 };
            s.colors.extend_from_slice(&[v, v, v * 0.92 + 0.04]);
        }
    }
    s
}

/// At an equal budget, starting from a set of gaussians too coarse for the
/// target, the credit-assigned controller ends well below both reference
/// strategies - the heuristic allowed to grow as fast as it likes, and MCMC -
/// and never exceeds the budget it was given.
///
/// Measured: heuristic 0.007773, MCMC 0.012083, hybrid 0.003967, all at 256
/// gaussians. The first hybrid split every large gaussian along its long axis
/// only, which leaves a coarse BLOB as coarse as it was in two directions
/// out of three; it measured 0.009053, behind the heuristic. Splitting by
/// shape is what the margin is.
#[test]
fn credit_assigned_density_beats_the_heuristic_at_an_equal_budget() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (64u32, 64u32);
    let truth = board(16, false);
    let cams = [
        Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([0.7, -0.25, 0.3], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([-0.7, 0.25, 0.3], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
    ];
    let t: Vec<TargetView> = cams.iter().map(|c| TargetView::new(*c, render(&g, &truth, c))).collect();
    let coarse = board(4, true);
    let base = FitCfg {
        iters: 200,
        lr_position: 1e-2,
        log_every: 0,
        densify_every: 20,
        densify_after: 10,
        densify_until: 160,
        densify_frac: 0.3,
        max_gaussians: 256,
        ..Default::default()
    };
    let (a, la) = fit(&g, Kernels::at(0), &coarse, &t, &FitCfg { strategy: Densify::Heuristic, ..base }, &mut |_, _| true);
    let (b, lb) = fit(&g, Kernels::at(0), &coarse, &t, &FitCfg { strategy: Densify::Hybrid, ..base }, &mut |_, _| true);
    let (c, lc) = fit(&g, Kernels::at(0), &coarse, &t, &FitCfg { strategy: Densify::Heuristic, densify_frac: 1.0, ..base }, &mut |_, _| true);
    let (d, ld) = fit(&g, Kernels::at(0), &coarse, &t, &FitCfg { strategy: Densify::Mcmc, noise: 1.0, ..base }, &mut |_, _| true);
    assert!(b.len() <= base.max_gaussians, "hybrid grew to {} past a budget of {}", b.len(), base.max_gaussians);
    assert!(
        lb < 0.7 * lc && lb < 0.7 * ld && lb < la,
        "at a budget of {}: hybrid {lb:.6} ({} gaussians), heuristic {lc:.6} ({}) or {la:.6} at its default \
         rate ({}), MCMC {ld:.6} ({})",
        base.max_gaussians, b.len(), c.len(), a.len(), d.len()
    );
}
