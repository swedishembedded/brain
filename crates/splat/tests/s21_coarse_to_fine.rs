// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Coarse-to-fine fitting (`FitCfg::coarse`): the first part of a fit sees
//! every photograph at half resolution. That is only sound if a half-size
//! target and its half-size camera describe the SAME image - intrinsics
//! halved exactly, pixels box-averaged - and if the fit that switches over
//! ends where a full-resolution fit does.
//!
//! Swedish Embedded AB implements 3D reconstruction training schedules for
//! its clients. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit, FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

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

fn render(g: &Gpu, s: &Splats, c: &Camera) -> Vec<f32> {
    let mut r = Renderer::new(g, Kernels::at(0), s.len(), c.width, c.height, 0);
    r.render(g, &GpuSplats::upload(g, s), c, &RenderOpts { eps2d: 0.0, ..Default::default() });
    rgba_to_rgb(&r.read_rgba(g, c.width, c.height))
}

#[test]
fn a_half_size_view_is_the_same_view() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let truth = board(12, false);
    let c = Camera::look_at([0.3, -0.2, 0.1], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, 96, 64);
    let full = TargetView::new(c, render(&g, &truth, &c));
    let half = full.half();
    assert_eq!((half.cam.width, half.cam.height), (48, 32));
    let direct = render(&g, &truth, &half.cam);
    let p = psnr(&direct, &half.rgb);
    assert!(p > 30.0, "rendering through the half camera and box-averaging the full render agree to {p:.1} dB");
}

#[test]
fn switching_resolution_part_way_matches_a_full_resolution_fit_for_less_compute() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (64u32, 64u32);
    let truth = board(16, false);
    let t: Vec<TargetView> = [[0.0, 0.0, 0.0], [0.7, -0.25, 0.3], [-0.7, 0.25, 0.3]]
        .iter()
        .map(|e| {
            let c = Camera::look_at(*e, [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h);
            TargetView::new(c, render(&g, &truth, &c))
        })
        .collect();
    let init = board(8, true);
    let base = FitCfg { iters: 160, lr: 1e-2, log_every: 0, ..Default::default() };
    let (a, _) = fit(&g, Kernels::at(0), &init, &t, &base, &mut |_, _| true);
    // At LESS compute: a half-resolution iteration renders and differentiates
    // a quarter of the pixels, so 220 iterations with the first half coarse
    // cost about 137 full-resolution ones against the reference's 160.
    let (b, _) = fit(&g, Kernels::at(0), &init, &t, &FitCfg { iters: 220, coarse: 0.5, ..base }, &mut |_, _| true);
    let score = |s: &Splats| t.iter().map(|v| psnr(&render(&g, s, &v.cam), &v.rgb)).sum::<f64>() / t.len() as f64;
    let (pa, pb) = (score(&a), score(&b));
    assert!(pb > pa - 0.5, "full resolution throughout: {pa:.2} dB; half resolution for the first half: {pb:.2} dB");
}
