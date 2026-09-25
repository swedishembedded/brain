// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Progressive spherical-harmonic bands (`FitCfg::sh_ramp`): view-dependent
//! colour is switched on one degree at a time, as 3D Gaussian Splatting
//! trains it, so it cannot explain away error that geometry has not had the
//! chance to yet. A band that has not switched on must neither colour a
//! splat nor move.
//!
//! Swedish Embedded AB implements 3D reconstruction training schedules for
//! its clients. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit, FitCfg, TargetView};
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

#[test]
fn bands_that_have_not_switched_on_stay_at_zero() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (32u32, 32u32);
    let mut truth = Splats::default();
    for i in 0..16 {
        let (x, y) = ((i % 4) as f32 * 0.5 - 0.75, (i / 4) as f32 * 0.5 - 0.75);
        truth.means.extend_from_slice(&[x, y, 3.0]);
        truth.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        truth.scales.extend_from_slice(&[0.2, 0.2, 0.2]);
        truth.opacities.push(0.9);
        truth.colors.extend_from_slice(&[0.2 + 0.04 * i as f32, 0.5, 0.7 - 0.03 * i as f32]);
    }
    let mut r = Renderer::new(&g, Kernels::at(0), truth.len(), w, h, 0);
    let gs = GpuSplats::upload(&g, &truth);
    let t: Vec<TargetView> = [[0.0, 0.0, 0.0], [0.6, 0.0, 0.2], [-0.6, 0.2, 0.2]]
        .iter()
        .map(|e| {
            let c = Camera::look_at(*e, [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 50.0, w, h);
            r.render(&g, &gs, &c, &RenderOpts::default());
            TargetView::new(c, rgba_to_rgb(&r.read_rgba(&g, w, h)))
        })
        .collect();
    let mut init = truth.clone();
    for c in init.colors.iter_mut() {
        *c = 0.5;
    }
    // A ramp twice the fit's length reaches degree 1 and no further.
    let cfg = FitCfg { iters: 30, lr: 1e-2, log_every: 0, sh_degree: 3, sh_ramp: 2.0, ..Default::default() };
    let (out, _) = fit(&g, Kernels::at(0), &init, &t, &cfg, &mut |_, _| true);
    let (degree, rest) = out.sh_rest.expect("an SH fit returns its coefficients");
    assert_eq!(degree, 3);
    let (mut low, mut high) = (0.0f32, 0.0f32);
    for coeffs in rest.chunks_exact(15) {
        low += coeffs[..3].iter().map(|v| v.abs()).sum::<f32>();
        high += coeffs[3..].iter().map(|v| v.abs()).sum::<f32>();
    }
    assert!(low > 0.0, "degree 1 was on for half the fit and never moved");
    assert_eq!(high, 0.0, "degrees 2 and 3 never switched on, yet their coefficients moved");
}
