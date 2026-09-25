// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Things that were not there in every photograph - a passer-by, a car, a
//! hand - are not part of the scene, and a fit that supervises them
//! faithfully bakes them in as floaters in front of the views that saw them.
//! With `FitCfg::transients` the fit recognizes them by their residual: a
//! large, spatially coherent region that one view insists on and the scene
//! the other views agree on cannot explain, and stops supervising it
//! (after RobustNeRF, Sabour et al. 2023). Scattered error - fine texture
//! the scene has not resolved yet - is not coherent, and stays supervised.
//!
//! Swedish Embedded AB implements 3D reconstruction of real-world captures,
//! including the people and traffic moving through them. If your team needs
//! expertise in robust radiance-field fitting then you can procure our
//! services by sending an email to info@swedishembedded.com.

use splat::opt::{fit, Densify, FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn board(cells: usize) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[-1.2 + step * (ix as f32 + 0.5), -1.2 + step * (iy as f32 + 0.5), 3.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.45, step * 0.45, step * 0.05]);
            s.opacities.push(0.99);
            let v = if (ix + iy) % 2 == 0 { 0.8 } else { 0.2 };
            s.colors.extend_from_slice(&[v, v * 0.9, v * 0.7]);
        }
    }
    s
}

#[test]
fn a_transient_in_some_photographs_is_not_baked_into_the_scene() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 48u32);
    let truth = board(12);
    let o = RenderOpts { ray: true, ..Default::default() };
    let cams: Vec<Camera> = (0..8)
        .map(|i| {
            let a = -0.35 + 0.1 * i as f32;
            Camera::look_at([2.0 * a.sin(), 0.2 * (i % 2) as f32, 3.0 - 2.0 * a.cos()], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 60.0, w, h)
        })
        .collect();
    let mut ren = Renderer::new(&g, ks, truth.len(), w, h, 0);
    let gs = GpuSplats::upload(&g, &truth);
    let clean: Vec<Vec<f32>> = cams
        .iter()
        .map(|c| {
            ren.render(&g, &gs, c, &o);
            rgba_to_rgb(&ren.read_rgba(&g, w, h))
        })
        .collect();
    // a bright block over a quarter of two photographs
    let disturbed = [2usize, 5];
    let targets: Vec<TargetView> = cams
        .iter()
        .zip(&clean)
        .enumerate()
        .map(|(i, (c, rgb))| {
            let mut rgb = rgb.clone();
            if disturbed.contains(&i) {
                for y in 8..28 {
                    for x in 10..34 {
                        rgb[((y * w as usize) + x) * 3..((y * w as usize) + x) * 3 + 3].copy_from_slice(&[0.95, 0.3, 0.9]);
                    }
                }
            }
            TargetView::new(*c, rgb)
        })
        .collect();
    let init = board(6);
    let base = FitCfg {
        iters: 400,
        lr_position: 1e-2,
        log_every: 0,
        strategy: Densify::Hybrid,
        densify_every: 25,
        densify_after: 25,
        densify_until: 300,
        max_gaussians: 400,
        ..Default::default()
    };
    let (plain, _) = fit(&g, ks, &init, &targets, &base, &mut |_, _| true);
    let (robust, _) = fit(&g, ks, &init, &targets, &FitCfg { transients: true, ..base }, &mut |_, _| true);
    // PSNR against the clean truth of view `i`, over the rectangle the
    // transient covered, or the whole frame
    let score = |s: &Splats, i: usize, region: bool| -> f64 {
        let mut r = Renderer::new(&g, ks, s.len(), w, h, 0);
        r.render(&g, &GpuSplats::upload(&g, s), &cams[i], &o);
        let img = rgba_to_rgb(&r.read_rgba(&g, w, h));
        if !region {
            return psnr(&img, &clean[i]);
        }
        let pick = |v: &[f32]| -> Vec<f32> { (8..28).flat_map(|y| (10..34).flat_map(move |x| (0..3).map(move |c| v[(y * w as usize + x) * 3 + c]))).collect() };
        psnr(&pick(&img), &pick(&clean[i]))
    };
    let mean = |s: &Splats, region: bool| disturbed.iter().map(|&i| score(s, i, region)).sum::<f64>() / disturbed.len() as f64;
    let (a, b) = (mean(&robust, true), mean(&plain, true));
    assert!(a > b + 3.0, "where the transient was, the clean truth renders at {a:.1} dB fitted robustly and {b:.1} dB fitted plainly");
    let (a, b) = (mean(&robust, false), mean(&plain, false));
    assert!(a >= b, "over the whole of the disturbed views: {a:.1} dB fitted robustly, {b:.1} dB fitted plainly");
}
