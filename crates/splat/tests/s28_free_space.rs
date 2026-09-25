// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Free-space carving (`splat::carve`): a camera that measured a surface at
//! some range saw the ray in front of it empty, so a gaussian lying wholly
//! there is a floater whatever colour it serves. Per gaussian, the views
//! that saw empty space where it sits and the views that saw it on their
//! surface.
//!
//! Swedish Embedded AB implements reconstruction whose geometry is held to
//! what every camera observed. If your team needs expertise in multi-view
//! geometry then you can procure our services by sending an email to
//! info@swedishembedded.com.

use splat::carve::{carve, Observations};
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn board() -> Splats {
    let mut s = Splats::default();
    for iy in 0..16 {
        for ix in 0..16 {
            s.means.extend_from_slice(&[-1.2 + 0.15 * ix as f32 + 0.075, -1.2 + 0.15 * iy as f32 + 0.075, 3.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.08, 0.08, 0.01]);
            s.opacities.push(0.98);
            s.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
        }
    }
    s
}

fn one(x: f32, y: f32, z: f32) -> Splats {
    Splats { means: vec![x, y, z], quats: vec![1.0, 0.0, 0.0, 0.0], scales: vec![0.03; 3], opacities: vec![0.9], colors: vec![0.9, 0.1, 0.1], sh_rest: None }
}

#[test]
fn a_floater_is_where_the_cameras_saw_empty_space() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let (w, h) = (64u32, 48u32);
    let cams: Vec<Camera> = [[-0.6f32, 0.0, 0.0], [0.0, 0.2, 0.0], [0.6, 0.0, 0.0], [0.0, -0.4, 0.2]]
        .iter()
        .map(|e| Camera::look_at(*e, [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 60.0, w, h))
        .collect();
    // what the cameras measured: the board's surface, rendered
    let surface = board();
    let o = RenderOpts { ray: true, ..Default::default() };
    let mut r = Renderer::new(&g, Kernels::at(0), surface.len(), w, h, 0);
    let gs = GpuSplats::upload(&g, &surface);
    let ranges: Vec<Vec<f32>> = cams
        .iter()
        .map(|c| {
            r.render(&g, &gs, c, &o);
            let rgba = r.read_rgba(&g, w, h);
            r.read_aux(&g, w, h).chunks_exact(5).zip(rgba.chunks_exact(4)).map(|(a, p)| if p[3] > 0.5 { a[0] } else { 0.0 }).collect()
        })
        .collect();
    let obs = Observations::new(&g, &cams, &ranges);
    // the board, a floater halfway to it, and a gaussian behind it
    let scene = splat::align::concat(&[surface.clone(), one(0.1, 0.05, 1.5), one(-0.2, 0.1, 4.5)]);
    let s = GpuSplats::upload(&g, &scene);
    let v = carve(&g, Kernels::at(0), &s, &obs, 0.02);
    let n = surface.len();
    let on_surface = (0..n).filter(|&i| v[i][0] == 0.0 && v[i][1] >= 2.0).count();
    assert!(on_surface as f64 > 0.9 * n as f64, "{on_surface} of the board's {n} gaussians are supported and unviolated");
    assert!(v[n][0] >= 3.0, "the floater: {} views saw empty space where it is", v[n][0]);
    assert_eq!(v[n][1], 0.0, "no view saw the floater on its surface");
    assert_eq!(v[n + 1][0], 0.0, "a gaussian behind the surface is occluded, not a violation");
}

/// Through a fit: floaters planted in front of a surface are gone after a
/// fit that carves against the views' range measurements (what stereo hands
/// the fit: here the true surface's range), and the surface is untouched.
/// The range loss is off, so the measurements reach the fit only through
/// carving. A floater every view images cannot be carved against the
/// scene's own render - each view renders the floater itself there - so
/// independent measurements are what carving needs.
#[test]
fn a_fit_that_carves_leaves_no_floater() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let (w, h) = (64u32, 48u32);
    let cams: Vec<Camera> = [[-0.8f32, 0.0, 0.0], [-0.4, 0.3, 0.0], [0.0, 0.0, 0.0], [0.4, -0.3, 0.0], [0.8, 0.0, 0.0], [0.0, 0.5, 0.2]]
        .iter()
        .map(|e| Camera::look_at(*e, [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 60.0, w, h))
        .collect();
    let truth = board();
    let o = RenderOpts { ray: true, ..Default::default() };
    let mut r = Renderer::new(&g, Kernels::at(0), truth.len(), w, h, 0);
    let gs = GpuSplats::upload(&g, &truth);
    let targets: Vec<splat::opt::TargetView> = cams
        .iter()
        .map(|c| {
            r.render(&g, &gs, c, &o);
            let rgba = r.read_rgba(&g, w, h);
            let range: Vec<f32> = r.read_aux(&g, w, h).chunks_exact(5).zip(rgba.chunks_exact(4)).map(|(a, p)| if p[3] > 0.5 { a[0] } else { 0.0 }).collect();
            splat::opt::TargetView::new(*c, splat::renderer::rgba_to_rgb(&rgba)).with_depth(range, None)
        })
        .collect();
    let floaters: Vec<Splats> = (0..6).map(|k| one(-0.3 + 0.12 * k as f32, 0.1 * (k % 2) as f32, 1.6)).collect();
    let mut parts = vec![truth.clone()];
    parts.extend(floaters);
    let init = splat::align::concat(&parts);
    let cfg = splat::opt::FitCfg {
        iters: 120,
        lr_position: 1e-3,
        log_every: 0,
        strategy: splat::opt::Densify::Hybrid,
        densify_every: 30,
        densify_after: 30,
        densify_until: 100,
        max_gaussians: init.len(),
        depth_weight: 0.0,
        ..Default::default()
    };
    let near = |s: &Splats| s.means.chunks_exact(3).zip(&s.opacities).filter(|(m, &o)| m[2] < 2.4 && o > 0.05).count();
    let (plain, _) = splat::opt::fit(&g, Kernels::at(0), &init, &targets, &cfg, &mut |_, _| true);
    let (carved, _) = splat::opt::fit(&g, Kernels::at(0), &init, &targets, &splat::opt::FitCfg { carve: true, ..cfg }, &mut |_, _| true);
    assert!(near(&plain) > 0, "without carving the floaters survive the fit ({} left) - the test needs them to", near(&plain));
    assert_eq!(near(&carved), 0, "{} floaters survive a fit that carves", near(&carved));
    let board_left = carved.means.chunks_exact(3).filter(|m| (m[2] - 3.0).abs() < 0.1).count();
    assert!(board_left as f64 > 0.95 * truth.len() as f64, "{board_left} of the board's {} gaussians remain", truth.len());
}
