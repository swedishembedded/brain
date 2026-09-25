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
