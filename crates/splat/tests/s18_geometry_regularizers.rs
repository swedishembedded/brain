// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Geometry regularizers (2D Gaussian Splatting, Huang et al. 2024): the
//! terms that make a fit represent a surface AS a surface.
//!
//! Both defects they target are invisible to an RGB loss from the training
//! views. Two semi-transparent layers composite to the same colour as one
//! opaque one, and a disc renders nearly the same from the front whichever
//! way its flat axis points. From a new view the first is fog and the second
//! is fur. So these tests start from exactly those two defects and measure
//! the geometry, not the image.
//!
//! Swedish Embedded AB implements surface-accurate 3D reconstruction for its
//! clients. If your team needs splat scenes whose geometry holds up away from
//! the capture path, you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::Lcg;
use gpu_core::Gpu;
use splat::geometry::axis;
use splat::opt::{fit, FitCfg, TargetView};
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// A textured slab of flat discs at depth `z`, normals along the view axis.
fn slab(cells: usize, z: f32, opacity: f32) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[-1.2 + step * (ix as f32 + 0.5), -1.2 + step * (iy as f32 + 0.5), z]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.55, step * 0.55, step * 0.08]);
            s.opacities.push(opacity);
            let v = if (ix / 2 + iy / 2) % 2 == 0 { 0.8 } else { 0.25 };
            s.colors.extend_from_slice(&[v, 0.5 * v + 0.2, 0.3]);
        }
    }
    s
}

fn views(g: &Gpu, truth: &Splats, w: u32, h: u32) -> Vec<TargetView> {
    let eyes = [[0.0, 0.0, 0.0], [0.7, 0.0, 0.3], [-0.7, 0.1, 0.3], [0.0, 0.7, 0.3], [0.1, -0.7, 0.3]];
    let mut r = Renderer::new(g, Kernels::at(0), truth.len(), w, h, 0);
    let gs = GpuSplats::upload(g, truth);
    eyes.iter()
        .map(|e| {
            let c = Camera::look_at(*e, [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 50.0, w, h);
            r.render(g, &gs, &c, &RenderOpts::default());
            TargetView::new(c, rgba_to_rgb(&r.read_rgba(g, w, h)))
        })
        .collect()
}

fn cfg(iters: usize) -> FitCfg {
    FitCfg { iters, lr_position: 1e-2, lr_color: 1e-2, lr_rotation: 1e-2, log_every: 0, max_growth: 0.0, ..Default::default() }
}

/// Opacity-weighted spread of the gaussians' depths.
fn depth_spread(s: &Splats) -> f32 {
    let wsum: f32 = s.opacities.iter().sum();
    let mean: f32 = (0..s.len()).map(|i| s.opacities[i] * s.means[i * 3 + 2]).sum::<f32>() / wsum;
    ((0..s.len()).map(|i| s.opacities[i] * (s.means[i * 3 + 2] - mean).powi(2)).sum::<f32>() / wsum).sqrt()
}

/// Two half-transparent copies of a surface, 0.4 apart in depth, render
/// almost like the surface. Depth distortion pulls them together; RGB alone
/// barely does.
#[test]
fn depth_distortion_collapses_layers_onto_one_surface() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (48u32, 48u32);
    let t = views(&g, &slab(10, 3.0, 0.95), w, h);
    let mut init = slab(10, 2.8, 0.7);
    let back = slab(10, 3.2, 0.7);
    init.means.extend_from_slice(&back.means);
    init.quats.extend_from_slice(&back.quats);
    init.scales.extend_from_slice(&back.scales);
    init.opacities.extend_from_slice(&back.opacities);
    init.colors.extend_from_slice(&back.colors);

    let (plain, _) = fit(&g, Kernels::at(0), &init, &t, &cfg(150), &mut |_, _| true);
    let with = FitCfg { distortion_weight: 1.0, ..cfg(150) };
    let (reg, _) = fit(&g, Kernels::at(0), &init, &t, &with, &mut |_, _| true);
    let (a, b, start) = (depth_spread(&plain), depth_spread(&reg), depth_spread(&init));
    assert!(
        b < 0.5 * a && b < 0.5 * start,
        "depth spread {start:.4} at the start, {a:.4} after an RGB-only fit and {b:.4} with depth \
         distortion; the term exists to collapse layers the RGB loss is content with"
    );
}

/// |cos| between each disc's flat axis and the surface normal, averaged by
/// how much each one actually renders as a disc: opacity x face area x
/// flatness. A gaussian the fit shrank to a sphere has no orientation and
/// renders almost nothing - the RGB loss answers an edge-on disc exactly that
/// way - so it must not count.
fn alignment(s: &Splats) -> f32 {
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for i in 0..s.len() {
        let q = &s.quats[i * 4..i * 4 + 4];
        let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let q = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
        let sc = &s.scales[i * 3..i * 3 + 3];
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| sc[a].total_cmp(&sc[b]));
        let (c, b, a) = (sc[order[0]], sc[order[1]], sc[order[2]]);
        let w = s.opacities[i] * a * b * (1.0 - c / b);
        num += w * axis(q, order[0])[2].abs();
        den += w;
    }
    num / den.max(1e-12)
}

/// Discs on a plane with their flat axes pointing anywhere. Normal
/// consistency against the rendered depth's own normals - and, separately, a
/// supervised normal prior - turns them onto the plane.
#[test]
fn normal_terms_turn_discs_onto_the_surface() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (48u32, 48u32);
    let truth = slab(10, 3.0, 0.95);
    let t = views(&g, &truth, w, h);
    let mut init = truth.clone();
    let mut rng = Lcg::new(11);
    for q in init.quats.chunks_exact_mut(4) {
        for v in q.iter_mut() {
            *v = rng.signed();
        }
    }
    let start = alignment(&init);
    let (plain, _) = fit(&g, Kernels::at(0), &init, &t, &cfg(150), &mut |_, _| true);
    let consistency = FitCfg { normal_consistency_weight: 0.1, ..cfg(150) };
    let (a, _) = fit(&g, Kernels::at(0), &init, &t, &consistency, &mut |_, _| true);
    // the plane's normal, facing each camera, in that camera's frame
    let with_prior: Vec<TargetView> = t
        .iter()
        .map(|v| {
            let vm = v.cam.viewmat();
            let n = [-vm[2], -vm[6], -vm[10]];
            v.clone().with_normals((0..(w * h) as usize).flat_map(|_| n).collect())
        })
        .collect();
    let prior = FitCfg { normal_prior_weight: 0.1, ..cfg(150) };
    let (b, _) = fit(&g, Kernels::at(0), &init, &with_prior, &prior, &mut |_, _| true);
    let (p, ca, cb) = (alignment(&plain), alignment(&a), alignment(&b));
    // Measured: 0.380 at the start, 0.605 RGB-only, 0.969 with consistency and
    // 0.999 with the prior, both at weight 0.1.
    assert!(
        ca > 0.9 && ca > p + 0.15,
        "alignment {start:.3} at the start, {p:.3} RGB-only, {ca:.3} with normal consistency"
    );
    assert!(cb > 0.9 && cb > p + 0.15, "alignment {p:.3} RGB-only, {cb:.3} with a normal prior");
}
