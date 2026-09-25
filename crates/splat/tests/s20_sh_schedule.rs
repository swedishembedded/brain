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
    let cfg = FitCfg { iters: 30, lr_position: 1e-2, log_every: 0, sh_degree: 3, sh_ramp: 2.0, ..Default::default() };
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

/// How much view-dependent colour a capture can support is set by how many
/// views it has. A degree-`d` expansion is `(d+1)²` coefficients per channel
/// per gaussian, and each gaussian is seen by a fraction of the capture: with
/// about as many coefficients as views observing it, SH memorizes the
/// training views instead of describing the surface. Measured on 24 views of
/// a diffuse synthetic scene, degree 3 lost 2.4 dB on held-out views against
/// degree 0 and degree 1 still lost 0.45; on a 13-photo real capture degree
/// 1 lost 0.4 dB held-out against degree 0.
///
/// So the sparse-start preset gives a capture degree `d` only once it has
/// `8 (d+1)²` views: flat colour below 32, full degree 3 from 128 - the
/// capture sizes 3DGS was defined on.
#[test]
fn a_sparse_capture_gets_flat_colour_and_a_dense_one_full_sh() {
    let degree = |views: usize| FitCfg::from_sparse_points(1000, 1000, views).sh_degree;
    assert_eq!(degree(13), 0);
    assert_eq!(degree(24), 0);
    assert_eq!(degree(31), 0);
    assert_eq!(degree(32), 1);
    assert_eq!(degree(72), 2);
    assert_eq!(degree(127), 2);
    assert_eq!(degree(128), 3);
    assert_eq!(degree(300), 3);
}

/// The rule above is per GAUSSIAN, not per capture: a gaussian's
/// view-dependent colour is fitted from the views that see it, so a corner
/// of a large capture that only a handful of photographs reach supports no
/// more than a small capture does. A fit gives each gaussian the degree its
/// own views support (`8 (d+1)²` of them, spread over enough of a turn to
/// tell directions apart), and a gaussian above that degree neither renders
/// nor learns the bands it cannot support.
#[test]
fn each_gaussian_gets_the_view_dependence_its_own_views_support() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let (w, h) = (24u32, 24u32);
    let blob = |s: &mut Splats, c: [f32; 3], k: usize| {
        for i in 0..k {
            let a = i as f32 * 2.4;
            s.means.extend_from_slice(&[c[0] + 0.25 * a.cos() * (i as f32 / k as f32).sqrt(), c[1] + 0.25 * a.sin() * (i as f32 / k as f32).sqrt(), c[2]]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.12, 0.12, 0.12]);
            s.opacities.push(0.9);
            s.colors.extend_from_slice(&[0.4, 0.5, 0.45]);
        }
    };
    let mut truth = Splats::default();
    blob(&mut truth, [0.0, 0.0, 0.0], 12); // A: at the centre of the ring
    blob(&mut truth, [0.0, -6.0, 0.0], 12); // B: far above it
    // strong view-dependent colour everywhere: degree 1, red along +x
    let mut rest = vec![0.0f32; truth.len() * 9];
    for c in rest.chunks_exact_mut(9) {
        c[2] = -0.6;
        c[3 + 2] = 0.3;
    }
    truth.sh_rest = Some((1, rest));
    // 36 cameras around A, too narrow to see B; 4 cameras close together
    // looking at B only
    let mut cams: Vec<Camera> = (0..36)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 36.0;
            Camera::look_at([4.0 * a.cos(), 0.0, 4.0 * a.sin()], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 30.0, w, h)
        })
        .collect();
    for i in 0..4 {
        let e = [0.3 * i as f32 - 0.45, -6.0, -4.0];
        cams.push(Camera::look_at(e, [0.0, -6.0, 0.0], [0.0, -1.0, 0.0], 30.0, w, h));
    }
    let mut r = Renderer::new(&g, Kernels::at(0), truth.len(), w, h, 0);
    let o = RenderOpts { ray: true, ..Default::default() };
    let targets: Vec<TargetView> = cams
        .iter()
        .map(|c| {
            let shaded = Splats { colors: splat::sh::shade(&truth, c.eye()).expect("has harmonics"), sh_rest: None, ..truth.clone() };
            r.render(&g, &GpuSplats::upload(&g, &shaded), c, &o);
            TargetView::new(*c, rgba_to_rgb(&r.read_rgba(&g, w, h)))
        })
        .collect();
    let init = Splats { sh_rest: None, ..truth.clone() };
    let cfg = FitCfg {
        iters: 80,
        lr_position: 0.0,
        log_every: 0,
        sh_degree: 1,
        strategy: splat::opt::Densify::Hybrid,
        densify_every: 10,
        densify_after: 10,
        densify_until: 60,
        max_gaussians: init.len(),
        ..Default::default()
    };
    let (out, _) = fit(&g, Kernels::at(0), &init, &targets, &cfg, &mut |_, _| true);
    let (_, rest) = out.sh_rest.expect("an SH fit returns its coefficients");
    let (mut a, mut b) = (0.0f32, 0.0f32);
    for (m, c) in out.means.chunks_exact(3).zip(rest.chunks_exact(9)) {
        let mag = c.iter().map(|v| v.abs()).sum::<f32>();
        if m[1] > -3.0 { a += mag } else { b += mag }
    }
    assert!(a > 0.1, "the gaussians 36 views see around a full turn learned no view dependence ({a})");
    assert_eq!(b, 0.0, "the gaussians only 4 nearby views see learned view dependence they cannot support");
}
