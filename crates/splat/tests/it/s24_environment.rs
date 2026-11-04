// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The environment a scene is seen against (`splat::env`): radiance by
//! direction, composited behind the gaussians along each pixel's own ray.
//!
//! 1. The basis is the real spherical harmonics: orthonormal over the sphere.
//! 2. The device composite is the host formula, through a lens.
//! 3. Its backward - the coefficients' gradient and the share it moves into
//!    the alpha channel for the rasterizer - is the derivative of the
//!    composite.
//! 4. A fit with an environment explains a sky with the sky, not with a
//!    haze of gaussians.
//!
//! Swedish Embedded AB implements 3D reconstruction of outdoor captures,
//! including the sky and the distant scenery around them. If your team needs
//! expertise in radiance fields then you can procure our services by sending
//! an email to info@swedishembedded.com.

use splat::env::{basis, coeffs, EnvDevice, EnvMap};
use splat::opt::{FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn fish(w: u32, h: u32) -> Camera {
    let c = Camera::look_at([0.2, -0.1, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 80.0, w, h);
    Camera { lens: camera::Lens::Fisheye { k: [0.02, -0.01, 0.0, 0.0] }, ..c }
}

fn sky(degree: u32) -> EnvMap {
    let mut e = EnvMap::uniform(degree, [0.45, 0.55, 0.7]);
    // brighter overhead (-y is up), a warm band toward +x, some texture
    e.coeffs[3 + 1] = -0.25; // Y_1^{-1} ~ y
    e.coeffs[3 * 3] = 0.15; // Y_1^{1} ~ x, red
    if degree >= 3 {
        e.coeffs[3 * 10 + 2] = 0.08;
        e.coeffs[3 * 13 + 1] = -0.06;
    }
    e
}

#[test]
fn the_basis_is_orthonormal_over_the_sphere() {
    let degree = 6;
    let n = coeffs(degree);
    let mut gram = vec![0.0f64; n * n];
    let samples = 40_000;
    for i in 0..samples {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / samples as f64;
        let r = (1.0 - z * z).sqrt();
        let phi = i as f64 * 2.399_963_229_728_653;
        let y = basis([r * phi.cos(), r * phi.sin(), z], degree);
        for a in 0..n {
            for b in 0..n {
                gram[a * n + b] += y[a] * y[b] * 4.0 * std::f64::consts::PI / samples as f64;
            }
        }
    }
    for a in 0..n {
        for b in 0..n {
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((gram[a * n + b] - want).abs() < 2e-3, "<Y_{a}, Y_{b}> = {} over the sphere", gram[a * n + b]);
        }
    }
}

#[test]
fn the_device_composite_is_the_host_formula_through_the_lens() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let ks = Kernels::at(0);
    let cam = fish(40, 30);
    let env = sky(4);
    // a render over black: half-transparent everywhere, so both terms count
    let rgba: Vec<f32> = (0..40 * 30).flat_map(|i| [0.1, 0.2, 0.05 * (i % 7) as f32, 0.3 + 0.01 * (i % 50) as f32]).collect();
    let img = g.storage_init("img", &rgba);
    let dev = EnvDevice::new(&g, &env);
    dev.composite(&g, &ks, &img, &cam, &RenderOpts::default());
    let got = g.read(&img, rgba.len());
    let k = cam.intrinsics();
    let m = cam.c2w;
    let mut worst = 0.0f32;
    for p in 0..40 * 30 {
        let Some(d) = k.unproject([(p % 40) as f64 + 0.5, (p / 40) as f64 + 0.5]) else { continue };
        let dw: [f32; 3] = std::array::from_fn(|i| (m[i * 4] as f64 * d[0] + m[i * 4 + 1] as f64 * d[1] + m[i * 4 + 2] as f64 * d[2]) as f32);
        let e = env.eval(dw);
        for c in 0..3 {
            let want = rgba[p * 4 + c] + (1.0 - rgba[p * 4 + 3]) * e[c];
            worst = worst.max((got[p * 4 + c] - want).abs());
        }
    }
    assert!(worst < 1e-4, "device composite differs from the host formula by {worst}");
}

#[test]
fn the_backward_is_the_derivative_of_the_composite() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let ks = Kernels::at(0);
    let cam = fish(24, 18);
    let env = sky(3);
    let n_px = 24 * 18;
    let rgba: Vec<f32> = (0..n_px).flat_map(|i| [0.1, 0.2, 0.3, 0.2 + 0.6 * ((i * 37 % 101) as f32 / 101.0)]).collect();
    let w: Vec<f32> = (0..n_px * 3).map(|i| ((i * 7919 % 97) as f32 / 97.0) - 0.4).collect();
    // L = sum w . C, with C the composite
    let loss = |env: &EnvMap, rgba: &[f32]| -> f64 {
        let img = g.storage_init("img", rgba);
        EnvDevice::new(&g, env).composite(&g, &ks, &img, &cam, &RenderOpts::default());
        let c = g.read(&img, rgba.len());
        (0..n_px).map(|p| (0..3).map(|k| (w[p * 3 + k] * c[p * 4 + k]) as f64).sum::<f64>()).sum()
    };
    let img = g.storage_init("img", &rgba);
    let mut dev = EnvDevice::new(&g, &env);
    dev.composite(&g, &ks, &img, &cam, &RenderOpts::default());
    let dimg = g.storage_init("dimg", &(0..n_px).flat_map(|p| [w[p * 3], w[p * 3 + 1], w[p * 3 + 2], 0.0]).collect::<Vec<f32>>());
    let grad = dev.backward(&g, &ks, &img, &dimg, &cam, &RenderOpts::default());
    let da = g.read(&dimg, n_px * 4);
    let h = 1e-2f32;
    for (k, &an) in grad.iter().enumerate() {
        let (mut a, mut b) = (env.clone(), env.clone());
        a.coeffs[k] += h;
        b.coeffs[k] -= h;
        let fd = (loss(&a, &rgba) - loss(&b, &rgba)) / (2.0 * h as f64);
        assert!((an - fd).abs() < 2e-3 * fd.abs().max(1.0), "coefficient {k}: analytic {an} vs central difference {fd}");
    }
    // the alpha share: dL/dalpha of the composite, per pixel
    for p in (0..n_px).step_by(17) {
        let (mut a, mut b) = (rgba.clone(), rgba.clone());
        a[p * 4 + 3] += h;
        b[p * 4 + 3] -= h;
        let fd = (loss(&env, &a) - loss(&env, &b)) / (2.0 * h as f64);
        assert!((da[p * 4 + 3] as f64 - fd).abs() < 1e-3 * fd.abs().max(1.0), "pixel {p}: alpha share {} vs central difference {fd}", da[p * 4 + 3]);
    }
}

/// A textured ball under a sky, seen from a ring of cameras. With an
/// environment the fit puts the sky where it belongs - at infinity - and
/// the background of a view it never trained on matches; without one, the
/// sky becomes gaussians hanging somewhere behind the ball.
#[test]
fn a_fit_explains_the_sky_with_the_environment() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (48u32, 36u32);
    let mut ball = Splats::default();
    for k in 0..300 {
        let z = 1.0 - 2.0 * (k as f32 + 0.5) / 300.0;
        let r = (1.0 - z * z).sqrt();
        let phi = k as f32 * 2.399_963;
        ball.means.extend_from_slice(&[0.6 * r * phi.cos(), 0.6 * r * phi.sin(), 4.0 + 0.6 * z]);
        ball.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        ball.scales.extend_from_slice(&[0.09; 3]);
        ball.opacities.push(0.95);
        ball.colors.extend_from_slice(&[0.5 + 0.4 * phi.sin(), 0.4, 0.3 + 0.3 * z]);
    }
    let truth_env = sky(3);
    let o = RenderOpts { ray: true, ..Default::default() };
    let ring = |a: f32| Camera::look_at([2.5 * a.sin(), -0.3, 4.0 - 2.5 * a.cos()], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 70.0, w, h);
    let shoot = |cam: &Camera| -> Vec<f32> {
        let mut r = Renderer::new(&g, ks, ball.len(), w, h, 0);
        r.render(&g, &GpuSplats::upload(&g, &ball), cam, &o);
        EnvDevice::new(&g, &truth_env).composite(&g, &ks, &r.img, cam, &o);
        rgba_to_rgb(&r.read_rgba(&g, w, h))
    };
    let train: Vec<TargetView> = (0..10).map(|i| ring(-0.9 + 0.2 * i as f32)).map(|c| TargetView::new(c, shoot(&c))).collect();
    let held = ring(0.05);
    let want = shoot(&held);
    let base = FitCfg { iters: 300, lr_position: 5e-3, log_every: 0, ..Default::default() };
    let with = splat::opt::fit_full(&g, ks, &ball, &train, &FitCfg { environment: Some(3), ..base }, &mut |_, _| true);
    let without = splat::opt::fit_full(&g, ks, &ball, &train, &base, &mut |_, _| true);
    let render = |s: &Splats, env: Option<&EnvMap>| -> Vec<f32> {
        let mut r = Renderer::new(&g, ks, s.len(), w, h, 0);
        r.render(&g, &GpuSplats::upload(&g, s), &held, &o);
        if let Some(e) = env {
            EnvDevice::new(&g, e).composite(&g, &ks, &r.img, &held, &o);
        }
        rgba_to_rgb(&r.read_rgba(&g, w, h))
    };
    let a = psnr(&render(&with.baked(), with.env.as_ref()), &want);
    let b = psnr(&render(&without.baked(), None), &want);
    assert!(a > 30.0 && a > b + 5.0, "a held-out view renders at {a:.1} dB with an environment and {b:.1} dB without");
}
