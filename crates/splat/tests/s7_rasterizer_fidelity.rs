// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the rasterizer itself can resolve, separated from what a scene
//! happens to contain.
//!
//! "The reconstruction is blurry" has two possible causes and they need
//! different fixes: the scene may not hold the detail, or the renderer may not
//! be putting it on screen. Nothing distinguished them, so the question was
//! settled by argument. This settles it by construction: a scene of ONE
//! gaussian per output pixel, placed on a plane at known depth and coloured
//! from a source image, is by construction a perfect representation of that
//! image. Whatever the renderer gives back is its own ceiling.
//!
//! # The anti-alias dilation is a blur, and it is not small
//!
//! `RenderOpts::eps2d` adds a constant to the diagonal of every splat's
//! screen-space covariance, in pixels squared. It is the reference 3DGS
//! convention and it exists for a reason - splats smaller than a pixel alias
//! badly as the camera moves - but it is a low-pass filter, and at the
//! reference default of 0.3 it removes most of the high-frequency content of a
//! scene whose splats are around a pixel across. That is exactly the regime a
//! feed-forward reconstruction lands in: one gaussian per source pixel, a
//! median projected size near 0.6 px.
//!
//! So the cost is measured here rather than left to be rediscovered. The test
//! fails if it changes in EITHER direction, because both mean something: a
//! bigger gap is a regression in the rasterizer, and a smaller one means the
//! dilation moved and every render in the repo just changed.
//!
//! Swedish Embedded AB implements GPU rasterizers whose resolving power is a
//! measured quantity rather than an impression. If your team needs rendering
//! quality held to a number, you can procure our services by sending an email
//! to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::quality::{psnr, sharpness_ratio};
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};

/// Fine texture over a smooth base - photograph-shaped, so the numbers here
/// mean the same thing they mean on a real scene.
fn source(w: usize, h: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let base = 0.25 + 0.3 * (x as f32 / w as f32) + 0.09 * (y as f32 / 18.0).sin();
            let tex = 0.28 * (((x / 2) + (y / 2)) % 2) as f32;
            for c in 0..3 {
                v[(y * w + x) * 3 + c] = (base + tex).clamp(0.0, 1.0);
            }
        }
    }
    v
}

/// One gaussian per pixel, on the z = 3 plane, sized so its screen-space
/// standard deviation is `std_px` pixels.
fn one_splat_per_pixel(img: &[f32], cam: &Camera, std_px: f32) -> Splats {
    let (w, h) = (cam.width as usize, cam.height as usize);
    let z = 3.0f32;
    let mut s = Splats::default();
    for y in 0..h {
        for x in 0..w {
            s.means.extend_from_slice(&[
                (x as f32 + 0.5 - cam.cx) * z / cam.fx,
                (y as f32 + 0.5 - cam.cy) * z / cam.fy,
                z,
            ]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            let sc = std_px * z / cam.fx;
            s.scales.extend_from_slice(&[sc, sc, sc]);
            s.opacities.push(1.0);
            let i = y * w + x;
            s.colors.extend_from_slice(&img[i * 3..i * 3 + 3]);
        }
    }
    s
}

fn render(g: &Gpu, s: &Splats, cam: &Camera, eps2d: f32) -> Vec<f32> {
    let ks = splat::Kernels::at(0);
    let o = RenderOpts { eps2d, ..Default::default() };
    let mut r = Renderer::new(g, ks, s.len(), cam.width, cam.height, s.len() * 16);
    let gs = GpuSplats::upload(g, s);
    r.render(g, &gs, cam, &o);
    r.read_rgba(g, cam.width, cam.height).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
}

fn camera(w: u32, h: u32) -> Camera {
    Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h)
}

/// With the dilation off, the rasterizer must give back very nearly the image
/// the scene encodes. This is the ceiling: if it is not near 1.0, no scene can
/// render sharply and the fault is in the rasterizer, not the reconstruction.
#[test]
fn the_rasterizer_can_resolve_one_splat_per_pixel() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (128u32, 128u32);
    let cam = camera(w, h);
    let img = source(w as usize, h as usize);
    let scene = one_splat_per_pixel(&img, &cam, 0.25);

    let out = render(&g, &scene, &cam, 0.0);
    let sharp = sharpness_ratio(&out, &img, w as usize, h as usize);
    let db = psnr(&out, &img);
    assert!(
        sharp > 0.90,
        "a scene of one splat per pixel renders at {sharp:.3}x its own source's high-frequency \
         content ({db:.1} dB). The rasterizer cannot resolve what it is given, so nothing \
         downstream can be sharp."
    );
    assert!(db > 35.0, "one splat per pixel reproduces its source at only {db:.1} dB");
}

/// And what the dilation costs, on the same scene. Both bounds matter: the
/// upper one catches the rasterizer getting blurrier, the lower one catches
/// the dilation quietly changing and every render in the repo with it.
#[test]
fn the_antialias_dilation_costs_most_of_the_fine_detail() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (128u32, 128u32);
    let cam = camera(w, h);
    let img = source(w as usize, h as usize);
    let scene = one_splat_per_pixel(&img, &cam, 0.25);
    let (wu, hu) = (w as usize, h as usize);

    let off = sharpness_ratio(&render(&g, &scene, &cam, 0.0), &img, wu, hu);
    let dflt = sharpness_ratio(&render(&g, &scene, &cam, RenderOpts::default().eps2d), &img, wu, hu);

    assert!(
        (0.20..0.45).contains(&dflt),
        "the default eps2d={} leaves {dflt:.3}x of the source's high-frequency content \
         (measured 0.298 when this was written; with the dilation off, {off:.3}). Outside that \
         range either the rasterizer changed or the dilation did - and the second silently \
         changes every render this repo produces.",
        RenderOpts::default().eps2d
    );
    assert!(
        off > dflt * 2.0,
        "the dilation now costs almost nothing ({off:.3} off vs {dflt:.3} on). If that is a \
         deliberate improvement, this test should record the new numbers; if it is not, the \
         dilation is no longer being applied."
    );
}

/// The dilation is part of the forward model `fit` inverts, not a display
/// setting applied afterwards, so the optimizer folds compensation for it into
/// the gaussians. Rendering a fitted scene at a DIFFERENT value un-does that
/// compensation: higher comes out blurred, lower comes out aliased.
///
/// Measured on a real six-photograph reconstruction fitted under the 0.3
/// default: rendered at 0.3 it carries 0.92 of the photographs' detail at 29.1
/// dB; at 0.05 it carries 1.90 - nearly twice the source's high-frequency
/// content, which is aliasing, not sharpness - at 24.4 dB. Both measures agree
/// that matching is right, and this pins the coupling so it cannot be
/// refactored apart.
#[test]
fn a_scene_must_be_rendered_at_the_dilation_it_was_fitted_under() {
    use splat::opt::{fit, FitCfg, TargetView};

    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = splat::Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let cam = camera(w, h);
    let (wu, hu) = (w as usize, h as usize);
    let img = source(wu, hu);
    let truth = one_splat_per_pixel(&img, &cam, 0.3);

    const FITTED_AT: f32 = 0.3;
    let target = TargetView { cam, rgb: render(&g, &truth, &cam, FITTED_AT) };

    // Start from flattened colours so the fit has to put the detail back.
    let mut init = truth.clone();
    for v in init.colors.iter_mut() {
        *v = 0.5 + (*v - 0.5) * 0.3;
    }
    let cfg = FitCfg { iters: 120, lr: 1e-2, log_every: 0, eps2d: FITTED_AT, ..Default::default() };
    let (fitted, _) = fit(&g, ks, &init, std::slice::from_ref(&target), &cfg, &mut |_, _| true);

    let matched = sharpness_ratio(&render(&g, &fitted, &cam, FITTED_AT), &target.rgb, wu, hu);
    let lower = sharpness_ratio(&render(&g, &fitted, &cam, 0.0), &target.rgb, wu, hu);

    assert!(
        (matched - 1.0).abs() < (lower - 1.0).abs(),
        "rendering the fitted scene at the dilation it was fitted under ({FITTED_AT}) gives \
         {matched:.3}x the target's detail, and rendering it at 0.0 gives {lower:.3}x - which is \
         CLOSER to 1.0. The fit no longer bakes in compensation for the dilation, so the two are \
         no longer coupled and `--eps2d` on `fit` and on `render` need not agree."
    );
    assert!(
        (0.75..1.3).contains(&matched),
        "a scene fitted and rendered at the same dilation reproduces its own target at only \
         {matched:.3}x its high-frequency content"
    );
}
