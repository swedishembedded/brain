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
    render_with(g, s, cam, eps2d, RenderOpts::default().antialiased)
}

fn render_with(g: &Gpu, s: &Splats, cam: &Camera, eps2d: f32, antialiased: bool) -> Vec<f32> {
    let ks = splat::Kernels::at(0);
    let o = RenderOpts { eps2d, antialiased, ..Default::default() };
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
        "the default low-pass (eps2d={}, compensation {}) leaves {dflt:.3}x of the source's \
         high-frequency content (measured 0.298 when this was written; with the filter off, \
         {off:.3}). Outside that range either the rasterizer changed or the filter did - and \
         the second silently changes every render this repo produces.",
        RenderOpts::default().eps2d,
        RenderOpts::default().antialiased
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
/// The gate is ACCURACY, not sharpness, and the difference matters here.
/// Rendered at the value it was fitted under this scene scores 27.4 dB and
/// carries 0.65 of the target's high-frequency content; rendered undilated it
/// scores 20.4 dB and carries 1.014, which reads as ideal and is aliasing. A
/// sharpness ratio near 1.0 is not evidence of anything on its own - the
/// quality gate in this suite exists because of precisely that - so the
/// coupling is pinned on the measure that cannot be fooled this way.
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
    let target = TargetView::new(cam, render(&g, &truth, &cam, FITTED_AT));

    // Start from flattened colours so the fit has to put the detail back.
    let mut init = truth.clone();
    for v in init.colors.iter_mut() {
        *v = 0.5 + (*v - 0.5) * 0.3;
    }
    let cfg = FitCfg { iters: 120, lr: 1e-2, log_every: 0, eps2d: FITTED_AT, ..Default::default() };
    let (fitted, _) = fit(&g, ks, &init, std::slice::from_ref(&target), &cfg, &mut |_, _| true);

    let rm = render(&g, &fitted, &cam, FITTED_AT);
    let rl = render(&g, &fitted, &cam, 0.0);
    let matched = sharpness_ratio(&rm, &target.rgb, wu, hu);

    let (db_m, db_l) = (psnr(&rm, &target.rgb), psnr(&rl, &target.rgb));
    assert!(
        db_m > db_l + 3.0,
        "rendering the fitted scene at the dilation it was fitted under ({FITTED_AT}) gives \
         {db_m:.1} dB and rendering it at 0.0 gives {db_l:.1} dB. The fit bakes compensation for \
         the dilation into the gaussians, so the two are supposed to be coupled and `--eps2d` on \
         `fit` and on `render` must agree - if they are not, something has decoupled them."
    );
    // And it must not have got there by going soft: the matched render is
    // allowed to carry less detail than the target, but not much less.
    assert!(
        (0.55..1.3).contains(&matched),
        "a scene fitted and rendered at the same dilation reproduces its own target at \
         {matched:.3}x its high-frequency content"
    );
}

/// Box-downsample by 2: the band-limited truth a half-resolution render is
/// supposed to approximate.
fn halve(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let (hw, hh) = (w / 2, h / 2);
    let mut out = vec![0.0f32; hw * hh * 3];
    for y in 0..hh {
        for x in 0..hw {
            for c in 0..3 {
                let s: f32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .map(|(dx, dy)| img[((2 * y + dy) * w + 2 * x + dx) * 3 + c])
                    .sum();
                out[(y * hw + x) * 3 + c] = s / 4.0;
            }
        }
    }
    out
}

/// Zooming OUT is the screen-space filter's job, and doing it without the
/// energy compensation is simply wrong.
///
/// At half the sampling rate each pixel covers four of the original, so the
/// honest answer is the box average. Inria's dilation inflates every splat's
/// footprint and leaves its opacity alone, so the result is both blurred and
/// too bright; Mip-Splatting's 2D Mip filter scales opacity by
/// `sqrt(|S| / |S + eps I|)` and lands much closer.
#[test]
fn zooming_out_needs_the_energy_the_dilation_throws_away() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (128u32, 128u32);
    let cam = camera(w, h);
    let img = source(w as usize, h as usize);
    // Sized so the splats tile the plane rather than sitting as separate
    // dots, which is the regime a real reconstruction lands in.
    let scene = one_splat_per_pixel(&img, &cam, 0.5);
    let truth = halve(&img, w as usize, h as usize);
    let half = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w / 2, h / 2);

    let inria = psnr(&render_with(&g, &scene, &half, 0.3, false), &truth);
    let mip = psnr(&render_with(&g, &scene, &half, 0.1, true), &truth);
    assert!(
        mip > inria + 1.5,
        "at half the sampling rate the Mip filter gives {mip:.1} dB against the dilation's \
         {inria:.1} dB - the compensation is not earning its place"
    );
}

/// What the 3D filter guarantees, stated directly.
///
/// A PSNR proxy for this is treacherous - it mostly measures whatever the
/// synthetic scene happens to look like - so the property is checked as a
/// property: after filtering, no gaussian is narrower than the finest detail
/// any of its own cameras could resolve, and none has been widened much past
/// it either. That is the whole contract, and it is what stops a fit from
/// hiding error in splats smaller than a pixel.
#[test]
fn band_limiting_leaves_no_gaussian_below_what_its_cameras_sampled() {
    let (w, h) = (96u32, 96u32);
    let cam = camera(w, h);
    let img = source(w as usize, h as usize);
    let scene = one_splat_per_pixel(&img, &cam, 0.08);
    let filtered = splat::mip::apply_3d_filter(&scene, &[cam], splat::mip::DEFAULT_SCALE);
    let sigma = splat::mip::smoothing_sigma(&scene, &[cam], splat::mip::DEFAULT_SCALE);

    let mut widened = 0;
    for (i, &sg) in sigma.iter().enumerate() {
        if sg <= 0.0 {
            continue;
        }
        widened += 1;
        for k in 0..3 {
            let got = filtered.scales[i * 3 + k];
            assert!(got >= sg, "gaussian {i} axis {k} is {got:.5}, below its own limit {:.5}", sg);
            let want = (scene.scales[i * 3 + k].powi(2) + sg * sg).sqrt();
            assert!((got - want).abs() < 1e-6, "gaussian {i} axis {k}: {got:.6} is not the convolution {want:.6}");
        }
        // widening spreads the mass, so it must not also brighten
        assert!(
            filtered.opacities[i] < scene.opacities[i],
            "gaussian {i} was widened without paying for it in opacity"
        );
    }
    assert!(widened > 1000, "the filter touched almost nothing ({widened})");
}

/// Where a back-projected gaussian belongs, to within half a pixel.
///
/// A pixel is an AREA, and the rasterizer samples it at its centre - pixel
/// `n` is sampled at `n + 0.5`. So unprojecting pixel `n` has to use `n + 0.5`
/// too, or every gaussian in the scene lands half a pixel up and half a pixel
/// left of the detail it was made from. That is 0.7 px diagonally, and a
/// feed-forward reconstruction's splats have a standard deviation around
/// 0.46 px, so the offset is roughly one and a half sigma: each pixel ends up
/// reading its neighbour's gaussian.
///
/// The damage does not look like a shift, which is what makes it survive
/// inspection. It looks like the scene is slightly out of focus, and it caps
/// reconstruction quality at a number low enough to be blamed on the model.
#[test]
fn unprojecting_a_pixel_must_use_its_centre() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (128u32, 128u32);
    let cam = camera(w, h);
    let img = source(w as usize, h as usize);

    // the same scene, built from pixel CORNERS instead of pixel centres
    let mut off = one_splat_per_pixel(&img, &cam, 0.46);
    let z = 3.0f32;
    for i in 0..off.len() {
        off.means[i * 3] -= 0.5 * z / cam.fx;
        off.means[i * 3 + 1] -= 0.5 * z / cam.fy;
    }
    let centred = one_splat_per_pixel(&img, &cam, 0.46);

    // Measured only at low dilation, and that is the point. At the reference
    // eps2d of 0.3 the offset scene scores HIGHER - 21.9 dB against 18.4 -
    // because a half-pixel diagonal offset spreads each pixel evenly over four
    // neighbours, which is a 2x2 box blur, and PSNR rewards smoothness once
    // the dilation has removed the detail that would have distinguished them.
    // So the default render setting does not merely hide this error, it
    // endorses it, and any check run there would have called the bug an
    // improvement.
    for (eps, floor) in [(0.0f32, 8.0f64), (0.05, 5.0)] {
        let a = psnr(&render(&g, &centred, &cam, eps), &img);
        let b = psnr(&render(&g, &off, &cam, eps), &img);
        assert!(
            a > b + floor,
            "at eps2d {eps}, centre-aligned gives {a:.1} dB and corner-aligned {b:.1} dB. Half a \
             pixel is supposed to cost a great deal here - if it does not, this test cannot \
             detect the error it exists to detect."
        );
    }

}
