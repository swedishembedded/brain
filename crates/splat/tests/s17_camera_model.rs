// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The photometric camera model: what stands between the radiance a scene
//! emits and the pixel values a photograph records.
//!
//! Real captures do not measure radiance consistently. Exposure, white balance
//! and lens vignetting change between frames or across one, and a fit that
//! models none of it has exactly one place to put the disagreement: the scene.
//! A view shot a stop brighter is explained by brighter gaussians, a darker
//! corner by a floater in front of it. The camera model decomposes the
//! difference into physically meaningful per-view and per-sensor factors
//! (after PPISP, Deutsch et al. 2026: exposure, white balance, chromatic
//! vignetting, response curve) so that the scene only has to explain what is
//! actually the same in every view.
//!
//! The targets here are produced by an independent formula in this file, not
//! by the model under test.
//!
//! Swedish Embedded AB implements photometrically calibrated 3D reconstruction
//! for its clients. If your team needs captures with changing exposure, white
//! balance or HDR brackets turned into consistent scenes, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::isp::{srgb_encode, ColorSpace, Isp, IspCfg};
use splat::opt::{fit_full, FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// A textured slab: `cells`x`cells` gaussians whose colours vary smoothly,
/// radiance scaled by `peak` so an HDR scene can exceed 1; `gray` replaces
/// the texture with one flat value.
fn slab(cells: usize, peak: f32, gray: Option<f32>) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[-1.2 + step * (ix as f32 + 0.5), -1.2 + step * (iy as f32 + 0.5), 3.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.5, step * 0.5, step * 0.2]);
            s.opacities.push(0.99);
            let (u, v) = (ix as f32 / cells as f32, iy as f32 / cells as f32);
            let c = if let Some(g) = gray {
                [g, g, g]
            } else {
                [
                    peak * (0.15 + 0.85 * u * u),
                    peak * (0.2 + 0.5 * v),
                    peak * (0.25 + 0.35 * (1.0 - u) * v),
                ]
            };
            s.colors.extend_from_slice(&c);
        }
    }
    s
}

/// Cameras aimed at DIFFERENT points of the slab, so a point of the surface
/// lands at a different image radius in each view. Aimed at one common
/// point, every surface point sits at about the same radius everywhere and a
/// radial falloff is indistinguishable from the surface being darker toward
/// its edges - no fit could, or should, tell those apart.
fn cams(w: u32, h: u32) -> Vec<Camera> {
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 3.0]),
        ([0.6, -0.2, 0.2], [0.5, -0.4, 3.0]),
        ([-0.6, 0.25, 0.25], [-0.5, 0.45, 3.0]),
        ([0.2, 0.5, 0.1], [0.45, 0.5, 3.0]),
        ([-0.3, -0.5, 0.0], [-0.5, -0.45, 3.0]),
    ]
    .iter()
    .map(|(e, t)| Camera::look_at(*e, *t, [0.0, -1.0, 0.0], 50.0, w, h))
    .collect()
}

fn render(g: &Gpu, s: &Splats, c: &Camera) -> Vec<f32> {
    let mut r = Renderer::new(g, Kernels::at(0), s.len(), c.width, c.height, 0);
    r.render(g, &GpuSplats::upload(g, s), c, &RenderOpts::default());
    rgba_to_rgb(&r.read_rgba(g, c.width, c.height))
}

/// The camera the targets were shot with, written out independently of
/// `splat::isp`: gain `2^ev * exp(wb)`, radial falloff `1 + a·r²` with `r`
/// normalised by the half-diagonal, then (optionally) sensor clipping and the
/// sRGB transfer.
fn shoot(radiance: &[f32], c: &Camera, ev: f32, wb: [f32; 3], vig: [f32; 3], encode: bool) -> Vec<f32> {
    let (w, h) = (c.width as usize, c.height as usize);
    let norm = 0.25 * (c.width as f32).powi(2) + 0.25 * (c.height as f32).powi(2);
    let mut out = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let r2 = ((x as f32 + 0.5 - c.cx).powi(2) + (y as f32 + 0.5 - c.cy).powi(2)) / norm;
            for ch in 0..3 {
                let p = (y * w + x) * 3 + ch;
                let v = radiance[p] * 2f32.powf(ev) * wb[ch].exp() * (1.0 + vig[ch] * r2);
                out[p] = if encode { srgb_encode(v.clamp(0.0, 1.0)) } else { v };
            }
        }
    }
    out
}

/// A short fit: colour and rotation step as fast as position, so the scene
/// settles within the few hundred iterations and what is left over is the
/// camera's.
fn base_cfg(iters: usize) -> FitCfg {
    FitCfg { iters, lr_position: 1e-2, lr_color: 1e-2, lr_rotation: 1e-2, log_every: 0, max_growth: 0.0, ..Default::default() }
}

/// Views shot at different exposures and white balances through a vignetting
/// lens. With the camera model the fit recovers the exposures it was never
/// told, and the scene it returns renders like the truth from the neutral
/// camera; without it, the scene absorbs the disagreement.
#[test]
fn exposure_white_balance_and_vignetting_are_explained_by_the_camera() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (48u32, 48u32);
    let truth = slab(12, 1.0, None);
    let cs = cams(w, h);
    let evs = [0.55f32, -0.45, 0.25, -0.35, 0.0];
    let wbs = [[0.08f32, 0.0, -0.08], [-0.05, 0.02, 0.03], [0.0, -0.06, 0.06], [-0.03, 0.04, -0.01], [0.0, 0.0, 0.0]];
    let vig = [-0.30f32, -0.34, -0.38];
    let targets: Vec<TargetView> = cs
        .iter()
        .enumerate()
        .map(|(i, c)| TargetView::new(*c, shoot(&render(&g, &truth, c), c, evs[i], wbs[i], vig, false)))
        .collect();
    let init = slab(12, 1.0, Some(0.4));

    let plain = base_cfg(240);
    let (a, _) = splat::opt::fit(&g, Kernels::at(0), &init, &targets, &plain, &mut |_, _| true);
    let isp_cfg = IspCfg { color_space: ColorSpace::Display, vignetting_after: 0.1, response_after: 1.0, ..IspCfg::default() };
    let with = FitCfg { isp: Some(isp_cfg), ..base_cfg(240) };
    let out = fit_full(&g, Kernels::at(0), &init, &targets, &with, &mut |_, _| true);
    let isp: &Isp = out.isp.as_ref().expect("a fit with a camera model returns it");

    // exposures are recovered up to the gauge (their mean), which the truth
    // above also has at zero
    let mean_ev = evs.iter().sum::<f32>() / evs.len() as f32;
    for (v, &ev) in evs.iter().enumerate() {
        let got = isp.exposure(v);
        assert!(
            (got - (ev - mean_ev)).abs() < 0.08,
            "view {v}: recovered exposure {got:+.3} EV against {:+.3} EV", ev - mean_ev
        );
    }
    // the lens: the recovered falloff has the right size, where the frame is
    // covered by the slab (the corners are background and carry no evidence)
    let r = 0.7f32;
    let seen = isp.vignetting(0, r);
    for ch in 0..3 {
        let want = 1.0 + vig[ch] * r * r;
        assert!(
            (seen[ch] - want).abs() < 0.04,
            "channel {ch}: transmission at r={r} is {:.3} against {want:.3}", seen[ch]
        );
    }
    // and the scene is the scene, seen from the neutral camera
    let held = Camera::look_at([0.3, 0.3, -0.2], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 50.0, w, h);
    let want = render(&g, &truth, &held);
    let (pa, pb) = (psnr(&render(&g, &a, &held), &want), psnr(&render(&g, &out.scene, &held), &want));
    assert!(
        pb > pa + 4.0,
        "a held-out view renders at {pb:.1} dB with the camera model and {pa:.1} dB without; the \
         model exists so the scene stops absorbing the camera"
    );
}

/// Exposure brackets are measurements of ONE radiance. Trained in scene-linear
/// space with the sRGB transfer inside the camera, a fit recovers radiance
/// above the clipping point of any single exposure; a display-referred fit
/// cannot represent it at all.
#[test]
fn exposure_brackets_recover_radiance_beyond_one_exposure() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (40u32, 40u32);
    let truth = slab(10, 2.5, None); // peak radiance 2.5: clipped at EV 0
    let cs = cams(w, h);
    let mut targets = Vec::new();
    for c in cs.iter().take(3) {
        let radiance = render(&g, &truth, c);
        for ev in [-1.5f32, 0.0, 1.5] {
            let t = TargetView::new(*c, shoot(&radiance, c, ev, [0.0; 3], [0.0; 3], true));
            targets.push(t.with_exposure(ev));
        }
    }
    // Adam moves a colour about one learning rate per step, so start where the
    // radiance is within reach of the iteration budget.
    let init = slab(10, 1.0, Some(1.2));
    let isp_cfg = IspCfg { color_space: ColorSpace::SceneLinear, vignetting_after: 1.0, response_after: 1.0, ..IspCfg::default() };
    let cfg = FitCfg { isp: Some(isp_cfg), ..base_cfg(260) };
    let out = fit_full(&g, Kernels::at(0), &init, &targets, &cfg, &mut |_, _| true);

    // The brightest quarter of the slab: its true radiance is 1.6..2.5 in the
    // red channel, beyond what the EV 0 frames recorded.
    let front = cs[0];
    let got = render(&g, &out.scene, &front);
    let want = render(&g, &truth, &front);
    let (mut num, mut den, mut n) = (0.0f64, 0.0f64, 0usize);
    for p in 0..(w * h) as usize {
        if want[p * 3] > 1.6 {
            num += (got[p * 3] - want[p * 3]).abs() as f64;
            den += want[p * 3] as f64;
            n += 1;
        }
    }
    assert!(n > 20, "the scene has no radiance above 1.6 to recover ({n} pixels)");
    let rel = num / den;
    assert!(
        rel < 0.12,
        "radiance above the EV 0 clipping point recovered with {:.1}% mean error over {n} pixels",
        100.0 * rel
    );
}
