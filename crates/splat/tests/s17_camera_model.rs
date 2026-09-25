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

use data::rng::Lcg;
use gpu_core::Gpu;
use splat::isp::{srgb_encode, ColorSpace, DeviceIsp, Encoding, GridCfg, Isp, IspCfg, NovelShot, Shot};
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
/// normalised by the half-diagonal, a colour matrix, a response curve, a
/// horizontal gain ramp from `1 - ramp` at the left edge to `1 + ramp` at the
/// right (a local tone change no global camera makes), then (optionally)
/// sensor clipping and the sRGB transfer.
struct Truth {
    ev: f32,
    wb: [f32; 3],
    vig: [f32; 3],
    ccm: [[f32; 3]; 3],
    curve: fn(f32) -> f32,
    ramp: f32,
    encode: bool,
}

const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl Truth {
    fn plain(ev: f32, wb: [f32; 3], vig: [f32; 3], encode: bool) -> Truth {
        Truth { ev, wb, vig, ccm: IDENTITY, curve: |y| y, ramp: 0.0, encode }
    }
}

fn shoot_with(radiance: &[f32], c: &Camera, t: &Truth) -> Vec<f32> {
    let (w, h) = (c.width as usize, c.height as usize);
    let norm = 0.25 * (c.width as f32).powi(2) + 0.25 * (c.height as f32).powi(2);
    let mut out = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let r2 = ((x as f32 + 0.5 - c.cx).powi(2) + (y as f32 + 0.5 - c.cy).powi(2)) / norm;
            let p = (y * w + x) * 3;
            let lit: [f32; 3] = std::array::from_fn(|ch| radiance[p + ch] * 2f32.powf(t.ev) * t.wb[ch].exp() * (1.0 + t.vig[ch] * r2));
            let ramp = 1.0 + t.ramp * (2.0 * (x as f32 + 0.5) / w as f32 - 1.0);
            for ch in 0..3 {
                let mixed = t.ccm[ch][0] * lit[0] + t.ccm[ch][1] * lit[1] + t.ccm[ch][2] * lit[2];
                let v = (t.curve)(mixed) * ramp;
                out[p + ch] = if t.encode { srgb_encode(v.clamp(0.0, 1.0)) } else { v };
            }
        }
    }
    out
}

fn shoot(radiance: &[f32], c: &Camera, ev: f32, wb: [f32; 3], vig: [f32; 3], encode: bool) -> Vec<f32> {
    shoot_with(radiance, c, &Truth::plain(ev, wb, vig, encode))
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

/// A pixel of a coarser level integrates the light of the four it covers.
/// Display-referred, the scene's colours ARE encoded values and averaging
/// them is what compositing does; scene-linear, the camera encodes after it
/// integrates, so a half-resolution sRGB target is the encoding of the mean
/// light - a black-and-white 2x2 block is encode(0.5) = 0.735, not 0.5.
#[test]
fn a_coarser_level_averages_light_in_a_scene_linear_fit() {
    let cam = Camera::look_at([0.0; 3], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 60.0, 2, 2);
    let rgb = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let t = TargetView::new(cam, rgb);
    let display = t.half_in(ColorSpace::Display).rgb;
    let linear = t.half_in(ColorSpace::SceneLinear).rgb;
    let linear_target = t.clone().with_encoding(Encoding::Linear).half_in(ColorSpace::SceneLinear).rgb;
    for c in 0..3 {
        assert!((display[c] - 0.5).abs() < 1e-6, "display-referred: {}", display[c]);
        assert!((linear[c] - srgb_encode(0.5)).abs() < 1e-5, "scene-linear against sRGB: {}", linear[c]);
        assert!((linear_target[c] - 0.5).abs() < 1e-6, "an already-linear target: {}", linear_target[c]);
    }
}

fn mat_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mat_inv(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    // the adjugate: entry (i, j) is the cofactor of (j, i)
    let cof = |r: usize, c: usize| {
        let (r0, r1, c0, c1) = ((r + 1) % 3, (r + 2) % 3, (c + 1) % 3, (c + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    std::array::from_fn(|i| std::array::from_fn(|j| cof(j, i) / det))
}

/// Two cameras photograph the same scene; the second renders colour through
/// its own colour correction matrix. Only a DIFFERENCE between sensors is
/// observable - the scene's colours absorb whatever the two have in common -
/// so what is recovered is the second's matrix relative to the first's.
#[test]
fn a_second_sensors_colour_matrix_is_recovered_relative_to_the_first() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (40u32, 40u32);
    let truth = slab(12, 0.8, None);
    let m_true = [[1.10f32, -0.06, -0.04], [-0.05, 1.08, -0.03], [0.02, -0.12, 1.10]];
    let mut targets = Vec::new();
    for c in cams(w, h) {
        let radiance = render(&g, &truth, &c);
        for sensor in 0..2 {
            let t = Truth { ccm: if sensor == 1 { m_true } else { IDENTITY }, ..Truth::plain(0.0, [0.0; 3], [0.0; 3], false) };
            targets.push(TargetView::new(c, shoot_with(&radiance, &c, &t)).with_sensor(sensor));
        }
    }
    let init = slab(12, 0.8, Some(0.35));
    let isp_cfg = IspCfg { vignetting_after: 1.0, ccm_after: 0.1, response_after: 1.0, ..IspCfg::default() };
    let out = fit_full(&g, Kernels::at(0), &init, &targets, &FitCfg { isp: Some(isp_cfg), ..base_cfg(240) }, &mut |_, _| true);
    let isp = out.isp.as_ref().expect("a fit with a camera model returns it");
    let rel = mat_mul(&isp.ccm(1), &mat_inv(&isp.ccm(0)));
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (rel[i][j] - m_true[i][j]).abs() < 0.03,
                "relative colour matrix entry ({i},{j}) recovered as {:.3} against {:.3}\n{}",
                rel[i][j],
                m_true[i][j],
                isp.summary()
            );
        }
    }
}

/// A tone curve that is not a power law (a power law is exactly an exposure
/// change and a scene change): Reinhard's `y / (y + k)`, scaled to 1 at 1.
fn tone(y: f32) -> f32 {
    1.3 * y / (y + 0.3)
}

/// Exposure brackets with known exposures pin down the response curve (the
/// classic Debevec-Malik setting): three exposures of one radiance land on
/// different parts of the curve. It is recovered up to the scene's colour -
/// each channel's radiance scale is the scene's to choose - so channel c's
/// `f_c(y)` is compared with `tone(k_c y)` for its best `k_c`.
#[test]
fn a_response_curve_is_recovered_from_exposure_brackets() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (40u32, 40u32);
    let truth = slab(10, 0.45, None);
    let cs = cams(w, h);
    let mut targets = Vec::new();
    for c in cs.iter().take(3) {
        let radiance = render(&g, &truth, c);
        for ev in [-1.0f32, 0.0, 1.0] {
            let t = Truth { curve: tone, ..Truth::plain(ev, [0.0; 3], [0.0; 3], false) };
            targets.push(TargetView::new(*c, shoot_with(&radiance, c, &t)).with_exposure(ev));
        }
    }
    let init = slab(10, 0.45, Some(0.3));
    let isp_cfg = IspCfg { exposure_after: 1.0, vignetting_after: 1.0, ccm_after: 1.0, response_after: 0.1, ..IspCfg::default() };
    let out = fit_full(&g, Kernels::at(0), &init, &targets, &FitCfg { isp: Some(isp_cfg), ..base_cfg(300) }, &mut |_, _| true);
    let isp = out.isp.as_ref().expect("a fit with a camera model returns it");
    // what the brackets measured: radiance 0.07..0.45 at -1..+1 EV
    let ys: Vec<f32> = (0..12).map(|i| 0.05 * 2f32.powf(i as f32 * 0.35)).collect();
    let scales = || (0..600).map(|i| 0.125 * 2f32.powf(i as f32 / 100.0));
    let off = |f: &dyn Fn(f32) -> f32, k: f32| -> f32 { ys.iter().map(|&y| (f(y) - tone(k * y)).abs()).fold(0.0, f32::max) };
    for c in 0..3 {
        let fitted = |y: f32| isp.response(0, y)[c];
        let (k, e) = scales().map(|k| (k, off(&fitted, k))).min_by(|a, b| a.1.total_cmp(&b.1)).expect("a scale");
        let identity = scales().map(|k| off(&|y| y, k)).fold(f32::INFINITY, f32::min);
        assert!(
            e < 0.02,
            "channel {c}: response curve recovered within {e:.3} of the truth (best scale {k:.3}); the identity curve is {identity:.3} off\n{}",
            isp.summary()
        );
    }
}

/// Views whose brightness ramps across the frame, differently in each (a
/// phone's local tone mapping): neither a per-view gain nor a lens explains
/// it, so without a local model the scene absorbs it. A per-view bilateral
/// grid does, and the scene seen from the neutral camera is the truth.
#[test]
fn a_bilateral_grid_explains_local_tone_the_global_camera_cannot() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let (w, h) = (48u32, 48u32);
    let truth = slab(12, 0.8, None);
    let cs = cams(w, h);
    let ramps = [0.3f32, -0.25, 0.2, -0.2, -0.05];
    let targets: Vec<TargetView> = cs
        .iter()
        .zip(ramps)
        .map(|(c, ramp)| TargetView::new(*c, shoot_with(&render(&g, &truth, c), c, &Truth { ramp, ..Truth::plain(0.0, [0.0; 3], [0.0; 3], false) })))
        .collect();
    let init = slab(12, 0.8, Some(0.35));
    let global = IspCfg { vignetting_after: 0.1, response_after: 1.0, ..IspCfg::default() };
    let local = IspCfg { grid: Some(GridCfg { cells: [8, 8, 4], after: 0.1, ..GridCfg::default() }), ..global };
    let fit = |isp: IspCfg| fit_full(&g, Kernels::at(0), &init, &targets, &FitCfg { isp: Some(isp), ..base_cfg(240) }, &mut |_, _| true);
    let (a, b) = (fit(global), fit(local));
    let training = |r: &splat::opt::FitResult| -> f64 {
        let isp = r.isp.as_ref().expect("camera model");
        let sum: f64 = targets.iter().enumerate().map(|(v, t)| psnr(&isp.render(Shot::Training(v), &t.cam, &render(&g, &r.scene, &t.cam)), &t.rgb)).sum();
        sum / targets.len() as f64
    };
    let (ta, tb) = (training(&a), training(&b));
    assert!(tb > ta + 3.0, "training views reproduce at {tb:.1} dB with the grid and {ta:.1} dB without");
    let held = Camera::look_at([0.3, 0.3, -0.2], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 50.0, w, h);
    let want = render(&g, &truth, &held);
    let (pa, pb) = (psnr(&render(&g, &a.scene, &held), &want), psnr(&render(&g, &b.scene, &held), &want));
    assert!(pb > pa + 2.0, "a held-out view renders at {pb:.1} dB with the grid and {pa:.1} dB without");
}

/// The device kernels are the host model: forward, the radiance adjoint and
/// every camera parameter's gradient, on a frame that is not a whole number
/// of workgroups, through two sensors, both colour spaces, a bilateral grid
/// and a novel view's neutral camera.
#[test]
fn the_device_camera_is_the_host_camera() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let (w, h) = (37u32, 23u32);
    let mut cam = Camera::look_at([0.0; 3], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 60.0, w, h);
    cam.cx += 3.0;
    let px = (w * h) as usize;
    for space in [ColorSpace::Display, ColorSpace::SceneLinear] {
        let cfg = IspCfg { color_space: space, grid: Some(GridCfg { cells: [5, 4, 3], ..GridCfg::default() }), ..IspCfg::default() };
        let mut isp = Isp::new(cfg, &[(0, 0.3, Encoding::Srgb), (1, -0.2, Encoding::Srgb)]);
        let mut rng = Lcg::new(17);
        let params: Vec<f32> = isp.params().iter().map(|_| rng.scaled(0.08)).collect();
        isp.set_params(&params);
        let rgba: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.7 } else { 0.02 + 1.1 * rng.unit() }).collect();
        let rgb = rgba_to_rgb(&rgba);
        let radiance = g.storage_init("test.radiance", &rgba);
        let io = g.storage((px * 4) as u64);
        let dev = DeviceIsp::new(&g, &isp, px);
        for shot in [Shot::Training(1), Shot::Novel(NovelShot::neutral(1, 0.4, Encoding::Srgb))] {
            dev.forward(&g, Kernels::at(0), &isp, shot, &cam, &radiance, &io);
            let got = rgba_to_rgb(&g.read(&io, px * 4));
            let want = isp.render(shot, &cam, &rgb);
            for i in 0..want.len() {
                assert!((got[i] - want[i]).abs() < 2e-5 * (1.0 + want[i].abs()), "{space:?} {shot:?} forward {i}: {} vs {}", got[i], want[i]);
            }
        }
        let up: Vec<f32> = (0..px * 3).map(|_| rng.signed()).collect();
        let drad = isp.backward(1, &cam, &rgb, &up);
        let host = isp.gradient().to_vec();
        isp.clear_gradient();
        g.write_f32(&io, &up.chunks_exact(3).flat_map(|u| [u[0], u[1], u[2], 0.25]).collect::<Vec<f32>>());
        dev.backward(&g, Kernels::at(0), &mut isp, 1, &cam, &radiance, &io);
        let got = g.read(&io, px * 4);
        for p in 0..px {
            assert_eq!(got[p * 4 + 3], 0.25, "the backward leaves alpha's upstream alone");
            for c in 0..3 {
                let (a, b) = (got[p * 4 + c], drad[p * 3 + c]);
                assert!((a - b).abs() < 2e-5 * (1.0 + b.abs()), "{space:?} d radiance {p}.{c}: {a} vs {b}");
            }
        }
        let device = isp.gradient();
        let scale = host.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "the backward reached the camera's parameters");
        for (k, (a, b)) in device.iter().zip(&host).enumerate() {
            assert!((a - b).abs() < 1e-3 * b.abs() + 1e-5 * scale, "{space:?} parameter {k}: device {a} vs host {b}");
        }
    }
}
