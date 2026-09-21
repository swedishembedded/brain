// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate on "is the reconstruction any good", as opposed to "did it run".
//!
//! Every other test in this crate checks a mechanism: gradients match an
//! oracle, a sort is stable, a PLY round-trips. None of them notices a scene
//! that reconstructs into a smear, because a smear is finite, sorted and
//! serialisable. The failure this catches was found by a person looking at a
//! picture, which is the part that does not scale.
//!
//! The property, stated so it cannot be satisfied by a blur: a scene fitted
//! against posed views must reproduce those views both ACCURATELY (PSNR) and
//! SHARPLY ([`quality::sharpness_ratio`] inside a band). Accuracy alone
//! passes a render that has lost every edge, because low-frequency content is
//! most of the energy in a photograph; sharpness alone passes a render that
//! is crisply wrong. Neither is sufficient and both are cheap.
//!
//! Swedish Embedded AB implements automated quality gates for 3D
//! reconstruction and rendering pipelines. If your team needs visual
//! regressions caught by CI rather than by a reviewer, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit, FitCfg, TargetView};
use splat::quality::{psnr, sharpness_ratio};
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// Floors the gate holds a reconstruction to, shared by the gate and by the
/// test that proves the gate has teeth - so the second cannot drift into
/// demonstrating something the first does not actually check.
///
/// Calibrated, not guessed: the fitted scene below lands at 29.0-31.5 dB and
/// 0.89-0.98 sharpness, and a 3x3 box blur of a CORRECT render lands at
/// 25.8 dB and 0.29. The accuracy floor sits under the blur deliberately, so
/// that the teeth test can show accuracy waving it through.
const ACCURACY_FLOOR_DB: f64 = 24.0;
const SHARPNESS_BAND: std::ops::Range<f64> = 0.70..1.60;
/// A view the fit never saw is held to a lower bar than one it optimized
/// against - it is a harder question, and the gap between the two is itself
/// the thing worth watching.
const HELD_OUT_FLOOR_DB: f64 = 21.0;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

/// A textured slab: gaussians on a grid in the z = 3 plane, coloured in a
/// checker so the scene HAS high-frequency content to lose. A cloud of random
/// blobs would make this gate vacuous - there would be no edges for a blur to
/// remove, and the sharpness ratio would sit at 1.0 whatever happened.
fn board(cells: usize) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            let (x, y) = (-1.2 + step * (ix as f32 + 0.5), -1.2 + step * (iy as f32 + 0.5));
            s.means.extend_from_slice(&[x, y, 3.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.42, step * 0.42, step * 0.42]);
            s.opacities.push(0.99);
            let on = (ix + iy) % 2 == 0;
            let v = if on { 0.88 } else { 0.12 };
            s.colors.extend_from_slice(&[v, v, v * 0.92 + 0.04]);
        }
    }
    s
}

fn views(g: &Gpu, ks: Kernels, truth: &Splats, w: u32, h: u32) -> (Vec<TargetView>, Renderer) {
    let cams = [
        Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([0.9, -0.3, 0.4], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([-0.9, 0.3, 0.4], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([0.2, 0.9, 0.5], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
    ];
    let o = RenderOpts::default();
    let mut ren = Renderer::new(g, ks, truth.len(), w, h, 0);
    let gs = GpuSplats::upload(g, truth);
    let t = cams
        .iter()
        .map(|c| {
            ren.render(g, &gs, c, &o);
            let img = ren.read_rgba(g, c.width, c.height);
            TargetView { cam: *c, rgb: img.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect() }
        })
        .collect();
    (t, ren)
}

/// The gate. A scene knocked out of alignment and fitted back must land close
/// to the photographs AND as sharp as them.
#[test]
fn a_fitted_scene_reproduces_its_views_sharply_and_not_merely_closely() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (72u32, 72u32);
    let truth = board(16);
    let (targets, _ren) = views(&g, ks, &truth, w, h);

    // Knock it out: colours toward flat grey, positions jittered. Flattening
    // the colours is what removes the edges, so the fit has to put the
    // high-frequency content BACK - it cannot pass by smoothing.
    let mut init = truth.clone();
    let mut r = Lcg(0xbeef);
    for v in init.colors.iter_mut() {
        *v = 0.5 + (*v - 0.5) * 0.25 + (r.next() - 0.5) * 0.1;
    }
    for v in init.means.iter_mut() {
        *v += (r.next() - 0.5) * 0.04;
    }

    let cfg = FitCfg { iters: 220, lr: 8e-3, log_every: 0, ..Default::default() };
    let (fitted, _) = fit(&g, ks, &init, &targets, &cfg, &mut |_, _| true);

    let o = RenderOpts::default();
    // `fit` may return more gaussians than it was given, so size the renderer
    // from what came back rather than from what went in.
    let mut ren = Renderer::new(&g, ks, fitted.len().max(truth.len()), w, h, 0);
    let gs = GpuSplats::upload(&g, &fitted);
    let (wu, hu) = (w as usize, h as usize);
    for (i, t) in targets.iter().enumerate() {
        ren.render(&g, &gs, &t.cam, &o);
        let rgba = ren.read_rgba(&g, w, h);
        let rgb: Vec<f32> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();

        let db = psnr(&rgb, &t.rgb);
        let sharp = sharpness_ratio(&rgb, &t.rgb, wu, hu);
        assert!(
            db > ACCURACY_FLOOR_DB,
            "view {i}: fitted scene reproduces its own target at only {db:.1} dB"
        );
        assert!(
            SHARPNESS_BAND.contains(&sharp),
            "view {i}: fitted scene renders at {sharp:.2}x the target's high-frequency content \
             ({db:.1} dB). Below the band is a smear, above it is speckle; PSNR sees neither \
             until it is severe."
        );
    }
}

/// And the gate has teeth: the SAME assertions, applied to a deliberately
/// blurred render of the correct scene, must fail on sharpness while passing
/// on accuracy. Without this, a band that happened to be too wide would look
/// exactly like a passing gate.
#[test]
fn the_gate_rejects_a_blurred_render_that_accuracy_alone_accepts() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (72u32, 72u32);
    let truth = board(16);
    let (targets, mut ren) = views(&g, ks, &truth, w, h);
    let (wu, hu) = (w as usize, h as usize);

    let o = RenderOpts::default();
    let gs = GpuSplats::upload(&g, &truth);
    ren.render(&g, &gs, &targets[0].cam, &o);
    let rgba = ren.read_rgba(&g, w, h);
    let exact: Vec<f32> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();

    // the same scene, rendered through a 3x3 box - "slightly out of focus"
    let mut soft = exact.clone();
    for y in 1..hu - 1 {
        for x in 1..wu - 1 {
            for c in 0..3 {
                let mut acc = 0.0;
                for dy in 0..3 {
                    for dx in 0..3 {
                        acc += exact[((y + dy - 1) * wu + (x + dx - 1)) * 3 + c];
                    }
                }
                soft[(y * wu + x) * 3 + c] = acc / 9.0;
            }
        }
    }

    let db = psnr(&soft, &exact);
    let sharp = sharpness_ratio(&soft, &exact, wu, hu);
    assert!(
        db > ACCURACY_FLOOR_DB,
        "the blurred render scores {db:.1} dB, BELOW the accuracy floor of {ACCURACY_FLOOR_DB} - so \
         accuracy alone would already have caught it and this test proves nothing. Either the floor \
         rose or the render got worse; check which before trusting the gate above."
    );
    assert!(
        !SHARPNESS_BAND.contains(&sharp),         "the blurred render kept {sharp:.2}x of the sharpness, inside the {SHARPNESS_BAND:?} band \
         the gate above allows - the band is too wide to catch a smear"
    );
}

/// A scene has to be right from somewhere it was NOT fitted.
///
/// Every other measure here renders the views the fit optimized against, and a
/// fit can satisfy those while producing something unusable. It reaches for
/// shapes that are invisible from the view that created them - a needle lined
/// up with a training ray costs nothing in that frame and streaks across every
/// other. Measured on a real reconstruction: 32.7 dB from its own cameras, and
/// a picture full of hairs the moment the camera moved.
///
/// So one view is held OUT of the fit and scored afterwards. That is the
/// number that says whether a scene models the subject or merely reproduces
/// its inputs.
#[test]
fn a_fitted_scene_is_still_right_from_a_view_it_never_saw() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (72u32, 72u32);
    let truth = board(16);
    let all = views(&g, ks, &truth, w, h).0;

    // fit on every view but the last, score on the last
    let (train, test) = all.split_at(all.len() - 1);
    let mut init = truth.clone();
    let mut r = Lcg(0xfeed);
    for v in init.colors.iter_mut() {
        *v = 0.5 + (*v - 0.5) * 0.25 + (r.next() - 0.5) * 0.1;
    }
    let cfg = FitCfg { iters: 200, lr: 8e-3, log_every: 0, ..Default::default() };
    let (fitted, _) = fit(&g, ks, &init, train, &cfg, &mut |_, _| true);
    let loose = FitCfg { iters: 200, lr: 8e-3, log_every: 0, max_aspect: 0.0, ..Default::default() };
    let (needly, _) = fit(&g, ks, &init, train, &loose, &mut |_, _| true);

    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, fitted.len().max(truth.len()), w, h, 0);
    let gs = GpuSplats::upload(&g, &fitted);
    let t = &test[0];
    ren.render(&g, &gs, &t.cam, &o);
    let rgba = ren.read_rgba(&g, w, h);
    let rgb: Vec<f32> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let (wu, hu) = (w as usize, h as usize);
    let db = psnr(&rgb, &t.rgb);
    let sharp = sharpness_ratio(&rgb, &t.rgb, wu, hu);
    // the same view, from a fit allowed to make needles
    let gs2 = GpuSplats::upload(&g, &needly);
    ren.render(&g, &gs2, &t.cam, &o);
    let r2 = ren.read_rgba(&g, w, h);
    let rgb2: Vec<f32> = r2.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    eprintln!("CAL held-out: clamped {db:.2} dB / {sharp:.3}  |  unclamped {:.2} dB / {:.3}",
              psnr(&rgb2, &t.rgb), sharpness_ratio(&rgb2, &t.rgb, wu, hu));
    assert!(
        db > HELD_OUT_FLOOR_DB,
        "held out of the fit, the scene renders at {db:.1} dB - it reproduces its inputs rather \
         than modelling the subject"
    );
    assert!(
        SHARPNESS_BAND.contains(&sharp),
        "held out of the fit, the scene carries {sharp:.2}x the target's high-frequency content; \
         above the band is the streaking a training-view metric cannot see"
    );
}
