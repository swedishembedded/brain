// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T13 gate: depth supervision and per-view masks in `splat fit`.
//!
//! The RGB loss is structurally blind to one axis. A gaussian's image-plane
//! gradient is orthogonal to its own viewing ray, so a scene placed at the
//! wrong DISTANCE - but scaled to subtend the same angle - renders the
//! identical image and has an exactly zero RGB gradient. An analytic
//! ground-truth harness measured that as a systematic 5.7% depth error on a
//! thin object, which is the number the headline test below reproduces
//! deliberately.
//!
//! The depth target is never ground truth: it comes from the same model whose
//! depth is wrong, so the term is an ANCHOR the caller weights per pixel
//! rather than a source of truth. The confidence channel is under test here
//! too, against a deliberately biased prior.
//!
//! Swedish Embedded AB implements differentiable renderers and the gradient
//! gates that keep them honest. If your team needs expertise in 3D
//! reconstruction or GPU autodiff then you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};
use splat::opt::{fit, FitCfg, TargetView};
use splat::renderer::{add_expected_depth_vjp, BwdScratch, GpuSplats, Renderer, SplatGrads};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

/// A jittered slab of `cells*cells` gaussians facing the camera.
///
/// Dense and opaque on purpose: an expected depth only means something where
/// the frame has accumulated alpha, so a scene of scattered blobs would leave
/// the test measuring a handful of pixels. The z jitter keeps the depth map
/// from being a constant the fit could satisfy by accident.
fn slab(cells: usize, seed: u64) -> Splats {
    let mut r = Lcg(seed);
    let span = 2.4f32;
    let step = span / cells as f32;
    let mut s = Splats::default();
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[
                -span / 2.0 + step * (ix as f32 + 0.5),
                -span / 2.0 + step * (iy as f32 + 0.5),
                3.2 + 0.8 * r.next(),
            ]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.6, step * 0.6, step * 0.6]);
            s.opacities.push(0.85 + 0.13 * r.next());
            s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
        }
    }
    s
}

/// A scene the forward is SMOOTH over, for finite differences.
///
/// The rasterizer has two genuine discontinuities - a splat drops out below
/// alpha 1/255, and a pixel stops walking once T falls to 1e-4 - and a dense
/// opaque scene sits on top of both: measured on the slab above, a central
/// difference on one scale reported -7.10 against a true 0.02, with the depth
/// term switched off entirely. That is the forward jumping, not the backward
/// being wrong, and no tolerance makes such a check meaningful. Six gaussians
/// at opacity ~0.6 never drive T below 4e-3, so nothing terminates and the
/// difference measures what it claims to.
fn smooth_scene(n: usize, seed: u64) -> Splats {
    let mut r = Lcg(seed);
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&[
            (r.next() - 0.5) * 1.6,
            (r.next() - 0.5) * 1.6,
            3.0 + r.next() * 1.5,
        ]);
        s.quats.extend_from_slice(&[0.5 + r.next(), r.next() - 0.5, r.next() - 0.5, r.next() - 0.5]);
        s.scales.extend_from_slice(&[
            0.25 + r.next() * 0.15,
            0.25 + r.next() * 0.15,
            0.25 + r.next() * 0.15,
        ]);
        s.opacities.push(0.5 + 0.15 * r.next());
        s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
    }
    s
}

/// The camera is at the WORLD ORIGIN looking down +Z, so scaling every mean
/// and every scale by the same factor is exactly a slide along each gaussian's
/// own viewing ray: identical image, different depth. That is the degeneracy
/// the depth term exists to break, and building the tests this way keeps them
/// from proving anything weaker.
fn cam(w: u32, h: u32) -> Camera {
    Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h)
}

fn slide_along_rays(s: &Splats, k: f32) -> Splats {
    let mut out = s.clone();
    for v in out.means.iter_mut() {
        *v *= k;
    }
    for v in out.scales.iter_mut() {
        *v *= k;
    }
    out
}

/// Render once and return (rgba `[px*4]`, expected depth `[px]`).
fn shoot(g: &Gpu, ks: Kernels, s: &Splats, c: &Camera, o: &RenderOpts) -> (Vec<f32>, Vec<f32>) {
    let mut ren = Renderer::new(g, ks, s.len(), c.width, c.height, 0);
    let gs = GpuSplats::upload(g, s);
    ren.render(g, &gs, c, o);
    (ren.read_rgba(g, c.width, c.height), ren.read_depth(g, c.width, c.height))
}

fn rgb_of(rgba: &[f32]) -> Vec<f32> {
    rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64
}

/// Pixels the truth actually covers - the only ones where an expected depth
/// is a measurement rather than a quotient of two near-zeros.
fn covered(truth_a: &[f32], truth: &[f32], i: usize) -> bool {
    truth_a[i * 4 + 3] >= 0.5 && truth[i] > 0.0
}

/// Mean SIGNED relative depth error over the covered pixels `keep` accepts.
/// Signed, because a prior biased in one direction has to be shown to have
/// dragged the fit THAT way rather than merely disturbed it.
fn depth_bias(got: &[f32], truth: &[f32], truth_a: &[f32], keep: impl Fn(usize) -> bool) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..truth.len() {
        if !covered(truth_a, truth, i) || !keep(i) {
            continue;
        }
        sum += ((got[i] - truth[i]) / truth[i]) as f64;
        n += 1;
    }
    assert!(n > 100, "not enough covered pixels to measure depth ({n})");
    sum / n as f64
}

/// Mean ABSOLUTE relative depth error over every covered pixel.
fn depth_err(got: &[f32], truth: &[f32], truth_a: &[f32]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..truth.len() {
        if !covered(truth_a, truth, i) {
            continue;
        }
        sum += ((got[i] - truth[i]).abs() / truth[i]) as f64;
        n += 1;
    }
    assert!(n > 100, "not enough covered pixels to measure depth ({n})");
    sum / n as f64
}

// ---------------------------------------------------------------------------
// 1. the VJP itself
// ---------------------------------------------------------------------------

/// L = Σ_px [ Σ_c wrgb·rgb + wd·D ] with D the per-pixel expected depth:
/// linear in both outputs, so the upstream gradients are the weights exactly
/// and every remaining discrepancy belongs to the renderer's backward.
fn depth_gradcheck(g: &Gpu) {
    let ks = Kernels::at(0);
    let base = smooth_scene(6, 0x51ce);
    let c = cam(32, 32);
    let o = RenderOpts { antialiased: false, eps2d: 0.3, ..Default::default() };
    let px = (c.width * c.height) as usize;

    // Fixed weights: part of the loss DEFINITION, so they do not move when a
    // parameter is perturbed. Depth is weighted only where the base scene has
    // accumulated enough alpha for an expected depth to mean anything.
    let (base_rgba, base_depth) = shoot(g, ks, &base, &c, &o);
    let mut r = Lcg(0x77aa);
    let wrgb: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { r.next() - 0.5 }).collect();
    let wd: Vec<f32> = (0..px)
        .map(|i| if base_rgba[i * 4 + 3] > 0.4 { r.next() - 0.5 } else { 0.0 })
        .collect();
    let supervised = wd.iter().filter(|v| **v != 0.0).count();
    assert!(supervised > 40, "too few depth-supervised pixels ({supervised})");

    let loss = |s: &Splats| -> f64 {
        let (rgba, depth) = shoot(g, ks, s, &c, &o);
        let mut l = 0.0f64;
        for i in 0..px {
            for k in 0..3 {
                l += (rgba[i * 4 + k] * wrgb[i * 4 + k]) as f64;
            }
            l += (depth[i] * wd[i]) as f64;
        }
        l
    };

    // ---- analytic ----
    let mut ren = Renderer::new(g, ks, base.len(), c.width, c.height, 0);
    let gs = GpuSplats::upload(g, &base);
    ren.render(g, &gs, &c, &o);
    let grads = SplatGrads::new(g, base.len());
    g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
    let mut dimg_h = wrgb.clone();
    let mut ddepth_h = vec![0.0f32; px];
    add_expected_depth_vjp(&wd, &base_depth, &base_rgba, &mut dimg_h, &mut ddepth_h);
    let dimg: DeviceBuffer = g.storage_init("dimg", &dimg_h);
    let ddepth: DeviceBuffer = g.storage_init("ddepth", &ddepth_h);
    let mut bscr = BwdScratch::new(g, base.len(), px, 0);
    ren.render_bwd(g, &gs, &c, &o, &dimg, Some(&ddepth), &mut bscr, &grads).expect("fits");
    let d_gauss = g.read(&grads.d_gauss, 10 * base.len());
    let d_opac = g.read(&grads.d_opac, base.len());

    // The step, and the size below which a gradient is not worth comparing.
    //
    // The step cannot be raised: a splat's truncation radius is `ceil`ed to
    // whole pixels and the composite drops splats below alpha 1/255, so at
    // h=1e-3 four of these gradients disagree by 60-80% with the CPU and the
    // GPU reporting the SAME wrong difference - the forward jumping, not the
    // backward being wrong.
    //
    // The step cannot be lowered either, and that fixes the cutoff. Each
    // rendered depth is an fp32 number near 3.5, so it carries ~4e-7 of
    // rounding; ~400 weighted pixels put ~2e-6 on the loss, and a central
    // difference divides that by 2h for a noise floor near 0.005. It shows up
    // directly as CPU and GPU finite differences that disagree with EACH
    // OTHER by that much while their analytic gradients match to five digits.
    // Only gradients well clear of that floor are asserted on; backend parity
    // below covers the rest, including the small ones.
    const H: f32 = 2e-4;
    const MIN_GRAD: f64 = 0.2;
    let mut rows: Vec<(f64, String)> = Vec::new();
    let fd_at = |up: &Splats, dn: &Splats| (loss(up) - loss(dn)) / (2.0 * H) as f64;
    for gi in 0..base.len() {
        let mut cases: Vec<(String, f64, f64)> = Vec::new();
        for k in 0..3 {
            // MEANS are the point: moving a gaussian along the ray is exactly
            // what the RGB term cannot ask for.
            let (mut up, mut dn) = (base.clone(), base.clone());
            up.means[gi * 3 + k] += H;
            dn.means[gi * 3 + k] -= H;
            cases.push((format!("mean[{gi}][{k}]"), d_gauss[gi * 10 + k] as f64, fd_at(&up, &dn)));

            let (mut up, mut dn) = (base.clone(), base.clone());
            up.scales[gi * 3 + k] += H;
            dn.scales[gi * 3 + k] -= H;
            cases.push((format!("scale[{gi}][{k}]"), d_gauss[gi * 10 + 3 + k] as f64, fd_at(&up, &dn)));
        }
        let (mut up, mut dn) = (base.clone(), base.clone());
        up.opacities[gi] += H;
        dn.opacities[gi] -= H;
        cases.push((format!("opacity[{gi}]"), d_opac[gi] as f64, fd_at(&up, &dn)));
        for (name, an, fd) in cases {
            if fd.abs() <= MIN_GRAD {
                continue;
            }
            let rel = (an - fd).abs() / an.abs().max(fd.abs());
            rows.push((rel, format!("{name}: analytic {an:.5} vs finite-difference {fd:.5}")));
        }
    }
    assert!(rows.len() > 10, "too few well-conditioned gradients exercised ({})", rows.len());
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let worst = rows[0].clone();
    if worst.0 >= 0.05 {
        for (r, m) in rows.iter().take(12) {
            println!("  {:6.1}%  {m}", 100.0 * r);
        }
    }
    println!("depth VJP: {} gradients, worst {:.2}%", rows.len(), 100.0 * worst.0);
    assert!(worst.0 < 0.05, "worst disagreement {:.1}%: {}", 100.0 * worst.0, worst.1);
}

#[test]
fn the_depth_vjp_matches_finite_differences_cpu() {
    depth_gradcheck(&Gpu::new_cpu(splat::PIPELINES));
}

#[test]
fn the_depth_vjp_matches_finite_differences_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    depth_gradcheck(&gpu_core::testgpu::dev(splat::PIPELINES));
}

/// Every gradient, not just the well-conditioned ones, has to be the same on
/// both backends.
///
/// The finite-difference gate above can only speak for gradients well clear
/// of its noise floor, which leaves the small ones unchecked by it. WGSL is
/// the source of truth and the CPU JIT and the GPU compile the same text, so
/// a disagreement between them is a kernel that reads or writes the record
/// stride differently on one of them - exactly the failure widening the
/// gradient record invites, and one that is silent otherwise.
#[test]
fn the_depth_gradients_are_the_same_on_both_backends() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let ks = Kernels::at(0);
    let base = smooth_scene(6, 0x51ce);
    let c = cam(32, 32);
    let o = RenderOpts { antialiased: false, eps2d: 0.3, ..Default::default() };
    let px = (c.width * c.height) as usize;
    let cpu = Gpu::new_cpu(splat::PIPELINES);

    // ONE upstream gradient, built on the CPU and handed to both, so the only
    // thing that can differ downstream is the backward itself.
    let (base_rgba, base_depth) = shoot(&cpu, ks, &base, &c, &o);
    let mut r = Lcg(0x3c0d);
    let wrgb: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { r.next() - 0.5 }).collect();
    let wd: Vec<f32> = (0..px)
        .map(|i| if base_rgba[i * 4 + 3] > 0.4 { r.next() - 0.5 } else { 0.0 })
        .collect();
    let mut dimg_h = wrgb.clone();
    let mut ddepth_h = vec![0.0f32; px];
    add_expected_depth_vjp(&wd, &base_depth, &base_rgba, &mut dimg_h, &mut ddepth_h);

    let grads_on = |g: &Gpu| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut ren = Renderer::new(g, ks, base.len(), c.width, c.height, 0);
        let gs = GpuSplats::upload(g, &base);
        ren.render(g, &gs, &c, &o);
        let grads = SplatGrads::new(g, base.len());
        g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
        let dimg: DeviceBuffer = g.storage_init("dimg", &dimg_h);
        let ddepth: DeviceBuffer = g.storage_init("ddepth", &ddepth_h);
        let mut bscr = BwdScratch::new(g, base.len(), px, 0);
        ren.render_bwd(g, &gs, &c, &o, &dimg, Some(&ddepth), &mut bscr, &grads).expect("fits");
        (
            g.read(&grads.d_gauss, 10 * base.len()),
            g.read(&grads.d_opac, base.len()),
            g.read(&grads.d_colors, 3 * base.len()),
        )
    };
    let a = grads_on(&cpu);
    let b = grads_on(&gpu_core::testgpu::dev(splat::PIPELINES));

    let mut worst = (0.0f32, String::new());
    for (name, x, y) in [("d_gauss", &a.0, &b.0), ("d_opac", &a.1, &b.1), ("d_colors", &a.2, &b.2)] {
        let scale = x.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (i, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            let rel = (p - q).abs() / scale;
            if rel > worst.0 {
                worst = (rel, format!("{name}[{i}]: cpu {p:.6} vs gpu {q:.6}"));
            }
        }
    }
    println!("backend parity: worst {:.2e} relative ({})", worst.0, worst.1);
    assert!(worst.0 < 1e-3, "backends disagree by {:.2e}: {}", worst.0, worst.1);
}

// ---------------------------------------------------------------------------
// 2. the headline defect
// ---------------------------------------------------------------------------

/// 5.7% too far along every viewing ray, rendering the RIGHT image.
///
/// The RGB loss is already at its minimum here - not merely small, identically
/// zero along this direction - so an RGB-only fit cannot move the scene. The
/// depth term must, and a test that does not check BOTH halves proves nothing.
#[test]
fn a_scene_at_the_wrong_depth_is_pulled_back_only_by_the_depth_term() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let truth = slab(7, 0x60a1);
    let c = cam(32, 32);
    // the fit's own forward model, whose depth is the range along each ray
    let o = RenderOpts { ray: true, ..Default::default() };
    let (truth_rgba, truth_depth) = shoot(&g, ks, &truth, &c, &o);

    const BIAS: f32 = 1.057;
    let init = slide_along_rays(&truth, BIAS);
    let (init_rgba, init_depth) = shoot(&g, ks, &init, &c, &o);

    // the premise: same image, wrong depth
    let rgb_gap = mse(&rgb_of(&init_rgba), &rgb_of(&truth_rgba));
    assert!(rgb_gap < 1e-8, "the displaced scene does not render the same image (mse {rgb_gap:.3e})");
    let e0 = depth_err(&init_depth, &truth_depth, &truth_rgba);
    assert!(
        (e0 - 0.057).abs() < 0.005,
        "premise broken: depth error {:.3}% is not the 5.7% defect",
        100.0 * e0
    );

    let target_rgb = rgb_of(&truth_rgba);
    let depth: Vec<f32> = (0..truth_depth.len())
        .map(|i| if covered(&truth_rgba, &truth_depth, i) { truth_depth[i] } else { 0.0 })
        .collect();
    let base_cfg = FitCfg {
        iters: 150,
        lr_position: 5e-3,
        log_every: 0,
        mip_scale: 0.0,
        max_scale_pixels: 0.0,
        max_needle: 0.0,
        ..Default::default()
    };

    let run = |dw: f32| -> f64 {
        let t = TargetView::new(c, target_rgb.clone()).with_depth(depth.clone(), None);
        let cfg = FitCfg { depth_weight: dw, ..base_cfg };
        let (fitted, _) = fit(&g, ks, &init, &[t], &cfg, &mut |_, _| true);
        let (_, d) = shoot(&g, ks, &fitted, &c, &o);
        depth_err(&d, &truth_depth, &truth_rgba)
    };

    let rgb_only = run(0.0);
    let with_depth = run(1.0);
    println!(
        "depth error: start {:.2}%  rgb-only {:.2}%  +depth {:.2}%",
        100.0 * e0, 100.0 * rgb_only, 100.0 * with_depth
    );
    assert!(
        rgb_only > 0.04,
        "the RGB loss alone moved the scene to within {:.2}% of the right depth - it is blind to \
         this axis by construction, so either the test setup or the loss changed",
        100.0 * rgb_only
    );
    assert!(
        with_depth < 0.015,
        "the depth term left {:.2}% depth error (started {:.2}%)",
        100.0 * with_depth, 100.0 * e0
    );
}

// ---------------------------------------------------------------------------
// 3. the prior is not ground truth
// ---------------------------------------------------------------------------

/// A deliberately biased depth prior must move only the pixels it is trusted
/// at.
///
/// The prior a caller has comes from the same model whose depth is wrong, so
/// the per-pixel confidence is the whole safety story. Note what is NOT
/// asserted: that a globally smaller confidence produces a smaller drift. It
/// does not, and cannot - AdamW's step is normalized per parameter, so
/// scaling every depth residual by the same constant leaves the step
/// identical, and along the viewing ray there is no RGB gradient competing
/// for the direction. Confidence is a RELATIVE weight between pixels, which is
/// what this measures and what a fusion's agreement count actually provides.
#[test]
fn a_biased_depth_prior_moves_only_the_pixels_it_is_trusted_at() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let truth = slab(7, 0x60a1);
    let c = cam(32, 32);
    // the fit's own forward model, whose depth is the range along each ray
    let o = RenderOpts { ray: true, ..Default::default() };
    let (truth_rgba, truth_depth) = shoot(&g, ks, &truth, &c, &o);
    let target_rgb = rgb_of(&truth_rgba);
    let px = (c.width * c.height) as usize;

    // A 10% systematic overshoot everywhere - the failure mode of feeding a
    // single view's model depth straight in - but believed only on the left.
    const BIAS: f32 = 1.10;
    let depth: Vec<f32> = (0..px)
        .map(|i| if covered(&truth_rgba, &truth_depth, i) { truth_depth[i] * BIAS } else { 0.0 })
        .collect();
    let left = |i: usize| i % 32 < 12;
    let right = |i: usize| i % 32 >= 20;
    let conf: Vec<f32> = (0..px).map(|i| if left(i) { 1.0 } else { 0.0 }).collect();

    let t = TargetView::new(c, target_rgb).with_depth(depth, Some(conf));
    let cfg = FitCfg {
        iters: 150,
        lr_position: 5e-3,
        log_every: 0,
        depth_weight: 1.0,
        mip_scale: 0.0,
        max_scale_pixels: 0.0,
        max_needle: 0.0,
        ..Default::default()
    };
    let (fitted, _) = fit(&g, ks, &truth, &[t], &cfg, &mut |_, _| true);
    let (_, d) = shoot(&g, ks, &fitted, &c, &o);

    // A band is left out of both halves: gaussians straddling the confidence
    // edge see trusted and distrusted pixels at once, and belong to neither.
    let trusted = depth_bias(&d, &truth_depth, &truth_rgba, left);
    let distrusted = depth_bias(&d, &truth_depth, &truth_rgba, right);
    println!("biased prior: trusted {:+.2}%  distrusted {:+.2}%", 100.0 * trusted, 100.0 * distrusted);
    assert!(
        trusted > 0.05,
        "the trusted half moved only {:+.2}% toward a {:+.0}% prior",
        100.0 * trusted, 100.0 * (BIAS - 1.0)
    );
    assert!(
        distrusted.abs() < 0.01,
        "the distrusted half drifted {:+.2}% anyway",
        100.0 * distrusted
    );
}

/// Where a prior disagrees with itself, the fit settles where the confidences
/// balance - which is what makes "trust this pixel this much" a dial and not
/// a switch.
///
/// Each gaussian here covers pixels carrying BOTH a +10% and a -10% depth
/// target. With the depth term's knee wider than the disagreement the loss is
/// quadratic in log range there, so the stationary point is the
/// confidence-weighted mean of the two: equal trust puts it at the truth, 3:1
/// trust halfway to the positive bias. Same gradient magnitudes, same
/// optimizer, only the weights differ - so unlike a global scale this is a
/// statement Adam cannot normalize away.
///
/// At the default knee a 10% disagreement is far past it, where the term's
/// influence is bounded: that is what keeps a wrong prior (a reflection, a
/// bad match) from dragging the geometry, so the 3:1 run then settles
/// towards the MORE trusted prior rather than at the mean - still between
/// the two, still decided by the confidences.
#[test]
fn conflicting_depth_priors_settle_where_their_confidences_balance() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let truth = slab(7, 0x60a1);
    let c = cam(32, 32);
    // the fit's own forward model, whose depth is the range along each ray
    let o = RenderOpts { ray: true, ..Default::default() };
    let (truth_rgba, truth_depth) = shoot(&g, ks, &truth, &c, &o);
    let target_rgb = rgb_of(&truth_rgba);
    let px = (c.width * c.height) as usize;

    const BIAS: f32 = 0.10;
    // checkerboard, so every gaussian sees both sides of the disagreement
    let up = |i: usize| (i / 32 + i % 32).is_multiple_of(2);
    let depth: Vec<f32> = (0..px)
        .map(|i| {
            if !covered(&truth_rgba, &truth_depth, i) {
                return 0.0;
            }
            truth_depth[i] * if up(i) { 1.0 + BIAS } else { 1.0 - BIAS }
        })
        .collect();

    let run = |w_up: f32, w_dn: f32, knee: f32| -> f64 {
        let conf: Vec<f32> = (0..px).map(|i| if up(i) { w_up } else { w_dn }).collect();
        let t = TargetView::new(c, target_rgb.clone()).with_depth(depth.clone(), Some(conf));
        let cfg = FitCfg {
            iters: 200,
            lr_position: 5e-3,
            log_every: 0,
            depth_weight: 1.0,
            depth_delta: knee,
            mip_scale: 0.0,
            max_scale_pixels: 0.0,
            max_needle: 0.0,
            ..Default::default()
        };
        let (fitted, _) = fit(&g, ks, &truth, &[t], &cfg, &mut |_, _| true);
        let (_, d) = shoot(&g, ks, &fitted, &c, &o);
        depth_bias(&d, &truth_depth, &truth_rgba, |_| true)
    };

    let wide = 1.0;
    let balanced = run(1.0, 1.0, wide);
    let leaning = run(3.0, 1.0, wide);
    let robust = run(3.0, 1.0, FitCfg::default().depth_delta);
    // (3-1)/(3+1) of a 10% bias
    let want = 0.5 * BIAS as f64;
    println!(
        "conflicting priors: balanced {:+.2}%  3:1 {:+.2}% (expected {:+.2}%)  3:1 at the default knee {:+.2}%",
        100.0 * balanced, 100.0 * leaning, 100.0 * want, 100.0 * robust
    );
    assert!(balanced.abs() < 0.02, "equal confidences did not cancel: {:+.2}%", 100.0 * balanced);
    assert!(
        (leaning - want).abs() < 0.025,
        "3:1 confidence settled at {:+.2}%, not near the weighted mean {:+.2}%",
        100.0 * leaning, 100.0 * want
    );
    assert!(
        robust > want && robust < BIAS as f64 * 1.02,
        "past the knee 3:1 confidence settled at {:+.2}%: bounded influence puts it between the \
         weighted mean and the more trusted prior",
        100.0 * robust
    );
}

// ---------------------------------------------------------------------------
// 4. masks
// ---------------------------------------------------------------------------

/// Masked-out pixels must not reach the loss, the gradient, or the
/// normalizer. Two targets differing ONLY where the mask is zero have to
/// produce the same scene and the same reported MSE; an all-ones mask has to
/// change nothing at all; and what is reported has to be the MSE of the
/// pixels that were kept, so a masked run stays comparable with an unmasked
/// one instead of looking better for having excluded half the frame.
#[test]
fn masked_pixels_reach_neither_the_loss_nor_the_gradient() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let truth = slab(6, 0x1d0e);
    let c = cam(32, 32);
    // the fit's own forward model, whose depth is the range along each ray
    let o = RenderOpts { ray: true, ..Default::default() };
    let (truth_rgba, _) = shoot(&g, ks, &truth, &c, &o);
    let target_rgb = rgb_of(&truth_rgba);
    let px = (c.width * c.height) as usize;

    let mut init = truth.clone();
    let mut r = Lcg(0xd00d);
    for v in init.colors.iter_mut() {
        *v = (*v + (r.next() - 0.5) * 0.4).clamp(0.0, 1.0);
    }

    // Right half masked out, and filled with something a fit would chase.
    let mask: Vec<f32> = (0..px).map(|i| if (i % 32) < 16 { 1.0 } else { 0.0 }).collect();
    let mut garbage = target_rgb.clone();
    for i in 0..px {
        if mask[i] == 0.0 {
            garbage[i * 3] = 1.0;
            garbage[i * 3 + 1] = 0.0;
            garbage[i * 3 + 2] = 1.0;
        }
    }
    let run = |rgb: Vec<f32>, m: Option<Vec<f32>>, iters: usize| -> (Splats, f32) {
        let mut t = TargetView::new(c, rgb);
        if let Some(m) = m {
            t = t.with_mask(m);
        }
        let cfg = FitCfg { iters, lr_position: 5e-3, log_every: 0, ..Default::default() };
        fit(&g, ks, &init, &[t], &cfg, &mut |_, _| true)
    };

    let (a, la) = run(target_rgb.clone(), Some(mask.clone()), 40);
    let (b, lb) = run(garbage, Some(mask.clone()), 40);
    assert_eq!(la, lb, "the masked-out pixels changed the reported loss");
    assert_eq!(a.means, b.means, "the masked-out pixels changed the fitted geometry");
    assert_eq!(a.colors, b.colors, "the masked-out pixels changed the fitted colours");

    // An all-ones mask is the unmasked fit exactly - normalizer included.
    let (c1, l1) = run(target_rgb.clone(), Some(vec![1.0; px]), 40);
    let (c0, l0) = run(target_rgb.clone(), None, 40);
    assert_eq!(l1, l0, "an all-ones mask changed the reported MSE ({l1} vs {l0})");
    assert_eq!(c1.means, c0.means, "an all-ones mask changed the fit");

    // The fit reports the objective of the scene it RETURNS, so the number
    // can be checked against a direct computation rather than against the
    // fit's own arithmetic.
    let (fitted, one) = run(target_rgb.clone(), Some(mask.clone()), 1);
    let mut ren = Renderer::new(&g, ks, fitted.len(), c.width, c.height, 0);
    let gs = GpuSplats::upload(&g, &fitted);
    ren.render(&g, &gs, &c, &o);
    let img = ren.read_rgba(&g, c.width, c.height);
    let mut kept = 0.0f64;
    let mut n = 0usize;
    for i in 0..px {
        if mask[i] == 0.0 {
            continue;
        }
        for k in 0..3 {
            kept += ((img[i * 4 + k] - target_rgb[i * 3 + k]) as f64).powi(2);
        }
        n += 1;
    }
    assert_eq!(n, px / 2, "the mask did not keep half the frame");
    let kept = kept / (n as f64 * 3.0);
    assert!(
        (kept - one as f64).abs() < 1e-6 * kept,
        "reported {one:.9} is not the masked MSE {kept:.9}"
    );
}
