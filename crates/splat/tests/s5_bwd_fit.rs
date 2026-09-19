// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T2 gate: rasterizer backward.
//!   1. gradcheck of render_bwd against a committed torch-autograd golden
//!      (tools/goldens/splat_dump_gradcheck.py; float64 oracle of the identical
//!      scene/loss). Finite differences are NOT used: the 1/255 truncation
//!      boundaries make them biased for scale/mean grads (the classic 3DGS
//!      finite-diff pitfall — verified: autograd matches our analytic grads
//!      where central differences are off by a factor of two);
//!   2. fit convergence: a perturbed scene optimized against renders of the
//!      ground truth must reduce MSE substantially.

use gpu_core::{DeviceBuffer, Gpu};
use splat::opt::{fit, FitCfg, TargetView};
use splat::renderer::{BwdScratch, GpuSplats, Renderer, SplatGrads};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

fn scene(n: usize, seed: u64) -> Splats {
    let mut r = Lcg(seed);
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&[
            (r.next() - 0.5) * 2.0,
            (r.next() - 0.5) * 2.0,
            3.0 + r.next() * 2.0,
        ]);
        s.quats.extend_from_slice(&[
            0.5 + r.next(),
            r.next() - 0.5,
            r.next() - 0.5,
            r.next() - 0.5,
        ]);
        s.scales.extend_from_slice(&[
            0.1 + r.next() * 0.15,
            0.1 + r.next() * 0.15,
            0.1 + r.next() * 0.15,
        ]);
        s.opacities.push(0.35 + 0.5 * r.next());
        s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
    }
    s
}

fn cam() -> Camera {
    Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 32, 32)
}

/// L = Σ_px Σ_c img_rgb · wimg (fixed random weights) — linear in the image,
/// so dL/dimg = wimg exactly.
fn loss(img: &[f32], wimg: &[f32]) -> f64 {
    let mut l = 0.0f64;
    for i in 0..img.len() / 4 {
        for c in 0..3 {
            l += (img[i * 4 + c] * wimg[i * 4 + c]) as f64;
        }
    }
    l
}

#[test]
fn gradcheck_vs_autograd() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let base = scene(6, 0x5eed);
    let c = cam();
    let o = RenderOpts::default();
    let px = (c.width * c.height) as usize;
    let mut r = Lcg(0xabcd);
    let wimg: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { r.next() - 0.5 }).collect();

    let render_loss = |s: &Splats| -> f64 {
        let mut ren = Renderer::new(&g, ks, s.len(), c.width, c.height, 0);
        let gs = GpuSplats::upload(&g, s);
        ren.render(&g, &gs, &c, &o);
        loss(&ren.read_rgba(&g, c.width, c.height), &wimg)
    };

    // analytic grads
    let mut ren = Renderer::new(&g, ks, base.len(), c.width, c.height, 0);
    let gs = GpuSplats::upload(&g, &base);
    ren.render(&g, &gs, &c, &o);
    let grads = SplatGrads::new(&g, base.len());
    g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
    let dimg: DeviceBuffer = g.storage_init("dimg", &wimg);
    let mut bscr = BwdScratch::new(&g, base.len(), px, 0);
    let nrecs = ren.render_bwd(&g, &gs, &c, &o, &dimg, &mut bscr, &grads).expect("fits");
    assert!(nrecs > 0);
    let d_gauss = g.read(&grads.d_gauss, 10 * base.len());
    let d_opac = g.read(&grads.d_opac, base.len());
    let d_colors = g.read(&grads.d_colors, 3 * base.len());

    let _ = render_loss; // (kept for local debugging)
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("golden/gradcheck.json")).unwrap();
    let get = |k: &str| -> Vec<f32> {
        golden[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
    };
    let (gm, gsc, gq, go, gc) =
        (get("d_means"), get("d_scales"), get("d_quats"), get("d_opac"), get("d_colors"));
    let mut checked = 0;
    let mut check = |name: &str, analytic: f32, expect: f32| {
        let denom = analytic.abs().max(expect.abs()).max(1e-3);
        let rel = (analytic - expect).abs() / denom;
        assert!(rel < 5e-3, "{name}: analytic {analytic:.6} vs autograd {expect:.6} (rel {rel:.4})");
        if expect.abs() > 1e-4 {
            checked += 1;
        }
    };
    for gi in 0..6 {
        for k in 0..3 {
            check(&format!("mean[{gi}][{k}]"), d_gauss[gi * 10 + k], gm[gi * 3 + k]);
            check(&format!("scale[{gi}][{k}]"), d_gauss[gi * 10 + 3 + k], gsc[gi * 3 + k]);
            check(&format!("color[{gi}][{k}]"), d_colors[gi * 3 + k], gc[gi * 3 + k]);
        }
        for k in 0..4 {
            check(&format!("quat[{gi}][{k}]"), d_gauss[gi * 10 + 6 + k], gq[gi * 4 + k]);
        }
        check(&format!("opacity[{gi}]"), d_opac[gi], go[gi]);
    }
    assert!(checked > 40, "too few non-trivial gradients exercised ({checked})");
}

#[test]
fn fit_recovers_perturbed_scene() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let truth = scene(12, 0x60a1);
    // three target views around the scene
    let cams = [
        Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 48, 48),
        Camera::look_at([1.5, -0.5, 0.5], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 48, 48),
        Camera::look_at([-1.5, 0.5, 0.5], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 48, 48),
    ];
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, truth.len(), 48, 48, 0);
    let gst = GpuSplats::upload(&g, &truth);
    let targets: Vec<TargetView> = cams
        .iter()
        .map(|c| {
            ren.render(&g, &gst, c, &o);
            let img = ren.read_rgba(&g, c.width, c.height);
            let rgb: Vec<f32> = img.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            TargetView { cam: *c, rgb }
        })
        .collect();

    // perturb colors + opacities + positions
    let mut init = truth.clone();
    let mut r = Lcg(0xd00d);
    for v in init.colors.iter_mut() {
        *v = (*v + (r.next() - 0.5) * 0.4).clamp(0.0, 1.0);
    }
    for v in init.means.iter_mut() {
        *v += (r.next() - 0.5) * 0.1;
    }
    let mse0 = {
        let gs = GpuSplats::upload(&g, &init);
        let mut acc = 0.0f64;
        for t in &targets {
            ren.render(&g, &gs, &t.cam, &o);
            let img = ren.read_rgba(&g, t.cam.width, t.cam.height);
            let px = (t.cam.width * t.cam.height) as usize;
            let mut l = 0.0f64;
            for i in 0..px {
                for c in 0..3 {
                    let d = img[i * 4 + c] - t.rgb[i * 3 + c];
                    l += (d * d) as f64;
                }
            }
            acc += l / (px as f64 * 3.0);
        }
        acc / targets.len() as f64
    };

    let cfg = FitCfg { iters: 120, lr: 5e-3, log_every: 0, ..Default::default() };
    let (_fitted, mse_end) = fit(&g, ks, &init, &targets, &cfg, &mut |_it, _mse| true);
    assert!(
        (mse_end as f64) < mse0 * 0.35,
        "fit did not converge: start {mse0:.6} end {mse_end:.6}"
    );
}

/// A scene's gradient-record count is a property of the scene and the camera -
/// how many gaussians each pixel's alpha-composite actually touches - and no
/// caller knows it before the forward pass has run. So the scratch cannot be
/// sized correctly up front, and sizing it by a per-pixel guess means a dense
/// enough scene aborts the whole fit. It must grow to whatever the pass needs,
/// and the gradients it then produces must be the same ones an amply-sized
/// scratch would have produced.
#[test]
fn a_scratch_too_small_for_the_scene_grows_instead_of_aborting() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let s = scene(24, 0x9a11);
    let c = cam();
    let o = RenderOpts::default();
    let px = (c.width * c.height) as usize;

    let mut r = Lcg(0x1234);
    let wimg: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { r.next() - 0.5 }).collect();

    let grads_for = |rec_cap: usize| -> (usize, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut ren = Renderer::new(&g, ks, s.len(), c.width, c.height, 0);
        let gs = GpuSplats::upload(&g, &s);
        ren.render(&g, &gs, &c, &o);
        let grads = SplatGrads::new(&g, s.len());
        g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
        let dimg: DeviceBuffer = g.storage_init("dimg", &wimg);
        let mut bscr = BwdScratch::new(&g, s.len(), px, rec_cap);
        let n = ren.render_bwd(&g, &gs, &c, &o, &dimg, &mut bscr, &grads).expect("fits");
        (
            n,
            g.read(&grads.d_gauss, 10 * s.len()),
            g.read(&grads.d_opac, s.len()),
            g.read(&grads.d_colors, 3 * s.len()),
        )
    };

    // One record per pixel is far below what 24 overlapping gaussians emit.
    let (n_small, gauss_small, opac_small, col_small) = grads_for(px);
    let (n_ample, gauss_ample, opac_ample, col_ample) = grads_for(64 * px);
    assert!(n_ample > px, "test is vacuous: the scene fits in the undersized scratch ({n_ample} records)");
    assert_eq!(n_small, n_ample, "record count changed with the scratch size");
    assert_eq!(gauss_small, gauss_ample, "gaussian grads differ after a grow");
    assert_eq!(opac_small, opac_ample, "opacity grads differ after a grow");
    assert_eq!(col_small, col_ample, "color grads differ after a grow");
}


/// Growing the record scratch to whatever the scene needs runs into a second
/// wall: one storage binding cannot exceed the device's
/// `max_storage_buffer_binding_size`, and a dense enough scene needs more
/// record bytes than that. The scratch must refuse it by NAME - what ran out,
/// what the device allows, and which knob shrinks the scene - rather than
/// growing past the limit and letting the backend reject the bind group with
/// a validation error that mentions neither gaussians nor records.
#[test]
fn a_scene_too_dense_for_one_binding_says_so_in_its_own_terms() {
    // what wgpu actually reports on this class of card: i32::MAX rounded down
    let limit = 2_147_483_644u64;
    let fits = splat::renderer::max_records_for_binding(limit);
    assert_eq!(fits, limit as usize / (splat::renderer::RECORD_WORDS * 4));

    let err = splat::renderer::record_capacity(fits + 1, limit)
        .expect_err("a count past the binding limit must not be silently allocated");
    for want in ["gradient record", "--prune", "2047 MiB"] {
        assert!(err.contains(want), "message {err:?} does not mention {want:?}");
    }
    // One below the limit is allocatable, and headroom never pushes it over.
    let cap = splat::renderer::record_capacity(fits - 1, limit).expect("fits");
    assert!(cap <= fits, "headroom grew the capacity past the binding limit ({cap} > {fits})");
    assert!(cap >= fits - 1, "capacity {cap} is below what the pass needs");
}



/// Cropping the camera is NOT a way around the record ceiling, and this pins
/// down why so the next person does not lose a day to it.
///
/// It looks exact: gradients are a sum over pixels, so rendering the frame in
/// horizontal strips and letting them accumulate should give what the whole
/// frame gives. It does not, because `splat_project.wgsl` clamps each
/// gaussian's 3D covariance against the FRUSTUM before projecting it
/// (`lim_y_pos = (height - cy)/fy + 0.3*tan_fovy`, the standard EWA
/// approximation). A cropped camera is a different frustum, so the projected
/// covariance - and every gradient through it - is a different function.
///
/// Measured here rather than asserted, so that if the projection ever stops
/// depending on the frustum this test fails and says banding became available.
#[test]
fn cropping_the_camera_does_not_decompose_the_backward() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let s = scene(16, 0x8a5d);
    let c = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 32, 32);
    let o = RenderOpts::default();
    let px = (c.width * c.height) as usize;
    let mut r = Lcg(0x77);
    let wimg: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { r.next() - 0.5 }).collect();

    let run = |bands: u32| -> Vec<f32> {
        let mut ren = Renderer::new(&g, ks, s.len(), c.width, c.height, 0);
        let gs = GpuSplats::upload(&g, &s);
        let grads = SplatGrads::new(&g, s.len());
        g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
        let mut bscr = BwdScratch::new(&g, s.len(), px, 0);
        let rows = c.height.div_ceil(bands);
        let mut y0 = 0;
        while y0 < c.height {
            let h = rows.min(c.height - y0);
            let band = Camera { cy: c.cy - y0 as f32, height: h, ..c };
            ren.render(&g, &gs, &band, &o);
            let n = (band.width * h * 4) as usize;
            let off = (y0 * c.width * 4) as usize;
            let dimg: DeviceBuffer = g.storage_init("dimg", &wimg[off..off + n]);
            ren.render_bwd(&g, &gs, &band, &o, &dimg, &mut bscr, &grads).expect("fits");
            y0 += h;
        }
        g.read(&grads.d_gauss, 10 * s.len())
    };

    let whole = run(1);
    let split = run(2);
    let scale = whole.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = whole.iter().zip(&split).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / scale;
    assert!(
        worst > 1e-3,
        "two cropped bands now reproduce the whole-frame gradient to {worst:.3e} relative. If the \
         projection no longer depends on the frustum, banding the backward by camera crop became \
         exact - which is the cheap way past the per-binding record ceiling, and worth taking."
    );
}
