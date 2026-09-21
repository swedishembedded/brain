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
    // The golden was produced by an autograd reference running Inria 3DGS:
    // plain dilation, no opacity compensation. Pin those, so this test keeps
    // checking the thing it has a reference for.
    let o = RenderOpts { antialiased: false, eps2d: 0.3, ..Default::default() };
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
/// record bytes than that. Row banding absorbs almost all of it, so what is
/// left is the case banding cannot help - one row of pixels past the ceiling -
/// and the scratch must refuse THAT by NAME: what ran out, what the device
/// allows, and which knob shrinks the scene, rather than growing past the limit
/// and letting the backend reject the bind group with a validation error that
/// mentions neither gaussians nor records.
#[test]
fn a_scene_too_dense_for_one_binding_says_so_in_its_own_terms() {
    // what wgpu actually reports on this class of card: i32::MAX rounded down
    let limit = 2_147_483_644u64;
    let fits = splat::renderer::max_records_for_binding(limit);
    assert_eq!(fits, limit as usize / (splat::renderer::RECORD_WORDS * 4));

    let err = splat::renderer::record_capacity(fits + 1, limit)
        .expect_err("a count past the binding limit must not be silently allocated");
    for want in ["gradient record", "one row of pixels", "--prune", "2047 MiB"] {
        assert!(err.contains(want), "message {err:?} does not mention {want:?}");
    }
    // One below the limit is allocatable, and headroom never pushes it over.
    let cap = splat::renderer::record_capacity(fits - 1, limit).expect("fits");
    assert!(cap <= fits, "headroom grew the capacity past the binding limit ({cap} > {fits})");
    assert!(cap >= fits - 1, "capacity {cap} is below what the pass needs");
}



/// A frame whose gradient records exceed one storage binding is still
/// differentiated, and gives the same gradients as one that fits.
///
/// `recs` is bound as a single storage buffer, so a dense enough scene cannot
/// be differentiated in one pass on the device at all, however much memory is
/// free - and a real capture reaches that ceiling long before it runs out of
/// VRAM. The loss is a sum over pixels, so the backward splits the ROWS until
/// each group fits and lets the groups accumulate.
///
/// It has to band the pixel walk and nothing else. Cropping the camera instead
/// would not work: `splat_project.wgsl` clamps each gaussian's covariance
/// against the frustum before projecting it (the standard EWA
/// `lim_y_pos = (height - cy)/fy + 0.3*tan_fovy`), so a shorter camera is a
/// different projection and a different function to differentiate.
#[test]
fn a_frame_past_the_record_ceiling_is_differentiated_in_bands_cpu() {
    band_equivalence(&Gpu::new_cpu(splat::PIPELINES));
}

/// The band origin rides in a uniform block the two backends lay out
/// independently, so the same check has to run on the device.
#[test]
fn a_frame_past_the_record_ceiling_is_differentiated_in_bands_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    band_equivalence(&gpu_core::testgpu::dev(splat::PIPELINES));
}

fn band_equivalence(g: &Gpu) {
    let ks = Kernels::at(0);
    let s = scene(16, 0x8a5d);
    let c = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 32, 32);
    let o = RenderOpts::default();
    let px = (c.width * c.height) as usize;
    let mut r = Lcg(0x77);
    let wimg: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { r.next() - 0.5 }).collect();

    // `limit` in bytes; zero means the device's own, which this scene is far
    // below. The banded run is squeezed to a few hundred records a pass.
    let run = |limit: u64| -> (Vec<f32>, usize) {
        let mut ren = Renderer::new(g, ks, s.len(), c.width, c.height, 0);
        let gs = GpuSplats::upload(g, &s);
        let grads = SplatGrads::new(g, s.len());
        g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
        let mut bscr = BwdScratch::new(g, s.len(), px, 0);
        if limit > 0 {
            bscr = bscr.with_binding_limit(limit);
        }
        let dimg: DeviceBuffer = g.storage_init("dimg", &wimg);
        ren.render(g, &gs, &c, &o);
        let n = ren.render_bwd(g, &gs, &c, &o, &dimg, &mut bscr, &grads).expect("bands must fit");
        (g.read(&grads.d_gauss, 10 * s.len()), n)
    };

    let (whole, n_whole) = run(0);
    let squeezed = 256 * (splat::renderer::RECORD_WORDS * 4) as u64;
    let (banded, n_banded) = run(squeezed);
    assert!(
        n_whole > 256,
        "test is vacuous: the frame emits {n_whole} records and never exceeds the forced ceiling"
    );
    assert_eq!(n_banded, n_whole, "banding changed how many records the frame emits");
    let scale = whole.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = whole.iter().zip(&banded).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / scale;
    assert!(
        worst < 1e-5,
        "banding the backward by pixel row changed the gradient by {worst:.3e} relative; it is a \
         sum over pixels and must not depend on how the rows are grouped"
    );
}

/// The 2D Mip filter's opacity compensation has to be differentiated too.
///
/// Dilating the screen-space covariance without touching opacity makes a
/// splat both blurrier and brighter, which is why Mip-Splatting scales opacity
/// by `sqrt(|S| / |S + eps I|)` to put the energy back. That factor depends on
/// the covariance, so it is part of the forward model and the backward has to
/// carry it - and it did not: the kernel said so in a comment and the advice
/// was to fit with compensation off. Fitting under a forward the backward does
/// not model optimises the wrong thing.
///
/// Finite differences, so this needs no external reference.
#[test]
fn the_mip_filters_opacity_compensation_is_differentiated() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let base = scene(5, 0x51ce);
    let c = cam();
    let o = RenderOpts { antialiased: true, eps2d: 0.1, ..Default::default() };
    let px = (c.width * c.height) as usize;
    let mut r = Lcg(0x77aa);
    let wimg: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { r.next() - 0.5 }).collect();

    let render_loss = |s: &Splats| -> f64 {
        let mut ren = Renderer::new(&g, ks, s.len(), c.width, c.height, 0);
        let gs = GpuSplats::upload(&g, s);
        ren.render(&g, &gs, &c, &o);
        loss(&ren.read_rgba(&g, c.width, c.height), &wimg)
    };

    let mut ren = Renderer::new(&g, ks, base.len(), c.width, c.height, 0);
    let gs = GpuSplats::upload(&g, &base);
    ren.render(&g, &gs, &c, &o);
    let grads = SplatGrads::new(&g, base.len());
    g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
    let dimg: DeviceBuffer = g.storage_init("dimg", &wimg);
    let mut bscr = BwdScratch::new(&g, base.len(), px, 0);
    ren.render_bwd(&g, &gs, &c, &o, &dimg, &mut bscr, &grads).expect("fits");
    let d_gauss = g.read(&grads.d_gauss, 10 * base.len());
    let d_opac = g.read(&grads.d_opac, base.len());

    // Scales and opacity are the two the compensation touches: opacity
    // directly, scales through the covariance the compensation is built from.
    // A splat's truncation radius is `ceil`ed to whole pixels, so a step large
    // enough to move it changes which tiles the splat touches and the loss
    // jumps. That is a real discontinuity in the forward, not a gradient
    // error, but it wrecks a finite difference: at h=1e-3 one gaussian here
    // reads -0.041 against a true -0.559, and at h=2e-4 it reads -0.55939.
    const H: f32 = 2e-4;
    let mut worst = (0.0f64, String::new());
    let mut rows: Vec<(f64, String)> = Vec::new();
    let mut checked = 0;
    for gi in 0..base.len() {
        for k in 0..3 {
            let mut up = base.clone();
            let mut dn = base.clone();
            up.scales[gi * 3 + k] += H;
            dn.scales[gi * 3 + k] -= H;
            let fd = (render_loss(&up) - render_loss(&dn)) / (2.0 * H) as f64;
            let an = d_gauss[gi * 10 + 3 + k] as f64;
            let rel = (an - fd).abs() / an.abs().max(fd.abs()).max(1e-2);
            if fd.abs() > 1e-2 {
                checked += 1;
                rows.push((rel, format!("scale[{gi}][{k}]: analytic {an:.5} vs finite-difference {fd:.5}")));
                if rel > worst.0 {
                    worst = (rel, format!("scale[{gi}][{k}]: analytic {an:.5} vs finite-difference {fd:.5}"));
                }
            }
        }
        let mut up = base.clone();
        let mut dn = base.clone();
        up.opacities[gi] += H;
        dn.opacities[gi] -= H;
        let fd = (render_loss(&up) - render_loss(&dn)) / (2.0 * H) as f64;
        let an = d_opac[gi] as f64;
        let rel = (an - fd).abs() / an.abs().max(fd.abs()).max(1e-2);
        if fd.abs() > 1e-2 {
            checked += 1;
            rows.push((rel, format!("opacity[{gi}]: analytic {an:.5} vs finite-difference {fd:.5}")));
            if rel > worst.0 {
                worst = (rel, format!("opacity[{gi}]: analytic {an:.5} vs finite-difference {fd:.5}"));
            }
        }
    }
    assert!(checked > 8, "too few non-trivial gradients exercised ({checked})");
    if worst.0 >= 0.05 {
        rows.sort_by(|a: &(f64, String), b| b.0.partial_cmp(&a.0).unwrap());
        for (r, m) in rows.iter().take(12) {
            println!("  {:6.1}%  {m}", 100.0 * r);
        }
    }
    assert!(worst.0 < 0.05, "worst disagreement {:.1}%: {}", 100.0 * worst.0, worst.1);
}
