// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T2 gate: rasterizer backward.
//!   1. gradcheck of render_bwd against a committed torch-autograd golden
//!      (tools/goldens/splat_dump_gradcheck.py; float64 oracle of the identical
//!      scene/loss). Finite differences are NOT used: the 1/255 truncation
//!      boundaries make them biased for scale/mean grads (the classic 3DGS
//!      finite-diff pitfall — verified: autograd matches our analytic grads
//!      where central differences are 2x off);
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
    let bscr = BwdScratch::new(&g, base.len(), px, 0);
    let nrecs = ren.render_bwd(&g, &gs, &c, &o, &dimg, &bscr, &grads);
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
    let (_fitted, mse_end) = fit(&g, ks, &init, &targets, &cfg);
    assert!(
        (mse_end as f64) < mse0 * 0.35,
        "fit did not converge: start {mse0:.6} end {mse_end:.6}"
    );
}
