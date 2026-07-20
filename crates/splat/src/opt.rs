// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3DGS scene optimization ("fit"): AdamW on gaussian parameters against
//! posed target images, driven by the atomic-free rasterizer backward. The
//! demonstrable use of `render_bwd` — and the training path for scenes.
//!
//! Parameterization: packed geometry `[N*10] = {means, scales(linear),
//! quats(raw)}` (matches `d_gauss`, so one AdamW dispatch covers it), plus
//! separate opacity/color buffers. Linear scales/opacities are clamped
//! host-side after each step (projected gradient) — simpler than log/logit
//! reparameterization and adequate for fitting; antialiased mode is not
//! supported by the backward (compensation chain unmodeled).

use gpu_core::{f, DeviceBuffer, Gpu};

use crate::renderer::{BwdScratch, GpuSplats, Renderer, SplatGrads};
use crate::types::{Camera, Mode, RenderOpts, Splats};
use crate::Kernels;

pub struct FitCfg {
    pub iters: usize,
    pub lr: f32,
    /// Clamp every step: scales into [min_scale, 0.3], opacity into [ε, 1-ε].
    pub min_scale: f32,
    pub log_every: usize,
}

impl Default for FitCfg {
    fn default() -> Self {
        FitCfg { iters: 200, lr: 5e-3, min_scale: 1e-4, log_every: 20 }
    }
}

/// One posed target view: camera + RGB f32 `[W*H*3]` in [0,1].
pub struct TargetView {
    pub cam: Camera,
    pub rgb: Vec<f32>,
}

/// Fit `init` against the targets; returns the optimized scene and the final
/// mean MSE across views.
pub fn fit(gpu: &Gpu, ks: Kernels, init: &Splats, targets: &[TargetView], cfg: &FitCfg) -> (Splats, f32) {
    assert!(!targets.is_empty());
    let n = init.len();
    let (maxw, maxh) = targets
        .iter()
        .fold((0u32, 0u32), |(mw, mh), t| (mw.max(t.cam.width), mh.max(t.cam.height)));
    let max_px = (maxw * maxh) as usize;

    // ---- parameter buffers ----
    let mut packed = Vec::with_capacity(n * 10);
    for i in 0..n {
        packed.extend_from_slice(&init.means[i * 3..i * 3 + 3]);
        packed.extend_from_slice(&init.scales[i * 3..i * 3 + 3]);
        packed.extend_from_slice(&init.quats[i * 4..i * 4 + 4]);
    }
    let p_geo = gpu.storage_init("fit.geo", &packed);
    let p_op = gpu.storage_init("fit.op", &init.opacities);
    let p_col = gpu.storage_init("fit.col", &init.colors);
    let means = gpu.storage(3 * n as u64);
    let scales = gpu.storage(3 * n as u64);
    let quats = gpu.storage(4 * n as u64);
    let adam = |numel: usize| (gpu.storage(numel as u64), gpu.storage(numel as u64));
    let (m_geo, v_geo) = adam(10 * n);
    let (m_op, v_op) = adam(n);
    let (m_col, v_col) = adam(3 * n);
    let grads = SplatGrads::new(gpu, n);
    let dimg = gpu.storage(4 * max_px as u64);
    let mut renderer = Renderer::new(gpu, ks, n, maxw, maxh, 0);
    let bscr = BwdScratch::new(gpu, n, max_px, 0);
    let opts = RenderOpts { mode: Mode::Color, ..Default::default() };

    let adamw_step = |bufs: [&DeviceBuffer; 4], numel: usize, t: i32| {
        let bc1 = 1.0 - 0.9f32.powi(t);
        let bc2 = 1.0 - 0.999f32.powi(t);
        let s = gpu.step(
            ks.adamw,
            &[bufs[0], bufs[1], bufs[2], bufs[3]],
            &[numel as u32, 0, f(cfg.lr), f(0.9), f(0.999), f(1e-8), f(0.0), f(bc1), f(bc2)],
            numel as u32,
        );
        gpu.submit(&[], &[s]);
    };

    let mut last_loss = 0.0f32;
    for it in 0..cfg.iters {
        // zero grads
        gpu.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors], &[]);
        let mut loss_sum = 0.0f64;
        for t in targets {
            let px = (t.cam.width * t.cam.height) as usize;
            // unpack params for the forward
            let unpack = gpu.step(
                ks.splat_unpack,
                &[&p_geo, &means, &scales, &quats],
                &[n as u32],
                n as u32,
            );
            gpu.submit(&[], &[unpack]);
            let gs = GpuSplats {
                n,
                means: means.clone(),
                quats: quats.clone(),
                scales: scales.clone(),
                opacities: p_op.clone(),
                colors: p_col.clone(),
            };
            renderer.render(gpu, &gs, &t.cam, &opts);
            // host loss: MSE over rgb; alpha unsupervised
            let img = renderer.read_rgba(gpu, t.cam.width, t.cam.height);
            let mut d = vec![0.0f32; px * 4];
            let scale = 2.0 / (px as f32 * 3.0);
            let mut lsum = 0.0f64;
            for i in 0..px {
                for c in 0..3 {
                    let diff = img[i * 4 + c] - t.rgb[i * 3 + c];
                    lsum += (diff * diff) as f64;
                    d[i * 4 + c] = scale * diff;
                }
            }
            loss_sum += lsum / (px as f64 * 3.0);
            gpu.write(&dimg, cast(&d));
            renderer.render_bwd(gpu, &gs, &t.cam, &opts, &dimg, &bscr, &grads);
        }
        let ts = it as i32 + 1;
        adamw_step([&p_geo, &grads.d_gauss, &m_geo, &v_geo], 10 * n, ts);
        adamw_step([&p_op, &grads.d_opac, &m_op, &v_op], n, ts);
        adamw_step([&p_col, &grads.d_colors, &m_col, &v_col], 3 * n, ts);
        // projected-gradient clamps (host; N is fit-sized)
        let mut geo = gpu.read(&p_geo, 10 * n);
        for i in 0..n {
            for k in 3..6 {
                geo[i * 10 + k] = geo[i * 10 + k].clamp(cfg.min_scale, 0.3);
            }
        }
        gpu.write(&p_geo, cast(&geo));
        let mut op = gpu.read(&p_op, n);
        for v in op.iter_mut() {
            *v = v.clamp(1e-4, 1.0 - 1e-4);
        }
        gpu.write(&p_op, cast(&op));

        last_loss = (loss_sum / targets.len() as f64) as f32;
        if cfg.log_every > 0 && (it % cfg.log_every == 0 || it + 1 == cfg.iters) {
            println!("fit iter {it:4}: mse {last_loss:.6}");
        }
    }

    // read back the optimized scene
    let geo = gpu.read(&p_geo, 10 * n);
    let op = gpu.read(&p_op, n);
    let col = gpu.read(&p_col, 3 * n);
    let mut out = Splats::default();
    for i in 0..n {
        out.means.extend_from_slice(&geo[i * 10..i * 10 + 3]);
        out.scales.extend_from_slice(&geo[i * 10 + 3..i * 10 + 6]);
        out.quats.extend_from_slice(&geo[i * 10 + 6..i * 10 + 10]);
        out.opacities.push(op[i]);
        out.colors.extend_from_slice(&col[i * 3..i * 3 + 3]);
    }
    (out, last_loss)
}

fn cast(v: &[f32]) -> &[u32] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u32, v.len()) }
}
