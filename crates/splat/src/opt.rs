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
    /// Largest ratio a gaussian's longest axis may have to its shortest,
    /// 0 = unconstrained.
    ///
    /// Bounding each axis says how BIG a gaussian may be and nothing about its
    /// SHAPE, and the optimizer exploits the difference: a needle lined up with
    /// a training view's ray lowers that view's loss while being invisible in
    /// it, and is a streak across every other view. That is not a small
    /// effect - a real 23-view scene went from 0.1% of gaussians above 10:1
    /// as the model emitted them to 40.7% after fitting, reaching 3000:1.
    pub max_aspect: f32,
    pub log_every: usize,
    /// Run density control every N iterations, 0 = never (a fixed set of
    /// gaussians, which is all this optimizer could ever do before).
    ///
    /// OFF by default, which is the conservative choice rather than the
    /// obvious one. Density control is the fix for a scene too SPARSE to hold
    /// its detail, and a feed-forward reconstruction is the opposite problem:
    /// it starts at one gaussian per source pixel per view, so a 48-frame
    /// video arrives at ~9.7M gaussians before anything is added. Growing that
    /// pushes the backward past the per-binding gradient-record ceiling, which
    /// fails the run outright. Turn it on for sparse scenes; reach for
    /// `splat::prune` on dense ones.
    pub densify_every: usize,
    /// Skip density control until this iteration, so the gradients it reads
    /// describe the scene rather than the first few steps of chaos.
    pub densify_after: usize,
    /// Fraction of gaussians, by positional-gradient magnitude, considered
    /// under-reconstructed at each density-control step.
    pub densify_frac: f32,
    /// Drop gaussians below this opacity at each density-control step.
    pub prune_opacity: f32,
    /// Refuse to grow past this many gaussians, 0 = the device decides. A
    /// runaway subdivision is a much worse failure than a soft scene.
    pub max_gaussians: usize,
    /// The anti-alias dilation the fit optimizes UNDER. It is part of the
    /// forward model being inverted, so the optimizer folds compensation for
    /// it into the gaussians: a scene fitted at one value and rendered at
    /// another comes out wrong in the direction of the difference. Render with
    /// the same value.
    pub eps2d: f32,
}

impl Default for FitCfg {
    fn default() -> Self {
        FitCfg {
            iters: 200,
            lr: 5e-3,
            min_scale: 1e-4,
            max_aspect: 10.0,
            log_every: 20,
            densify_every: 0,
            densify_after: 30,
            densify_frac: 0.05,
            prune_opacity: 0.02,
            max_gaussians: 0,
            eps2d: RenderOpts::default().eps2d,
        }
    }
}

/// One posed target view: camera + RGB f32 `[W*H*3]` in [0,1].
pub struct TargetView {
    pub cam: Camera,
    pub rgb: Vec<f32>,
}

/// Fit `init` against the targets; returns the optimized scene and the final
/// mean MSE across views.
///
/// `on_step(iter, mse)` is polled once per completed iteration, inside this
/// loop (not bolted on from outside) - returning `false` aborts early, at the
/// end of whichever iteration just ran. `crates/cli/src/splat_cli.rs::fit_cmd`
/// passes a closure that only prints and always returns `true`;
/// `splat::caps::fit` polls the invocation's cancel token from its own.
pub fn fit(gpu: &Gpu, ks: Kernels, init: &Splats, targets: &[TargetView], cfg: &FitCfg, on_step: &mut dyn FnMut(usize, f32) -> bool) -> (Splats, f32) {
    assert!(!targets.is_empty());
    // With density control off there is exactly ONE stage, and this is the
    // function it always was - same buffers, same Adam state, start to finish.
    let stage_len = if cfg.densify_every == 0 { cfg.iters } else { cfg.densify_every };
    let mut scene = init.clone();
    let mut loss = 0.0f32;
    let mut done = 0usize;
    let mut stop = false;
    while done < cfg.iters && !stop {
        let iters = stage_len.min(cfg.iters - done);
        let (next, l, grad, aborted) = fit_stage(gpu, ks, &scene, targets, cfg, iters, done, on_step);
        scene = next;
        loss = l;
        done += iters;
        stop = aborted;
        if cfg.densify_every > 0 && done >= cfg.densify_after && done < cfg.iters && !stop {
            let before = scene.len();
            densify(&mut scene, &grad, cfg);
            if cfg.log_every > 0 && scene.len() != before {
                println!("fit iter {done:4}: density control {before} -> {} gaussians", scene.len());
            }
        }
    }
    (scene, loss)
}

/// Grow the scene where the loss is still pulling hardest, and drop what has
/// gone transparent.
///
/// `grad` is the accumulated magnitude of each gaussian's positional gradient.
/// Upstream 3DGS thresholds that at an absolute 2e-4; a FRACTION is used here
/// instead because this loss is normalized per pixel and per view, so the
/// absolute scale of a gradient depends on image size and view count and no
/// constant transfers between scenes. A fraction also bounds growth by
/// construction, which an absolute threshold does not.
///
/// Large gaussians SPLIT (two children at 1/1.6 the scale, offset along the
/// parent's dominant axis) and small ones CLONE, following the reference: a
/// big gaussian covering detail it cannot represent needs subdividing, while a
/// small one in an under-populated region needs a neighbour.
fn densify(scene: &mut Splats, grad: &[f32], cfg: &FitCfg) {
    let n = scene.len();
    if n == 0 || grad.len() != n {
        return;
    }
    let cap = if cfg.max_gaussians > 0 { cfg.max_gaussians } else { usize::MAX };
    if n >= cap {
        return;
    }

    // the gradient threshold, as a fraction of the population
    let want = ((n as f32 * cfg.densify_frac) as usize).min(cap - n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| grad[b].total_cmp(&grad[a]));
    let chosen: std::collections::HashSet<usize> = order.into_iter().take(want).collect();

    // "large" relative to THIS scene, so the rule does not depend on units
    let mut sizes: Vec<f32> = (0..n).map(|i| scene.scales[i * 3..i * 3 + 3].iter().fold(0.0f32, |m, &v| m.max(v))).collect();
    sizes.sort_by(f32::total_cmp);
    let big = sizes[(n as f32 * 0.8) as usize % n];

    let mut out = Splats::default();
    let push = |o: &mut Splats, i: usize, dm: [f32; 3], shrink: f32| {
        for (k, d) in dm.iter().enumerate() {
            o.means.push(scene.means[i * 3 + k] + d);
        }
        o.quats.extend_from_slice(&scene.quats[i * 4..i * 4 + 4]);
        for k in 0..3 {
            o.scales.push((scene.scales[i * 3 + k] * shrink).max(cfg.min_scale));
        }
        o.opacities.push(scene.opacities[i]);
        o.colors.extend_from_slice(&scene.colors[i * 3..i * 3 + 3]);
    };
    for i in 0..n {
        if scene.opacities[i] < cfg.prune_opacity {
            continue; // pruned: too transparent to be carrying anything
        }
        let s = &scene.scales[i * 3..i * 3 + 3];
        let axis = (0..3).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
        if chosen.contains(&i) && s[axis] >= big {
            // split: two smaller children straddling the parent's long axis,
            // rotated into world space by the parent's own orientation
            let q = &scene.quats[i * 4..i * 4 + 4];
            let nq = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-8);
            let (w, x, y, z) = (q[0] / nq, q[1] / nq, q[2] / nq, q[3] / nq);
            let col = match axis {
                0 => [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
                1 => [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
                _ => [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
            };
            let d = s[axis] * 0.5;
            push(&mut out, i, [col[0] * d, col[1] * d, col[2] * d], 1.0 / 1.6);
            push(&mut out, i, [-col[0] * d, -col[1] * d, -col[2] * d], 1.0 / 1.6);
        } else if chosen.contains(&i) {
            // clone: a second gaussian for the optimizer to walk off the first
            push(&mut out, i, [0.0; 3], 1.0);
            push(&mut out, i, [0.0; 3], 1.0);
        } else {
            push(&mut out, i, [0.0; 3], 1.0);
        }
    }
    if !out.is_empty() {
        *scene = out;
    }
}

/// One run of the optimizer over a FIXED set of gaussians. Returns the scene,
/// the last loss, each gaussian's accumulated positional-gradient magnitude,
/// and whether `on_step` asked to stop.
#[allow(clippy::too_many_arguments)]
fn fit_stage(
    gpu: &Gpu,
    ks: Kernels,
    init: &Splats,
    targets: &[TargetView],
    cfg: &FitCfg,
    iters: usize,
    it0: usize,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> (Splats, f32, Vec<f32>, bool) {
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
    let mut bscr = BwdScratch::new(gpu, n, max_px, 0);
    let opts = RenderOpts { mode: Mode::Color, eps2d: cfg.eps2d, ..Default::default() };

    // `adamw.wgsl` (M6.4) binds param/grad/m/v PLUS a per-tensor `numel`
    // descriptor and a device-resident grad-scale coefficient, and reads its
    // hyperparameters from an 8-field uniform (lr, beta1, beta2, eps, wd,
    // bc1, bc2, scale); see `crates/optim::Optim::build`/`::step`, the
    // canonical caller this mirrors. `desc`/`coef` are write-once (numel is
    // fixed for the run; no grad clipping here so coef is always 1.0);
    // `hparams` is rewritten once per iteration (only bc1/bc2 move).
    let unit_coef = gpu.storage(1);
    gpu.write(&unit_coef, &[f(1.0)]);
    let mk_desc = |numel: usize| {
        // Shape AND contents from `adamw.wgsl`'s own declaration of them - a
        // descriptor one word short of what the kernel reads does not fail,
        // it zeroes the update. No LoRA+ groups here, so every tensor is 1.0.
        let words = kernels::adamw_desc(numel, 1.0);
        let d = gpu.storage(words.len() as u64);
        gpu.write(&d, &words);
        d
    };
    let desc_geo = mk_desc(10 * n);
    let desc_op = mk_desc(n);
    let desc_col = mk_desc(3 * n);
    let hparams = gpu.uniform_dynamic(8);

    let adamw_step = |bufs: [&DeviceBuffer; 4], desc: &DeviceBuffer, numel: usize| {
        let s = gpu.step_buf(
            ks.adamw,
            &hparams,
            &[bufs[0], bufs[1], bufs[2], bufs[3], desc, &unit_coef],
            numel as u32,
        );
        gpu.submit(&[], &[s]);
    };

    let mut last_loss = 0.0f32;
    let mut aborted = false;
    // Positional-gradient magnitude per gaussian, averaged over the tail of
    // the stage. Read back over a WINDOW rather than every iteration: one
    // iteration is noisy and every iteration would be 40 bytes per gaussian
    // per step off the device for a signal that is used once.
    let window = iters.clamp(1, 4);
    let mut gsum = vec![0.0f32; n];
    for it in 0..iters {
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
            renderer
                .render_bwd(gpu, &gs, &t.cam, &opts, &dimg, &mut bscr, &grads)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        // Adam's bias correction counts from the start of THIS stage, because
        // its moments do too: m and v are fresh buffers per stage, and pairing
        // zeroed moments with a bias correction for a much later timestep is
        // not a well-formed Adam step. Staging still costs something - measured
        // at ~3.5% worse final loss than a single unbroken run, from losing the
        // momentum - which density control has to earn back before it pays.
        let ts = it as i32 + 1;
        let bc1 = 1.0 - 0.9f32.powi(ts);
        let bc2 = 1.0 - 0.999f32.powi(ts);
        gpu.write(&hparams, &[f(cfg.lr), f(0.9), f(0.999), f(1e-8), f(0.0), f(bc1), f(bc2), f(1.0)]);
        adamw_step([&p_geo, &grads.d_gauss, &m_geo, &v_geo], &desc_geo, 10 * n);
        adamw_step([&p_op, &grads.d_opac, &m_op, &v_op], &desc_op, n);
        adamw_step([&p_col, &grads.d_colors, &m_col, &v_col], &desc_col, 3 * n);
        // projected-gradient clamps (host; N is fit-sized)
        let mut geo = gpu.read(&p_geo, 10 * n);
        for i in 0..n {
            for k in 3..6 {
                geo[i * 10 + k] = geo[i * 10 + k].clamp(cfg.min_scale, 0.3);
            }
            if cfg.max_aspect > 1.0 {
                // Raise the short axes to the longest one's fair share rather
                // than shrinking the long axis: shrinking would fight the
                // gradient that grew it, while a floor simply refuses the
                // degenerate shape.
                let lo = geo[i * 10 + 3].max(geo[i * 10 + 4]).max(geo[i * 10 + 5]) / cfg.max_aspect;
                for k in 3..6 {
                    geo[i * 10 + k] = geo[i * 10 + k].max(lo);
                }
            }
        }
        gpu.write(&p_geo, cast(&geo));
        let mut op = gpu.read(&p_op, n);
        for v in op.iter_mut() {
            *v = v.clamp(1e-4, 1.0 - 1e-4);
        }
        gpu.write(&p_op, cast(&op));

        if cfg.densify_every > 0 && it + window >= iters {
            let dg = gpu.read(&grads.d_gauss, 10 * n);
            for i in 0..n {
                let (a, b, c) = (dg[i * 10], dg[i * 10 + 1], dg[i * 10 + 2]);
                gsum[i] += (a * a + b * b + c * c).sqrt();
            }
        }

        last_loss = (loss_sum / targets.len() as f64) as f32;
        let global = it0 + it;
        if cfg.log_every > 0 && (global.is_multiple_of(cfg.log_every) || global + 1 == cfg.iters) {
            println!("fit iter {global:4}: mse {last_loss:.6}");
        }
        if !on_step(global, last_loss) {
            aborted = true;
            break;
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
    (out, last_loss, gsum, aborted)
}

fn cast(v: &[f32]) -> &[u32] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u32, v.len()) }
}
