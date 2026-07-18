// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end ZipDepth training: forward -> masked L1 -> backward -> AdamW.
//!
//! Deliberately placeholder-grade on DATA and LOSS, full-fidelity on the LOOP:
//! the synthetic generator and plain L1 are stand-ins (real pairs and the
//! SSI + gradient loss slot into the same two dispatch sites), but the loop
//! itself is the real thing — the same forward/backward the master gradcheck
//! proves, `masked_l1`/`masked_l1_grad` for the loss (host-summed numerator,
//! exactly like every other global reduction in brain), and the shared
//! on-device AdamW (`optim::Optim`, single submit per step, no host readback
//! of gradients). `tests/p4_train.rs` pins that overfitting one batch halves
//! the loss — the loop provably LEARNS, which is the property a placeholder
//! must not fake.
//!
//! The synthetic scenes carry a real depth cue: nearer rectangles are painted
//! BRIGHTER (`albedo * (0.35 + 0.65*inv_depth)`), so intensity predicts
//! inverse depth and a conv net can genuinely generalize from it — the
//! overfit test is then a learning test, not noise memorisation.

use std::collections::HashMap;

use gpu_core::{f, Gpu};
use optim::Optim;
use paramstore::ParamStore;
use vision::Ctx;

use crate::config::ZipConfig;
use crate::model::ZipDepth;

/// Training knobs. `h`/`w` are the model input size (multiples of 32);
/// `fixed_batch` repeats one batch every step (the overfit sanity mode),
/// otherwise each step draws fresh scenes (seed-rotated).
#[derive(Clone, Copy, Debug)]
pub struct TrainCfg {
    pub steps: u32,
    pub batch: u32,
    pub h: u32,
    pub w: u32,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    pub fixed_batch: bool,
}

/// What a run reports: the loss at the first and last step.
#[derive(Clone, Copy, Debug)]
pub struct Trained {
    pub first_loss: f32,
    pub last_loss: f32,
}

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 40) as f32 / (1u64 << 24) as f32
}

/// One synthetic RGB -> inverse-depth pair: 6 axis-aligned rectangles over a
/// far background, painted far-to-near (nearer occludes), each shaded by its
/// own inverse depth so brightness correlates with nearness. Returns
/// (CHW `[3*h*w]` RGB in [0,1], `[h*w]` inverse depth).
pub fn synth_pair(seed: u64, h: u32, w: u32) -> (Vec<f32>, Vec<f32>) {
    let (h, w) = (h as usize, w as usize);
    let hw = h * w;
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    // Background: far, dim.
    let bg_inv = 0.12 + 0.05 * lcg(&mut s);
    let bg_alb = [0.15 + 0.1 * lcg(&mut s), 0.15 + 0.1 * lcg(&mut s), 0.15 + 0.1 * lcg(&mut s)];
    let shade = |alb: f32, inv: f32| alb * (0.35 + 0.65 * inv);
    let mut rgb = vec![0.0f32; 3 * hw];
    let mut inv = vec![bg_inv; hw];
    for c in 0..3 {
        rgb[c * hw..(c + 1) * hw].fill(shade(bg_alb[c], bg_inv));
    }
    // 6 rectangles, painted in increasing inverse depth (near last => occludes).
    let mut rects: Vec<(usize, usize, usize, usize, f32, [f32; 3])> = (0..6)
        .map(|_| {
            let x0 = (lcg(&mut s) * 0.8 * w as f32) as usize;
            let y0 = (lcg(&mut s) * 0.8 * h as f32) as usize;
            let rw = 2 + (lcg(&mut s) * 0.5 * w as f32) as usize;
            let rh = 2 + (lcg(&mut s) * 0.5 * h as f32) as usize;
            let iv = 0.25 + 0.75 * lcg(&mut s);
            let alb = [0.25 + 0.75 * lcg(&mut s), 0.25 + 0.75 * lcg(&mut s), 0.25 + 0.75 * lcg(&mut s)];
            (x0, y0, rw, rh, iv, alb)
        })
        .collect();
    rects.sort_by(|a, b| a.4.total_cmp(&b.4));
    for (x0, y0, rw, rh, iv, alb) in rects {
        for y in y0..(y0 + rh).min(h) {
            for x in x0..(x0 + rw).min(w) {
                inv[y * w + x] = iv;
                for c in 0..3 {
                    rgb[c * hw + y * w + x] = shade(alb[c], iv);
                }
            }
        }
    }
    (rgb, inv)
}

/// A whole batch, concatenated along N. Scene seeds derive from
/// `(base_seed, step, image-in-batch)`; `fixed_batch` pins step to 0.
fn synth_batch(t: &TrainCfg, step: u32) -> (Vec<f32>, Vec<f32>) {
    let eff = if t.fixed_batch { 0 } else { step as u64 };
    let mut xs = Vec::with_capacity((t.batch * 3 * t.h * t.w) as usize);
    let mut ys = Vec::with_capacity((t.batch * t.h * t.w) as usize);
    for b in 0..t.batch as u64 {
        let (x, y) = synth_pair(t.seed ^ (eff * 131 + b * 7919), t.h, t.w);
        xs.extend_from_slice(&x);
        ys.extend_from_slice(&y);
    }
    (xs, ys)
}

/// Run the loop: build the model at `(t.h, t.w)`, seed the params from `init`
/// (fresh `init_weights` or a loaded checkpoint for fine-tune), and step.
/// `on_step(step, loss)` fires every step (progress printing, test capture).
/// Returns the trained `ParamStore` (the caller saves it) and the loss report.
pub fn train_loop(
    gpu: &Gpu,
    cfg: ZipConfig,
    t: &TrainCfg,
    init: &HashMap<String, Vec<f32>>,
    mut on_step: impl FnMut(u32, f32),
) -> (ParamStore, Trained) {
    let ctx = Ctx::new(gpu, crate::net::ids());
    let m = ZipDepth::build_hw(&ctx, cfg, t.batch, t.h, t.w, true);
    // Track BN running stats so eval-mode inference on the saved weights works.
    m.set_update_running(true);
    let ps = ParamStore::new(gpu, m.param_list(), init);
    let (adamw, gn, gs, cc, gsb) = crate::net::optim_ids();
    let opt = Optim::new(adamw, gn, gs, cc, gsb);
    let (k_l1, k_l1g) = crate::net::loss_ids();

    let total = m.out_shape.numel();
    let xb = gpu.storage(m.in_shape.numel() as u64);
    let tgt = gpu.storage(total as u64);
    let mask = gpu.storage_init("mask", &vec![1.0f32; total as usize]);
    let l1 = gpu.storage(total as u64);
    let d_out = gpu.storage(total as u64);

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..t.steps {
        let (x, y) = synth_batch(t, step);
        gpu.write(&xb, bytemuck::cast_slice(&x));
        gpu.write(&tgt, bytemuck::cast_slice(&y));
        ps.zero_grads(gpu);
        m.forward(&ctx, &ps, &xb);

        // loss = mean(|pred - tgt| * mask): device per-element terms, host sum.
        let s = gpu.step(k_l1, &[m.out(), &tgt, &mask, &l1], &[total], total);
        gpu.submit(&[], &[s]);
        let loss = gpu.read(&l1, total as usize).iter().sum::<f32>() / total as f32;

        // dL/dpred = sign(pred - tgt) * mask / total, then the model backward.
        let s = gpu.step(k_l1g, &[m.out(), &tgt, &mask, &d_out], &[total, f(1.0 / total as f32)], total);
        gpu.submit(&[], &[s]);
        m.backward(&ctx, &ps, &xb, &d_out);
        opt.step(gpu, &ps, step + 1, t.lr, t.wd, 0.9, 0.999, 1e-8, Some(1.0), 1.0);
        gpu.poll_wait();

        if step == 0 {
            first = loss;
        }
        last = loss;
        on_step(step, loss);
    }
    (ps, Trained { first_loss: first, last_loss: last })
}
