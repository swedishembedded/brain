// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end ZipDepth training: forward -> scale-and-shift-invariant loss -> backward -> AdamW.
//!
//! The objective is the MiDaS one: an SSI MSE plus multi-scale gradient matching on inverse depth, per image, over a
//! validity mask ([`crate::loss`] derives it and its exact gradient). Its per-pixel arithmetic runs on the device;
//! the per-image reductions are summed on the host, and the gradient lands in the buffer the model's backward
//! reads. The weights are updated by the shared on-device AdamW (`optim::Optim`, one submit per step, no host
//! readback of gradients).
//!
//! Batches come from a [`BatchSource`]:
//!
//! * [`crate::dataset::DepthDataset`] reads a real paired RGB / depth directory, with its validity masks, in shuffled
//!   epochs (format and validation in that module);
//! * [`Synthetic`] draws seeded scenes: rectangles over a far background, nearer ones painted brighter
//!   (`albedo * (0.35 + 0.65 * inv_depth)`), so intensity predicts inverse depth and a conv net can generalise from
//!   it. It needs no files, so it serves the overfit test (`tests/it/p4_train.rs` pins that one fixed batch halves the
//!   loss), smoke runs and CI.
//!
//! `TrainCfg::fixed_batch` draws one batch and repeats it every step, whichever the source.

use std::collections::HashMap;

use data::rng::Lcg;
use gpu_core::Gpu;
use optim::Optim;
use paramstore::ParamStore;
use vision::Ctx;

use crate::config::ZipConfig;
use crate::loss::DepthLoss;
use crate::model::ZipDepth;

/// MiDaS's weight of the gradient-matching term.
pub const DEFAULT_GRAD_WEIGHT: f32 = 0.5;

/// Training knobs. `h`/`w` are the model input size (multiples of 32);
/// `fixed_batch` repeats one batch every step (the overfit sanity mode),
/// otherwise each step draws a fresh batch from the source.
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
    /// Weight `alpha` of the gradient-matching term in `L_ssi + alpha * L_grad` ([`DEFAULT_GRAD_WEIGHT`] is MiDaS's).
    pub grad_weight: f32,
    /// What the backward keeps of the forward; `None` keeps every unit's intermediates. A choice the net cannot
    /// run is refused by name before the first step ([`crate::ZipDepth::set_saved_policy`]).
    pub saved: Option<model::saved::SavedPolicy>,
}

/// What a run reports: the total loss at the first and last step.
#[derive(Clone, Copy, Debug)]
pub struct Trained {
    pub first_loss: f32,
    pub last_loss: f32,
}

/// One synthetic RGB -> inverse-depth pair: 6 axis-aligned rectangles over a
/// far background, painted far-to-near (nearer occludes), each shaded by its
/// own inverse depth so brightness correlates with nearness. Returns
/// (CHW `[3*h*w]` RGB in [0,1], `[h*w]` inverse depth).
pub fn synth_pair(seed: u64, h: u32, w: u32) -> (Vec<f32>, Vec<f32>) {
    let (h, w) = (h as usize, w as usize);
    let hw = h * w;
    let mut s = Lcg::new(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
    // Background: far, dim.
    let bg_inv = 0.12 + 0.05 * s.unit();
    let bg_alb = [0.15 + 0.1 * s.unit(), 0.15 + 0.1 * s.unit(), 0.15 + 0.1 * s.unit()];
    let shade = |alb: f32, inv: f32| alb * (0.35 + 0.65 * inv);
    let mut rgb = vec![0.0f32; 3 * hw];
    let mut inv = vec![bg_inv; hw];
    for c in 0..3 {
        rgb[c * hw..(c + 1) * hw].fill(shade(bg_alb[c], bg_inv));
    }
    // 6 rectangles, painted in increasing inverse depth (near last => occludes).
    let mut rects: Vec<(usize, usize, usize, usize, f32, [f32; 3])> = (0..6)
        .map(|_| {
            let x0 = (s.unit() * 0.8 * w as f32) as usize;
            let y0 = (s.unit() * 0.8 * h as f32) as usize;
            let rw = 2 + (s.unit() * 0.5 * w as f32) as usize;
            let rh = 2 + (s.unit() * 0.5 * h as f32) as usize;
            let iv = 0.25 + 0.75 * s.unit();
            let alb = [0.25 + 0.75 * s.unit(), 0.25 + 0.75 * s.unit(), 0.25 + 0.75 * s.unit()];
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

/// One batch, concatenated along N: CHW RGB `[N*3*h*w]`, inverse depth `[N*h*w]` and its 0/1 validity mask
/// `[N*h*w]`.
#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub rgb: Vec<f32>,
    pub inv: Vec<f32>,
    pub mask: Vec<f32>,
}

/// Where a training run's batches come from.
pub trait BatchSource {
    /// The next `batch` images at the model input size `h x w`. An error aborts the run (a dataset file that fails to
    /// decode names itself in it).
    fn next_batch(&mut self, batch: u32, h: u32, w: u32) -> Result<Batch, String>;
}

/// Seeded synthetic scenes ([`synth_pair`]); the source that needs no files. Scene seeds derive from
/// `(seed, draw number, image in batch)`, so a run is reproducible and every draw differs.
pub struct Synthetic {
    seed: u64,
    draws: u64,
}

impl Synthetic {
    pub fn new(seed: u64) -> Synthetic {
        Synthetic { seed, draws: 0 }
    }
}

impl BatchSource for Synthetic {
    fn next_batch(&mut self, batch: u32, h: u32, w: u32) -> Result<Batch, String> {
        let mut out = Batch::default();
        for b in 0..u64::from(batch) {
            let (rgb, inv) = synth_pair(self.seed ^ (self.draws * 131 + b * 7919), h, w);
            out.rgb.extend_from_slice(&rgb);
            out.inv.extend_from_slice(&inv);
        }
        out.mask = vec![1.0; out.inv.len()];
        self.draws += 1;
        Ok(out)
    }
}

/// Run the loop: build the model at `(t.h, t.w)`, seed the params from `init`
/// (fresh `init_weights` or a loaded checkpoint for fine-tune), and step, drawing
/// each step's batch from `source`.
/// `on_step(step, loss)` fires every step with the total loss (progress printing, test capture);
/// returning false stops the run with an error naming the step.
/// Returns the trained `ParamStore` (the caller saves it) and the loss report, or the refusal of `t.saved`, of a
/// batch the source cannot deliver, or of an image the loss cannot fit (fewer than two valid pixels, a NaN).
pub fn train_loop(
    gpu: &Gpu,
    cfg: ZipConfig,
    t: &TrainCfg,
    init: &HashMap<String, Vec<f32>>,
    source: &mut dyn BatchSource,
    mut on_step: impl FnMut(u32, f32) -> bool,
) -> Result<(ParamStore, Trained), String> {
    let ctx = Ctx::new(gpu, crate::net::ids());
    let m = ZipDepth::build_hw(&ctx, cfg, t.batch, t.h, t.w, true);
    if let Some(policy) = t.saved {
        m.set_saved_policy(policy)?;
    }
    // Track BN running stats so eval-mode inference on the saved weights works.
    m.set_update_running(true);
    let ps = ParamStore::new(gpu, m.param_list(), init);
    let (adamw, gn, gs, cc, gsb) = crate::net::optim_ids();
    let opt = Optim::new(adamw, gn, gs, cc, gsb);

    let total = m.out_shape.numel();
    let xb = gpu.storage(m.in_shape.numel() as u64);
    let tgt = gpu.storage(total as u64);
    let mask = gpu.storage(total as u64);
    let d_out = gpu.storage(total as u64);
    let loss_fn = DepthLoss::new(gpu, t.batch, t.h, t.w);

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..t.steps {
        if step == 0 || !t.fixed_batch {
            let b = source.next_batch(t.batch, t.h, t.w)?;
            if (b.rgb.len(), b.inv.len(), b.mask.len()) != (m.in_shape.numel() as usize, total as usize, total as usize) {
                return Err(format!(
                    "batch source returned {} / {} / {} values (rgb / inverse depth / mask), expected {} / {total} / {total}",
                    b.rgb.len(),
                    b.inv.len(),
                    b.mask.len(),
                    m.in_shape.numel()
                ));
            }
            if let Some(bad) = b.mask.iter().position(|&m| m != 0.0 && m != 1.0) {
                return Err(format!("batch source: mask value {} at {bad} is not 0 or 1", b.mask[bad]));
            }
            if let Some(bad) = b.inv.iter().chain(&b.rgb).position(|v| !v.is_finite()) {
                return Err(format!("batch source: non-finite value at {bad} of its inverse depth then RGB values"));
            }
            gpu.write(&xb, bytemuck::cast_slice(&b.rgb));
            gpu.write(&tgt, bytemuck::cast_slice(&b.inv));
            gpu.write(&mask, bytemuck::cast_slice(&b.mask));
        }
        ps.zero_grads(gpu);
        m.forward(&ctx, &ps, &xb);

        // loss and dL/dpred (host-summed per-image reductions, device per-pixel work), then the model backward.
        let loss = loss_fn.eval(gpu, m.out(), &tgt, &mask, t.grad_weight, &d_out)?.total(t.grad_weight);
        m.backward(&ctx, &ps, &xb, &d_out);
        opt.step(gpu, &ps, step + 1, t.lr, t.wd, optim::Adam::default(), Some(1.0), 1.0);
        gpu.poll_wait();

        if step == 0 {
            first = loss;
        }
        last = loss;
        if !on_step(step, loss) {
            return Err(format!("cancelled after step {}", step + 1));
        }
    }
    Ok((ps, Trained { first_loss: first, last_loss: last }))
}
