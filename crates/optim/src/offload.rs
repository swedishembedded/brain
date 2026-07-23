// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Off-device AdamW: the optimiser state (`m`/`v` + a master copy of the
//! weights) lives in **system RAM**, and the update runs on the CPU. Only the
//! weight and gradient stay on the GPU — where the forward/backward need them.
//!
//! Why: full fp32 AdamW keeps 4 buffers per parameter on the GPU
//! (weight+grad+m+v = 4×model). The moments are touched **once per step**, never
//! during forward/backward, so they don't need HBM bandwidth. Moving them to the
//! box's 177 GB of RAM cuts GPU optimiser state to 2×model (weight+grad) and lets
//! models far larger than 24 GB of VRAM train — the classic ZeRO-Offload idea.
//!
//! Per step: read each grad off the GPU, run the exact AdamW update (same math as
//! `adamw.wgsl`) over host-resident `m`/`v`/master-weights with rayon across the
//! 48 cores, and write the updated weight back. The 0.6B round-trip is ~4.8 GB
//! over PCIe (~0.3 s) + a memory-bound CPU pass — cheap next to a training step,
//! and it removes the VRAM pressure that was throttling full fine-tuning.

use gpu_core::Gpu;
use paramstore::ParamStore;
use rayon::prelude::*;

/// Host-resident AdamW state for the offloaded parameters.
pub struct OffloadAdam {
    /// Per param: (name, master weights, m, v). Order matches `ps.offload`.
    state: Vec<(String, Vec<f32>, Vec<f32>, Vec<f32>)>,
}

impl OffloadAdam {
    /// Initialise from the store's current (GPU-resident) offloaded weights.
    pub fn new(gpu: &Gpu, ps: &ParamStore) -> OffloadAdam {
        let state = ps
            .offload
            .iter()
            .map(|(name, numel)| {
                let w = gpu.read(ps.w(name), *numel); // master copy in RAM
                (name.clone(), w, vec![0.0f32; *numel], vec![0.0f32; *numel])
            })
            .collect();
        OffloadAdam { state }
    }

    /// One AdamW step over the offloaded params. `extra_scale` divides the grads
    /// (grad-accumulation averaging); `clip` is a global grad-norm clip computed
    /// host-side across exactly these params. Matches `adamw.wgsl` element-wise.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        gpu: &Gpu,
        ps: &ParamStore,
        t: u32,
        lr: f32,
        wd: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        clip: Option<f32>,
        extra_scale: f32,
    ) {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        // Pull every grad to the host once (the only device->host traffic).
        let grads: Vec<Vec<f32>> =
            self.state.iter().map(|(name, w, _, _)| gpu.read(ps.g(name), w.len())).collect();

        // Global grad-norm clip (over the offloaded set) — matches the GPU
        // clip-coef path: coef = min(1, max_norm / (||g||*scale)).
        let gscale = if extra_scale != 0.0 { 1.0 / extra_scale } else { 1.0 };
        let scale = if let Some(max_norm) = clip {
            let sq: f64 = grads
                .par_iter()
                .map(|g| g.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>())
                .sum();
            let norm = (sq.sqrt() as f32) * gscale;
            gscale * (max_norm / norm.max(max_norm)).min(1.0)
        } else {
            gscale
        };

        // Element-wise AdamW per param, parallel across the 48 cores.
        self.state
            .par_iter_mut()
            .zip(grads.par_iter())
            .for_each(|((_, w, m, v), g)| {
                for i in 0..w.len() {
                    let gi = g[i] * scale;
                    let mi = beta1 * m[i] + (1.0 - beta1) * gi;
                    let vi = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                    m[i] = mi;
                    v[i] = vi;
                    let mhat = mi / bc1;
                    let vhat = vi / bc2;
                    let mut wi = w[i];
                    wi -= lr * wd * wi;
                    wi -= lr * mhat / (vhat.sqrt() + eps);
                    w[i] = wi;
                }
            });

        // Push updated weights back to the GPU (the only host->device traffic).
        for (name, w, _, _) in &self.state {
            gpu.write(ps.w(name), bytemuck::cast_slice(w));
        }
    }

    /// Sum of squares of the offloaded gradients (host-side), excluding any named
    /// params. For pipeline-parallel training the global grad-norm is the sum of
    /// each stage's `grad_sq`; a replicated (tied) weight is excluded on all but
    /// one stage so it is counted exactly once, matching the single-device norm.
    pub fn grad_sq(&self, gpu: &Gpu, ps: &ParamStore, exclude: &[&str]) -> f64 {
        let grads: Vec<Vec<f32>> = self
            .state
            .iter()
            .filter(|(n, _, _, _)| !exclude.contains(&n.as_str()))
            .map(|(n, w, _, _)| gpu.read(ps.g(n), w.len()))
            .collect();
        grads
            .par_iter()
            .map(|g| g.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>())
            .sum()
    }

    /// One AdamW step multiplying every gradient by a **caller-supplied** `scale`
    /// (which already folds in grad-accumulation averaging and any global clip
    /// coefficient) — no internal norm computation. Used by the pipeline optimiser
    /// so all stages apply one globally-reduced clip coefficient, keeping tied
    /// replicas bit-identical. Element-wise identical to [`Self::step`].
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_scale(
        &mut self,
        gpu: &Gpu,
        ps: &ParamStore,
        t: u32,
        lr: f32,
        wd: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        scale: f32,
    ) {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        let grads: Vec<Vec<f32>> =
            self.state.iter().map(|(name, w, _, _)| gpu.read(ps.g(name), w.len())).collect();
        self.state
            .par_iter_mut()
            .zip(grads.par_iter())
            .for_each(|((_, w, m, v), g)| {
                for i in 0..w.len() {
                    let gi = g[i] * scale;
                    let mi = beta1 * m[i] + (1.0 - beta1) * gi;
                    let vi = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                    m[i] = mi;
                    v[i] = vi;
                    let mhat = mi / bc1;
                    let vhat = vi / bc2;
                    let mut wi = w[i];
                    wi -= lr * wd * wi;
                    wi -= lr * mhat / (vhat.sqrt() + eps);
                    w[i] = wi;
                }
            });
        for (name, w, _, _) in &self.state {
            gpu.write(ps.w(name), bytemuck::cast_slice(w));
        }
    }

    /// The current master weights (host copy), for saving a checkpoint without a
    /// device read-back.
    pub fn master(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.state.iter().map(|(n, w, _, _)| (n.as_str(), w.as_slice()))
    }
}
