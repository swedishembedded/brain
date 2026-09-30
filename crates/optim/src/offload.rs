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
//! over PCIe plus a memory-bound CPU pass - cheap next to a training step,
//! and it removes the VRAM pressure that was throttling full fine-tuning.

use gpu_core::Gpu;
use paramstore::ParamStore;
use backend_cpu::par;

/// Host-resident AdamW state for the offloaded parameters.
/// One offloaded parameter's host-side state:
/// `(name, master fp32 weights, Adam m, Adam v)`.
pub type OffloadSlot = (String, Vec<f32>, Vec<f32>, Vec<f32>);

pub struct OffloadAdam {
    /// Per param: (name, master weights, m, v). Order matches `ps.offload`.
    state: Vec<OffloadSlot>,
}

impl OffloadAdam {
    /// Initialise from the store's current (GPU-resident) offloaded weights.
    pub fn new(gpu: &Gpu, ps: &ParamStore) -> OffloadAdam {
        // LoRA (the only source of a non-1.0 lr_mult today) and Role::Offload
        // never coexist in this tree (a LoRA-training ParamStore assigns
        // every non-adapter tensor Role::Frozen, never Role::Offload) - guard
        // the untested combination loudly rather than silently applying the
        // GPU path's per-tensor lr but not this host path's, if that ever
        // changes.
        for (name, _) in &ps.offload {
            assert_eq!(ps.lr_mult_of(name), 1.0, "OffloadAdam: {name} has a non-1.0 lr_mult, which this host path does not yet apply");
        }
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
        adam: crate::Adam,
        clip: Option<f32>,
        extra_scale: f32,
    ) {
        let bc = adam.bias_corrections(t);

        // Pull every grad to the host once (the only device->host traffic).
        let grads: Vec<Vec<f32>> =
            self.state.iter().map(|(name, w, _, _)| gpu.read(ps.g(name), w.len())).collect();

        // Global grad-norm clip (over the offloaded set) — matches the GPU
        // clip-coef path: coef = min(1, max_norm / (||g||*scale)).
        let gscale = if extra_scale != 0.0 { 1.0 / extra_scale } else { 1.0 };
        let scale = if let Some(max_norm) = clip {
            let sq: f64 = par::sum_sq_f64(&grads);
            let norm = (sq.sqrt() as f32) * gscale;
            gscale * (max_norm / norm.max(max_norm)).min(1.0)
        } else {
            gscale
        };

        // Element-wise AdamW per param, parallel across the 48 cores.
        par::zip_each(&mut self.state, &grads, |(_, w, m, v), g| {
                for i in 0..w.len() {
                    adam.update(bc, lr, wd, g[i] * scale, &mut w[i], &mut m[i], &mut v[i]);
                }
            });

        // Push updated weights back to the GPU (the only host->device traffic).
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
