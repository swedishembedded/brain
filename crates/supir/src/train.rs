// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SUPIR's **training** graph: trunk + adaptors + frozen-backbone-shaped SDXL
//! UNet, all recorded into ONE reverse-mode tape, plus an MSE loss head.
//!
//! **Composition, again - the same story [`sdxlunet::train`]'s own module doc
//! tells.** This file adds no kernel and no block. The forward is exactly
//! [`crate::model::Supir::new_train`]'s (trunk via `Rec::down_path`/`mid_block`
//! under `Rec::set_prefix`, adaptors via [`crate::adaptors::Adaptors`]'s
//! [`vae::blocks::skipfuse::SkipFuse`] impl, backbone via
//! [`sdxlunet::model::Unet::record_into`]); the reverse is
//! [`vae::blocks::grad::Trace::backward`], the same shared tape walk
//! `sdxlunet::train::UnetTrainer` drives. What this file owns is the loss
//! head and the wiring between them - `sdxlunet::train`'s own MSE-loss
//! pattern, restated because SUPIR's graph has a different set of inputs
//! (`sample` AND `hint`, not just `sample`) and a different kernel set (the
//! `edm_mix`/`scale_row` slots [`crate::model::KERNELS`] already carries).
//!
//! Naming note: the design brief for this file named it `grad.rs`/
//! `modelgrad.rs`, the pattern `s3dit`/`flux2`/`wan` use for a model whose
//! forward is a HAND-WRITTEN device kernel sequence with no automatic
//! differentiation - those crates need an independent host f64
//! forward-plus-analytic-backward reimplementation as ground truth, because
//! nothing else exercises their backward math. SUPIR's trainable path is not
//! that: every op the trunk and the adaptors use (`conv`/`gn`/`silu`/`mul`/
//! `add`/`concat`/`mix`, the transformer stages) already goes through
//! [`vae::blocks::Builder`], which is tape-recording and already has a
//! generic reverse walk - the exact situation `sdxlunet`'s own `train.rs`
//! module doc describes ("not one new kernel... this checks the result").
//! Reimplementing the whole trunk (SDXL's down+mid blocks: resnets,
//! transformer blocks, GroupNorm, attention) a second time in host f64 would
//! duplicate `check_unet`'s own already-proven coverage of that exact code,
//! for no additional correctness signal - the risk here is the NEW plumbing
//! (`Rec::set_prefix`/`take_temb_act`/`set_temb_act`, the `SkipFuse` seam,
//! and whether gradients correctly ACCUMULATE where a trunk hidden state
//! feeds more than one adaptor), which a device-vs-device finite-difference
//! check on the actual recorded graph exercises directly - the same shape
//! [`sdxlunet::train::UnetTrainer`] and `gradcheck::unet` already established
//! as this workspace's gate for "a model built from already-differentiable
//! blocks". So this crate keeps `sdxlunet`'s own file name (`train.rs`) and
//! its own trainer struct ([`SupirTrainer`]), not a second, unrelated
//! naming scheme borrowed from crates whose graphs are shaped differently.

use gpu_core::{DeviceBuffer, Gpu, Step};
use vae::blocks::grad::{BwdIds, Grads, Reverse, Trace};
use vae::blocks::Tensors;

use crate::config::SupirConfig;
use crate::model::Supir;

/// Where [`vae::blocks::BWD_KERNELS`] sits in [`TRAIN_KERNELS`] - right after
/// [`crate::model::KERNELS`] (which already carries the `edm_mix`/`scale_row`
/// forward-set slots [`vae::blocks::Builder::set_mix_ids`] needs).
const BWD_BASE: usize = crate::model::KERNELS.len();
const TAIL: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

const K_MSE_VALUE: usize = TAIL;
const K_MSE_GRAD: usize = TAIL + 1;

/// This model's TRAINING kernel set: [`crate::model::KERNELS`], then the
/// shared block backward set, then the loss pair. `scale_row` is NOT
/// restated here - it is already one of `crate::model::KERNELS`'s own two
/// appended slots, and `vae::blocks::BWD_KERNELS` does not itself contain
/// it (the caller supplies `edm_mix`/`scale_row` out of band via
/// `MixIds` - see `crate::model`'s own doc) - so there is no duplicate name
/// for the CPU backend's Cranelift JIT to reject.
pub const TRAIN_KERNELS: [(&str, &str); TAIL + 2] = train_kernel_set();

pub const TRAIN_PIPELINES: &[(&str, &str)] = &TRAIN_KERNELS;

const fn train_kernel_set() -> [(&'static str, &'static str); TAIL + 2] {
    let mut k = [("", ""); TAIL + 2];
    let mut i = 0;
    while i < crate::model::KERNELS.len() {
        k[i] = crate::model::KERNELS[i];
        i += 1;
    }
    let mut j = 0;
    while j < vae::blocks::BWD_KERNELS.len() {
        k[BWD_BASE + j] = vae::blocks::BWD_KERNELS[j];
        j += 1;
    }
    k[K_MSE_VALUE] = ("mse_value", kernels::MSE_VALUE);
    k[K_MSE_GRAD] = ("mse_grad", kernels::MSE_GRAD);
    k
}

/// A trainable SUPIR graph (trunk + adaptors + backbone) at one latent
/// resolution and one text-token count, at a fixed `control_scale` - see
/// [`crate::adaptors::Adaptors`]'s field doc for why the scale is a host
/// constant baked into the graph, same as the eval-mode [`Supir`].
pub struct SupirTrainer {
    gpu: Gpu,
    cfg: SupirConfig,
    supir: Supir,
    trace: Trace,
    grads: Grads,
    target: DeviceBuffer,
    loss: DeviceBuffer,
    d_out: DeviceBuffer,
    fwd: Vec<Step>,
    rev: Vec<Step>,
    rev_clears: Vec<DeviceBuffer>,
    n_out: u32,
}

impl SupirTrainer {
    /// Record the forward + reverse. `gpu` must carry [`TRAIN_KERNELS`].
    pub fn new(gpu: Gpu, cfg: SupirConfig, tensors: &Tensors, h: u32, w: u32, t_enc: u32, control_scale: f32) -> SupirTrainer {
        let supir = Supir::new_train(gpu.share_or_new(TRAIN_PIPELINES), cfg.clone(), tensors, h, w, t_enc, control_scale);
        let n_out = cfg.backbone.out_channels * h * w;

        let target = gpu.storage(n_out as u64);
        let loss = gpu.storage(n_out as u64);
        let d_out = gpu.storage(n_out as u64);

        let trace = supir.trace().clone();
        let grads = trace.alloc_grads(&gpu);

        let mut fwd = supir.steps().to_vec();
        fwd.push(gpu.step(K_MSE_VALUE, &[supir.out(), &target, &loss], &[n_out], n_out));

        let mut rev = vec![gpu.step(K_MSE_GRAD, &[supir.out(), &target, &d_out], &[n_out], n_out)];
        let reverse: Reverse = trace.backward(&gpu, BwdIds::at(BWD_BASE), &grads, supir.out(), &d_out);
        rev.extend(reverse.steps.clone());

        SupirTrainer {
            gpu,
            cfg,
            supir,
            trace,
            grads,
            target,
            loss,
            d_out,
            fwd,
            rev,
            rev_clears: reverse.clears.clone(),
            n_out,
        }
    }

    pub fn config(&self) -> &SupirConfig {
        &self.cfg
    }

    /// Every trainable tensor this graph reads - trunk (`control_model.*`),
    /// adaptors (`project_modules.*`) AND the backbone (unprefixed) - in
    /// first-use order. `crate::finetune` filters this by name to implement
    /// the adaptor-only ("frozen encoder") recipe; nothing here itself
    /// freezes anything, matching `UnetTrainer::params`'s own "everything
    /// recorded is trainable" contract.
    pub fn params(&self) -> &[(String, u64)] {
        self.trace.params()
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let len = self.len_of(name);
        self.gpu.read(self.trace.weight(name), len)
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.trace.weight(name), data);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let len = self.len_of(name);
        self.gpu.read(self.grads.g(name), len)
    }

    fn len_of(&self, name: &str) -> usize {
        self.params()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| *l as usize)
            .unwrap_or_else(|| panic!("supir train: no parameter {name}"))
    }

    /// Write the graph's inputs and the MSE target. `sample`/`hint` are
    /// `[in_channels · H · W]` - see [`Supir::run`]'s doc for the rest.
    #[allow(clippy::too_many_arguments)]
    pub fn set_inputs(&self, sample: &[f32], hint: &[f32], enc: &[f32], timestep: f32, pooled: &[f32], time_ids: &[f32], target: &[f32]) {
        assert_eq!(target.len(), self.n_out as usize, "supir train: target must be [out_channels, h, w]");
        self.supir.write_inputs(sample, hint, timestep, enc, pooled, time_ids);
        self.gpu.write_f32(&self.target, target);
    }

    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd);
        self.gpu.read(&self.loss, self.n_out as usize).iter().sum()
    }

    pub fn zero_grads(&self) {
        let zeros: Vec<DeviceBuffer> = self.grads.all().into_iter().cloned().collect();
        self.gpu.submit(&zeros.iter().collect::<Vec<_>>(), &[]);
    }

    pub fn backward(&self) {
        self.gpu.submit(&self.rev_clears.iter().collect::<Vec<_>>(), &self.rev);
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// `dL/d(out)` after a [`Self::backward`] - exposed for tests that check
    /// the loss head in isolation, mirroring
    /// [`sdxlunet::train::UnetTrainer::d_out`].
    pub fn d_out(&self) -> &DeviceBuffer {
        &self.d_out
    }
}

/// Shared test-only fixture: a [`SupirConfig::tiny`] instance plus a merged
/// backbone+delta tensor map with deterministic synthetic weights - the
/// same "frozen backbone + SUPIR delta" merge `crate::model`'s own
/// `tiny_forward_is_finite` test performs, written once here so
/// `crate::lora`'s and `crate::finetune`'s tests (in different files of the
/// same crate) do not each re-derive it.
#[cfg(test)]
pub(crate) fn tiny_setup(seed: u64) -> (SupirConfig, Tensors) {
    let cfg = SupirConfig::tiny();
    let mut tensors = sdxlunet::init::init_weights(&cfg.backbone, seed);
    tensors.extend(crate::init::init_weights(&cfg, seed ^ 0x5350_4952));
    (cfg, tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_training_kernel_set_has_no_duplicate_names() {
        let mut seen = std::collections::HashSet::new();
        for (name, _) in TRAIN_KERNELS {
            assert!(!name.is_empty(), "TRAIN_KERNELS has an unfilled slot");
            assert!(seen.insert(name), "TRAIN_KERNELS registers '{name}' twice");
        }
    }

    #[test]
    fn the_inference_set_is_a_prefix_of_the_training_set() {
        for (i, (name, _)) in crate::model::KERNELS.iter().enumerate() {
            assert_eq!(TRAIN_KERNELS[i].0, *name, "slot {i} differs between the inference and training sets");
        }
    }
}
