// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SDXL UNet **training** graph: SSA forward + hand-written reverse.
//!
//! **Composition, again.** This module adds no kernel, no block and no second
//! model. The forward is exactly [`crate::model::Unet`]'s, recorded by the same
//! [`crate::model::Rec`] with [`vae::blocks::Builder::set_train`] on; the reverse
//! is [`vae::blocks::grad::Trace::backward`], the shared tape walk `vqgan` and
//! `AutoencoderKL` already use. What this file owns is the loss head and the
//! wiring between them.
//!
//! # What closing this backward actually took
//!
//! The conv half of the UNet was already differentiable: every conv, GroupNorm,
//! SiLU, add and upsample goes through `vae::blocks::Builder`, which records an
//! `Op` tape the shared reverse walks. The **transformer** half was not, and not
//! because its adjoints were missing - `matmul_dx`/`matmul_dw`, `bias_grad`,
//! `gelu_erf_bwd`, `layernorm_dx`/`_dgamma`/`_dbeta`, `add_chan_bcast_dv`,
//! `concat_split` and the `attn_bwd_*_cross` quartet all already existed for the
//! decoder LMs' backwards. It was that `Rec` emitted those stages with
//! `Builder::push_step`, which appends a step to the forward list and records
//! NOTHING on the tape.
//!
//! That is not a missing feature, it is a silent wrong answer waiting to happen:
//! `Trace::backward` skips any op whose output no consumer claimed, so a pushed
//! step in the middle of a differentiated chain **breaks the chain**, and every
//! parameter upstream of it gets a **zero** gradient with no error at all.
//! `Builder::push_step`'s own doc says so. The fix was therefore to give the
//! shared builder real recorders for those stages (`linear`, `layernorm`,
//! `gelu_erf`, `mul`, `add_chan`, `self_attn`, `cross_attn`, and a tape entry on
//! the existing `concat`) and to route `Rec` through them.
//!
//! # Two things about the forward that train mode has to change
//!
//! 1. **Flash attention is not differentiable here.** `flash_attn_bidir` never
//!    materialises the softmax, and the adjoint quartet binds exactly that. So a
//!    recording builder takes the materialised `attn_*_bidir` path however
//!    cooperative the device is (`Rec::self_attention` asks `Builder::is_train`).
//!    Outside train mode the choice is unchanged - flash where it is available,
//!    because the score slab is `heads·T²`.
//! 2. **Every attention site needs its OWN softmax slab.** The eval graph reused
//!    one pair across all sites, which is right when nothing reads them again and
//!    silently wrong when the adjoint does: two sites sharing one `probs` would
//!    each differentiate against the other's. `Builder::self_attn`/`cross_attn`
//!    allocate per call out of the activation pool, which train mode disables -
//!    so distinct in train mode, recycled in eval mode, no cost either way.
//!
//! # The loss
//!
//! Plain **MSE against a target sample**, `mean (out - target)²` over the
//! `out_channels·H·W` output. That is the epsilon-prediction diffusion objective
//! with the noise supplied by the caller, and it is deliberately the simplest
//! thing that exercises every parameter: a gradient gate wants full coverage of
//! the graph, not a faithful training recipe. A real fine-tune adds the noise
//! schedule, timestep sampling and SNR weighting on top of this same
//! forward/reverse pair.

use gpu_core::{DeviceBuffer, Gpu, Step};
use vae::blocks::grad::{BwdIds, Grads, Reverse, Trace};
use vae::blocks::Tensors;

use crate::config::UNetConfig;
use crate::model::Unet;

/// Where [`vae::blocks::BWD_KERNELS`] sits in [`TRAIN_KERNELS`] - right after
/// the inference set ([`crate::model::KERNELS`]).
const BWD_BASE: usize = crate::model::KERNELS.len();
const TAIL: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

const K_MSE_VALUE: usize = TAIL;
const K_MSE_GRAD: usize = TAIL + 1;

/// This model's TRAINING kernel set: the inference set, then the shared block
/// backward set, then the loss pair.
///
/// `axpy` and every other shared-reverse kernel come from `BWD_KERNELS` and are
/// NOT restated here: a second registration of one kernel name is what the CPU
/// backend's Cranelift JIT rejects outright (`DuplicateDefinition`), which is
/// silently fine on a GPU and a hard failure on `BRAIN_DEVICE=cpu`.
pub const TRAIN_KERNELS: [(&str, &str); TAIL + 2] = train_kernel_set();

/// [`TRAIN_KERNELS`] as a `'static` slice - what `gpu_core::testgpu::dev` and
/// `Gpu::new_like` want.
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

/// A trainable SDXL UNet at one latent resolution and one text-token count: one
/// forward step list, one reverse step list, one gradient buffer per tensor.
///
/// Both lists are recorded ONCE, at construction. A training step is two
/// submits, exactly like the inference graph is one - no per-step rebuilding.
pub struct UnetTrainer {
    gpu: Gpu,
    cfg: UNetConfig,
    unet: Unet,
    trace: Trace,
    grads: Grads,
    /// The MSE target, written per step.
    target: DeviceBuffer,
    /// PER-ELEMENT loss terms, summed on the host.
    ///
    /// `mse_value` writes `(pred[i] - tgt[i])^2 / n` into `out[i]`, one
    /// invocation per element, and the caller sums - the same shape the
    /// cross-entropy value kernel uses, and what keeps the host-side reduction
    /// a plain sum. This buffer is therefore `n_out` long, not 1. Sizing it 1
    /// and dispatching one thread raises no error at all: it silently makes the
    /// "loss" the first element's term alone, which is exactly wrong enough
    /// that a finite-difference check then measures a real but meaningless
    /// quantity and reports the whole graph as broken.
    loss: DeviceBuffer,
    /// `dL/d(out)`, the seed of the reverse walk.
    d_out: DeviceBuffer,
    fwd: Vec<Step>,
    rev: Vec<Step>,
    rev_clears: Vec<DeviceBuffer>,
    n_out: u32,
}

impl UnetTrainer {
    /// Record the forward + reverse for a `[out_channels, h, w]` output and
    /// `t_enc` text tokens. `gpu` must carry [`TRAIN_KERNELS`].
    pub fn new(gpu: Gpu, cfg: UNetConfig, tensors: &Tensors, h: u32, w: u32, t_enc: u32) -> UnetTrainer {
        let unet = Unet::new_train(gpu.share_or_new(TRAIN_PIPELINES), cfg.clone(), tensors, h, w, t_enc);
        let n_out = cfg.out_channels * h * w;

        let target = gpu.storage(n_out as u64);
        let loss = gpu.storage(n_out as u64);
        let d_out = gpu.storage(n_out as u64);

        let trace = unet.trace().clone();
        let grads = trace.alloc_grads(&gpu);

        // Forward = the recorded graph, then the scalar loss.
        let mut fwd = unet.steps().to_vec();
        // `mse_value` Params: [n]; bufs [pred, tgt, out] - ONE INVOCATION PER
        // ELEMENT, writing `(pred-tgt)^2/n` into `out[i]`; the host sums.
        fwd.push(gpu.step(K_MSE_VALUE, &[unet.out(), &target, &loss], &[n_out], n_out));

        // Reverse = seed `d_out` from the loss, then walk the tape.
        // `mse_grad` Params: [n]; bufs [pred, tgt, d_pred] - ASSIGNS.
        let mut rev = vec![gpu.step(K_MSE_GRAD, &[unet.out(), &target, &d_out], &[n_out], n_out)];
        let reverse: Reverse = trace.backward(&gpu, BwdIds::at(BWD_BASE), &grads, unet.out(), &d_out);
        rev.extend(reverse.steps.clone());

        UnetTrainer {
            gpu,
            cfg,
            unet,
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

    pub fn config(&self) -> &UNetConfig {
        &self.cfg
    }

    /// Every trainable tensor this graph reads, `(name, length in floats)`, in
    /// first-use order.
    pub fn params(&self) -> &[(String, u64)] {
        self.trace.params()
    }

    /// The device buffer holding a parameter - what a finite-difference check
    /// perturbs.
    pub fn weight(&self, name: &str) -> &DeviceBuffer {
        self.trace.weight(name)
    }

    /// The gradient buffer for a parameter.
    pub fn grad(&self, name: &str) -> &DeviceBuffer {
        self.grads.g(name)
    }

    /// Write the four graph inputs and the MSE target.
    pub fn set_inputs(&self, sample: &[f32], enc: &[f32], timestep: f32, pooled: &[f32], time_ids: &[f32], target: &[f32]) {
        assert_eq!(target.len(), self.n_out as usize, "unet train: target must be [out_channels, h, w]");
        // ONE implementation of the two host-side embeddings, shared with
        // `Unet::run` - see its doc for why that matters to a gradient gate.
        self.unet.write_inputs(sample, timestep, enc, pooled, time_ids);
        self.gpu.write_f32(&self.target, target);
    }

    /// Run the forward. Returns `mean (out - target)^2`.
    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd);
        // `mse_value` already divided each term by `n`, so the host reduction
        // is a plain sum - see the `loss` field's own doc.
        self.gpu.read(&self.loss, self.n_out as usize).iter().sum()
    }

    /// Zero every parameter gradient. Must run once per optimizer step, BEFORE
    /// [`Self::backward`]: the weight-gradient kernels read-modify-write, so
    /// clearing them inside the reverse submit would drop every contribution
    /// before the last.
    pub fn zero_grads(&self) {
        let zeros: Vec<DeviceBuffer> = self.grads.all().into_iter().cloned().collect();
        self.gpu.submit(&zeros.iter().collect::<Vec<_>>(), &[]);
    }

    /// Run the reverse. Requires a [`Self::forward`] at the same point (the SSA
    /// forward buffers ARE the backprop cache) and a [`Self::zero_grads`].
    ///
    /// The activation-gradient buffers are cleared by this submit, not by
    /// `zero_grads`: they are ASSIGNED into temps and folded in with `axpy`, so
    /// each must start at zero every step, while the parameter gradients must
    /// survive across the whole reverse.
    pub fn backward(&self) {
        self.gpu.submit(&self.rev_clears.iter().collect::<Vec<_>>(), &self.rev);
    }

    /// `dL/d(out)` after a [`Self::backward`] - exposed for tests that check the
    /// loss head in isolation from the graph.
    pub fn d_out(&self) -> &DeviceBuffer {
        &self.d_out
    }

    /// Read a parameter's current values - what a finite-difference sweep
    /// restores after perturbing.
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let len = self
            .params()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| *l)
            .unwrap_or_else(|| panic!("unet train: no parameter {name}"));
        self.gpu.read(self.trace.weight(name), len as usize)
    }

    /// Overwrite a parameter's values.
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.trace.weight(name), data);
    }

    /// Read a parameter's accumulated gradient.
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let len = self
            .params()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| *l)
            .unwrap_or_else(|| panic!("unet train: no parameter {name}"));
        self.gpu.read(self.grads.g(name), len as usize)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The training kernel set must not name one kernel twice: the CPU JIT
    /// rejects a duplicate definition outright, so this is a hard failure on
    /// `BRAIN_DEVICE=cpu` and silently fine on a GPU - i.e. exactly the kind of
    /// defect that reaches main.
    #[test]
    fn the_training_kernel_set_has_no_duplicate_names() {
        let mut seen = std::collections::HashSet::new();
        for (name, _) in TRAIN_KERNELS {
            assert!(!name.is_empty(), "TRAIN_KERNELS has an unfilled slot");
            assert!(seen.insert(name), "TRAIN_KERNELS registers '{name}' twice");
        }
    }

    /// The inference set is a PREFIX of the training set, so every slot constant
    /// in `crate::model` addresses the same kernel under both.
    #[test]
    fn the_inference_set_is_a_prefix_of_the_training_set() {
        for (i, (name, _)) in crate::model::KERNELS.iter().enumerate() {
            assert_eq!(TRAIN_KERNELS[i].0, *name, "slot {i} differs between the inference and training sets");
        }
    }
}
