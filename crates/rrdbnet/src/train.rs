// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RRDBNet **training** graph: SSA forward + hand-written reverse.
//!
//! Composition, not a second model: the forward is exactly [`crate::model::
//! Rrdb`]'s, recorded by [`crate::model::Rrdb::new_train`] with [`vae::blocks::
//! Builder::set_train`] on; the reverse is [`vae::blocks::grad::Trace::
//! backward`], the same shared tape walk `sdxlunet`/`vqgan`/`AutoencoderKL`
//! already use. What this file owns is the loss head and the wiring between
//! them - the same split `crates/sdxlunet/src/train.rs` uses.
//!
//! # What closing this backward actually took
//!
//! Every conv, the channel concat and the nearest-2x upsample were already
//! differentiable through `vae::blocks::Builder` - `check_rrdbnet` gates a
//! genuinely NEW defect class instead: `crate::model`'s `lrelu`/`residual`
//! helpers used to call `Builder::push_step` for the LeakyReLU activation and
//! the `x + 0.2 * f(x)` residual, which records a forward step but NOTHING on
//! the reverse-mode tape. `Trace::backward` walks the tape and silently skips
//! any op whose output no consumer claimed, so a `push_step`'d stage in the
//! middle of a differentiated chain breaks the chain - every conv weight
//! upstream of an activation (i.e. every weight in the net) got a silent ZERO
//! gradient, not merely an unchecked one. This is the SECOND occurrence of
//! that exact bug class in this repo (the first was SDXL's transformer half,
//! `vae::blocks::BWD_KERNELS`'s own doc). The fix was the same shape: give the
//! shared builder real recorders (`Builder::leaky_relu`, `Builder::
//! residual_scale`) and route `crate::model` through them.
//!
//! # The loss
//!
//! Plain **MSE against a random HR target**, `mean (out - target)²` over the
//! `out_channels·H·W` output - deliberately not a faithful Real-ESRGAN
//! training recipe (no discriminator, no perceptual loss, see AGENTS.md): the
//! point is full-graph gradient coverage, not a real fine-tune. `out` is the
//! network's UNCLAMPED `conv_last` output ([`crate::model::Rrdb::out`]) - the
//! `run`-path `[0,1]` clamp is a host-side readback step, never a graph op, so
//! there is no kink to worry about here.

use gpu_core::{DeviceBuffer, Gpu, Step};
use vae::blocks::grad::{BwdIds, Grads, Reverse, Trace};
use vae::blocks::Tensors;

use crate::config::RrdbConfig;
use crate::model::Rrdb;

/// Where [`vae::blocks::BWD_KERNELS`] sits in [`TRAIN_KERNELS`] - right after
/// the inference set ([`crate::model::KERNELS`]).
const BWD_BASE: usize = crate::model::KERNELS.len();
const TAIL: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

const K_MSE_VALUE: usize = TAIL;
const K_MSE_GRAD: usize = TAIL + 1;

/// This model's TRAINING kernel set: the inference set, then the shared block
/// backward set (now carrying `leaky_relu_bwd`/`scale_add_dexp`, see the
/// module doc), then the loss pair.
///
/// `axpy` and every other shared-reverse kernel come from `BWD_KERNELS` and are
/// NOT restated here: a second registration of one kernel name is what the CPU
/// backend's Cranelift JIT rejects outright (`DuplicateDefinition`), silently
/// fine on a GPU and a hard failure on `BRAIN_DEVICE=cpu`.
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

/// A trainable RRDBNet at one input size: one forward step list, one reverse
/// step list, one gradient buffer per tensor. Both are recorded ONCE, at
/// construction - a training step is two submits, exactly like the inference
/// graph is one.
pub struct RrdbTrainer {
    gpu: Gpu,
    cfg: RrdbConfig,
    rrdb: Rrdb,
    trace: Trace,
    grads: Grads,
    /// The MSE target, written per step.
    target: DeviceBuffer,
    /// PER-ELEMENT loss terms, summed on the host - see `sdxlunet::train::
    /// UnetTrainer::loss`'s own doc for why this is `n_out` long, not 1.
    loss: DeviceBuffer,
    /// `dL/d(out)`, the seed of the reverse walk.
    d_out: DeviceBuffer,
    fwd: Vec<Step>,
    rev: Vec<Step>,
    rev_clears: Vec<DeviceBuffer>,
    n_out: u32,
}

impl RrdbTrainer {
    /// Record the forward + reverse for a `[in_channels, h, w]` input. `gpu`
    /// must carry [`TRAIN_KERNELS`].
    pub fn new(gpu: Gpu, cfg: RrdbConfig, tensors: &Tensors, h: u32, w: u32) -> RrdbTrainer {
        let rrdb = Rrdb::new_train(gpu.share_or_new(TRAIN_PIPELINES), cfg.clone(), tensors, h, w);
        let (oh, ow) = rrdb.out_hw();
        let n_out = cfg.out_channels * oh * ow;

        let target = gpu.storage(n_out as u64);
        let loss = gpu.storage(n_out as u64);
        let d_out = gpu.storage(n_out as u64);

        let trace = rrdb.trace().clone();
        let grads = trace.alloc_grads(&gpu);

        // Forward = the recorded graph, then the scalar loss.
        let mut fwd = rrdb.steps().to_vec();
        // `mse_value` Params: [n]; bufs [pred, tgt, out] - ONE INVOCATION PER
        // ELEMENT, writing `(pred-tgt)^2/n` into `out[i]`; the host sums.
        fwd.push(gpu.step(K_MSE_VALUE, &[rrdb.out(), &target, &loss], &[n_out], n_out));

        // Reverse = seed `d_out` from the loss, then walk the tape.
        // `mse_grad` Params: [n]; bufs [pred, tgt, d_pred] - ASSIGNS.
        let mut rev = vec![gpu.step(K_MSE_GRAD, &[rrdb.out(), &target, &d_out], &[n_out], n_out)];
        let reverse: Reverse = trace.backward(&gpu, BwdIds::at(BWD_BASE), &grads, rrdb.out(), &d_out);
        rev.extend(reverse.steps.clone());

        RrdbTrainer {
            gpu,
            cfg,
            rrdb,
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

    pub fn config(&self) -> &RrdbConfig {
        &self.cfg
    }

    /// Output size `(h, w)` - what [`Self::set_inputs`]'s `target` must be
    /// shaped `[out_channels, h, w]` for.
    pub fn out_hw(&self) -> (u32, u32) {
        self.rrdb.out_hw()
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

    /// Write the input image and the MSE target (both `[channels, h, w]` CHW,
    /// at the graph's own `in`/`out` sizes).
    pub fn set_inputs(&self, chw: &[f32], target: &[f32]) {
        assert_eq!(target.len(), self.n_out as usize, "rrdb train: target must be [out_channels, oh, ow]");
        self.gpu.write_f32(self.rrdb.input(), chw);
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

    /// `dL/d(out)` after a [`Self::backward`] - exposed for tests that check
    /// the loss head in isolation from the graph (mirrors `sdxlunet::train::
    /// UnetTrainer::d_out`).
    pub fn d_out(&self) -> &DeviceBuffer {
        &self.d_out
    }

    /// Read a parameter's current values - what a finite-difference sweep
    /// restores after perturbing.
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let len = self.len_of(name);
        self.gpu.read(self.trace.weight(name), len)
    }

    /// Overwrite a parameter's values.
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.trace.weight(name), data);
    }

    /// Read a parameter's accumulated gradient.
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let len = self.len_of(name);
        self.gpu.read(self.grads.g(name), len)
    }

    fn len_of(&self, name: &str) -> usize {
        self.params()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| *l as usize)
            .unwrap_or_else(|| panic!("rrdb train: no parameter {name}"))
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
    /// `BRAIN_DEVICE=cpu` and silently fine on a GPU - exactly the kind of
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
