// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The logit scale of [`crate::training::TokenObjectives::set_logit_scale`]:
//! one device scalar and the in-place scaling step a model appends after its
//! LM head, so the softmax cross-entropy, its gradient and the per-token
//! log-probability readout all see `scale * logits`.
//!
//! A sampler that draws at temperature `T` draws from `softmax(logits / T)`;
//! an RL objective whose importance ratio compares the trainer's log-probability
//! with the sampler's must score under that same distribution, which is
//! `scale = 1/T`. The gradient is `scale * (softmax(scale * z) - onehot)`: the
//! model folds the extra `scale` into the CE gradient's existing `1/count`
//! factor ([`LogitScale::grad_factor`]), so the backward needs no new step.
//!
//! A model owns `Option<LogitScale>`; `None` (the default, scale `1.0`) records
//! no step and allocates nothing, the same opt-in shape as [`crate::lossw::WeightedCe`].
//!
//! Swedish Embedded AB implements training stacks whose policy-gradient
//! objectives score samples under the distribution that produced them. If your
//! team needs expertise in on-policy RL for language models you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::cell::Cell;

use gpu_core::{DeviceBuffer, Gpu, Step};

/// The device scalar and the host copy of the factor the logits are scaled by.
pub struct LogitScale {
    value: Cell<f32>,
    buf: DeviceBuffer,
}

impl LogitScale {
    /// A scale of `1.0` (the identity) on `gpu`.
    pub fn new(gpu: &Gpu) -> LogitScale {
        let buf = gpu.storage(1);
        gpu.write_f32(&buf, &[1.0]);
        LogitScale { value: Cell::new(1.0), buf }
    }

    /// Set the factor. Takes effect on the next forward/backward; the recorded
    /// steps read it from the device, so nothing is rebuilt.
    pub fn set(&self, gpu: &Gpu, scale: f32) {
        assert!(scale.is_finite() && scale > 0.0, "the logit scale must be finite and positive, got {scale}");
        self.value.set(scale);
        gpu.write_f32(&self.buf, &[scale]);
    }

    /// The device scalar holding the factor, for a consumer that applies it
    /// inside its own steps (see `HeadCe::set_scale_buffer`).
    pub fn buffer(&self) -> &DeviceBuffer {
        &self.buf
    }

    pub fn value(&self) -> f32 {
        self.value.get()
    }

    /// Append the step scaling the first `numel` elements of `logits` in place
    /// (`grad_scale_buf`, the model's registered kernel index for it).
    pub fn hook(&self, gpu: &Gpu, steps: &mut Vec<Step>, grad_scale_buf: usize, logits: &DeviceBuffer, numel: u32) {
        steps.push(gpu.step(grad_scale_buf, &[logits, &self.buf], &[numel], numel));
    }

    /// The CE gradient's scalar factor for a batch of `count` supervised
    /// positions: `scale / count` (`d/dz CE(scale * z)` carries one factor of
    /// `scale` beyond the softmax difference).
    pub fn grad_factor(&self, count: f32) -> f32 {
        self.value.get() / count
    }
}
