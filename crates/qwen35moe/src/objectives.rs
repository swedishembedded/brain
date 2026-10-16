// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-token objectives for the Qwen3.5 mixture-of-experts decoder: the
//! `model::training::TokenObjectives` capability DPO, GRPO and distillation
//! stand on.
//!
//! The loss is per token and the router and experts sit below it, so a
//! per-token weight on the logit gradient composes with the mixture-of-experts
//! backward untouched: the weights scale `d_logits` (the shared
//! `model::lossw::WeightedCe` row-scaling step) and everything under it
//! differentiates the weighted loss. The logit scale is one device scalar the
//! logits are multiplied by after the head matmul (`model::logitscale`), and
//! the cross-entropy gradient's count carries the extra factor.
//!
//! Swedish Embedded AB implements solutions for reward-driven and preference
//! fine-tuning of mixture-of-experts language models on a single GPU for its
//! clients. If your team needs expertise in DPO, GRPO or distillation over
//! sparse expert decoders then you can procure our services by sending an
//! email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::logitscale::LogitScale;
use model::lossw::WeightedCe;
use model::training::TokenObjectives;

use super::{Qwen35, GRAD_SCALE_BUF, SCALE_ROW};

/// What a model has opted into; the default pays for neither.
#[derive(Default)]
pub(super) struct State {
    weighted: Option<WeightedCe>,
    scale: Option<LogitScale>,
}

impl State {
    /// Append the step scaling the first `numel` logits by the logit scale
    /// (nothing when none is set).
    pub(super) fn scale_logits(&self, gpu: &Gpu, steps: &mut Vec<Step>, logits: &DeviceBuffer, numel: u32) {
        if let Some(s) = &self.scale {
            s.hook(gpu, steps, GRAD_SCALE_BUF, logits, numel);
        }
    }

    /// The count the cross-entropy gradient divides by: `count / scale`, so
    /// the gradient carries the factor `scale` the scaled logits add.
    pub(super) fn grad_divisor(&self, count: f32) -> f32 {
        self.scale.as_ref().map_or(count, |s| count / s.value())
    }

    /// The row-weighted gradient buffer the head backward must read, or `d_logits` itself.
    pub(super) fn weigh<'a>(&'a self, gpu: &Gpu, steps: &mut Vec<Step>, d_logits: &'a DeviceBuffer) -> &'a DeviceBuffer {
        match &self.weighted {
            Some(w) => w.hook(gpu, steps, SCALE_ROW, d_logits),
            None => d_logits,
        }
    }

    /// The scalar the (weighted) gradient differentiates.
    pub(super) fn loss(&self, gpu: &Gpu, nll: &[f32], count: f32) -> f32 {
        match &self.weighted {
            Some(w) => w.loss(gpu, nll, count),
            None => nll.iter().sum::<f32>() / count,
        }
    }

    pub(super) fn reset_weights(&self, gpu: &Gpu, n: usize) {
        if let Some(w) = &self.weighted {
            w.write(gpu, &vec![1.0; n]);
        }
    }

    pub(super) fn write_weights(&self, gpu: &Gpu, weights: &[f32]) {
        self.weighted.as_ref().expect("qwen35moe: set_loss_weights requires enable_weighted_loss() first").write(gpu, weights);
    }
}

impl Qwen35 {
    /// Whether this build can serve [`TokenObjectives`]: a training build that
    /// owns the head.
    pub(super) fn offers_token_objectives(&self) -> bool {
        self.is_train && self.shard.head
    }
}

impl TokenObjectives for Qwen35 {
    fn enable_weighted_loss(&mut self) {
        if self.objectives.weighted.is_some() {
            return;
        }
        let (n, v) = ((self.b * self.t) as u64, self.cfg.vocab as u64);
        self.objectives.weighted = Some(WeightedCe::new(&self.gpu, n, v));
        self.objectives.reset_weights(&self.gpu, n as usize);
    }

    fn batch_token_logprobs(&self) -> Vec<f32> {
        self.gpu.read(&self.ce_buf, (self.b * self.t) as usize).into_iter().map(|nll| -nll).collect()
    }

    fn set_loss_weights(&self, weights: &[f32]) {
        self.objectives.write_weights(&self.gpu, weights);
    }

    fn set_loss_denominator(&self, tokens: f32) {
        assert!(tokens > 0.0, "the loss denominator must be positive, got {tokens}");
        self.count.set(tokens);
    }

    fn set_logit_scale(&mut self, scale: f32) {
        assert!(scale.is_finite() && scale > 0.0, "the logit scale must be finite and positive, got {scale}");
        if self.objectives.scale.is_none() {
            if scale == 1.0 {
                return;
            }
            self.objectives.scale = Some(LogitScale::new(&self.gpu));
        }
        self.objectives.scale.as_ref().expect("created above").set(&self.gpu, scale);
    }
}
