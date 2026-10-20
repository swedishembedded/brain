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

use gpu_core::{DeviceBuffer, Gpu};
use model::head_ce::RowWeights;
use model::logitscale::LogitScale;
use model::training::TokenObjectives;

use super::Qwen35;

/// What a model has opted into; the default pays for neither.
#[derive(Default)]
pub(super) struct State {
    weighted: Option<Weighted>,
    scale: Option<LogitScale>,
}

/// The per-token weight row (`[b*t]`) and the scratch the head writes the weighted logit gradient to.
struct Weighted {
    weights: DeviceBuffer,
    scratch: DeviceBuffer,
}

impl State {
    /// The head backward's row weights, when weighted loss is enabled.
    pub(super) fn row_weights(&self) -> Option<RowWeights<'_>> {
        self.weighted.as_ref().map(|w| RowWeights { weights: &w.weights, scratch: &w.scratch })
    }

    /// The CE gradient's scalar factor for a batch of `count` supervised tokens.
    pub(super) fn grad_factor(&self, count: f32) -> f32 {
        self.scale.as_ref().map_or(1.0 / count, |s| s.grad_factor(count))
    }

    /// The scalar the weighted gradient differentiates, `sum_i w_i * nll_i / count`;
    /// the plain mean when weighted loss is not enabled.
    pub(super) fn loss(&self, gpu: &Gpu, nll: &[f32], count: f32) -> f32 {
        match &self.weighted {
            Some(w) => {
                let weights = gpu.read(&w.weights, nll.len());
                nll.iter().zip(&weights).map(|(l, w)| l * w).sum::<f32>() / count
            }
            None => nll.iter().sum::<f32>() / count,
        }
    }
}

impl Qwen35 {
    /// Whether this build can serve [`TokenObjectives`]: a training build that owns the head.
    pub(super) fn offers_token_objectives(&self) -> bool {
        self.head_ce.is_some()
    }

    /// Give every supervised position weight 1 (a [`model::Batch::Lm`] on a
    /// weighted-loss model must reproduce the unweighted gradient).
    pub(super) fn reset_loss_weights(&self) {
        if let Some(w) = &self.objectives.weighted {
            self.gpu.write_f32(&w.weights, &vec![1.0; (self.b * self.t) as usize]);
        }
    }

    /// Write per-position weights for the batch already set (`[b*t]`).
    pub(super) fn write_loss_weights(&self, weights: &[f32]) {
        let w = self.objectives.weighted.as_ref().expect("qwen35moe: set_loss_weights requires enable_weighted_loss() first");
        self.gpu.write_f32(&w.weights, weights);
    }
}

impl TokenObjectives for Qwen35 {
    fn head_chunk_rows(&self) -> Option<u32> {
        self.head_ce.as_ref().map(model::head_ce::HeadCe::rows_per_chunk)
    }

    fn enable_weighted_loss(&mut self) {
        if self.objectives.weighted.is_some() {
            return;
        }
        let head = self.head_ce.as_ref().expect("qwen35moe: weighted loss needs the chunked head of a training build");
        let weighted = Weighted { weights: self.gpu.storage((self.b * self.t) as u64), scratch: head.row_weight_scratch(&self.gpu) };
        self.objectives.weighted = Some(weighted);
        self.reset_loss_weights();
    }

    fn batch_token_logprobs(&self) -> Vec<f32> {
        self.gpu.read(&self.ce_buf, (self.b * self.t) as usize).into_iter().map(|nll| -nll).collect()
    }

    fn set_loss_weights(&self, weights: &[f32]) {
        self.write_loss_weights(weights);
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
            let state = LogitScale::new(&self.gpu);
            let buf = state.buffer().clone();
            self.head_ce.as_mut().expect("qwen35moe: a logit scale needs the chunked head of a training build").set_scale_buffer(Some(buf));
            self.objectives.scale = Some(state);
        }
        self.objectives.scale.as_ref().expect("created above").set(&self.gpu, scale);
    }
}
