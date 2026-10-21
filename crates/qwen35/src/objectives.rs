// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-token objectives for the Qwen3.5 decoder: the
//! `model::training::TokenObjectives` capability that DPO, GRPO and
//! distillation stand on, over the chunked LM head the training build already
//! has.
//!
//! * **Weighted loss.** The head's recorded backward scales each token's
//!   logit gradient by a per-token weight (`model::head_ce::RowWeights`), so
//!   the weighted loss costs one `[b*t]` weight row and one `[rows, vocab]`
//!   scratch, nothing per layer.
//! * **Log-probability readout.** The forward leaves every token's
//!   cross-entropy in `ce_buf`; `batch_token_logprobs` is its negation.
//! * **Logit scale.** One device scalar the head multiplies the logits by
//!   after the head GEMM (`HeadCe::set_scale_buffer`), so the loss, its
//!   gradient and the readout all see `scale * logits`; the CE gradient's
//!   `1/count` carries the extra factor (`LogitScale::grad_factor`). This
//!   model records no tape, so a new scale takes effect on the next call.
//!
//! The capability exists on a training build with the chunked head: not on an
//! inference build, a pipeline stage without the head, or a build with the
//! multi-token-prediction head (its loss has a second term no weight reaches).
//!
//! Swedish Embedded AB implements solutions for reward-driven and preference
//! fine-tuning of hybrid linear-attention language models on a single GPU for
//! its clients. If your team needs expertise in DPO, GRPO or distillation
//! over Gated-DeltaNet decoders then you can procure our services by sending
//! an email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};
use model::head_ce::RowWeights;
use model::logitscale::LogitScale;
use model::training::TokenObjectives;

use super::Qwen35;

/// What a model has opted into; the default is plain unweighted, unscaled
/// training, which pays for neither.
#[derive(Default)]
pub(super) struct State {
    weighted: Option<Weighted>,
    scale: Option<LogitScale>,
}

/// The per-token weight row (`[b*t]`, padded positions included) and the
/// scratch the head writes the weighted logit gradient to.
struct Weighted {
    weights: DeviceBuffer,
    scratch: DeviceBuffer,
}

impl State {
    /// The head backward's row weights, when weighted loss is enabled.
    /// Whether per-position loss weights are enabled (they are laid out at the full row length).
    pub(super) fn is_weighted(&self) -> bool {
        self.weighted.is_some()
    }

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
    /// Whether this build can serve [`TokenObjectives`].
    pub(super) fn offers_token_objectives(&self) -> bool {
        self.head_ce.is_some()
    }

    /// Every sequence's first `t_user` positions of a per-position buffer
    /// that is padded to `t` per sequence.
    fn unpad<T: Copy>(&self, padded: &[T]) -> Vec<T> {
        let (tu, tp) = (self.t_user as usize, self.t as usize);
        if tu == tp {
            return padded.to_vec();
        }
        (0..self.b as usize).flat_map(|s| padded[s * tp..s * tp + tu].iter().copied()).collect()
    }

    /// The inverse of [`Self::unpad`], filling padded positions with `fill`.
    fn pad<T: Copy>(&self, values: &[T], fill: T) -> Vec<T> {
        let (tu, tp, b) = (self.t_user as usize, self.t as usize, self.b as usize);
        assert_eq!(values.len(), b * tu, "qwen35: {} per-position values for {b} sequences of {tu}", values.len());
        let mut padded = vec![fill; b * tp];
        for s in 0..b {
            padded[s * tp..s * tp + tu].copy_from_slice(&values[s * tu..(s + 1) * tu]);
        }
        padded
    }

    /// Give every supervised position weight 1 (a [`model::Batch::Lm`] on a
    /// weighted-loss model must reproduce the unweighted gradient).
    pub(super) fn reset_loss_weights(&self) {
        if let Some(w) = &self.objectives.weighted {
            self.gpu.write_f32(&w.weights, &vec![1.0; (self.b * self.t) as usize]);
        }
    }

    /// Write per-position weights for the batch already set (`[b*t_user]`).
    pub(super) fn write_loss_weights(&self, weights: &[f32]) {
        let w = self.objectives.weighted.as_ref().expect("qwen35: set_loss_weights requires enable_weighted_loss() first");
        self.gpu.write_f32(&w.weights, &self.pad(weights, 0.0));
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
        let head = self.head_ce.as_ref().expect("qwen35: weighted loss needs the chunked head of a training build without MTP");
        let weighted = Weighted { weights: self.gpu.storage((self.b * self.t) as u64), scratch: head.row_weight_scratch(&self.gpu) };
        self.objectives.weighted = Some(weighted);
        self.reset_loss_weights();
    }

    fn batch_token_logprobs(&self) -> Vec<f32> {
        // Only the batch's computed prefix holds fresh values; the rest of the row is unsupervised.
        let mut nll = self.gpu.read(&self.ce_buf, (self.b * self.run_len()) as usize);
        nll.resize((self.b * self.t) as usize, 0.0);
        self.unpad(&nll).into_iter().map(|l| -l).collect()
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
            self.head_ce.as_mut().expect("qwen35: a logit scale needs the chunked head of a training build without MTP").set_scale_buffer(Some(buf));
            self.objectives.scale = Some(state);
        }
        self.objectives.scale.as_ref().expect("created above").set(&self.gpu, scale);
    }
}
