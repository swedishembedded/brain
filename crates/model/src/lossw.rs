// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reward/advantage-weighted cross-entropy: the shared buffers-plus-
//! `scale_row`-step-plus-weighted-forward recipe every [`crate::Head::
//! TokenClassifier`] model's [`crate::Batch::LmWeighted`] opt-in needs (see
//! that variant's own doc comment for the contract). `qwen3::Qwen::
//! enable_weighted_loss` carried this inline as ~60 lines across three call
//! sites (buffer allocation, the backward's row-scaling step, and the
//! forward's weighted-sum branch); this hoists all three so a second (and
//! future) `Head::TokenClassifier` model adopts weighted-loss training in a
//! few lines instead of copying that trio verbatim.
//!
//! A model owns `Option<WeightedCe>` on itself (`None` = ordinary
//! unweighted training, zero extra buffers and zero extra kernel dispatch -
//! the same opt-in shape as `enable_mrope`/`enable_mm_splice`). `Some` is
//! built once, from the model's own `enable_weighted_loss(&mut self)`,
//! sized for that model's `n = b·t` rows and `v` = vocab.
//!
//! Swedish Embedded AB implements shared, gradient-checked training
//! primitives like this one so an architecture team adopting a new training
//! objective does not re-derive (and re-gradcheck) the same weighted-CE math
//! per model. If your team needs a new training regime wired onto an
//! existing model, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Per-position weighted cross-entropy state: an `[n]` weight row and an
/// `[n·v]` scratch buffer for the weighted CE gradient (`n = b·t`, `v` =
/// vocab).
pub struct WeightedCe {
    n: u64,
    v: u64,
    loss_weights: DeviceBuffer,
    d_logits_weighted: DeviceBuffer,
}

impl WeightedCe {
    /// Allocate the two buffers weighted-loss training needs. Call once,
    /// from the model's own opt-in method (after its ordinary unweighted
    /// buffers already exist), sized for that model's `n = b·t` rows and
    /// `v` = vocab.
    pub fn new(gpu: &Gpu, n: u64, v: u64) -> Self {
        Self { n, v, loss_weights: gpu.storage(n), d_logits_weighted: gpu.storage(n * v) }
    }

    /// Append the row-scaling step (the model's own registered kernel index
    /// for `scale_row.wgsl`) that scales the just-computed unweighted
    /// `d_logits` per ROW (token position) by the weights last written via
    /// [`Self::write`], into this instance's own scratch buffer - NOT in
    /// place (see `scale_row.wgsl`'s own doc comment). Returns that scratch
    /// buffer: every downstream backward step (the head's dw/dx and beyond)
    /// must read THIS instead of the raw `d_logits` passed in.
    pub fn hook<'a>(&'a self, gpu: &Gpu, steps: &mut Vec<Step>, scale_row: usize, d_logits: &DeviceBuffer) -> &'a DeviceBuffer {
        let total = (self.n * self.v) as u32;
        steps.push(gpu.step(scale_row, &[d_logits, &self.loss_weights, &self.d_logits_weighted], &[total, self.v as u32], total));
        &self.d_logits_weighted
    }

    /// Write the per-position CE-gradient weights (`[n]`) for the next
    /// backward.
    pub fn write(&self, gpu: &Gpu, weights: &[f32]) {
        assert_eq!(weights.len() as u64, self.n, "WeightedCe::write: expected {} weights, got {}", self.n, weights.len());
        gpu.write_f32(&self.loss_weights, weights);
    }

    /// The scalar weighted loss `Σ loss_weights[i]·ce_loss[i] / count` - the
    /// SAME scalar the (weighted) gradient [`Self::hook`] wired up
    /// differentiates (the `Model::forward` contract): per-row terms don't
    /// cross, so a per-row scalar factor commutes with `d/d(logits)`.
    /// `losses` is the per-row unweighted CE loss the model's own loss
    /// kernel already wrote (`ce_buf`), read back once by the caller for
    /// both the weighted and unweighted paths. Reads `loss_weights` back
    /// from the device rather than trusting a cached host copy of
    /// [`Self::write`]'s argument - one extra small host read per forward,
    /// kept simple and impossible to let drift from what [`Self::hook`]'s
    /// step actually read.
    pub fn loss(&self, gpu: &Gpu, losses: &[f32], count: f32) -> f32 {
        let w = gpu.read(&self.loss_weights, losses.len());
        losses.iter().zip(&w).map(|(l, wi)| l * wi).sum::<f32>() / count
    }
}
