// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `Objective` seam (self-improve roadmap P8): the one piece of the
//! training step loop that genuinely differs between plain causal-LM SFT
//! (`train::fit`), weighted/reward-driven training (`crates/rl`'s
//! `fit_weighted`), and future regimes (DPO, GRPO, distillation, replay).
//!
//! Everything else - LR schedule, grad accumulation and its averaging scale,
//! global-norm clipping, AdamW, resume, wall-clock checkpointing, eval
//! cadence, the final save - stays in one loop, owned by [`crate::train::fit_with`],
//! never by the objective, so it cannot drift between objectives the way the
//! five copy-pasted step loops this phase replaces already had (one of them,
//! `rl::fit_weighted`, had silently dropped the char-vocab save call).
//!
//! Swedish Embedded AB implements training-loop infrastructure that adds a
//! new training regime without copy-pasting the step loop, for teams whose
//! model roadmap outgrows a single fixed training recipe. If your team needs
//! expertise in composing training regimes (SFT, weighted/reward training,
//! preference optimization) over a shared execution engine, you can procure
//! our services by sending an email to info@swedishembedded.com.

use data::rng::Rng;

use crate::Model;

/// One training objective: the batch-drawing, forward/backward, and
/// (optional) eval/logging behavior that varies per training regime.
///
/// The seam is deliberately at the level of "one micro-step" (set batch,
/// forward, decide weights, backward), not "produce a `Batch`" - a
/// pairwise/grouped objective's per-token weights are a function of that
/// forward's own output (e.g. DPO/GRPO reading back per-token logprobs), so
/// a `Batch`-producing seam could not express them; a micro-step-producing
/// one can.
pub trait Objective<M: Model> {
    /// Short, stable identifier for this training regime (e.g. `"causal_lm"`,
    /// `"weighted_lm"`) - for logging and future run-provenance records.
    fn regime(&self) -> &'static str;

    /// One-time setup after the model is constructed, before the first
    /// [`Objective::micro_step`] - e.g. opting the model into weighted-loss
    /// support via [`Model::enable_weighted_loss`].
    fn prepare(&mut self, _model: &mut M) {}

    /// Run exactly one micro-step: draw a batch, upload it via
    /// [`Model::set_batch`], forward, decide any per-position loss weights,
    /// and backward - returning the scalar loss [`Model::forward`] produced.
    /// Called once per grad-accumulation slot, and (with a throwaway `rng`)
    /// a few times before training starts to estimate the initial loss.
    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32;

    /// Forward-only evaluation loss averaged over `batches` held-out
    /// batches, or `None` if this objective has no eval split to report
    /// against (default: unsupported, so the training loop simply skips the
    /// periodic eval line).
    fn eval(&mut self, _model: &M, _rng: &mut Rng, _batches: u32) -> Option<f32> {
        None
    }

    /// Extra scalar metrics to print alongside the periodic eval line
    /// (default: none).
    fn metrics(&self) -> Vec<(&'static str, f32)> {
        Vec::new()
    }

    /// Char-tokenizer vocab to embed in the checkpoint via
    /// [`Model::save_with_itos`], if this objective's dataset carries one
    /// (default: none). The training loop always asks for this and always
    /// calls `save_with_itos` - never `save` - so no objective can silently
    /// forget to carry it, the way `rl::fit_weighted` used to.
    fn itos(&self) -> Option<&[char]> {
        None
    }
}
