// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Mixture<M>`: a probability-weighted draw over sub-`Objective`s per
//! micro-step - the weakest member of this objective family, deliberately
//! so: it adds no gradient math of its own, only a host-side categorical
//! draw ([`draw_arm`]) that decides WHICH already-existing
//! `Objective::micro_step` runs this micro-step. Paired with [`Anchor`] (a
//! fixed, unweighted causal-LM dataset - a frozen sample of the base
//! model's own SFT data) as one of the mixed-in arms, this is the
//! anti-forgetting mechanism this training-regime family leans on: an
//! L2-to-reference penalty would need a new argument threaded through
//! `Model::adamw_step` across every implementor, out of scope here; LoRA's
//! bounded-rank delta is the implicit regularizer the rest of this
//! self-improvement path already relies on, and interleaving anchor steps
//! with the primary objective's own steps is the cheap, zero-new-math way
//! to keep the anchor data's own loss from drifting upward while training
//! continues.
//!
//! Swedish Embedded AB builds the training-regime machinery that keeps a
//! self-improving model from forgetting its own base capabilities while it
//! trains on task-specific reward signal. If your team needs anti-
//! forgetting-aware continual training on top of a from-scratch training
//! stack, you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use model::{Batch, Model, Objective, IGNORE};

/// i32 targets from the loader (`-1` = ignore) reinterpreted as the model's
/// `u32` IGNORE sentinel. Mirrors the same one-line helper already
/// duplicated in `model::train` and `rl::load_weighted` - not worth a
/// shared crate for one line copied a third time.
fn targets_to_u32(y: &[i32]) -> Vec<u32> {
    y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect()
}

/// A fixed anchor dataset trained as ordinary unweighted causal LM
/// (`Batch::Lm`) - the anti-forgetting arm of a [`Mixture`]. Typically a
/// frozen sample of the base model's own SFT data, held out so its own
/// held-out loss is a meaningful anti-forgetting signal.
pub struct Anchor {
    data: TokenDataset,
    batch_cfg: BatchConfig,
}

impl Anchor {
    pub fn new(data: TokenDataset, batch_cfg: BatchConfig) -> Self {
        Anchor { data, batch_cfg }
    }
}

impl<M: Model> Objective<M> for Anchor {
    fn regime(&self) -> &'static str {
        "anchor_causal_lm"
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let (x, y) = self.data.get_batch(&self.batch_cfg, rng);
        let targets = targets_to_u32(&y);
        model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
        let loss = model.forward();
        model.backward();
        loss
    }

    fn eval(&mut self, model: &M, rng: &mut Rng, batches: u32) -> Option<f32> {
        let mut total = 0.0;
        for _ in 0..batches.max(1) {
            let (x, y) = self.data.get_batch(&self.batch_cfg, rng);
            let targets = targets_to_u32(&y);
            model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
            total += model.forward();
        }
        Some(total / batches.max(1) as f32)
    }
}

/// Pick an index into `weights` with probability proportional to
/// `weights[i]`, from one uniform draw of `rng`. Pure host-side arithmetic -
/// the entire "mixture" mechanism: [`Mixture::micro_step`] calls this once
/// per micro-step and delegates completely to the chosen arm's own
/// [`Objective::micro_step`], adding no gradient math of its own.
/// `weights` need not already sum to `1` - normalized here by their sum.
pub fn draw_arm(weights: &[f32], rng: &mut Rng) -> usize {
    assert!(!weights.is_empty(), "draw_arm: at least one weight required");
    let total: f32 = weights.iter().sum();
    assert!(total > 0.0, "draw_arm: weights must sum to > 0");
    let u = rng.next_f32() * total;
    let mut acc = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u < acc {
            return i;
        }
    }
    // Floating-point edge case only (u == total exactly): last arm.
    weights.len() - 1
}

/// A probability-weighted draw over sub-[`Objective`]s per micro-step -
/// zero new gradient math, [`Mixture::micro_step`] delegates entirely to
/// whichever arm [`draw_arm`] selects that step. [`Mixture::prepare`] runs
/// on every arm (so e.g. a weighted-loss primary arm still calls
/// `Model::enable_weighted_loss`); `eval`/`itos` delegate to the FIRST arm
/// (the primary objective, by convention) since averaging heterogeneous
/// regimes' losses is not a meaningful number.
pub struct Mixture<M: Model> {
    weights: Vec<f32>,
    arms: Vec<Box<dyn Objective<M>>>,
}

impl<M: Model> Mixture<M> {
    /// `arms`: `(probability, objective)` pairs, at least one, every
    /// probability `> 0`. Probabilities need not already sum to `1` -
    /// [`draw_arm`] normalizes by their sum.
    pub fn new(arms: Vec<(f32, Box<dyn Objective<M>>)>) -> Self {
        assert!(!arms.is_empty(), "Mixture needs at least one arm");
        let (weights, arms): (Vec<f32>, Vec<Box<dyn Objective<M>>>) = arms.into_iter().unzip();
        assert!(weights.iter().all(|&w| w > 0.0), "Mixture arm probabilities must be > 0");
        Mixture { weights, arms }
    }
}

impl<M: Model> Objective<M> for Mixture<M> {
    fn regime(&self) -> &'static str {
        "mixture"
    }

    fn prepare(&mut self, model: &mut M) {
        for arm in &mut self.arms {
            arm.prepare(model);
        }
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let idx = draw_arm(&self.weights, rng);
        self.arms[idx].micro_step(model, rng)
    }

    fn eval(&mut self, model: &M, rng: &mut Rng, batches: u32) -> Option<f32> {
        self.arms.first_mut()?.eval(model, rng, batches)
    }

    fn itos(&self) -> Option<&[char]> {
        self.arms.first().and_then(|a| a.itos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seeded draw counts over many draws must match the configured
    /// probabilities within a fixed tolerance - the gate criterion this
    /// phase's spec states explicitly.
    #[test]
    fn draw_arm_matches_configured_probabilities_over_many_draws() {
        let weights = [0.7, 0.2, 0.1];
        let mut rng = Rng::new(42);
        let n = 200_000;
        let mut counts = [0u32; 3];
        for _ in 0..n {
            counts[draw_arm(&weights, &mut rng)] += 1;
        }
        let tol = 0.01; // 1 percentage point, generously above sampling noise
        for (i, &w) in weights.iter().enumerate() {
            let frac = counts[i] as f64 / n as f64;
            assert!(
                (frac - w as f64).abs() < tol,
                "arm {i}: expected fraction ~{w}, got {frac} (counts {counts:?})"
            );
        }
    }

    /// Same seed, same weights -> the same draw sequence, every time -
    /// `Mixture` must be reproducible, not merely correctly distributed.
    #[test]
    fn draw_arm_is_deterministic_for_a_fixed_seed() {
        let weights = [0.5, 0.5];
        let mut rng_a = Rng::new(7);
        let mut rng_b = Rng::new(7);
        let seq_a: Vec<usize> = (0..1000).map(|_| draw_arm(&weights, &mut rng_a)).collect();
        let seq_b: Vec<usize> = (0..1000).map(|_| draw_arm(&weights, &mut rng_b)).collect();
        assert_eq!(seq_a, seq_b);
    }

    /// Weights `[7, 2, 1]` (sum 10) must draw identically to the already-
    /// normalized `[0.7, 0.2, 0.1]` for the same rng stream - callers should
    /// never need to pre-normalize their configured probabilities.
    #[test]
    fn draw_arm_normalizes_weights_that_do_not_already_sum_to_one() {
        let raw = [7.0, 2.0, 1.0];
        let norm = [0.7, 0.2, 0.1];
        let mut rng_raw = Rng::new(99);
        let mut rng_norm = Rng::new(99);
        for _ in 0..500 {
            assert_eq!(draw_arm(&raw, &mut rng_raw), draw_arm(&norm, &mut rng_norm));
        }
    }
}
