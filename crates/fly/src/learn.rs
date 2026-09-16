// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Episodes, reward, and the controls that make "it learned" checkable.
//!
//! The whole difficulty of this milestone is that a plastic network's
//! behaviour changes over time whether or not it is learning anything useful.
//! Weights drift, activity wanders, and an episode-to-episode improvement
//! appears in a system that is doing nothing of the kind. So the apparatus
//! here is built around the controls rather than around the learner: every
//! condition runs the same episodes through the same code, and only the one
//! thing under test differs.

use crate::Fly;

/// What one episode produced.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Episode {
    /// Net forward displacement of the body over the episode, in the model's
    /// own length units. This is the objective: a fly that walks forward
    /// scores, one that stands still or falls over does not.
    pub distance: f64,
    /// Total spikes, so a condition that simply went quiet is distinguishable
    /// from one that moved less.
    pub spikes: u64,
    /// Proprioceptor spikes, same purpose for the sensory channel.
    pub proprio_spikes: u64,
}

/// How reward becomes a neuromodulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardConfig {
    /// Ticks per episode.
    pub ticks: u32,
    /// Descending command held for the episode.
    pub command: f32,
    /// Exponential-moving-average rate for the reward baseline.
    ///
    /// The neuromodulator is a reward PREDICTION ERROR, not a reward: without
    /// a baseline, a constantly-rewarded network potentiates every eligible
    /// synapse without ever distinguishing a good tick from an average one,
    /// which is potentiation dressed as learning.
    pub baseline_rate: f64,
    /// Scales the prediction error into the modulator.
    pub modulator_gain: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig { ticks: 300, command: 2.0, baseline_rate: 0.01, modulator_gain: 50.0 }
    }
}

/// Which condition an episode is run under.
///
/// These are not options; they are the control matrix. A result that appears
/// under `Learning` and also under `ShuffledReward` is not learning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Condition {
    /// Plasticity on, reward delivered when it is earned.
    Learning,
    /// Plasticity off. Weights cannot move.
    Frozen,
    /// Plasticity on, but the modulator is delivered at the WRONG time: the
    /// same values in a shuffled order, so the distribution is identical and
    /// only the correlation with behaviour is destroyed. This is the control
    /// that separates learning from potentiation.
    ShuffledReward,
}

/// Run one episode and return what it produced.
///
/// The fly is reset first, so episodes are independent and an improvement
/// cannot come from a body that happens to have fallen into a better pose.
pub fn episode(fly: &mut Fly, cfg: RewardConfig, condition: Condition, rng: &mut Lcg) -> Result<Episode, String> {
    fly.reset();
    let cmd = vec![cfg.command; fly.descending_count()];
    fly.set_descending(&cmd)?;
    fly.set_plasticity(condition != Condition::Frozen);

    let start = fly.qpos();
    let mut baseline = 0.0f64;
    let mut ep = Episode::default();
    // Pre-drawn so that ShuffledReward delivers the SAME distribution as
    // Learning, just uncorrelated with what the fly did.
    let mut deltas: Vec<f32> = Vec::with_capacity(cfg.ticks as usize);

    let mut last_x = start.first().copied().unwrap_or(0.0);
    for _ in 0..cfg.ticks {
        let t = fly.step()?;
        ep.spikes += t.total_spikes as u64;
        ep.proprio_spikes += fly.proprioceptor_spikes() as u64;

        let x = fly.qpos().first().copied().unwrap_or(0.0);
        let reward = x - last_x;
        last_x = x;
        baseline += cfg.baseline_rate * (reward - baseline);
        let delta = ((reward - baseline) * cfg.modulator_gain as f64) as f32;
        deltas.push(delta);

        match condition {
            Condition::Learning => fly.modulate(delta),
            // Deliver a modulator drawn from what this episode has already
            // produced, at a time unrelated to what just happened.
            Condition::ShuffledReward => {
                let pick = deltas[rng.index(deltas.len())];
                fly.modulate(pick);
            }
            Condition::Frozen => {}
        }
    }

    let end = fly.qpos();
    ep.distance = end.first().copied().unwrap_or(0.0) - start.first().copied().unwrap_or(0.0);
    Ok(ep)
}

/// A small deterministic PRNG for the shuffled-reward control.
///
/// `data::rng::Lcg` is this workspace's test PRNG, but `crates/data` is a
/// heavier dependency than this crate wants for one index draw, and the
/// shuffled-reward control needs to be reproducible rather than
/// cryptographic.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Lcg {
        Lcg(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() >> 33) as usize % n
    }
}
