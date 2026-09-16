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

use crate::reference::{ImitationReward, Reference};
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

/// What the creature is being asked to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Objective {
    /// Net forward displacement of the body.
    ///
    /// Kept because the ceiling was measured under it, and because it is the
    /// honest demonstration of why it is the wrong objective: a single
    /// coordinated lunge scores as well as a gait, and a direct search under
    /// this reward suppressed the cord's recurrent circuitry and drove sensory
    /// input straight to the muscles. That is a reflex, and it is what this
    /// reward asks for.
    Displacement,
    /// Track a recorded fly walking, DeepMimic style.
    ///
    /// What flybody's own walking task uses, and what the imitation-learning
    /// literature settled on precisely because it does not need the reward
    /// engineering the alternative does. The reward is dense - every tick has
    /// a target - where displacement is nearly flat until something moves.
    Imitate {
        /// Which recorded snippet to follow.
        snippet: usize,
        reward: ImitationReward,
    },
}

/// How reward becomes a neuromodulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardConfig {
    /// What to reward.
    pub objective: Objective,
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
        RewardConfig {
            objective: Objective::Displacement,
            ticks: 300,
            command: 2.0,
            baseline_rate: 0.01,
            modulator_gain: 50.0,
        }
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
    /// Plasticity on and reward delivered correctly, but the WIRING is a
    /// degree-matched shuffle of the connectome. The structural control: if
    /// this does as well, the published wiring was not what mattered.
    ///
    /// Selected when the fly is built, not here - the graph is fixed at
    /// construction - so this variant exists to label the condition rather
    /// than to change what `episode` does.
    ShuffledConnectome,
}

impl Condition {
    /// Whether weights may move under this condition.
    pub fn plastic(self) -> bool {
        self != Condition::Frozen
    }

    /// Whether the modulator this condition delivers is the one that was
    /// earned.
    pub fn reward_is_honest(self) -> bool {
        matches!(self, Condition::Learning | Condition::ShuffledConnectome)
    }
}

/// Run one episode and return what it produced.
///
/// The fly is reset first, so episodes are independent and an improvement
/// cannot come from a body that happens to have fallen into a better pose.
pub fn episode(fly: &mut Fly, cfg: RewardConfig, condition: Condition, rng: &mut Lcg) -> Result<Episode, String> {
    episode_with(fly, cfg, condition, None, rng)
}

/// The same, with a reference trajectory available for [`Objective::Imitate`].
///
/// Separate entry point rather than an `Option` on `RewardConfig` because a
/// reference is data with a lifetime, and threading it through a `Copy` config
/// would make the config borrow.
pub fn episode_with(
    fly: &mut Fly,
    cfg: RewardConfig,
    condition: Condition,
    reference: Option<&Reference>,
    rng: &mut Lcg,
) -> Result<Episode, String> {
    fly.reset();
    let cmd = vec![cfg.command; fly.descending_count()];
    fly.set_descending(&cmd)?;
    fly.set_plasticity(condition.plastic());

    // Reference-state initialisation: an imitation episode starts ON the
    // trajectory it is asked to follow.
    if let Objective::Imitate { snippet, .. } = cfg.objective {
        let r = reference.ok_or("Objective::Imitate needs a reference trajectory")?;
        let (nq, nv) = fly.dims();
        r.check_matches(nq, nv)?;
        let (q, v) = r.frame(snippet, 0).ok_or_else(|| format!("snippet {snippet} is empty"))?;
        let q: Vec<f64> = q.iter().map(|x| *x as f64).collect();
        let v: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        fly.set_pose(&q, &v)?;
    }

    let start = fly.qpos();
    let mut baseline = 0.0f64;
    let mut ep = Episode::default();
    // Pre-drawn so that ShuffledReward delivers the SAME distribution as
    // Learning, just uncorrelated with what the fly did.
    let mut deltas: Vec<f32> = Vec::with_capacity(cfg.ticks as usize);

    let mut last_x = start.first().copied().unwrap_or(0.0);
    for tick in 0..cfg.ticks as usize {
        let t = fly.step()?;
        ep.spikes += t.total_spikes as u64;
        ep.proprio_spikes += fly.proprioceptor_spikes() as u64;

        let reward = match cfg.objective {
            Objective::Displacement => {
                let x = fly.qpos().first().copied().unwrap_or(0.0);
                let r = x - last_x;
                last_x = x;
                r
            }
            Objective::Imitate { snippet, reward } => {
                // One reference frame per control tick - the dataset is
                // sampled at exactly the control period, so `tick` indexes it
                // directly. Past the end of a snippet the reward is zero
                // rather than clamped to the last frame, which would pay a
                // creature for standing still at the end.
                let r = reference.expect("checked above");
                match r.frame(snippet, tick) {
                    Some((rq, rv)) => {
                        reward.total_over(&fly.qpos(), &fly.qvel(), rq, rv, r.moving_dofs())
                    }
                    None => 0.0,
                }
            }
        };
        baseline += cfg.baseline_rate * (reward - baseline);
        let delta = ((reward - baseline) * cfg.modulator_gain as f64) as f32;
        deltas.push(delta);

        match condition {
            Condition::Learning | Condition::ShuffledConnectome => fly.modulate(delta),
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
    /// A standard normal, by Box-Muller. The search needs a symmetric
    /// perturbation; a uniform one biases every step toward the corners of the
    /// box it samples.
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() >> 33) as usize % n
    }
}

/// A search over per-cell-type gains, with the wiring held fixed.
///
/// This is the CEILING INSTRUMENT, and it exists to answer a question the
/// control matrix cannot: when the local learning rule fails to produce
/// walking, is the rule weak, or can this structure not do this task with this
/// reward and this body? Without an answer, a negative result is ambiguous and
/// therefore not much of a result.
///
/// It is not a surrogate-gradient method, and the substitution is deliberate.
/// A gradient path would have to differentiate through MuJoCo, which this
/// binding does not expose and which would mean either a differentiable body
/// or a policy-gradient estimator - both larger projects than the question
/// needs. What the question needs is an upper bound on what these parameters
/// can achieve under a stronger optimiser than a local rule, and a direct
/// search gives that.
///
/// The parameters are per-presynaptic-SUPER-CLASS gains rather than per-synapse
/// weights, which is what makes the search tractable: eleven numbers against
/// 5.3 million, at roughly three seconds per evaluation. It is also the
/// standard shape for a connectome-constrained model - structure fixed,
/// a small number of biologically meaningful gains free - rather than a
/// convenience. The cost is real and worth naming: a gain search cannot
/// express anything the cell-type partition cannot, so it is a LOWER bound on
/// what the full weight space could do, and a negative result from it is
/// weaker evidence than a negative result from a per-synapse optimiser.
pub struct GainSearch {
    /// Which gain group each edge belongs to, by its presynaptic neuron.
    edge_group: Vec<u8>,
    groups: Vec<String>,
    /// Signed, scaled weights at unit gain.
    base: Vec<f32>,
}

impl GainSearch {
    pub fn new(c: &connectome::Connectome, weight_scale: f32) -> GainSearch {
        let mut groups: Vec<String> = Vec::new();
        let mut of_neuron: Vec<u8> = Vec::with_capacity(c.neurons.len());
        for n in &c.neurons {
            let key = if n.super_class.is_empty() { "<none>" } else { n.super_class.as_str() };
            let idx = match groups.iter().position(|g| g == key) {
                Some(i) => i,
                None => {
                    groups.push(key.to_string());
                    groups.len() - 1
                }
            };
            of_neuron.push(idx as u8);
        }
        let base = c.signed_csc(weight_scale).w;
        let edge_group = c.csc.pre.iter().map(|&p| of_neuron[p as usize]).collect();
        GainSearch { edge_group, groups, base }
    }

    pub fn groups(&self) -> &[String] {
        &self.groups
    }

    /// The weight vector for a given gain setting.
    pub fn weights(&self, gains: &[f32]) -> Vec<f32> {
        self.base
            .iter()
            .zip(&self.edge_group)
            .map(|(w, &g)| w * gains.get(g as usize).copied().unwrap_or(1.0))
            .collect()
    }

    /// A neutral starting point: every gain at 1, which reproduces the
    /// connectome exactly as imported.
    pub fn unit_gains(&self) -> Vec<f32> {
        vec![1.0; self.groups.len()]
    }
}
