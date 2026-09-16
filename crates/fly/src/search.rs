// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Searching the parameters a connectome does not contain.
//!
//! ## Why there is a search at all
//!
//! A connectome is a wiring diagram. It says which cell contacts which and how
//! many times, and it does not say what a synapse is WORTH, what a membrane's
//! time constants are, or how excitable a cell is. Those are the parameters
//! that decide whether a circuit oscillates, and they are not in any file.
//! They have to be found.
//!
//! ## Why an evolution strategy and not a gradient
//!
//! The objective is an episode: a thousand control ticks through a spiking
//! network and a physics engine, scored at the end by whether the animal
//! walked. Nothing in that is differentiable - MuJoCo is not, a spike is not -
//! so the choice is between a gradient estimator that needs a differentiable
//! surrogate for both, and a method that only needs to be able to RUN the
//! episode. This is the second. It is also what the previous instrument here
//! was reaching for: `learn::GainSearch` hill-climbs one coordinate at a time,
//! which needs `2d` evaluations to take one step and stalls on any ridge that
//! is not axis-aligned.
//!
//! The estimator is the standard mirrored-sampling natural-gradient ES: draw
//! `n/2` perturbations, evaluate each in BOTH directions, and step along the
//! rank-weighted average. Three properties earn it here.
//!
//! * **Mirrored pairs** halve the variance of the estimate for the same number
//!   of episodes, because the difference of a pair cancels everything about
//!   the objective that is even in the perturbation - including the large
//!   constant offset an episodic score usually has.
//! * **Rank weighting** rather than the raw score. An episode score with a
//!   product in it (see `Objective::Walk`) is heavy-tailed: one lucky episode
//!   can be twenty times the median, and under raw weighting that single
//!   sample would set the direction of the whole step. Ranks make the update
//!   invariant to any monotone rescaling of the objective, which is exactly
//!   the information a noisy simulator is entitled to give.
//! * **Common random numbers**: every candidate in a generation is evaluated
//!   with the SAME episode seed. The comparison within a generation is then
//!   between parameter vectors rather than between dice rolls, which is what
//!   makes a small real difference visible against a large stochastic one.
//!
//! ## What it does not do
//!
//! It does not learn. It is an offline search over parameters, run by a person
//! with a compute budget, and it stands in this repo as the CEILING
//! instrument: what the best parameters over this structure can do, against
//! which a local plasticity rule's own result can be read. Confusing the two
//! is the thing the whole `learn` module is arranged to prevent. Without a
//! ceiling, "the local rule is weak" and "this structure cannot do the task"
//! produce the same negative result and are not distinguishable from it.

use crate::learn::Lcg;

/// A tunable knob with a name and a range.
///
/// Ranges rather than free parameters, because every one of these has a
/// physical meaning with values that are not merely worse but meaningless: a
/// decay outside `[0, 1)` is not a slow synapse, it is a divergence, and a
/// negative gain inverts a transmitter the data has already told us about.
#[derive(Clone, Debug, PartialEq)]
pub struct Knob {
    pub name: String,
    pub lo: f32,
    pub hi: f32,
}

impl Knob {
    pub fn new(name: impl Into<String>, lo: f32, hi: f32) -> Knob {
        Knob { name: name.into(), lo, hi }
    }

    /// Map an unbounded search coordinate into the knob's range.
    ///
    /// A squashing function rather than a clamp, so the search never lands on
    /// a boundary and stops receiving gradient information there - which is
    /// what a clamp does, and it is how a bounded search silently becomes a
    /// search over a subspace.
    pub fn of(&self, z: f32) -> f32 {
        let u = 0.5 * (1.0 + (z * 0.5).tanh());
        self.lo + u * (self.hi - self.lo)
    }

    /// The search coordinate a given value corresponds to. The inverse of
    /// [`Knob::of`], for starting a search at a known-good setting.
    pub fn at(&self, value: f32) -> f32 {
        let u = ((value - self.lo) / (self.hi - self.lo)).clamp(1e-4, 1.0 - 1e-4);
        2.0 * (2.0 * u - 1.0).atanh()
    }
}

/// A mirrored-sampling natural-gradient evolution strategy.
///
/// Deliberately not generic over the objective's type: it takes an `f64` and
/// maximises it, and everything about what that number means lives with the
/// caller.
pub struct Es {
    knobs: Vec<Knob>,
    mean: Vec<f32>,
    /// Exploration radius, in search coordinates.
    sigma: f32,
    /// Step size along the estimated natural gradient.
    rate: f32,
    rng: Lcg,
    generation: u32,
}

/// One generation's outcome.
#[derive(Clone, Debug)]
pub struct Generation {
    pub index: u32,
    /// Best score seen in this generation.
    pub best: f64,
    /// Mean score over the generation, which is what the step actually moves.
    pub mean: f64,
    /// The parameters that scored `best`.
    pub best_params: Vec<f32>,
    /// The distribution's centre after the update.
    pub centre: Vec<f32>,
}

impl Es {
    /// Start from a named point in knob space.
    pub fn new(knobs: Vec<Knob>, start: &[f32], sigma: f32, rate: f32, seed: u64) -> Result<Es, String> {
        if start.len() != knobs.len() {
            return Err(format!("{} knobs but {} starting values", knobs.len(), start.len()));
        }
        let mean = knobs.iter().zip(start).map(|(k, v)| k.at(*v)).collect();
        Ok(Es { knobs, mean, sigma, rate, rng: Lcg::new(seed), generation: 0 })
    }

    pub fn knobs(&self) -> &[Knob] {
        &self.knobs
    }

    /// The knob values at the distribution's centre: the current best guess.
    pub fn centre(&self) -> Vec<f32> {
        self.knobs.iter().zip(&self.mean).map(|(k, z)| k.of(*z)).collect()
    }

    /// Run one generation of `pairs * 2` evaluations.
    ///
    /// `evaluate` is handed the knob VALUES, already in range, and an episode
    /// seed that is the same for every candidate in this generation.
    pub fn step(&mut self, pairs: usize, mut evaluate: impl FnMut(&[f32], u64) -> f64) -> Generation {
        let d = self.mean.len();
        let seed = self.rng.next_u64();
        let mut eps: Vec<Vec<f32>> = Vec::with_capacity(pairs);
        let mut scored: Vec<(f64, usize, bool)> = Vec::with_capacity(pairs * 2);
        let (mut best, mut best_params) = (f64::NEG_INFINITY, self.centre());

        for i in 0..pairs {
            let e: Vec<f32> = (0..d).map(|_| self.rng.normal()).collect();
            for mirrored in [false, true] {
                let sign = if mirrored { -1.0 } else { 1.0 };
                let values: Vec<f32> = self
                    .knobs
                    .iter()
                    .zip(&self.mean)
                    .zip(&e)
                    .map(|((k, z), x)| k.of(z + sign * self.sigma * x))
                    .collect();
                let score = evaluate(&values, seed);
                if score > best {
                    best = score;
                    best_params = values;
                }
                scored.push((score, i, mirrored));
            }
            eps.push(e);
        }

        // Rank weights, centred so a generation that is uniformly good does
        // not translate the whole distribution: only the ORDER within it can
        // move the centre.
        let mean_score = scored.iter().map(|s| s.0).sum::<f64>() / scored.len().max(1) as f64;
        let mut order: Vec<usize> = (0..scored.len()).collect();
        order.sort_by(|&a, &b| scored[a].0.total_cmp(&scored[b].0));
        let n = scored.len() as f32;
        let mut weight = vec![0.0f32; scored.len()];
        for (rank, &idx) in order.iter().enumerate() {
            weight[idx] = (rank as f32 + 0.5) / n - 0.5;
        }

        let mut grad = vec![0.0f32; d];
        for (k, &(_, pair, mirrored)) in scored.iter().enumerate() {
            let sign = if mirrored { -1.0 } else { 1.0 };
            for (g, x) in grad.iter_mut().zip(&eps[pair]) {
                *g += weight[k] * sign * x;
            }
        }
        let scale = self.rate / (n * self.sigma).max(1e-9);
        for (z, g) in self.mean.iter_mut().zip(&grad) {
            *z += scale * g;
        }

        self.generation += 1;
        Generation { index: self.generation, best, mean: mean_score, best_params, centre: self.centre() }
    }
}
