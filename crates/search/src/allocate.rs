// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where the next unit of search budget goes.
//!
//! A campaign has several fundamentally different ways to produce a candidate
//! (return to an archived cell and perturb, hunt a specific unmet goal, refine
//! an elite locally, recombine two, sample a learned proposal policy) and no
//! way to know in advance which of them is currently worth running. Worse, the
//! answer *moves*: early on, fresh proposals and novelty dominate; once there
//! is something good to work with, local refinement does; near a strong
//! candidate, verification does.
//!
//! So the question is not "which operator is best" but
//!
//! ```text
//! which operator has the greatest expected archive gain per second, NOW?
//! ```
//!
//! which is a bandit problem, and this is UCB1 over it. The measured rate is
//! per **second**, never per call: two operators finding equally much are not
//! equally good if one takes twenty times as long to do it, and an allocator
//! that cannot see that spends the whole campaign on the expensive one.

/// What one run of an operator bought.
///
/// `fresh` and `improved` are WEIGHTS rather than counts, and that is the
/// whole point of them. Counting every new niche as one rewards whichever
/// operator can manufacture the most cells, which is not the same operator as
/// the one advancing the search. Measured on `samples/decision/doom`: a
/// wall-pressing operator produced 44 513 steps of cheap cells around ground
/// already covered against a playing operator's 23 272, scored a comparable
/// rate on a per-count metric, and took a third of the budget while the
/// playing operator was the one reaching new parts of the level.
///
/// So a caller credits each admission with how far along it is relative to
/// the best the archive holds - see [`Gain::admitted`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Gain {
    /// Niches nothing had reached before, weighted by depth.
    pub fresh: f64,
    /// Niches already held, now reached better or faster, weighted by depth.
    pub improved: f64,
    /// What it cost in wall clock.
    pub seconds: f64,
}

/// What a fresh niche is worth against an improved one.
///
/// A fresh niche is the search reaching somewhere it has never been, which is
/// what the whole apparatus exists for; an improvement is a time coming down
/// in a place already known, which matters and matters less. Four to one is a
/// judgement, not a measurement - it is here as one named constant rather than
/// spread across the operators so that changing it is one edit.
const FRESH_WORTH: f64 = 4.0;
const IMPROVED_WORTH: f64 = 1.0;

/// The exploration constant. Textbook UCB1 uses `sqrt(2)`; the rates here are
/// normalised into `0..=1` before it is applied, so the usual value applies
/// without rescaling.
const EXPLORATION: f64 = std::f64::consts::SQRT_2;

/// A clock reading of zero is resolution, not an infinitely productive
/// operator. Divided through unguarded it produces an infinite rate and that
/// arm takes the entire remaining budget.
const MIN_SECONDS: f64 = 1e-3;

/// The least an admission may be credited with, however shallow.
///
/// Never zero. A cell at the level's front door is not worth much and it is
/// not worth NOTHING - it is where a search that has not started yet has to
/// begin, and an operator that only ever produces them would otherwise show a
/// rate of exactly zero and be indistinguishable from one that produces no
/// cells at all.
const SHALLOWEST: f64 = 0.05;

impl Gain {
    /// Credit one admission, weighted by how far along the cell is relative
    /// to the best the archive holds.
    pub fn admitted(&mut self, fresh: bool, reached: f32, best: f32) {
        let depth = if best > 0.0 {
            (reached as f64 / best as f64).clamp(SHALLOWEST, 1.0)
        } else {
            SHALLOWEST
        };
        if fresh {
            self.fresh += depth;
        } else {
            self.improved += depth;
        }
    }

    /// The scalar an arm is credited with, before dividing by time.
    pub fn value(&self) -> f64 {
        FRESH_WORTH * self.fresh + IMPROVED_WORTH * self.improved
    }

    fn rate(&self) -> f64 {
        self.value() / self.seconds.max(MIN_SECONDS)
    }
}

#[derive(Clone, Debug)]
struct Arm {
    name: String,
    draws: u64,
    seconds: f64,
    value: f64,
}

/// One row of [`Allocator::report`] - what a campaign prints so a human can
/// see where the budget went.
#[derive(Clone, Debug)]
pub struct Spent {
    pub name: String,
    pub draws: u64,
    pub seconds: f64,
    /// Archive gain per second, the quantity the allocation is made on.
    pub rate: f64,
}

/// UCB1 over search operators, scored on archive gain per second.
#[derive(Clone, Debug)]
pub struct Allocator {
    arms: Vec<Arm>,
    rounds: u64,
}

impl Allocator {
    pub fn new(names: &[&str]) -> Allocator {
        Allocator {
            arms: names
                .iter()
                .map(|n| Arm { name: (*n).to_string(), draws: 0, seconds: 0.0, value: 0.0 })
                .collect(),
            rounds: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.arms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arms.is_empty()
    }

    pub fn name(&self, arm: usize) -> &str {
        &self.arms[arm].name
    }

    /// Which operator to run next.
    ///
    /// Every arm is tried once before any is repeated - an untried arm has no
    /// rate, and treating "no measurement" as "no gain" is how an operator
    /// that would have opened the next region gets written off before it ever
    /// ran.
    pub fn choose(&mut self) -> usize {
        if let Some(i) = self.arms.iter().position(|a| a.draws == 0) {
            return i;
        }
        let top = self.arms.iter().map(|a| self.rate_of(a)).fold(f64::MIN_POSITIVE, f64::max);
        let total = (self.rounds.max(1) as f64).ln();
        let mut best = (0usize, f64::NEG_INFINITY);
        for (i, a) in self.arms.iter().enumerate() {
            // Normalised so the exploration term is on a comparable scale to
            // the reward whatever units the gain is measured in.
            let exploit = self.rate_of(a) / top;
            let explore = EXPLORATION * (total / a.draws as f64).sqrt();
            let score = exploit + explore;
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    }

    fn rate_of(&self, a: &Arm) -> f64 {
        a.value / a.seconds.max(MIN_SECONDS)
    }

    /// Record what a run of `arm` bought and what it cost.
    pub fn credit(&mut self, arm: usize, gain: Gain) {
        let Some(a) = self.arms.get_mut(arm) else { return };
        a.draws += 1;
        a.seconds += gain.seconds.max(MIN_SECONDS);
        a.value += gain.value();
        self.rounds += 1;
        // `rate()` is the per-call view; the arm accumulates totals so its
        // rate is over everything it has ever done rather than the last call.
        let _ = gain.rate();
    }

    pub fn report(&self) -> Vec<Spent> {
        self.arms
            .iter()
            .map(|a| Spent {
                name: a.name.clone(),
                draws: a.draws,
                seconds: a.seconds,
                rate: self.rate_of(a),
            })
            .collect()
    }
}
