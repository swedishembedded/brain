// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Playing the policy: one forward pass per move, for many instances at once.
//!
//! There is no search here and that is the point of the crate. No frontier,
//! no priority queue, no node expansions, no heuristic evaluated on
//! candidates. The policy names a move, the move is played, and the state it
//! lands on is the next query. The cost of an instance is its solution length
//! in forward passes.
//!
//! Instances are rolled out in LOCKSTEP so a batch shares each pass: a
//! network this small is latency-bound on a single state and throughput-bound
//! on a thousand, so solving a thousand cubes costs barely more than solving
//! one.

use crate::{Net, StateSpace};

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Reached the goal, by these moves.
    Solved(Vec<usize>),
    /// Ran out of steps. The moves it did play, for inspection.
    Stuck(Vec<usize>),
}

impl Outcome {
    pub fn solved(&self) -> bool {
        matches!(self, Outcome::Solved(_))
    }
    pub fn moves(&self) -> &[usize] {
        match self {
            Outcome::Solved(m) | Outcome::Stuck(m) => m,
        }
    }
}

/// How a rollout is played.
#[derive(Clone, Copy, Debug)]
pub struct Rollout {
    /// Give up after this many moves. A cap is not a fallback - an instance
    /// that hits it is reported STUCK and counted against the policy.
    pub max_steps: usize,
    /// Refuse a move the generator would have called redundant after the
    /// previous one (undoing it, or on a cube turning the same face twice).
    ///
    /// This is a mask over the policy's own output, not a search: it removes
    /// moves that provably waste a step, and it costs nothing. It is separate
    /// because its effect belongs in the measurement - a policy that needs it
    /// is a policy that has learned to dither.
    pub forbid_redundant: bool,
    /// Sample from the policy instead of taking its argmax, at this
    /// temperature. Zero means argmax.
    ///
    /// Still one forward pass per move and still no search - what it changes
    /// is the failure MODE. A greedy rollout that reaches a state whose
    /// argmax leads back where it came from will loop there until the step
    /// cap; sampling turns that into a walk biased by the policy, which
    /// escapes. Reported separately from greedy, because a policy that only
    /// works when sampled has not learned the decision, it has learned a
    /// direction.
    pub temperature: f32,
    /// Seed for that sampling. A run is reproducible or it is not evidence.
    pub seed: u64,
}

impl Default for Rollout {
    fn default() -> Rollout {
        Rollout { max_steps: 64, forbid_redundant: true, temperature: 0.0, seed: 1 }
    }
}

/// Roll `starts` out together. One forward pass per lockstep move.
///
/// Instances that finish early stop consuming moves but keep their row, which
/// wastes a fraction of each pass and buys a graph whose shape never changes.
pub fn solve_batch<S: StateSpace>(
    space: &S,
    net: &Net,
    starts: &[S::State],
    how: Rollout,
) -> Vec<Outcome> {
    let rows = net.rows as usize;
    assert!(starts.len() <= rows, "a batch rolls out at most {rows} instances");
    let width = space.feature_len();
    let n_moves = space.moves();

    let mut states: Vec<S::State> = starts.to_vec();
    let mut played: Vec<Vec<usize>> = vec![Vec::new(); starts.len()];
    let mut done: Vec<bool> = states.iter().map(|s| space.is_goal(s)).collect();
    let mut last: Vec<Option<usize>> = vec![None; starts.len()];
    let mut features = vec![0.0f32; rows * width];
    let mut rng = crate::data::Rng::new(how.seed);

    for _ in 0..how.max_steps {
        if done.iter().all(|&d| d) {
            break;
        }
        features.iter_mut().for_each(|f| *f = 0.0);
        for (r, s) in states.iter().enumerate() {
            space.write_features(s, &mut features[r * width..(r + 1) * width]);
        }
        let probs = net.policy(&features);

        for r in 0..states.len() {
            if done[r] {
                continue;
            }
            let row = &probs[r * n_moves..(r + 1) * n_moves];
            let allowed = |m: usize| {
                !(how.forbid_redundant && last[r].is_some_and(|l| space.redundant(l, m)))
            };
            let best = if how.temperature > 0.0 {
                // Renormalise over the allowed moves and draw. `probs` is
                // already a softmax, so a temperature is a power.
                let inv = 1.0 / how.temperature;
                let mut acc = 0.0f64;
                let mut weights = vec![0.0f64; n_moves];
                for (m, &p) in row.iter().enumerate() {
                    if allowed(m) {
                        let w = (p.max(1e-12) as f64).powf(inv as f64);
                        weights[m] = w;
                        acc += w;
                    }
                }
                if acc <= 0.0 {
                    continue;
                }
                let mut u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * acc;
                let mut pick = usize::MAX;
                for (m, &w) in weights.iter().enumerate() {
                    if w > 0.0 {
                        u -= w;
                        if u <= 0.0 {
                            pick = m;
                            break;
                        }
                    }
                }
                if pick == usize::MAX {
                    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
                } else {
                    pick
                }
            } else {
                let mut best = usize::MAX;
                let mut best_p = f32::NEG_INFINITY;
                for (m, &p) in row.iter().enumerate() {
                    if allowed(m) && p > best_p {
                        best_p = p;
                        best = m;
                    }
                }
                if best == usize::MAX {
                    continue;
                }
                best
            };
            states[r] = space.apply(&states[r], best);
            played[r].push(best);
            last[r] = Some(best);
            if space.is_goal(&states[r]) {
                done[r] = true;
            }
        }
    }

    played
        .into_iter()
        .zip(done)
        .map(|(m, ok)| if ok { Outcome::Solved(m) } else { Outcome::Stuck(m) })
        .collect()
}
