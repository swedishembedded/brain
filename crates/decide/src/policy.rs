// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Treating a prediction as an ACTION: the sequential-decision objective.
//!
//! [`crate::loss`] fits a distribution to a known-correct option. This module
//! is for the case where there is no correct option at a given moment - only
//! an outcome that arrives later, and a model that has to commit to a
//! probability at every step on the way there.
//!
//! The formulation, following *SalesRLAgent* (arXiv 2503.23303) §IV.A:
//!
//! ```text
//! state  s_t  the conversation up to turn t
//! action a_t  "conversion probability estimates" - the committed probability
//! reward r_t  the ACCURACY of that estimate
//! ```
//!
//! ## The reward has to be a proper scoring rule, and this is not a detail
//!
//! "Reward measures the accuracy of the prediction" admits two readings, and
//! only one of them works.
//!
//! Read it as *pay 1 when the model calls the outcome right*, sample a binary
//! action from the policy, and run a policy gradient: the expected ascent
//! direction works out to `(2q - 1) * p * (1 - p)` for an outcome rate `q`,
//! which has the sign of `q - 1/2` at **every** `p`. So the objective is
//! maximized by the MODE. A state whose conversations close 70% of the time
//! trains to 1.0, not to 0.7, and the model ends up accurate and completely
//! uncalibrated - which is the one thing this architecture exists not to be.
//!
//! Read it instead as a **proper scoring rule on the committed probability** -
//! `r = 1 - (p - y)^2`, the negative Brier score - and the expected reward is
//! maximized exactly at `p = q`. That is the reading implemented here, and it
//! is also the more literal one: the paper's action is the probability
//! estimate itself, not a bet placed with it.
//!
//! `the_policy_optimum_is_the_conditional_rate` is the test that separates the
//! two, and it is the test that caught the first reading.
//!
//! ## What is left of PPO, honestly
//!
//! The conversation is REPLAYED. The action cannot change what the customer
//! says next, so there is no state distribution to shift, and every turn's
//! return is its own reward. In that setting the sequential-decision objective
//! reduces to a trust-region-regularized proper scoring rule - and that
//! reduction is a consequence of the environment not responding, not a
//! simplification chosen here.
//!
//! What survives is exactly the two properties the paper asks for, and they
//! are not cosmetic:
//!
//! * **Conservative policy updates** - the committed probability may not move
//!   further than `clip` from the one that collected the batch, in PPO's own
//!   clipped form. Applied to a deterministic action, so the gradient is the
//!   reward's own rather than a likelihood ratio's.
//! * **Strong policy regularization** - an entropy term resisting collapse
//!   onto a confident answer before the conversation has said anything.
//!
//! Plus the one genuinely sequential piece: `discount` weights a turn by how
//! near it is to the outcome, so a probability committed ten turns from the
//! close is held to a looser standard than the one committed at it.
//!
//! Host arithmetic, for the same reason [`crate::loss`] is: two options and
//! one scalar per turn never register against a six-layer encode.

/// How conservative the update is.
#[derive(Clone, Copy, Debug)]
pub struct PolicyConfig {
    /// The trust region: the committed probability may not move more than this
    /// fraction away from the one that collected the batch.
    pub clip: f32,
    /// Weight on the entropy bonus.
    pub entropy: f32,
    /// Per-turn discount toward the outcome. A turn `k` steps before the end
    /// is weighted `gamma^k`, so early small talk is not held to the same
    /// standard as the closing exchange. `1.0` weights every turn equally.
    pub gamma: f32,
}

impl Default for PolicyConfig {
    /// PPO's usual clip, entropy low enough to regularize without flattening a
    /// decision the conversation really has made, and a discount mild enough
    /// that a median-length conversation's first turn still carries about a
    /// third of the weight of its last.
    fn default() -> PolicyConfig {
        PolicyConfig { clip: 0.2, entropy: 0.01, gamma: 0.92 }
    }
}

/// A `Noul` scores exactly one slot, so its score vector has one element and
/// this is the only index there is.
///
/// Named rather than written as a literal `0` because the reason it is 1 and
/// not 2 is the whole subject of `primitives::Question::slots`: two slots make
/// the state signal common-mode and a softmax cancels it.
pub const PROPOSITION: usize = 0;

/// One replayed turn.
#[derive(Clone, Copy, Debug)]
pub struct Turn {
    /// The probability the collecting policy committed to. The trust region is
    /// measured against this.
    pub old_prob: f32,
    /// How the conversation ended.
    pub outcome: bool,
    /// How many turns remain after this one - what `gamma` is raised to.
    pub turns_to_end: u32,
}

/// The negative Brier score of committing `p` when the outcome was `y`.
///
/// Proper: its expectation over `y ~ Bernoulli(q)` is maximized at `p = q`,
/// which is the entire reason it and not a hit rate is the reward here.
pub fn reward(p: f32, outcome: bool) -> f32 {
    let d = p - f32::from(outcome);
    1.0 - d * d
}

/// Loss and `dL/d(score)` for one turn, on the single proposition score.
///
/// `scores` is the one-element score vector a `Noul` produces.
pub fn policy_loss(scores: &[f32], turn: &Turn, cfg: &PolicyConfig) -> (f32, Vec<f32>) {
    assert_eq!(scores.len(), 1, "a noul question scores exactly one slot");
    let p = crate::primitives::sigmoid(scores[PROPOSITION]) as f64;
    let y = f64::from(turn.outcome);
    let discount = (cfg.gamma as f64).powi(turn.turns_to_end as i32);

    // --- the trust region ---
    // The committed probability, against the one that collected this turn.
    // Outside the band the surrogate is flat, exactly as PPO's clipped
    // objective is flat outside its own - the update stops rather than being
    // scaled down, which is what makes it a region and not a penalty.
    let old = (turn.old_prob as f64).clamp(1e-6, 1.0);
    let ratio = p / old;
    let (lo, hi) = (1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64);
    // Only clip in the direction the update is trying to move: a policy that
    // has drifted high and is being pulled back down must still be able to
    // move. Clipping both ways would freeze it wherever it landed.
    let in_region = if p < y { ratio <= hi } else { ratio >= lo };

    // --- reward and entropy ---
    let r = 1.0 - (p - y) * (p - y);
    let ent = bernoulli_entropy(p);
    let beta = cfg.entropy as f64;
    let loss = (-discount * r - beta * ent) as f32;

    // --- gradient, through the logistic: dp/dz = p (1 - p) ---
    let dp = p * (1.0 - p);
    // d(-r)/dz = 2 (p - y) dp/dz.
    let d_r = if in_region { discount * 2.0 * (p - y) * dp } else { 0.0 };
    // dH/dp = ln((1-p)/p), and the loss SUBTRACTS beta * H.
    let d_ent = -beta * ((1.0 - p).max(1e-12) / p.max(1e-12)).ln() * dp;
    (loss, vec![(d_r + d_ent) as f32])
}

/// `H(p)` for a Bernoulli, in nats.
fn bernoulli_entropy(p: f64) -> f64 {
    let t = |q: f64| if q > 0.0 { -q * q.ln() } else { 0.0 };
    t(p) + t(1.0 - p)
}

/// Binary cross-entropy against a SOFT target, on the single proposition
/// score - the supervised phase's objective.
///
/// The dataset labels every turn with a conversion probability, not a class,
/// so the warm start fits the probability rather than an argmax of it. Fitting
/// a hard label here would throw away exactly the calibration the trajectory
/// carries, and the policy phase would have to learn it back from a scalar
/// reward.
///
/// The gradient is `sigmoid(z) - target`: one subtraction, no curvature to
/// tune, and - unlike the two-slot form this replaced - no cancellation
/// between two nearly-parallel option queries.
pub fn bce_loss(scores: &[f32], target: f32) -> (f32, Vec<f32>) {
    assert_eq!(scores.len(), 1, "a noul question scores exactly one slot");
    let p = crate::primitives::sigmoid(scores[PROPOSITION]) as f64;
    let y = target.clamp(0.0, 1.0) as f64;
    let loss = -(y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln());
    (loss as f32, vec![(p - y) as f32])
}

/// The entropy of a soft binary target - the floor [`bce_loss`] cannot go
/// below.
///
/// Worth having because the raw cross-entropy against a probabilistic label is
/// close to unreadable as a progress number: a label of 0.5 costs `ln 2` even
/// from a perfect model, so "loss 0.65" says nothing on its own. Subtracting
/// this gives the KL divergence, which is 0 exactly when the model matches the
/// label and is comparable across datasets with different label spreads.
pub fn target_entropy(target: f32) -> f32 {
    bernoulli_entropy(target.clamp(0.0, 1.0) as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::sigmoid;

    fn fd(f: impl Fn(&[f32]) -> f32, z: &[f32]) -> Vec<f32> {
        let h = 1e-3;
        (0..z.len())
            .map(|i| {
                let (mut a, mut b) = (z.to_vec(), z.to_vec());
                a[i] += h;
                b[i] -= h;
                (f(&a) - f(&b)) / (2.0 * h)
            })
            .collect()
    }

    fn check(name: &str, f: impl Fn(&[f32]) -> f32, z: &[f32], got: &[f32]) {
        let want = fd(&f, z);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 2e-3 * (1.0 + w.abs()),
                "{name}: d[{i}] analytic {g} vs finite-difference {w}"
            );
        }
    }

    /// **The central claim of this module**: the objective's optimum is the
    /// CONDITIONAL RATE, not a confident 0 or 1.
    ///
    /// Tabular, so it tests the objective and nothing else. Two states whose
    /// outcomes fall 70/30 and 20/80; a model that could see which
    /// conversation it was in would go deterministic, and one that can only
    /// see the state must land on the rate.
    ///
    /// This is the test that caught a wrong reward. Paying 1 for calling the
    /// outcome right instead of scoring the committed probability sends state
    /// 0 to 0.999 - accurate, and useless.
    #[test]
    fn the_policy_optimum_is_the_conditional_rate() {
        let rates = [0.7f32, 0.2];
        // No discount and no entropy: this is a statement about the REWARD,
        // and both of those deliberately move the optimum (see below).
        let cfg = PolicyConfig { clip: 1e9, entropy: 0.0, gamma: 1.0 };
        let mut z = [0.0f32; 2];
        let mut rng = data::rng::Rng::new(4);
        let n = 60000;
        for t in 0..n {
            let s = (rng.next_u64() % 2) as usize;
            let outcome = rng.next_f32() < rates[s];
            let old = sigmoid(z[s]);
            let turn = Turn { old_prob: old, outcome, turns_to_end: 0 };
            let (_, d) = policy_loss(&z[s..=s], &turn, &cfg);
            // Decaying step. A CONSTANT one does not converge here, it random
            // walks around the optimum, and the walk is biased because the map
            // from score to probability is curved - which reads as a real
            // miscalibration when it is only the optimizer.
            let lr = 2.0 / (1.0 + t as f32 / 3000.0);
            z[s] -= lr * d[0];
        }
        for (s, &want) in rates.iter().enumerate() {
            let got = sigmoid(z[s]);
            assert!(
                (got - want).abs() < 0.03,
                "state {s}: policy settled on P(yes) = {got:.3}, the outcomes fall at {want:.3}"
            );
        }
    }

    /// A hit-rate reward is what the formulation must NOT use, so the reason
    /// is pinned here rather than left in a comment: its expected ascent
    /// direction has the sign of `q - 1/2` at every `p`, so it can only end at
    /// 0 or 1.
    #[test]
    fn a_hit_rate_reward_would_have_no_interior_optimum() {
        for q in [0.7f64, 0.2] {
            for p in [0.3f64, 0.5, 0.8, 0.95] {
                let ascent = (2.0 * q - 1.0) * p * (1.0 - p);
                assert_eq!(
                    ascent > 0.0,
                    q > 0.5,
                    "q={q} p={p}: a hit-rate reward's direction must depend only on q"
                );
            }
        }
        // ...whereas the proper reward this module uses turns over at p = q.
        for q in [0.7f32, 0.2] {
            let expected = |p: f32| q * reward(p, true) + (1.0 - q) * reward(p, false);
            assert!(expected(q) > expected(q - 0.1), "reward is not maximized at the rate");
            assert!(expected(q) > expected(q + 0.1), "reward is not maximized at the rate");
        }
    }

    #[test]
    fn the_policy_gradient_matches_finite_differences() {
        let cfg = PolicyConfig { clip: 1e9, entropy: 0.05, gamma: 0.9 };
        for (z, turn, what) in [
            ([0.3f32], Turn { old_prob: 0.57, outcome: true, turns_to_end: 0 }, "converted, at the end"),
            ([-0.4], Turn { old_prob: 0.40, outcome: false, turns_to_end: 4 }, "lost, discounted"),
            ([-1.5], Turn { old_prob: 0.18, outcome: true, turns_to_end: 2 }, "low commitment"),
        ] {
            let (_, d) = policy_loss(&z, &turn, &cfg);
            check(what, |s| policy_loss(s, &turn, &cfg).0, &z, &d);
        }
    }

    /// The trust region has to bite in the direction of travel and NOT in the
    /// other: a policy that has drifted must still be able to come back.
    #[test]
    fn the_trust_region_stops_travel_but_not_return() {
        let cfg = PolicyConfig { clip: 0.2, entropy: 0.0, gamma: 1.0 };
        // Committed ~0.9, collected at 0.5: the ratio is ~1.8, far above the
        // band. The outcome says go higher still, so the update must stop.
        let z = [2.2f32];
        assert!(sigmoid(z[0]) > 0.85, "fixture is not where the test needs it");
        let (_, d) = policy_loss(&z, &Turn { old_prob: 0.5, outcome: true, turns_to_end: 0 }, &cfg);
        assert!(d[0].abs() < 1e-7, "the trust region did not stop an upward move: {d:?}");
        // Same drift, opposite outcome: it must be free to move back down.
        let (_, d) = policy_loss(&z, &Turn { old_prob: 0.5, outcome: false, turns_to_end: 0 }, &cfg);
        assert!(d[0].abs() > 1e-4, "the trust region froze a correction: {d:?}");
    }

    /// The discount has to actually weight a turn by its distance from the end.
    #[test]
    fn the_discount_weights_turns_by_distance_to_the_outcome() {
        let cfg = PolicyConfig { clip: 1e9, entropy: 0.0, gamma: 0.5 };
        let z = [0.2f32];
        let near = policy_loss(&z, &Turn { old_prob: 0.55, outcome: true, turns_to_end: 0 }, &cfg).1[0].abs();
        let far = policy_loss(&z, &Turn { old_prob: 0.55, outcome: true, turns_to_end: 3 }, &cfg).1[0].abs();
        assert!((far / near - 0.125).abs() < 1e-4, "three turns out should weigh 0.5^3, got {}", far / near);
    }

    /// The entropy term must NOT be free: it deliberately pulls the optimum
    /// toward the middle, and a caller raising it should see that happen
    /// rather than discover it later as a flat trajectory.
    #[test]
    fn entropy_regularization_pulls_toward_the_middle() {
        let settle = |beta: f32| -> f32 {
            let cfg = PolicyConfig { clip: 1e9, entropy: beta, gamma: 1.0 };
            let mut z = [0.0f32];
            let mut rng = data::rng::Rng::new(6);
            for t in 0..40000 {
                let outcome = rng.next_f32() < 0.9;
                let old = sigmoid(z[0]);
                let (_, d) = policy_loss(&z, &Turn { old_prob: old, outcome, turns_to_end: 0 }, &cfg);
                z[0] -= (1.0 / (1.0 + t as f32 / 3000.0)) * d[0];
            }
            sigmoid(z[0])
        };
        let (plain, regularized) = (settle(0.0), settle(0.3));
        assert!((plain - 0.9).abs() < 0.03, "unregularized settled at {plain:.3}, outcomes are 0.9");
        assert!(regularized < plain - 0.02, "entropy {regularized:.3} did not pull below {plain:.3}");
    }

    /// Fitting a soft target must reproduce the target, not its argmax: a
    /// model trained on a 0.7 label should say 0.7.
    #[test]
    fn the_soft_target_optimum_is_the_target() {
        for want in [0.7f32, 0.05, 0.5] {
            let mut z = [0.0f32];
            for _ in 0..4000 {
                let (_, d) = bce_loss(&z, want);
                z[0] -= 0.5 * d[0];
            }
            assert!((sigmoid(z[0]) - want).abs() < 1e-3, "converged to {}, wanted {want}", sigmoid(z[0]));
        }
    }

    #[test]
    fn the_soft_target_gradient_matches_finite_differences() {
        for target in [0.3f32, 0.5, 0.98] {
            let z = [0.4f32];
            let (_, d) = bce_loss(&z, target);
            check("soft target", |s| bce_loss(s, target).0, &z, &d);
        }
    }

    /// The reported number has to be zero for a model that matches the label,
    /// whatever the label is - that is the whole reason it is reported instead
    /// of the cross-entropy.
    #[test]
    fn the_divergence_from_a_matched_target_is_zero() {
        for q in [0.5f32, 0.7, 0.05] {
            let z = [(q / (1.0 - q)).ln()];
            let (ce, _) = bce_loss(&z, q);
            let kl = ce - target_entropy(q);
            assert!(kl.abs() < 1e-4, "target {q}: matched model reports divergence {kl}");
        }
        let (ce, _) = bce_loss(&[0.0], 0.8);
        assert!(ce - target_entropy(0.8) > 0.1, "a mismatched model reported no divergence");
    }

    /// The whole reason this module works on ONE score: two nearly-parallel
    /// option slots make the state signal common-mode, and a softmax over them
    /// subtracts it out. A logistic on a single score keeps it.
    ///
    /// Measured on the released checkpoint, the yes-slot and no-slot scores
    /// moved by 0.29 across three very different conversations while their
    /// DIFFERENCE moved by 0.0066. These are those numbers.
    #[test]
    fn a_softmax_over_two_slots_cancels_what_a_logistic_keeps() {
        let observed = [(0.5740f32, 0.6161f32), (0.6482, 0.6969), (0.3587, 0.4008)];
        let softmax_p: Vec<f32> = observed.iter().map(|&(y, n)| crate::loss::softmax(&[y, n])[0]).collect();
        let logistic_p: Vec<f32> = observed.iter().map(|&(y, _)| sigmoid(y)).collect();
        let spread = |v: &[f32]| v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min);
        assert!(spread(&softmax_p) < 0.005, "the softmax somehow kept the signal: {softmax_p:?}");
        assert!(
            spread(&logistic_p) > 0.06,
            "the logistic lost the signal too: {logistic_p:?} (spread {})",
            spread(&logistic_p)
        );
    }
}
