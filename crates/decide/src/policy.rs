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
    /// Weight on the divergence from a REFERENCE policy - the anchor.
    ///
    /// A policy gradient started from a cloned policy has no reason to stay
    /// near it. The two phases optimise different objectives, so once the
    /// cloning stops, the update pulls only toward return; where return is
    /// sparse and its estimate noisy, that pull is mostly noise, and the
    /// demonstrated behaviour is walked away from rather than improved on.
    /// Holding the update near a fixed reference is how RLHF keeps a tuned
    /// model near the one it was tuned from, and it is the same problem.
    ///
    /// `0.0` - the default, and what every caller that has not thought about
    /// it gets - is no anchor and exactly the behaviour this had before.
    pub anchor: f32,
    /// How far the policy may drift from the one that COLLECTED the batch
    /// before the remaining passes over it are abandoned.
    ///
    /// Clipping alone is not a trust region. It zeroes the gradient of any
    /// sample that has moved too far, but the samples still inside the band
    /// keep pushing, and several passes over a batch in minibatches is
    /// hundreds of optimizer steps - so a policy can end an iteration a long
    /// way from the one whose data justified the step. Every reference
    /// implementation guards this by measuring the divergence and stopping,
    /// and it is the guard this had no version of.
    ///
    /// `0.0` disables it. Spinning Up's default is 0.01.
    pub target_kl: f32,
}

impl Default for PolicyConfig {
    /// PPO's usual clip, entropy low enough to regularize without flattening a
    /// decision the conversation really has made, and a discount mild enough
    /// that a median-length conversation's first turn still carries about a
    /// third of the weight of its last.
    fn default() -> PolicyConfig {
        PolicyConfig { clip: 0.2, entropy: 0.01, gamma: 0.92, anchor: 0.0, target_kl: 0.02 }
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

/// One step of a CONTROL policy: which option was taken, how likely it was
/// when it was taken, and how much better the outcome turned out than expected.
#[derive(Clone, Copy, Debug)]
pub struct Act {
    /// `pi_old(a)` - the probability the collecting policy gave this action.
    /// PPO's denominator, and what lets one rollout be reused for several
    /// update passes.
    pub old_prob: f32,
    /// Which option was taken, indexing the score vector.
    pub action: usize,
    /// The advantage: return-to-go against a baseline.
    pub advantage: f32,
}

/// What the reference policy would have done here, over this step's own
/// options. `None` when there is no anchor to hold to.
pub type Reference<'a> = Option<&'a [f32]>;

/// How far the policy has moved from the one that collected this step, as
/// `KL(pi_old || pi_new)` estimated from the single action taken.
///
/// Schulman's k3 estimator, `(r - 1) - ln r` with `r = pi_new(a) / pi_old(a)`:
/// unbiased, never negative, and far quieter than the `-ln r` estimator - a
/// stopping rule built on a quantity that can come out negative stops on
/// noise.
pub fn drift(scores: &[f32], act: &Act) -> f32 {
    let p = crate::loss::softmax(scores);
    let old = (act.old_prob as f64).clamp(1e-8, 1.0);
    let r = ((p[act.action] as f64) / old).max(1e-8);
    (((r - 1.0) - r.ln()) as f32).max(0.0)
}

/// Clipped policy-gradient loss and `dL/d(score)` for one control step, over a
/// categorical policy of any width.
///
/// **This is the case where maximizing reward is the right objective**, and it
/// is worth saying next to [`policy_loss`], where it is not. A probability
/// estimate is graded by a proper scoring rule because its optimum must be the
/// conditional RATE. A control policy is graded by return because its optimum
/// should be the best available ACTION - converging onto one option is the
/// goal, not a failure. The entropy term is what keeps it from getting there
/// before it has explored.
///
/// The other difference is the one that makes this genuinely reinforcement
/// learning: the environment RESPONDS. An action changes what state the next
/// decision is made from, so the policy shifts its own data distribution, and
/// PPO's trust region is doing real work rather than degenerating into a
/// regularizer on a fixed dataset.
///
/// `scores` are the raw option scores, one per option the caller supplied.
/// **The option set may differ at every step** - only `action` has to index
/// into this call's own.
pub fn choice_loss(scores: &[f32], act: &Act, cfg: &PolicyConfig) -> (f32, Vec<f32>) {
    choice_loss_anchored(scores, act, None, cfg)
}

/// [`choice_loss`], plus a pull toward what `reference` would have done.
///
/// The divergence is the full `KL(pi || reference)` over the option set, not
/// a one-sample estimate of it: the set is a handful of options wide and
/// every score is already in hand, so the exact quantity costs nothing and
/// carries none of the variance an estimator would add to a gradient that is
/// already noisy.
pub fn choice_loss_anchored(
    scores: &[f32],
    act: &Act,
    reference: Reference,
    cfg: &PolicyConfig,
) -> (f32, Vec<f32>) {
    assert!(act.action < scores.len(), "action {} is outside the {} options", act.action, scores.len());
    let p = crate::loss::softmax(scores);
    let a = act.action;
    let adv = act.advantage as f64;
    let old = (act.old_prob as f64).clamp(1e-8, 1.0);
    let ratio = p[a] as f64 / old;

    // --- clipped surrogate ---
    let (lo, hi) = (1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64);
    let clipped = ratio.clamp(lo, hi);
    let (unclipped_obj, clipped_obj) = (ratio * adv, clipped * adv);
    // PPO maximizes the MINIMUM of the two, so the loss is its negation. Where
    // the clipped branch wins the objective no longer depends on the ratio and
    // the gradient is exactly zero - that is the trust region, not an
    // approximation of one.
    let take_unclipped = unclipped_obj <= clipped_obj;
    let surrogate = -unclipped_obj.min(clipped_obj);

    // --- entropy bonus ---
    let ent: f64 = -p.iter().map(|&pi| (pi as f64) * (pi as f64).max(1e-12).ln()).sum::<f64>();
    let beta = cfg.entropy as f64;

    // --- the anchor ---
    let kappa = cfg.anchor as f64;
    let reference = reference.filter(|r| kappa > 0.0 && r.len() == scores.len());
    let kl: f64 = match reference {
        Some(r) => p
            .iter()
            .zip(r.iter())
            .map(|(&pi, &ri)| {
                let (pi, ri) = (pi as f64, (ri as f64).max(1e-12));
                if pi <= 0.0 { 0.0 } else { pi * (pi / ri).ln() }
            })
            .sum(),
        None => 0.0,
    };
    let loss = (surrogate - beta * ent + kappa * kl) as f32;

    // --- gradient, through the softmax jacobian dp_j/dz_i = p_j(d_ij - p_i) ---
    let mut d = vec![0.0f32; scores.len()];
    for (i, di) in d.iter_mut().enumerate() {
        let pi = p[i] as f64;
        let delta = if i == a { 1.0 } else { 0.0 };
        // d(ratio)/dz_i = ratio * (delta_ia - p_i).
        let d_surr = if take_unclipped { -adv * ratio * (delta - pi) } else { 0.0 };
        // dH/dz_i = -p_i (ln p_i + H), and the loss SUBTRACTS beta * H.
        let d_ent = beta * pi * ((pi.max(1e-12)).ln() + ent);
        // d(KL)/dz_i = p_i (ln(p_i / r_i) - KL), and the loss ADDS kappa * KL.
        let d_kl = match reference {
            Some(r) => {
                let ri = (r[i] as f64).max(1e-12);
                kappa * pi * ((pi.max(1e-12) / ri).ln() - kl)
            }
            None => 0.0,
        };
        *di = (d_surr + d_ent + d_kl) as f32;
    }
    (loss, d)
}

/// Discounted return-to-go for each step of one episode.
pub fn returns_to_go(rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rewards.len()];
    let mut acc = 0.0f32;
    for i in (0..rewards.len()).rev() {
        acc = rewards[i] + gamma * acc;
        out[i] = acc;
    }
    out
}

/// Centre and scale a batch of returns into advantages.
///
/// The baseline is the batch mean, which is the cheapest unbiased choice: a
/// baseline may depend on anything except the action taken, and a statistic of
/// the whole batch does not. Scaling by the standard deviation keeps the step
/// size meaningful when an environment's reward scale changes - without it,
/// tuning the learning rate means re-tuning it per reward function.
///
/// A batch with no spread returns all zeros rather than dividing by it: every
/// step did equally well, so there is nothing to push toward.
pub fn normalize(advantages: &mut [f32]) {
    if advantages.is_empty() {
        return;
    }
    let n = advantages.len() as f32;
    let mean = advantages.iter().sum::<f32>() / n;
    let var = advantages.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>() / n;
    let sd = var.sqrt();
    if sd < 1e-6 {
        advantages.fill(0.0);
        return;
    }
    for a in advantages.iter_mut() {
        *a = (*a - mean) / sd;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stopping rule is only as good as the quantity it stops on: zero
    /// when the policy has not moved, never negative, and growing with the
    /// distance travelled in either direction.
    #[test]
    fn the_drift_measures_travel_from_the_policy_that_collected_the_step() {
        let scores = [1.0f32, 0.0, 0.0];
        let p = crate::loss::softmax(&scores);

        // Collected by this very policy: no travel.
        let still = Act { old_prob: p[0], action: 0, advantage: 1.0 };
        assert!(drift(&scores, &still) < 1e-6, "{}", drift(&scores, &still));

        // Moved, in either direction, costs something - and the estimator is
        // never negative, which a stopping rule built on `-ln r` would be.
        let up = Act { old_prob: p[0] * 0.5, action: 0, advantage: 1.0 };
        let down = Act { old_prob: (p[0] * 2.0).min(1.0), action: 0, advantage: 1.0 };
        assert!(drift(&scores, &up) > 0.0);
        assert!(drift(&scores, &down) > 0.0);

        // And further is further.
        let far = Act { old_prob: p[0] * 0.1, action: 0, advantage: 1.0 };
        assert!(drift(&scores, &far) > drift(&scores, &up));
    }

    /// The anchor has to pull toward the reference and nowhere else: off by
    /// default, zero when the policy already agrees with the reference, and
    /// pushing the taken action's score DOWN when the policy has drifted onto
    /// something the reference thought unlikely.
    #[test]
    fn the_anchor_pulls_a_drifted_policy_back_and_is_otherwise_silent() {
        let act = Act { old_prob: 0.5, action: 0, advantage: 1.0 };
        let scores = [2.0f32, 0.0, 0.0];
        let here = crate::loss::softmax(&scores);

        // Off unless asked for, and then bit-identical to the unanchored form.
        let off = PolicyConfig { anchor: 0.0, ..PolicyConfig::default() };
        let (l0, g0) = choice_loss(&scores, &act, &off);
        let (l1, g1) = choice_loss_anchored(&scores, &act, Some(&here), &off);
        assert_eq!(l0, l1);
        assert_eq!(g0, g1);

        // Anchored to ITSELF costs nothing and bends nothing: KL(p || p) = 0.
        let on = PolicyConfig { anchor: 0.5, ..PolicyConfig::default() };
        let (l2, g2) = choice_loss_anchored(&scores, &act, Some(&here), &on);
        assert!((l2 - l0).abs() < 1e-5, "{l2} vs {l0}");
        for (a, b) in g2.iter().zip(&g0) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }

        // Anchored to a reference that preferred a DIFFERENT option: the loss
        // rises, and the gradient on the option the policy has drifted onto
        // is more positive than it was - which is the direction that lowers
        // its score.
        let elsewhere = crate::loss::softmax(&[0.0f32, 2.0, 0.0]);
        let (l3, g3) = choice_loss_anchored(&scores, &act, Some(&elsewhere), &on);
        assert!(l3 > l0, "diverging from the reference has to cost: {l3} vs {l0}");
        assert!(
            g3[0] > g0[0],
            "the drifted option's score should be pushed down: {} vs {}",
            g3[0],
            g0[0]
        );
    }
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
        let cfg = PolicyConfig { clip: 1e9, entropy: 0.0, gamma: 1.0, anchor: 0.0, target_kl: 0.02 };
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
        let cfg = PolicyConfig { clip: 1e9, entropy: 0.05, gamma: 0.9, anchor: 0.0, target_kl: 0.02 };
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
        let cfg = PolicyConfig { clip: 0.2, entropy: 0.0, gamma: 1.0, anchor: 0.0, target_kl: 0.02 };
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
        let cfg = PolicyConfig { clip: 1e9, entropy: 0.0, gamma: 0.5, anchor: 0.0, target_kl: 0.02 };
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
            let cfg = PolicyConfig { clip: 1e9, entropy: beta, gamma: 1.0, anchor: 0.0, target_kl: 0.02 };
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

    /// The control objective's gradient, over a width a fixed-head policy could
    /// not have, and in both PPO branches.
    #[test]
    fn the_choice_gradient_matches_finite_differences() {
        let cfg = PolicyConfig { clip: 0.2, entropy: 0.05, gamma: 1.0, anchor: 0.0, target_kl: 0.02 };
        for (z, act, what) in [
            (vec![0.3f32, -0.4, 0.9], Act { old_prob: 0.30, action: 0, advantage: 1.2 }, "in region, positive"),
            (vec![0.3f32, -0.4, 0.9], Act { old_prob: 0.30, action: 2, advantage: -0.8 }, "in region, negative"),
            // A collecting probability far from the current policy pushes the
            // ratio outside the trust region in each direction.
            (vec![3.0f32, -1.0, -1.0], Act { old_prob: 0.02, action: 0, advantage: 0.9 }, "clipped above"),
            (vec![-3.0f32, 1.0, 1.0], Act { old_prob: 0.90, action: 0, advantage: -0.9 }, "clipped below"),
            // Widths a fixed output layer would have to be rebuilt for.
            (vec![0.1f32; 7], Act { old_prob: 1.0 / 7.0, action: 5, advantage: 0.5 }, "seven options"),
            (vec![0.2f32, -0.1], Act { old_prob: 0.55, action: 1, advantage: 0.4 }, "two options"),
        ] {
            let (_, d) = choice_loss(&z, &act, &cfg);
            check(what, |s| choice_loss(s, &act, &cfg).0, &z, &d);
        }
    }

    /// A clipped ratio must contribute NO gradient - the trust region is the
    /// whole reason a rollout may be reused for several passes.
    #[test]
    fn a_clipped_choice_ratio_stops_the_gradient() {
        let cfg = PolicyConfig { clip: 0.2, entropy: 0.0, gamma: 1.0, anchor: 0.0, target_kl: 0.02 };
        let z = vec![4.0f32, 0.0, 0.0];
        let act = Act { old_prob: 0.02, action: 0, advantage: 1.0 };
        let (_, d) = choice_loss(&z, &act, &cfg);
        assert!(d.iter().all(|v| v.abs() < 1e-7), "the clipped branch produced a gradient: {d:?}");
        // ...and in the region it must not be zero, or the case above passes
        // for a function that never learns.
        let act = Act { old_prob: 0.33, action: 0, advantage: 1.0 };
        let (_, d) = choice_loss(&[0.1f32, 0.0, 0.0], &act, &cfg);
        assert!(d.iter().any(|v| v.abs() > 1e-4), "in-region gradient vanished: {d:?}");
    }

    /// **The control objective converges onto the BEST action** - which is the
    /// opposite of what `policy_loss` must do, and the difference is the point.
    ///
    /// A three-armed bandit whose middle arm pays best. A probability estimator
    /// graded by a proper scoring rule would settle on the payout RATES; a
    /// control policy graded by return should end up taking arm 1 almost
    /// always.
    #[test]
    fn the_control_optimum_is_the_best_action() {
        let payouts = [0.1f32, 0.9, 0.3];
        let cfg = PolicyConfig { clip: 0.2, entropy: 0.001, gamma: 1.0, anchor: 0.0, target_kl: 0.02 };
        let mut z = vec![0.0f32; 3];
        let mut rng = data::rng::Rng::new(8);
        for t in 0..8000 {
            let p = crate::loss::softmax(&z);
            // Sample an arm from the policy.
            let (mut u, mut a) = (rng.next_f32(), 2usize);
            for (i, &pi) in p.iter().enumerate() {
                if u < pi {
                    a = i;
                    break;
                }
                u -= pi;
            }
            let reward = if rng.next_f32() < payouts[a] { 1.0 } else { 0.0 };
            // Baseline: the policy's own expected payout, which depends on the
            // state and the policy but never on the action drawn.
            let expected: f32 = p.iter().zip(&payouts).map(|(&pi, &q)| pi * q).sum();
            let act = Act { old_prob: p[a], action: a, advantage: reward - expected };
            let (_, d) = choice_loss(&z, &act, &cfg);
            let lr = 0.5 / (1.0 + t as f32 / 2000.0);
            for (zi, di) in z.iter_mut().zip(&d) {
                *zi -= lr * di;
            }
        }
        let p = crate::loss::softmax(&z);
        assert!(p[1] > 0.9, "the policy did not commit to the best arm: {p:?}");
    }

    /// Returns-to-go must accumulate BACKWARD from the end, discounted. Getting
    /// the direction wrong credits an action with what happened before it.
    #[test]
    fn returns_accumulate_backward_from_the_end() {
        let r = [0.0f32, 0.0, 1.0];
        let g = returns_to_go(&r, 0.5);
        assert!((g[2] - 1.0).abs() < 1e-6, "terminal return {}", g[2]);
        assert!((g[1] - 0.5).abs() < 1e-6, "one step out {}", g[1]);
        assert!((g[0] - 0.25).abs() < 1e-6, "two steps out {}", g[0]);
        // Undiscounted, every step carries the whole episode's reward.
        assert_eq!(returns_to_go(&[1.0, 2.0, 3.0], 1.0), vec![6.0, 5.0, 3.0]);
    }

    /// Advantages must be centred, so a batch where everything went equally
    /// well pushes nowhere.
    #[test]
    fn advantages_are_centred_and_a_flat_batch_pushes_nowhere() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0];
        normalize(&mut a);
        let mean: f32 = a.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "not centred: {a:?}");
        let sd = (a.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((sd - 1.0).abs() < 1e-5, "not scaled: {a:?}");
        // No spread: every step did equally well and there is nothing to learn
        // from, which must not become a division by zero.
        let mut flat = vec![2.5f32; 5];
        normalize(&mut flat);
        assert!(flat.iter().all(|&v| v == 0.0), "a flat batch produced {flat:?}");
        let mut empty: Vec<f32> = Vec::new();
        normalize(&mut empty);
    }

    /// Fitting a soft target must reproduce the target, not its argmax: a
    /// model trained on a 0.7 label should say 0.7.    /// Fitting a soft target must reproduce the target, not its argmax: a
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
