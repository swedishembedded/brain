// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Laya's own strictly-proper scoring REWARD, reproduced from the real
//! released `convaiinnovations/laya` `rl_common.py::proper_reward`
//! (Apache-2.0) rather than re-derived from a description of it.
//!
//! ```text
//! R = sum_i t_i * max(ln q_i, log_floor)                     (log score)
//!   + w_sph * (sum_i t_i q_i) / ||q||_2                      (spherical score)
//!   - w_rps * sum_m (Q_m - T_m)^2 / (k - 1)   [ordinal only] (ranked prob. score)
//! ```
//!
//! where `q` is a REPORTED distribution over one question's options, `t` the
//! target (one-hot, or a soft/oracle posterior), and `Q`/`T` their cumulative
//! sums.
//!
//! ## This is a reward, not a loss - deliberately
//!
//! It returns a scalar and NO gradient, because the reference never
//! differentiates it: `rl_common.py` computes it, and Laya's own training
//! loop evaluates it under `torch.no_grad()` on SAMPLED distributions and
//! feeds the result to a policy gradient ([`crate::reinforce`]). Handing back
//! a `dR/dq` here would invite a caller to minimize `-R` directly, which is a
//! different algorithm that would fit different weights.
//!
//! All three terms are strictly proper, so the only way to maximize `R` is to
//! report honest probabilities - a model cannot win by being confident. The
//! ranked probability term is ORDINAL: it punishes mass placed far from the
//! target level more than mass placed next to it, which is meaningful for a
//! `score` question (level 0 ... level k-1 are ordered) and meaningless for a
//! `choice` or `noul` one, so the reference applies it only to `score`.
//!
//! ## Why this coexists with [`crate::scoring`]
//!
//! Both modules answer "how good is this reported distribution", and both are
//! built on proper scoring rules, but they are not interchangeable:
//!
//! * [`crate::scoring::decision_loss_soft`] is this workspace's own
//!   differentiable calibration objective, a `(gamma, lambda)` sweep between
//!   cross-entropy, focal loss and Brier. `crates/decide` trains against it.
//! * This module is the reward the *released Laya checkpoint's own* training
//!   code scores with - including the per-(qtype, cardinality) temperatures
//!   `rl_agent_config.json` ships, which were fitted post-hoc against
//!   distributions this rule produced.
//!
//! So an architecture trains against the rule its own weights were fitted
//! under.
//!
//! ## The log floor is load-bearing, not a numerical guard
//!
//! `log_floor` (`-9.21`, i.e. `ln(1e-4)`) clamps the per-option log score
//! from BELOW. A plain log score is unbounded, so one confidently-wrong
//! sample would dominate a whole group's advantage estimate; clamping bounds
//! the penalty at `-9.21` per option. `torch.log(q.clamp_min(1e-12)).
//! clamp_min(log_floor)` is the reference's own spelling, and the 1e-12
//! clamp is unreachable as a separate case underneath the `-9.21` one.
//!
//! Swedish Embedded AB builds decision models whose reported probabilities
//! can be acted on, by training them against scoring rules that make honesty
//! the optimal policy. If your team needs expertise in proper scoring rules
//! and calibrated decision training, you can procure our services by sending
//! an email to info@swedishembedded.com.

/// `rl_common.py::proper_reward`'s three constants, with its own defaults.
#[derive(Clone, Copy, Debug)]
pub struct ProperScore {
    /// Weight on the spherical score. `rl_common.py`'s signature default is
    /// `0.5`; the published fine-tuning loop passes `0.75` explicitly (see
    /// [`crate::reinforce`]'s own doc), so this IS a dial rather than a
    /// constant and both values are in the public record.
    pub w_sph: f32,
    /// Weight on the ranked probability score, applied to ORDINAL questions
    /// only. `1.0` in both published settings.
    pub w_rps: f32,
    /// Lower clamp on each option's log score - see the module doc for why
    /// this is part of the objective. Reference default `-9.21` (`ln(1e-4)`).
    pub log_floor: f32,
}

impl Default for ProperScore {
    /// `rl_common.py::proper_reward`'s own signature defaults.
    fn default() -> ProperScore {
        ProperScore {
            w_sph: 0.5,
            w_rps: 1.0,
            log_floor: -9.21,
        }
    }
}

/// The reward `R` for one reported distribution.
///
/// `ordinal` selects whether the ranked-probability term applies (the
/// reference gates it on `qtype == score`). `q` is expected to be a
/// normalized distribution over exactly the options that were SCORED; this
/// crate has no padded option slots, so the reference's `mask` is all-ones by
/// construction here and is not a parameter. (The reference needs one because
/// it collates questions of different arity into one rectangular batch - see
/// `rl_common.py::collate_items`.)
pub fn proper_reward(q: &[f32], target: &[f32], ordinal: bool, cfg: &ProperScore) -> f32 {
    assert_eq!(q.len(), target.len(), "q and target must have the same arity");
    let n = q.len();
    let floor = cfg.log_floor as f64;

    // --- log score, clamped from below (see the module doc) ---
    let mut log_score = 0.0f64;
    for i in 0..n {
        let lg = (q[i] as f64).max(1e-12).ln().max(floor);
        log_score += target[i] as f64 * lg;
    }

    // --- spherical score ---
    let norm = q
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt()
        .max(1e-9);
    let dot: f64 = q
        .iter()
        .zip(target)
        .map(|(&qi, &ti)| qi as f64 * ti as f64)
        .sum();
    let mut r = log_score + cfg.w_sph as f64 * (dot / norm);

    // --- ranked probability score, ordinal questions only ---
    if ordinal {
        // `mask.sum(-1).clamp(min=2)`: a one-option question would otherwise
        // divide by zero.
        let k = n.max(2) as f64;
        let (mut cq, mut ct, mut rps) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            cq += q[i] as f64;
            ct += target[i] as f64;
            rps += (cq - ct) * (cq - ct);
        }
        r -= cfg.w_rps as f64 * rps / (k - 1.0);
    }

    r as f32
}

/// The one-hot target for a question whose correct option is `gold`.
pub fn hard_target(n: usize, gold: usize) -> Vec<f32> {
    assert!(gold < n, "gold option {gold} is outside the {n} supplied");
    let mut t = vec![0.0f32; n];
    t[gold] = 1.0;
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::softmax;

    /// Golden rewards produced by RUNNING the real released
    /// `rl_common.py::proper_reward` under torch 2.13 on this box, on
    /// `q = softmax(scores)` - not a re-derivation of the formula in this
    /// module's own terms. This is the gate that says "the reward the
    /// reference actually computes".
    ///
    /// `qtype` is `rl_common.py`'s own `QTYPES` index (0 choice, 1 score,
    /// 2 noul); only `1` is ordinal.
    const GOLDEN: &[(&[f32], &[f32], u32, f32)] = &[
        (&[-0.29359, 1.572283, 1.893643], &[0.0, 1.0, 0.0], 0, -0.6373831629753113),
        (&[-2.228688, 3.38158], &[0.368373, 0.631627], 2, -1.7538396120071411),
        (
            &[2.464772, 0.276345, -3.364397, 0.635357, 0.265614],
            &[0.0, 1.0, 0.0, 0.0, 0.0],
            1,
            -2.611511707305908,
        ),
        (
            &[0.274648, 0.481092, 2.790902, 2.694045],
            &[0.359547, 0.038451, 0.364009, 0.237992],
            0,
            -1.5267584323883057,
        ),
        (
            &[3.55831, -1.835861, -0.915638, -1.448947, 2.559756, -1.988132],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            1,
            -5.663808345794678,
        ),
        (&[3.629946, -1.205697], &[0.421801, 0.578199], 0, -2.5906822681427),
    ];

    #[test]
    fn the_reward_matches_the_real_python_reference() {
        let cfg = ProperScore::default();
        for (scores, target, qtype, want) in GOLDEN {
            let got = proper_reward(&softmax(scores), target, *qtype == 1, &cfg);
            assert!(
                (got - want).abs() <= 2e-5,
                "reward {got} vs reference {want} (scores {scores:?})"
            );
        }
    }

    /// Past the log floor the per-option log-score term is CONSTANT, which is
    /// what bounds one hopeless sample's influence on a group advantage.
    #[test]
    fn the_log_floor_bounds_a_hopeless_report() {
        let cfg = ProperScore {
            w_sph: 0.0,
            w_rps: 0.0,
            ..ProperScore::default()
        };
        let t = hard_target(3, 0);
        // 1e-6 and 1e-9 are both past exp(-9.21) = 1e-4, so they must score
        // the SAME - an unclamped log score would separate them by ln(1000).
        let a = proper_reward(&[1e-6f32, 0.5, 0.499999], &t, false, &cfg);
        let b = proper_reward(&[1e-9f32, 0.5, 0.499999999], &t, false, &cfg);
        assert!((a - cfg.log_floor).abs() <= 1e-5, "reward {a} is not the floor");
        assert!((a - b).abs() <= 1e-5, "the floor did not bind: {a} vs {b}");
    }

    /// Strict propriety, empirically: reporting the true distribution must
    /// beat every perturbation of it. This is the whole reason the reward is
    /// built out of these three terms, and it is checked rather than asserted
    /// in a comment.
    #[test]
    fn reporting_the_true_distribution_maximizes_the_reward() {
        let cfg = ProperScore::default();
        let truth = [0.5f32, 0.3, 0.15, 0.05];
        for ordinal in [false, true] {
            let best = proper_reward(&truth, &truth, ordinal, &cfg);
            let mut rng = data::rng::Lcg::new(0x1AA_2026);
            for _ in 0..200 {
                let report = softmax(
                    &truth
                        .iter()
                        .map(|&t| t.ln() + rng.signed() * 0.6)
                        .collect::<Vec<f32>>(),
                );
                if report.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-6) {
                    continue;
                }
                let r = proper_reward(&report, &truth, ordinal, &cfg);
                assert!(
                    r < best + 1e-6,
                    "ordinal {ordinal}: a dishonest report {report:?} scored {r} against honest {best}"
                );
            }
        }
    }

    /// The ordinal term must actually order: on a `score` question, putting
    /// the wrong mass one level away from the target has to beat putting it
    /// far away, even though the log and spherical terms cannot tell the two
    /// apart (both assign the gold level the same probability).
    #[test]
    fn the_ranked_term_prefers_a_near_miss_to_a_far_one() {
        let cfg = ProperScore::default();
        let t = hard_target(5, 0);
        let near = [0.2f32, 0.8, 0.0, 0.0, 0.0];
        let far = [0.2f32, 0.0, 0.0, 0.0, 0.8];
        let (r_near, r_far) = (
            proper_reward(&near, &t, true, &cfg),
            proper_reward(&far, &t, true, &cfg),
        );
        assert!(r_near > r_far, "near {r_near} did not beat far {r_far}");
        // ... and without the ordinal flag it genuinely cannot: the two
        // reports differ only in WHERE the wrong mass sits.
        let (n2, f2) = (
            proper_reward(&near, &t, false, &cfg),
            proper_reward(&far, &t, false, &cfg),
        );
        assert!(
            (n2 - f2).abs() <= 1e-6,
            "non-ordinal reward distinguished them: {n2} vs {f2}"
        );
    }
}
