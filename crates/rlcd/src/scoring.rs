// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The calibration objective, and the softmax it is built on.
//!
//! Host arithmetic, deliberately - a question has at most a few hundred
//! options, so the cost never registers against a model forward pass, and
//! keeping it here is what lets any decision model (an encoder-plus-head like
//! `brain-decide`, or a spliced head like Laya's) drive the same objective
//! through nothing more than its own `&[f32]` option scores.
//!
//! One objective with two dials:
//!
//! ```text
//! L = (1 - lambda) * focal(gamma) + lambda * brier
//! ```
//!
//! `gamma = 0` makes the focal term EXACTLY cross-entropy, so `(0, 0)` is the
//! plain-CE control rather than an approximation of it - which is what makes
//! a calibration sweep a comparison rather than four unrelated runs.
//!
//! Why these two. Cross-entropy is a proper scoring rule and is calibrated on
//! the training set, then goes overconfident on held-out data through the
//! generalization gap. Focal loss is improper but empirically ends up better
//! calibrated at test time, because it under-confidences the training set by
//! exactly the amount the gap re-inflates. Brier is a proper scoring rule that
//! penalizes the whole distribution rather than only the correct option's
//! mass. Which combination wins is measured, not assumed.
//!
//! ## Soft targets
//!
//! [`decision_loss_soft`] takes a full target **distribution** over the
//! options, not a single gold index - this is what lets a model train
//! directly against an exact oracle posterior (`P(fault) = 2/3`, not
//! "option 2"). [`decision_loss`] is the one-hot special case, kept as a
//! convenience wrapper so every existing hard-label caller is unchanged.
//!
//! The Brier term already summed over every option
//! (`sum_i (p_i - y_i)^2`), so it generalizes to a soft target by replacing
//! the indicator `y_i` with `target[i]` - no change in form. The focal/CE
//! term did not: the hard-label version only ever evaluated at the gold
//! option, `t = p[gold]`. Its soft generalization is defined here as
//!
//! ```text
//! L_focal = sum_i target_i * (1 - p_i)^gamma * (-ln p_i)
//! ```
//!
//! which reduces to the hard-label formula exactly when `target` is one-hot
//! (only the gold term survives), and to soft cross-entropy exactly at
//! `gamma = 0` (`sum_i target_i * -ln p_i`).

/// The objective's two dials.
#[derive(Clone, Copy, Debug)]
pub struct LossConfig {
    /// Focal exponent. `0.0` is cross-entropy exactly.
    pub gamma: f32,
    /// Weight on the Brier term, in `[0, 1]`.
    pub lambda: f32,
}

impl Default for LossConfig {
    /// Cross-entropy - the control every other setting is measured against.
    fn default() -> LossConfig {
        LossConfig {
            gamma: 0.0,
            lambda: 0.0,
        }
    }
}

impl LossConfig {
    pub fn cross_entropy() -> LossConfig {
        LossConfig::default()
    }

    /// The setting the calibration literature reports as the best single
    /// choice for test-time calibration at equal accuracy.
    pub fn focal(gamma: f32) -> LossConfig {
        LossConfig { gamma, lambda: 0.0 }
    }

    pub fn with_brier(mut self, lambda: f32) -> LossConfig {
        self.lambda = lambda;
        self
    }
}

/// Numerically stable softmax over one question's option scores.
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = scores
        .iter()
        .map(|&z| ((z - max) as f64).exp() as f32)
        .collect();
    let sum: f64 = p.iter().map(|&v| v as f64).sum();
    // A zero sum is unreachable after the max shift (at least one term is
    // exp(0)), so this guards a corrupt input rather than a normal path.
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for v in &mut p {
        *v = (*v as f64 * inv) as f32;
    }
    p
}

/// Softmax and its logarithm together, in f64, from one max-shifted pass.
///
/// The log is the half that stays honest when an option saturates: `p`
/// underflows to zero at a logit gap around 745, while `ln p` is merely
/// -745 and still exact. Anything needing `ln p` from `p` alone has to
/// invent a floor, and a floor is what silently flattens a gradient.
fn softmax_with_log(scores: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let shifted: Vec<f64> = scores.iter().map(|&z| z as f64 - max).collect();
    let lse = shifted.iter().map(|&s| s.exp()).sum::<f64>().ln();
    let logp: Vec<f64> = shifted.iter().map(|&s| s - lse).collect();
    let p: Vec<f64> = logp.iter().map(|&l| l.exp()).collect();
    (p, logp)
}

/// `p * d/dp` of the per-option focal term `(1 - p)^gamma * (-ln p)`.
///
/// Returned as the PRODUCT rather than as the slope alone, because the
/// slope carries a `1/p` that diverges as an option saturates while every
/// downstream use multiplies by `p` again. Forming the product cancels that
/// analytically: at `gamma = 0` it is exactly `-1` at any `p`, so plain
/// cross-entropy keeps its `p - target` gradient however confident the
/// prediction got. Clamping `p` instead - which is what this did - makes
/// the cancellation inexact and drives the gradient to zero precisely for
/// the confidently WRONG predictions that most need one.
fn focal_slope_times_p(p: f64, logp: f64, gamma: f64) -> f64 {
    if gamma == 0.0 {
        return -1.0;
    }
    let one_minus = (1.0 - p).max(0.0);
    // `(1-p)^(gamma-1) * p*ln p` tends to 0 as `p -> 1` for every
    // `gamma > 0` - `ln p` vanishes faster than the power can diverge - so
    // taking the limit here is what keeps `gamma < 1` from forming inf * 0.
    let ramp = if one_minus > 0.0 { one_minus.powf(gamma - 1.0) * p * logp } else { 0.0 };
    gamma * ramp - one_minus.powf(gamma)
}

/// Loss and `dL/d(score)` for one question against a full target
/// **distribution** over its options (need not be one-hot; expected to sum
/// to 1 - the caller owns that invariant, the same way it already owns
/// `scores` being finite).
///
/// Returned together because they share the softmax: computing them apart
/// would run it twice and invite the two to drift.
pub fn decision_loss_soft(scores: &[f32], target: &[f32], cfg: &LossConfig) -> (f32, Vec<f32>) {
    assert_eq!(
        scores.len(),
        target.len(),
        "scores and target must have the same arity"
    );
    let (p, logp) = softmax_with_log(scores);
    let n = p.len();
    let (g, lam) = (cfg.gamma as f64, cfg.lambda as f64);

    // --- soft focal / cross-entropy: L = sum_i target_i * (1-p_i)^g * -ln(p_i) ---
    // `-logp[i]` rather than `-p[i].ln()`: the two agree wherever `p` is
    // representable and only the former survives where it is not.
    let mut focal = 0.0f64;
    let mut psl = vec![0.0f64; n]; // p_i * d/dp_i of the per-option focal term
    for i in 0..n {
        let one_minus = (1.0 - p[i]).max(0.0);
        focal += target[i] as f64 * one_minus.powf(g) * (-logp[i]);
        psl[i] = focal_slope_times_p(p[i], logp[i], g);
    }

    // --- Brier: sum over ALL options against the full target distribution ---
    let brier: f64 = p
        .iter()
        .zip(target)
        .map(|(&pi, &ti)| {
            let d = pi - ti as f64;
            d * d
        })
        .sum();

    let loss = ((1.0 - lam) * focal + lam * brier) as f32;

    // --- gradient through the softmax jacobian: dp_j/dz_i = p_j(delta_ij - p_i) ---
    // Each objective term is `sum_k f_k(p_k)`, so dL/dp_k = target_k * slope_k
    // for the focal half; chaining through the jacobian collapses the sum
    // over k into one dot product per output index. Both places the slope
    // appears it is already multiplied by a `p`, so both read `psl`
    // directly and the `1/p` never has to be formed - see
    // [`focal_slope_times_p`].
    let focal_dot: f64 = (0..n).map(|k| target[k] as f64 * psl[k]).sum();
    let brier_dot: f64 = p
        .iter()
        .zip(target)
        .map(|(&pj, &tj)| 2.0 * (pj - tj as f64) * pj)
        .sum();

    let mut d = vec![0.0f32; n];
    for i in 0..n {
        let pi = p[i];
        let d_focal = target[i] as f64 * psl[i] - pi * focal_dot;
        let d_brier = 2.0 * pi * (pi - target[i] as f64) - pi * brier_dot;
        d[i] = ((1.0 - lam) * d_focal + lam * d_brier) as f32;
    }
    (loss, d)
}

/// Loss and `dL/d(score)` for one question whose correct option is `gold` -
/// the one-hot special case of [`decision_loss_soft`], kept so every
/// hard-label caller needs no change.
pub fn decision_loss(scores: &[f32], gold: usize, cfg: &LossConfig) -> (f32, Vec<f32>) {
    assert!(
        gold < scores.len(),
        "gold option {gold} is outside the {} supplied",
        scores.len()
    );
    let mut target = vec![0.0f32; scores.len()];
    target[gold] = 1.0;
    decision_loss_soft(scores, &target, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    fn ce_reference(scores: &[f32], gold: usize) -> (f32, Vec<f32>) {
        let p = softmax(scores);
        let loss = -(p[gold].max(1e-12)).ln();
        let d: Vec<f32> = p
            .iter()
            .enumerate()
            .map(|(i, &pi)| pi - if i == gold { 1.0 } else { 0.0 })
            .collect();
        (loss, d)
    }

    /// The pre-lift formula this module replaced, transcribed verbatim as a
    /// private reference so the M1 gate below has something independent to
    /// check the new soft-target formula against on a one-hot target.
    fn old_hard_label_reference(scores: &[f32], gold: usize, cfg: &LossConfig) -> (f32, Vec<f32>) {
        let p = softmax(scores);
        let n = p.len();
        let t = (p[gold] as f64).max(1e-12);
        let (g, lam) = (cfg.gamma as f64, cfg.lambda as f64);

        let one_minus = (1.0 - t).max(0.0);
        let focal = -one_minus.powf(g) * t.ln();
        let dfocal_dt = if g == 0.0 {
            -1.0 / t
        } else {
            g * one_minus.powf(g - 1.0) * t.ln() - one_minus.powf(g) / t
        };

        let brier: f64 = p
            .iter()
            .enumerate()
            .map(|(i, &pi)| {
                let y = if i == gold { 1.0 } else { 0.0 };
                let d = pi as f64 - y;
                d * d
            })
            .sum();

        let loss = ((1.0 - lam) * focal + lam * brier) as f32;

        let brier_dot: f64 = p
            .iter()
            .enumerate()
            .map(|(j, &pj)| {
                let y = if j == gold { 1.0 } else { 0.0 };
                2.0 * (pj as f64 - y) * pj as f64
            })
            .sum();
        let mut d = vec![0.0f32; n];
        for (i, di) in d.iter_mut().enumerate() {
            let pi = p[i] as f64;
            let delta = if i == gold { 1.0 } else { 0.0 };
            let d_focal = dfocal_dt * (p[gold] as f64) * (delta - pi);
            let y = if i == gold { 1.0 } else { 0.0 };
            let d_brier = 2.0 * pi * (pi - y) - pi * brier_dot;
            *di = ((1.0 - lam) * d_focal + lam * d_brier) as f32;
        }
        (loss, d)
    }

    /// M1's gate: the lift must not change behavior. On a one-hot target the
    /// new general soft-target formula has to agree with the exact formula it
    /// replaced, loss and gradient, over randomized scores and every
    /// `(gamma, lambda)` setting the calibration sweep exercises - not just
    /// the handful of fixed vectors the other tests below use.
    #[test]
    fn soft_path_matches_the_pre_lift_hard_label_formula() {
        let mut rng = Lcg::new(20_260_922);
        let settings = [
            LossConfig::cross_entropy(),
            LossConfig::focal(3.0),
            LossConfig::cross_entropy().with_brier(0.5),
            LossConfig::focal(2.5).with_brier(0.7),
        ];
        for _ in 0..200 {
            let n = 2 + (rng.next_u32() as usize % 6);
            let scores = rng.vec_scaled(n, 3.0);
            let gold = rng.next_u32() as usize % n;
            for cfg in settings {
                let (l_new, d_new) = decision_loss(&scores, gold, &cfg);
                let (l_old, d_old) = old_hard_label_reference(&scores, gold, &cfg);
                assert!(
                    (l_new - l_old).abs() <= 1e-5,
                    "loss {l_new} vs {l_old} (n={n}, gold={gold}, cfg={cfg:?})"
                );
                for (a, b) in d_new.iter().zip(&d_old) {
                    assert!(
                        (a - b).abs() <= 1e-5,
                        "grad {a} vs {b} (n={n}, gold={gold}, cfg={cfg:?})"
                    );
                }
            }
        }
    }

    /// The control has to BE cross-entropy, not resemble it: the sweep's
    /// whole point is that the four settings differ in one dial each.
    #[test]
    fn gamma_zero_lambda_zero_is_cross_entropy() {
        let scores = [0.4f32, -1.2, 2.0, 0.1, -0.3];
        for gold in 0..scores.len() {
            let (l, d) = decision_loss(&scores, gold, &LossConfig::cross_entropy());
            let (rl, rd) = ce_reference(&scores, gold);
            assert!((l - rl).abs() <= 1e-6, "loss {l} vs {rl}");
            for (a, b) in d.iter().zip(&rd) {
                assert!((a - b).abs() <= 1e-6, "grad {a} vs {b}");
            }
        }
    }

    /// Soft cross-entropy's gradient has the textbook closed form `p -
    /// target`, for a genuinely non-one-hot target - the case the pre-lift
    /// formula could not express at all.
    #[test]
    fn soft_cross_entropy_gradient_is_p_minus_target() {
        let scores = [0.4f32, -1.2, 2.0, 0.1, -0.3];
        let target = [0.5f32, 0.2, 0.1, 0.1, 0.1];
        let p = softmax(&scores);
        let (_, d) = decision_loss_soft(&scores, &target, &LossConfig::cross_entropy());
        for i in 0..scores.len() {
            let expect = p[i] - target[i];
            assert!(
                (d[i] - expect).abs() <= 1e-5,
                "grad[{i}] {} vs p-target {expect}",
                d[i]
            );
        }
    }

    /// A saturated prediction must still produce cross-entropy's real
    /// gradient. This is the case a finite-difference check cannot see: the
    /// clamp that flattens the gradient flattens the LOSS by the same
    /// amount, so numeric and analytic agree with each other and are both
    /// wrong. The reference here is the closed form `p - target`, which is
    /// exact at every logit gap.
    #[test]
    fn cross_entropy_keeps_its_gradient_when_a_prediction_saturates() {
        let target = [0.8f32, 0.2];
        // 27.6 nats is where p dips under the old 1e-12 floor; the last two
        // are far past anything f32 can represent as a normal number.
        for gap in [10.0f32, 27.0, 30.0, 100.0, 400.0] {
            let scores = [-gap, 0.0];
            let p = softmax(&scores);
            let (loss, d) = decision_loss_soft(&scores, &target, &LossConfig::cross_entropy());
            for i in 0..2 {
                let expect = p[i] - target[i];
                assert!(
                    (d[i] - expect).abs() <= 1e-5,
                    "gap {gap}: grad[{i}] {} should be p-target {expect}",
                    d[i]
                );
            }
            // The loss itself must keep growing with the gap rather than
            // saturating at -ln(1e-12): a confidently wrong prediction is
            // not equally wrong at 30 nats and at 400.
            assert!(loss.is_finite() && loss >= 0.8 * gap * 0.99, "gap {gap}: loss {loss} saturated");
        }
    }

    /// Finite differences over the scores, for every setting in the sweep AND
    /// a genuinely soft target - the gradient is hand-derived through the
    /// softmax jacobian and that is exactly the kind of derivation that is
    /// plausibly wrong.
    #[test]
    fn the_gradient_matches_finite_differences() {
        let scores = [0.4f32, -1.2, 2.0, 0.1, -0.3];
        let targets: [[f32; 5]; 2] = [[0.0, 0.0, 1.0, 0.0, 0.0], [0.4, 0.1, 0.2, 0.2, 0.1]];
        let settings = [
            LossConfig::cross_entropy(),
            LossConfig::focal(3.0),
            LossConfig::cross_entropy().with_brier(0.5),
            LossConfig::focal(3.0).with_brier(0.5),
        ];
        for cfg in settings {
            for target in &targets {
                let (_, d) = decision_loss_soft(&scores, target, &cfg);
                for i in 0..scores.len() {
                    let eps = 1e-3f32;
                    let mut up = scores;
                    let mut dn = scores;
                    up[i] += eps;
                    dn[i] -= eps;
                    let num = (decision_loss_soft(&up, target, &cfg).0
                        - decision_loss_soft(&dn, target, &cfg).0)
                        / (2.0 * eps);
                    let tol = 2e-3 + 2e-2 * d[i].abs().max(num.abs());
                    assert!(
                        (d[i] - num).abs() <= tol,
                        "gamma {} lambda {} target {target:?} i {i}: analytic {} vs numeric {num}",
                        cfg.gamma,
                        cfg.lambda,
                        d[i]
                    );
                }
            }
        }
    }

    /// The gradient of a softmax-normalized objective sums to zero: shifting
    /// every score by the same amount cannot change any probability.
    #[test]
    fn the_gradient_is_shift_invariant() {
        let scores = [0.4f32, -1.2, 2.0, 0.1];
        for cfg in [
            LossConfig::cross_entropy(),
            LossConfig::focal(3.0).with_brier(0.5),
        ] {
            let (_, d) = decision_loss(&scores, 2, &cfg);
            let sum: f32 = d.iter().sum();
            assert!(sum.abs() <= 1e-5, "gradient sums to {sum}, not 0");
        }
    }

    /// Focal loss must be SMALLER than cross-entropy on an already-confident
    /// example - that down-weighting is the entire mechanism.
    #[test]
    fn focal_down_weights_a_confident_example() {
        let confident = [6.0f32, 0.0, 0.0];
        let ce = decision_loss(&confident, 0, &LossConfig::cross_entropy()).0;
        let focal = decision_loss(&confident, 0, &LossConfig::focal(3.0)).0;
        assert!(
            focal < ce,
            "focal {focal} should be below CE {ce} on a confident example"
        );
    }
}
