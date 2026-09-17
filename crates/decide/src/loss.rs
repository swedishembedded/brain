// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision objective, and the softmax it is built on.
//!
//! Host arithmetic, deliberately. A question has at most a few hundred options
//! and this is the last step of a six-layer encoder, so the cost does not
//! register - and keeping it here is what lets the option count be genuinely
//! defined at request time, since no kernel has to have an opinion about it.
//!
//! One objective with two dials:
//!
//! ```text
//! L = (1 - lambda) * focal(gamma) + lambda * brier
//! ```
//!
//! `gamma = 0` makes the focal term EXACTLY cross-entropy, so `(0, 0)` is the
//! plain-CE control rather than an approximation of it - which is what makes
//! the calibration sweep a comparison rather than four unrelated runs.
//!
//! Why these two. Cross-entropy is a proper scoring rule and is calibrated on
//! the training set, then goes overconfident on held-out data through the
//! generalization gap. Focal loss is improper but empirically ends up better
//! calibrated at test time, because it under-confidences the training set by
//! exactly the amount the gap re-inflates. Brier is a proper scoring rule that
//! penalizes the whole distribution rather than only the gold option's mass.
//! Which combination wins is measured, not assumed.

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
        LossConfig { gamma: 0.0, lambda: 0.0 }
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
    let mut p: Vec<f32> = scores.iter().map(|&z| ((z - max) as f64).exp() as f32).collect();
    let sum: f64 = p.iter().map(|&v| v as f64).sum();
    // A zero sum is unreachable after the max shift (the gold term is exp(0)),
    // so this guards a corrupt input rather than a normal path.
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for v in &mut p {
        *v = (*v as f64 * inv) as f32;
    }
    p
}

/// Loss and `dL/d(score)` for one question whose correct option is `gold`.
///
/// Returned together because they share the softmax: computing them apart
/// would run it twice and invite the two to drift.
pub fn decision_loss(scores: &[f32], gold: usize, cfg: &LossConfig) -> (f32, Vec<f32>) {
    assert!(gold < scores.len(), "gold option {gold} is outside the {} supplied", scores.len());
    let p = softmax(scores);
    let n = p.len();
    let t = (p[gold] as f64).max(1e-12);
    let (g, lam) = (cfg.gamma as f64, cfg.lambda as f64);

    // --- focal / cross-entropy ---
    // L_f = -(1-t)^gamma * log t. At gamma = 0 both the value and the
    // derivative collapse to cross-entropy's, with no special case.
    let one_minus = (1.0 - t).max(0.0);
    let focal = -one_minus.powf(g) * t.ln();
    // dL_f/dt, then chained through the softmax below.
    let dfocal_dt = if g == 0.0 {
        -1.0 / t
    } else {
        g * one_minus.powf(g - 1.0) * t.ln() - one_minus.powf(g) / t
    };

    // --- Brier: sum over ALL options, not just the gold one ---
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

    // --- gradient through the softmax jacobian ---
    // dp_j/dz_i = p_j (delta_ij - p_i).
    // Focal touches only p_gold, so its chain is one term.
    // Brier touches every p_j, so its chain carries the shared sum.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ce_reference(scores: &[f32], gold: usize) -> (f32, Vec<f32>) {
        let p = softmax(scores);
        let loss = -(p[gold].max(1e-12)).ln();
        let d: Vec<f32> = p.iter().enumerate().map(|(i, &pi)| pi - if i == gold { 1.0 } else { 0.0 }).collect();
        (loss, d)
    }

    /// The control has to BE cross-entropy, not resemble it: the sweep's whole
    /// point is that the four settings differ in one dial each.
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

    /// Finite differences over the scores, for every setting in the sweep -
    /// the gradient is hand-derived through the softmax jacobian and that is
    /// exactly the kind of derivation that is plausibly wrong.
    #[test]
    fn the_gradient_matches_finite_differences() {
        let scores = [0.4f32, -1.2, 2.0, 0.1, -0.3];
        let settings = [
            LossConfig::cross_entropy(),
            LossConfig::focal(3.0),
            LossConfig::cross_entropy().with_brier(0.5),
            LossConfig::focal(3.0).with_brier(0.5),
        ];
        for cfg in settings {
            for gold in 0..scores.len() {
                let (_, d) = decision_loss(&scores, gold, &cfg);
                for i in 0..scores.len() {
                    let eps = 1e-3f32;
                    let mut up = scores;
                    let mut dn = scores;
                    up[i] += eps;
                    dn[i] -= eps;
                    let num = (decision_loss(&up, gold, &cfg).0 - decision_loss(&dn, gold, &cfg).0) / (2.0 * eps);
                    let tol = 2e-3 + 2e-2 * d[i].abs().max(num.abs());
                    assert!(
                        (d[i] - num).abs() <= tol,
                        "gamma {} lambda {} gold {gold} i {i}: analytic {} vs numeric {num}",
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
        for cfg in [LossConfig::cross_entropy(), LossConfig::focal(3.0).with_brier(0.5)] {
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
        assert!(focal < ce, "focal {focal} should be below CE {ce} on a confident example");
    }
}
