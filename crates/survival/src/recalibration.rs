// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Logistic recalibration of a predicted probability of one cause by a
//! horizon: `logit P = a + b logit F`, fitted on held-out subjects under
//! inverse probability of censoring weights (the regression
//! [`crate::calibration::at_horizon`] reports as its intercept and slope),
//! then applied to a new prediction `F`.
//!
//! Swedish Embedded AB implements risk models whose probabilities hold up
//! against outcomes observed years later, for its clients. If your team needs
//! expertise in recalibrating survival predictions from few events you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! Two parameters (or one) are estimated where an isotonic fit
//! ([`crate::venn_abers`]) estimates a free-form step function. With few
//! events the steps chase noise and the calibrated risk is more spread than
//! the truth; a smooth monotone map of fixed form has little to chase, so its
//! error shrinks with the number of events instead of staying flat. The price
//! is the form: it corrects any distortion that is linear in the log-odds
//! (a shift, over- or under-confidence) and nothing else. It has no interval.
//!
//! [`Slope`] says how much of the form is estimated: only the intercept
//! (calibration in the large, the slope stays the model's), both, or both only
//! where the data show the slope is not one. Estimating a slope from a few
//! hundred events is itself noisy, which is what the third option protects a
//! well calibrated model from.

use crate::calibration::{labelled, logistic, logit, sigmoid, Labelled};
use crate::estimate::Step;
use crate::Obs;

/// The largest intercept or slope a fit may have: past it the weighted
/// outcomes are (nearly) separated by the prediction, the fit runs away, and
/// the "calibration" would be a step in disguise.
const MAX_PARAMETER: f64 = 25.0;

/// How the slope of a [`Recalibration`] is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Slope {
    /// Estimate intercept and slope.
    Free,
    /// Estimate the intercept only; the slope is one (calibration in the
    /// large).
    Fixed,
    /// Estimate both, keep the slope only if it differs from one by more than
    /// this many standard errors (sandwich estimate under the weights), else
    /// estimate the intercept alone.
    Evidence(f64),
}

/// A fitted logistic recalibration: `logit P = intercept + slope * logit F`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Recalibration {
    /// The intercept `a`.
    pub intercept: f64,
    /// The slope `b`: positive, so the map keeps the order of the risks.
    pub slope: f64,
}

/// The standard error of the fitted slope: the sandwich estimate
/// `H^-1 U H^-1` of the weighted logistic fit (NaN when singular).
fn slope_se(l: &Labelled, a: f64, b: f64) -> f64 {
    let (mut haa, mut hab, mut hbb) = (0.0, 0.0, 0.0);
    let (mut uaa, mut uab, mut ubb) = (0.0, 0.0, 0.0);
    for ((&x, &y), &v) in l.x.iter().zip(&l.y).zip(&l.v) {
        let p = sigmoid(a + b * x);
        let w = v * p * (1.0 - p);
        haa += w;
        hab += w * x;
        hbb += w * x * x;
        let r = v * (y - p);
        uaa += r * r;
        uab += r * r * x;
        ubb += r * r * x * x;
    }
    let det = haa * hbb - hab * hab;
    // Row 2 of H^-1 is (-hab, haa) / det; the slope's variance is that row
    // times U times its transpose.
    let (r0, r1) = (-hab / det, haa / det);
    (r0 * r0 * uaa + 2.0 * r0 * r1 * uab + r1 * r1 * ubb).sqrt()
}

impl Recalibration {
    /// Fit on subjects whose predicted probability of `cause` by `t` is
    /// `cif[i]`, with `g` the censoring distribution of the same subjects.
    /// `None` when the data give no usable map: no spread in the predictions,
    /// a separable or singular fit (see `MAX_PARAMETER`), or a slope that is not
    /// positive.
    pub fn at_horizon(
        cif: &[f64],
        obs: &[Obs],
        cause: usize,
        t: f64,
        g: &Step,
        slope: Slope,
    ) -> Option<Recalibration> {
        assert_eq!(cif.len(), obs.len(), "one prediction per subject");
        let l = labelled(cif, obs, cause, t, g);
        let fixed = |l: &Labelled| {
            let (a, _) = logistic(&l.x, &l.y, &l.v, true);
            (a.abs() <= MAX_PARAMETER).then_some(Recalibration { intercept: a, slope: 1.0 })
        };
        if slope == Slope::Fixed {
            return fixed(&l);
        }
        let (a, b) = logistic(&l.x, &l.y, &l.v, false);
        if !(a.abs() <= MAX_PARAMETER && b > 0.0 && b <= MAX_PARAMETER) {
            return None;
        }
        if let Slope::Evidence(z) = slope {
            let se = slope_se(&l, a, b);
            if !(se.is_finite() && (b - 1.0).abs() > z * se) {
                return fixed(&l);
            }
        }
        Some(Recalibration { intercept: a, slope: b })
    }

    /// The calibrated probability for a predicted probability `p`.
    pub fn apply(&self, p: f64) -> f64 {
        sigmoid(self.intercept + self.slope * logit(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::at_horizon;
    use crate::estimate::censoring;

    fn uniforms(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
    }

    /// Subjects with rate `0.2 exp(x)`, censored at rate 0.1; the score is the
    /// true risk by `t` with its log-odds mapped by `a + b logit`.
    fn distorted(n: usize, t: f64, a: f64, b: f64, seed: u64) -> (Vec<f64>, Vec<Obs>) {
        let mut u = uniforms(seed);
        let (mut score, mut obs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let rate = 0.2 * (3.0 * u() - 1.5).exp();
            let (time, c) = (-u().ln() / rate, -u().ln() / 0.1);
            obs.push(if time <= c { Obs::event(time, 0) } else { Obs::censored(c) });
            score.push(sigmoid(a + b * logit(1.0 - (-rate * t).exp())));
        }
        (score, obs)
    }

    #[test]
    fn a_hand_computable_intercept_shift_is_found_exactly() {
        // Uncensored, all subjects predicted 0.5 (so logit 0): the
        // intercept-only fit is the logit of the observed event share, here
        // 3 of 4 by t = 1.
        let obs = [Obs::event(0.5, 0), Obs::event(0.6, 0), Obs::event(0.7, 0), Obs::censored(2.0)];
        let cif = [0.5; 4];
        let r = Recalibration::at_horizon(&cif, &obs, 0, 1.0, &censoring(&[]), Slope::Fixed).unwrap();
        assert!((r.intercept - 3.0f64.ln()).abs() < 1e-9, "{r:?}");
        assert_eq!(r.slope, 1.0);
        assert!((r.apply(0.5) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn a_known_distortion_is_undone_and_the_truth_is_left_alone() {
        let t = 3.0;
        let (a, b) = (-0.5, 0.6);
        let (cal, cal_obs) = distorted(20_000, t, a, b, 3);
        let (test, test_obs) = distorted(20_000, t, a, b, 4);
        let (g, g_test) = (censoring(&cal_obs), censoring(&test_obs));
        let fit = Recalibration::at_horizon(&cal, &cal_obs, 0, t, &g, Slope::Free).unwrap();
        // The inverse of the distortion: logit F = (logit P - a) / b.
        assert!((fit.slope - 1.0 / b).abs() < 0.1 && (fit.intercept - (-a / b)).abs() < 0.12, "{fit:?}");
        let after: Vec<f64> = test.iter().map(|&p| fit.apply(p)).collect();
        let cal_after = at_horizon(&after, &test_obs, 0, t, &g_test, 10);
        assert!((cal_after.slope - 1.0).abs() < 0.08 && cal_after.intercept.abs() < 0.1, "{cal_after:?}");
        assert!((cal_after.oe_ratio - 1.0).abs() < 0.05, "{cal_after:?}");

        let (truth, truth_obs) = distorted(20_000, t, 0.0, 1.0, 5);
        let fit = Recalibration::at_horizon(&truth, &truth_obs, 0, t, &censoring(&truth_obs), Slope::Free).unwrap();
        assert!((fit.slope - 1.0).abs() < 0.08 && fit.intercept.abs() < 0.1, "{fit:?}");
    }

    #[test]
    fn the_evidence_policy_keeps_a_slope_only_where_the_data_demand_it() {
        let t = 3.0;
        let z = Slope::Evidence(2.0);
        // Calibrated predictions: the slope is not kept (an intercept alone),
        // on every one of several seeds, where a free slope strays by chance.
        for seed in 10..16 {
            let (s, o) = distorted(600, t, 0.0, 1.0, seed);
            let r = Recalibration::at_horizon(&s, &o, 0, t, &censoring(&o), z).unwrap();
            assert_eq!(r.slope, 1.0, "seed {seed}: {r:?}");
        }
        // Strongly over-confident predictions: the slope is kept and corrects.
        let (s, o) = distorted(4000, t, 0.0, 2.0, 20);
        let r = Recalibration::at_horizon(&s, &o, 0, t, &censoring(&o), z).unwrap();
        assert!((r.slope - 0.5).abs() < 0.1, "{r:?}");
    }

    #[test]
    fn predictions_without_spread_or_with_a_reversed_order_give_no_map() {
        let obs: Vec<Obs> = (0..20)
            .map(|i| if i % 2 == 0 { Obs::event(1.0, 0) } else { Obs::censored(5.0) })
            .collect();
        let g = censoring(&obs);
        assert!(Recalibration::at_horizon(&[0.3; 20], &obs, 0, 2.0, &g, Slope::Free).is_none());
        // Risk falls where events happen: a negative slope is not a calibration.
        let cif: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 0.1 } else { 0.4 }).collect();
        assert!(Recalibration::at_horizon(&cif, &obs, 0, 2.0, &g, Slope::Free).is_none());
    }
}
