// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibration: do the predicted probabilities mean what they say?
//!
//! - [`d_calibration`] (Haider et al. 2020): if survival curves are right,
//!   each subject's predicted survival at its own observed time is uniform on
//!   `[0, 1]`. A censored subject spreads its mass over the bins at or below
//!   its predicted survival at censoring, in proportion to where the true
//!   value could lie. A chi-square test against uniform bins decides.
//! - [`at_horizon`]: for the probability of one cause by a horizon `t`, the
//!   observed rate (Aalen-Johansen) against the mean prediction, a logistic
//!   recalibration `logit P = a + b logit F` fitted with inverse probability
//!   of censoring weights (slope `b = 1`, intercept `a = 0` when calibrated;
//!   a slope below one means predictions too extreme), and a table of
//!   observed against expected by risk group.
//!
//! Sampling weights enter every sum; the D-calibration chi-square is scaled to
//! the Kish effective sample size, because a sum of weights is not a count of
//! independent observations.

use crate::estimate::{aalen_johansen, kaplan_meier, Step};
use crate::special::chi2_sf;
use crate::Obs;

/// D-calibration result.
#[derive(Clone, Debug, PartialEq)]
pub struct DCalibration {
    /// Share of (effective) mass per bin; uniform is `1 / bins` each.
    pub shares: Vec<f64>,
    /// Pearson chi-square against uniform, on the effective sample size.
    pub statistic: f64,
    /// Its upper tail probability (`bins - 1` degrees of freedom).
    pub p_value: f64,
}

/// D-calibration of predicted all-cause survival. `surv_at_time[i]` is subject
/// `i`'s predicted probability of no event by its own observed time.
pub fn d_calibration(surv_at_time: &[f64], obs: &[Obs], bins: usize) -> DCalibration {
    assert_eq!(surv_at_time.len(), obs.len(), "one prediction per subject");
    assert!(bins >= 2, "at least two bins");
    let b = bins as f64;
    let bin_of = |u: f64| ((u.clamp(0.0, 1.0) * b) as usize).min(bins - 1);
    let mut mass = vec![0.0; bins];
    for (&u, o) in surv_at_time.iter().zip(obs) {
        let u = u.clamp(0.0, 1.0);
        let j = bin_of(u);
        if o.cause.is_some() || u <= 0.0 {
            mass[j] += o.weight;
        } else {
            // Censored: the true value is uniform on [0, u].
            let lower = j as f64 / b;
            mass[j] += o.weight * (u - lower) / u;
            for m in mass.iter_mut().take(j) {
                *m += o.weight * (1.0 / b) / u;
            }
        }
    }
    let total: f64 = mass.iter().sum();
    let (sw, sw2): (f64, f64) = obs.iter().fold((0.0, 0.0), |(a, c), o| {
        (a + o.weight, c + o.weight * o.weight)
    });
    let n_eff = sw * sw / sw2;
    let shares: Vec<f64> = mass.iter().map(|m| m / total).collect();
    let expected = 1.0 / b;
    let statistic = n_eff
        * shares
            .iter()
            .map(|s| (s - expected).powi(2) / expected)
            .sum::<f64>();
    DCalibration {
        p_value: chi2_sf(statistic, b - 1.0),
        shares,
        statistic,
    }
}

/// One risk group's calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    /// Weight of the subjects in the group.
    pub weight: f64,
    /// Mean predicted probability.
    pub expected: f64,
    /// Aalen-Johansen estimate within the group.
    pub observed: f64,
}

/// Calibration of the probability of one cause by a horizon.
///
/// [`HorizonCalibration::ece`] is the expected calibration error of the
/// risk-group table: with `groups` of equal weight it is the equal-mass
/// binned ECE, with the observed side an Aalen-Johansen estimate (censoring-
/// and competing-risk-aware) and not a raw event share.
#[derive(Clone, Debug, PartialEq)]
pub struct HorizonCalibration {
    /// Aalen-Johansen cumulative incidence by the horizon.
    pub observed: f64,
    /// Mean predicted probability.
    pub expected: f64,
    /// `observed / expected`.
    pub oe_ratio: f64,
    /// Recalibration intercept `a` of `logit P = a + b logit F`.
    pub intercept: f64,
    /// Recalibration slope `b`.
    pub slope: f64,
    /// Calibration-in-the-large: `a` with `b` fixed at one.
    pub intercept_in_the_large: f64,
    /// Observed against expected per risk group, lowest risk first.
    pub groups: Vec<Group>,
    /// Weighted mean absolute gap between observed and expected over the groups.
    pub mean_abs_gap: f64,
}

impl HorizonCalibration {
    /// The expected calibration error: the weighted mean absolute gap between
    /// observed and expected risk over the risk groups.
    pub fn ece(&self) -> f64 {
        self.mean_abs_gap
    }
}

pub(crate) fn logit(p: f64) -> f64 {
    // Predictions of exactly 0 or 1 have no logit; move them inside by a hair.
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    (p / (1.0 - p)).ln()
}

pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The longest Newton step of the recalibration fit, on the log-odds scale.
const MAX_NEWTON_STEP: f64 = 5.0;

/// Weighted logistic regression of `y` on `[1, x]` (or on the offset `x`
/// with slope fixed to one when `fixed_slope`), by damped Newton steps; NaN
/// when the system has no solution.
pub(crate) fn logistic(x: &[f64], y: &[f64], v: &[f64], fixed_slope: bool) -> (f64, f64) {
    let (mut a, mut b) = (0.0, 1.0);
    for _ in 0..100 {
        let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..x.len() {
            let p = sigmoid(a + b * x[i]);
            let r = v[i] * (y[i] - p);
            let w = v[i] * p * (1.0 - p);
            ga += r;
            gb += r * x[i];
            haa += w;
            hab += w * x[i];
            hbb += w * x[i] * x[i];
        }
        let (da, db) = if fixed_slope {
            (ga / haa, 0.0)
        } else {
            let det = haa * hbb - hab * hab;
            ((hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det)
        };
        // A singular system (no spread in the predictions, separation) has no
        // fit: not measured, never a number from a division by zero.
        if !(da.is_finite() && db.is_finite()) {
            return (f64::NAN, f64::NAN);
        }
        // Large weights can overshoot: no step longer than a few units.
        let scale = (da.abs().max(db.abs()) / MAX_NEWTON_STEP).max(1.0);
        a += da / scale;
        b += db / scale;
        if da.abs() + db.abs() < 1e-12 {
            break;
        }
    }
    (a, b)
}

/// The subjects whose outcome by `t` is known, as a binary regression: the
/// log-odds `x` of each one's predicted probability, its outcome `y` (an event
/// of `cause` by `t`) and its inverse-probability-of-censoring weight `v`
/// (sampling weight over `G` at its event time, or at `t` for a subject still
/// event-free). A subject censored before `t` drops out, and so does one whose
/// `G` is zero where it is needed.
pub(crate) struct Labelled {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub v: Vec<f64>,
}

pub(crate) fn labelled(cif: &[f64], obs: &[Obs], cause: usize, t: f64, g: &Step) -> Labelled {
    let mut l = Labelled { x: vec![], y: vec![], v: vec![] };
    for (f, o) in cif.iter().zip(obs) {
        let (yi, gi) = if o.time <= t {
            match o.cause {
                Some(c) => (if c == cause { 1.0 } else { 0.0 }, g.at(o.time)),
                None => continue,
            }
        } else {
            (0.0, g.at(t))
        };
        if gi > 0.0 {
            l.x.push(logit(*f));
            l.y.push(yi);
            l.v.push(o.weight / gi);
        }
    }
    l
}

/// Calibration of `cif` (each subject's predicted probability of `cause` by
/// `t`) against the outcomes, with `g` the censoring distribution from the
/// training data and `groups` risk groups of equal weight.
pub fn at_horizon(
    cif: &[f64],
    obs: &[Obs],
    cause: usize,
    t: f64,
    g: &Step,
    groups: usize,
) -> HorizonCalibration {
    assert_eq!(cif.len(), obs.len(), "one prediction per subject");
    let wsum: f64 = obs.iter().map(|o| o.weight).sum();
    let expected = cif.iter().zip(obs).map(|(f, o)| f * o.weight).sum::<f64>() / wsum;
    let observed = aalen_johansen(obs, cause).at(t);
    let Labelled { x, y, v } = labelled(cif, obs, cause, t, g);
    let (intercept, slope) = logistic(&x, &y, &v, false);
    let (intercept_in_the_large, _) = logistic(&x, &y, &v, true);
    // Risk groups of equal weight, by prediction.
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by(|&a, &b| cif[a].partial_cmp(&cif[b]).expect("finite predictions"));
    let mut table = Vec::with_capacity(groups);
    let (mut start, mut acc) = (0usize, 0.0);
    for (k, &i) in order.iter().enumerate() {
        acc += obs[i].weight;
        let boundary = acc >= wsum * (table.len() + 1) as f64 / groups as f64 - 1e-9;
        if boundary || k + 1 == order.len() {
            let members: Vec<usize> = order[start..=k].to_vec();
            let gw: f64 = members.iter().map(|&i| obs[i].weight).sum();
            let sub: Vec<Obs> = members.iter().map(|&i| obs[i]).collect();
            table.push(Group {
                weight: gw,
                expected: members.iter().map(|&i| cif[i] * obs[i].weight).sum::<f64>() / gw,
                observed: aalen_johansen(&sub, cause).at(t),
            });
            start = k + 1;
        }
    }
    let mean_abs_gap = table
        .iter()
        .map(|g| g.weight * (g.observed - g.expected).abs())
        .sum::<f64>()
        / wsum;
    HorizonCalibration {
        observed,
        expected,
        oe_ratio: observed / expected,
        intercept,
        slope,
        intercept_in_the_large,
        groups: table,
        mean_abs_gap,
    }
}

/// Every subject's predicted survival at its own observed time, from a
/// per-subject survival function - the input [`d_calibration`] takes.
pub fn survival_at_own_time(obs: &[Obs], survival: impl Fn(usize, f64) -> f64) -> Vec<f64> {
    obs.iter()
        .enumerate()
        .map(|(i, o)| survival(i, o.time))
        .collect()
}

/// The marginal Kaplan-Meier curve used as everyone's prediction: a
/// calibrated-by-construction reference for [`d_calibration`] in tests and
/// as a baseline.
pub fn marginal_survival(obs: &[Obs]) -> Step {
    kaplan_meier(obs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::censoring;

    fn exp_sample(n: usize, rate: f64, seed: u64) -> Vec<Obs> {
        let mut s = seed;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let t = -u().ln() / rate;
                let c = -u().ln() / (rate * 0.5);
                if t <= c {
                    Obs::event(t, 0)
                } else {
                    Obs::censored(c)
                }
            })
            .collect()
    }

    #[test]
    fn the_true_model_is_d_calibrated_and_a_wrong_one_is_not() {
        let obs = exp_sample(4000, 0.2, 3);
        let truth: Vec<f64> = obs.iter().map(|o| (-0.2 * o.time).exp()).collect();
        let good = d_calibration(&truth, &obs, 10);
        assert!(good.p_value > 0.01, "true survival rejected: {good:?}");
        let wrong: Vec<f64> = obs.iter().map(|o| (-0.4 * o.time).exp()).collect();
        assert!(d_calibration(&wrong, &obs, 10).p_value < 1e-6);
    }

    #[test]
    fn the_true_probability_has_unit_slope_and_an_overconfident_one_does_not() {
        // Subject i has rate 0.2 * exp(x_i), x_i spread over [-1.5, 1.5].
        let mut s = 11u64;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let n = 30_000;
        let (mut obs, mut rate) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let r = 0.2 * (3.0 * u() - 1.5).exp();
            let (t, c) = (-u().ln() / r, -u().ln() / 0.1);
            obs.push(if t <= c {
                Obs::event(t, 0)
            } else {
                Obs::censored(c)
            });
            rate.push(r);
        }
        let g = censoring(&obs);
        let t = 3.0;
        let truth: Vec<f64> = rate.iter().map(|r| 1.0 - (-r * t).exp()).collect();
        let cal = at_horizon(&truth, &obs, 0, t, &g, 10);
        assert!(
            (cal.slope - 1.0).abs() < 0.06 && cal.intercept.abs() < 0.08,
            "{cal:?}"
        );
        assert!(
            (cal.oe_ratio - 1.0).abs() < 0.03 && cal.mean_abs_gap < 0.02,
            "{cal:?}"
        );
        // Doubling every log-odds makes the predictions too extreme: slope near 1/2.
        let extreme: Vec<f64> = truth.iter().map(|&p| sigmoid(2.0 * logit(p))).collect();
        let over = at_horizon(&extreme, &obs, 0, t, &g, 10);
        assert!((over.slope - 0.5).abs() < 0.05, "{over:?}");
    }

    #[test]
    fn calibration_in_the_large_ignores_a_slope_error_the_two_parameter_intercept_absorbs() {
        let mut s = 21u64;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let n = 40_000;
        let (mut obs, mut rate) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let r = 0.2 * (3.0 * u() - 1.5).exp();
            let (t, c) = (-u().ln() / r, -u().ln() / 0.1);
            obs.push(if t <= c { Obs::event(t, 0) } else { Obs::censored(c) });
            rate.push(r);
        }
        let g = censoring(&obs);
        let t = 0.5; // low risks: far from the 50% at which the intercept is read
        let truth: Vec<f64> = rate.iter().map(|r| 1.0 - (-r * t).exp()).collect();
        let mean_logit = truth.iter().map(|&p| logit(p)).sum::<f64>() / n as f64;
        // Pull the log-odds towards their mean: centred on the truth, spread too small.
        let squeezed: Vec<f64> = truth
            .iter()
            .map(|&p| sigmoid(mean_logit + 0.6 * (logit(p) - mean_logit)))
            .collect();
        let cal = at_horizon(&squeezed, &obs, 0, t, &g, 10);
        assert!(cal.slope > 1.4, "{cal:?}");
        assert!(cal.intercept.abs() > 0.8, "the two-parameter intercept absorbs the slope: {cal:?}");
        assert!(cal.intercept_in_the_large.abs() < 0.25, "the average risk is right: {cal:?}");
        // Squeezing the log-odds moves the mean risk a little (Jensen), hence the loose band.
        assert!((cal.oe_ratio - 1.0).abs() < 0.2, "{cal:?}");
        // A shifted average, by contrast, shows in calibration in the large.
        let shifted: Vec<f64> = truth.iter().map(|&p| sigmoid(logit(p) - 0.7)).collect();
        let off = at_horizon(&shifted, &obs, 0, t, &g, 10);
        assert!((off.intercept_in_the_large - 0.7).abs() < 0.15, "{off:?}");
    }

    #[test]
    fn predictions_without_spread_have_no_slope() {
        let obs: Vec<Obs> = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    Obs::event(1.0, 0)
                } else {
                    Obs::censored(5.0)
                }
            })
            .collect();
        let g = censoring(&obs);
        let cal = at_horizon(&[0.3; 20], &obs, 0, 2.0, &g, 2);
        assert!(cal.slope.is_nan(), "{cal:?}");
    }
}
