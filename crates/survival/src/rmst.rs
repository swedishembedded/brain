// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Restricted mean survival time (RMST) and a reference life table.
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in turning survival curves into an expected time lived and
//! checking it against what happened, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! - [`km_rmst`]: the area under a Kaplan-Meier [`Step`] up to `tau`.
//! - [`curve_rmst`]: the area under a predicted curve given as survival at
//!   whole years 1, 2, ... (survival 1 at year 0), linear between them.
//! - [`calibration`]: predicted RMST against the Kaplan-Meier RMST observed in
//!   equal-weight groups of subjects ordered by prediction.
//! - [`Gompertz`]: a hazard `exp(a + b * age)` fitted by weighted maximum
//!   likelihood with delayed entry, which turns an RMST into the age at which
//!   the reference table gives the same time ("equivalent age").

use crate::estimate::{kaplan_meier, Step};
use crate::Obs;

/// Area under a Kaplan-Meier (or any survival) step function on `[0, tau]`.
pub fn km_rmst(s: &Step, tau: f64) -> f64 {
    assert!(tau > 0.0, "the restriction time must be positive");
    let (times, _) = s.points();
    let mut area = 0.0;
    let mut prev = 0.0;
    for &t in times.iter().take_while(|&&t| t < tau) {
        area += s.at(prev) * (t - prev);
        prev = t;
    }
    area + s.at(prev) * (tau - prev)
}

/// Area under a survival curve known at whole years `1..=yearly.len()` (and
/// 1 at year 0), linear between them, up to `tau` years.
pub fn curve_rmst(yearly: &[f64], tau: f64) -> f64 {
    assert!(tau > 0.0, "the restriction time must be positive");
    assert!(
        tau <= yearly.len() as f64,
        "the curve ends at year {} before tau {tau}",
        yearly.len()
    );
    let mut area = 0.0;
    let mut left = 1.0;
    for (k, &right) in yearly.iter().enumerate() {
        let width = (tau - k as f64).clamp(0.0, 1.0);
        if width == 0.0 {
            break;
        }
        let end = left + (right - left) * width;
        area += width * (left + end) / 2.0;
        left = right;
    }
    area
}

/// One group of the RMST calibration table.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    /// Total sampling weight of the group.
    pub weight: f64,
    /// Weighted mean of the predicted RMST.
    pub expected: f64,
    /// Kaplan-Meier RMST observed in the group.
    pub observed: f64,
}

/// Calibration of predicted RMST.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub groups: Vec<Group>,
    /// Weighted mean of `|observed - expected|` over groups.
    pub mean_abs_gap: f64,
    /// Weighted least-squares line of observed on expected over the groups
    /// (slope 1 and intercept 0 when calibrated).
    pub slope: f64,
    pub intercept: f64,
}

/// Group subjects into `groups` equal-weight bins by `predicted` RMST and
/// compare each bin's mean prediction with the Kaplan-Meier RMST of its
/// subjects up to `tau`.
pub fn calibration(predicted: &[f64], obs: &[Obs], tau: f64, groups: usize) -> Calibration {
    assert_eq!(predicted.len(), obs.len(), "one prediction per subject");
    assert!(groups >= 2, "at least two groups");
    let total: f64 = obs.iter().map(|o| o.weight).sum();
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by(|&a, &b| {
        predicted[a]
            .partial_cmp(&predicted[b])
            .expect("finite predictions")
    });
    let mut out = Vec::with_capacity(groups);
    let (mut members, mut weight, mut sum) = (Vec::<Obs>::new(), 0.0, 0.0);
    let (mut seen, mut current) = (0.0, 0usize);
    let mut flush = |members: &mut Vec<Obs>, weight: &mut f64, sum: &mut f64| {
        if *weight > 0.0 {
            out.push(Group {
                weight: *weight,
                expected: *sum / *weight,
                observed: km_rmst(&kaplan_meier(members), tau),
            });
        }
        members.clear();
        *weight = 0.0;
        *sum = 0.0;
    };
    for &i in &order {
        let w = obs[i].weight;
        // A subject belongs to the bin its weight midpoint falls in.
        let bin = ((((seen + w / 2.0) / total) * groups as f64).floor() as usize).min(groups - 1);
        if bin != current {
            flush(&mut members, &mut weight, &mut sum);
            current = bin;
        }
        seen += w;
        weight += w;
        sum += w * predicted[i];
        members.push(obs[i]);
    }
    flush(&mut members, &mut weight, &mut sum);
    let gw: f64 = out.iter().map(|g| g.weight).sum();
    let mean_abs_gap = out
        .iter()
        .map(|g| g.weight * (g.observed - g.expected).abs())
        .sum::<f64>()
        / gw;
    let mx = out.iter().map(|g| g.weight * g.expected).sum::<f64>() / gw;
    let my = out.iter().map(|g| g.weight * g.observed).sum::<f64>() / gw;
    let sxx: f64 = out
        .iter()
        .map(|g| g.weight * (g.expected - mx).powi(2))
        .sum();
    let sxy: f64 = out
        .iter()
        .map(|g| g.weight * (g.expected - mx) * (g.observed - my))
        .sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { f64::NAN };
    Calibration {
        groups: out,
        mean_abs_gap,
        slope,
        intercept: my - slope * mx,
    }
}

/// Hazard `exp(a + b * age)` per year of age.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gompertz {
    pub a: f64,
    pub b: f64,
}

/// A subject followed from `entry` to `exit` (ages in years), `died` at exit.
#[derive(Clone, Copy, Debug)]
pub struct Follow {
    pub entry: f64,
    pub exit: f64,
    pub died: bool,
    pub weight: f64,
}

impl Gompertz {
    /// Weighted maximum likelihood with delayed entry. For a given `b` the
    /// maximising `a` is closed form, so the profile likelihood in `b` is
    /// maximised by golden-section search on `[1e-4, 0.5]` per year.
    pub fn fit(follow: &[Follow]) -> Gompertz {
        let deaths: f64 = follow.iter().filter(|f| f.died).map(|f| f.weight).sum();
        assert!(deaths > 0.0, "a hazard cannot be fitted without deaths");
        let exposure = |b: f64| -> f64 {
            follow
                .iter()
                .map(|f| f.weight * ((b * f.exit).exp() - (b * f.entry).exp()) / b)
                .sum()
        };
        let profile = |b: f64| -> f64 {
            // log-likelihood at the best a: sum d*b*exit + D*(ln D - ln E) - D
            let drift: f64 = follow
                .iter()
                .filter(|f| f.died)
                .map(|f| f.weight * b * f.exit)
                .sum();
            drift + deaths * (deaths.ln() - exposure(b).ln()) - deaths
        };
        let (mut lo, mut hi) = (1e-4_f64, 0.5_f64);
        let phi = (5f64.sqrt() - 1.0) / 2.0;
        let (mut x1, mut x2) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
        let (mut f1, mut f2) = (profile(x1), profile(x2));
        while hi - lo > 1e-10 {
            if f1 < f2 {
                lo = x1;
                x1 = x2;
                f1 = f2;
                x2 = lo + phi * (hi - lo);
                f2 = profile(x2);
            } else {
                hi = x2;
                x2 = x1;
                f2 = f1;
                x1 = hi - phi * (hi - lo);
                f1 = profile(x1);
            }
        }
        let b = (lo + hi) / 2.0;
        Gompertz {
            a: (deaths / exposure(b)).ln(),
            b,
        }
    }

    /// Probability of surviving `u` more years from `age`.
    pub fn survival(&self, age: f64, u: f64) -> f64 {
        (-(self.a + self.b * age).exp() * ((self.b * u).exp() - 1.0) / self.b).exp()
    }

    /// Expected years lived in the next `tau` from `age` (Simpson's rule).
    pub fn rmst_from(&self, age: f64, tau: f64) -> f64 {
        assert!(tau > 0.0, "the restriction time must be positive");
        let n = 400;
        let h = tau / n as f64;
        let mut sum = self.survival(age, 0.0) + self.survival(age, tau);
        for k in 1..n {
            sum += self.survival(age, k as f64 * h) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    /// The age in `[lo, hi]` whose RMST over `tau` equals `rmst`, and whether
    /// the answer had to be clamped to a bound because `rmst` lies outside.
    pub fn equivalent_age(&self, rmst: f64, tau: f64, lo: f64, hi: f64) -> (f64, bool) {
        // RMST falls with age, so the bound with the larger RMST is `lo`.
        if rmst >= self.rmst_from(lo, tau) {
            return (lo, rmst > self.rmst_from(lo, tau));
        }
        if rmst <= self.rmst_from(hi, tau) {
            return (hi, rmst < self.rmst_from(hi, tau));
        }
        let (mut a, mut b) = (lo, hi);
        for _ in 0..60 {
            let m = (a + b) / 2.0;
            if self.rmst_from(m, tau) > rmst {
                a = m;
            } else {
                b = m;
            }
        }
        ((a + b) / 2.0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn km_rmst_by_hand() {
        // Events at 1 and 3, one censoring at 2: S = 1 on [0,1), 2/3 on [1,3), then 0.
        let obs = [Obs::event(1.0, 0), Obs::censored(2.0), Obs::event(3.0, 0)];
        let s = kaplan_meier(&obs);
        // S(1) = 2/3; at 3 one at risk -> 0.
        assert!((km_rmst(&s, 3.0) - (1.0 + 2.0 * 2.0 / 3.0)).abs() < 1e-12);
        assert!((km_rmst(&s, 0.5) - 0.5).abs() < 1e-12);
        assert!((km_rmst(&s, 10.0) - (1.0 + 2.0 * 2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn curve_rmst_is_the_trapezoid_area() {
        // Exponential hazard 0.1: sampled yearly, trapezoid is close to the closed form.
        let yearly: Vec<f64> = (1..=10).map(|k| (-0.1 * k as f64).exp()).collect();
        let exact = (1.0 - (-1.0f64).exp()) / 0.1;
        assert!((curve_rmst(&yearly, 10.0) - exact).abs() < 0.01);
        // Linear inside a year.
        assert!((curve_rmst(&[0.5], 0.5) - 0.5 * (1.0 + 0.75) / 2.0).abs() < 1e-12);
        // Survival 1 throughout gives tau.
        assert!((curve_rmst(&[1.0; 5], 4.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "before tau")]
    fn curve_rmst_refuses_a_short_curve() {
        curve_rmst(&[0.9, 0.8], 3.0);
    }

    #[test]
    fn integer_weights_equal_copies() {
        let a = [
            Obs::event(1.0, 0).weighted(2.0),
            Obs::censored(2.0).weighted(3.0),
            Obs::event(4.0, 0),
        ];
        let b = [
            Obs::event(1.0, 0),
            Obs::event(1.0, 0),
            Obs::censored(2.0),
            Obs::censored(2.0),
            Obs::censored(2.0),
            Obs::event(4.0, 0),
        ];
        assert!((km_rmst(&kaplan_meier(&a), 5.0) - km_rmst(&kaplan_meier(&b), 5.0)).abs() < 1e-12);
    }

    #[test]
    fn calibration_of_an_exact_prediction_has_no_gap() {
        // Two groups with known no-censoring survival times; predictions equal their RMST.
        let mut obs = Vec::new();
        let mut pred = Vec::new();
        for _ in 0..50 {
            obs.push(Obs::event(2.0, 0));
            pred.push(2.0);
        }
        for _ in 0..50 {
            obs.push(Obs::censored(10.0));
            pred.push(5.0);
        }
        let c = calibration(&pred, &obs, 5.0, 2);
        assert_eq!(c.groups.len(), 2);
        assert!((c.groups[0].observed - 2.0).abs() < 1e-12);
        assert!((c.groups[1].observed - 5.0).abs() < 1e-12);
        assert!(c.mean_abs_gap < 1e-12);
        assert!((c.slope - 1.0).abs() < 1e-12 && c.intercept.abs() < 1e-12);
        // Predictions that are too optimistic show a gap.
        let off: Vec<f64> = pred.iter().map(|p| p + 1.0).collect();
        assert!((calibration(&off, &obs, 5.0, 2).mean_abs_gap - 1.0).abs() < 1e-12);
    }

    /// Deterministic generator so the test needs no random-number dependency.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    #[test]
    fn a_gompertz_hazard_is_recovered_under_delayed_entry() {
        let truth = Gompertz { a: -10.0, b: 0.09 };
        let mut s = 7u64;
        let mut follow = Vec::new();
        while follow.len() < 60_000 {
            let entry = 20.0 + 60.0 * lcg(&mut s);
            // Draw the age at death given alive at entry, by inverting the survival function.
            let u = lcg(&mut s);
            let target = -u.ln() * truth.b / (truth.a + truth.b * entry).exp();
            let wait = (1.0 + target).ln() / truth.b;
            let death = entry + wait;
            let end = entry + 15.0;
            follow.push(Follow {
                entry,
                exit: death.min(end),
                died: death <= end,
                weight: 1.0,
            });
        }
        let fit = Gompertz::fit(&follow);
        assert!((fit.b - truth.b).abs() < 0.004, "b {}", fit.b);
        assert!((fit.a - truth.a).abs() < 0.35, "a {}", fit.a);
    }

    #[test]
    fn equivalent_age_inverts_the_rmst_and_is_monotone() {
        let g = Gompertz { a: -10.0, b: 0.09 };
        let (age, clamped) = g.equivalent_age(g.rmst_from(63.0, 10.0), 10.0, 18.0, 85.0);
        assert!(!clamped && (age - 63.0).abs() < 1e-6, "{age}");
        assert!(g.rmst_from(40.0, 10.0) > g.rmst_from(70.0, 10.0));
        // Beyond the table's range the answer is clamped and says so.
        assert_eq!(g.equivalent_age(9.99, 10.0, 18.0, 85.0), (18.0, true));
        assert_eq!(g.equivalent_age(0.1, 10.0, 18.0, 85.0), (85.0, true));
        // Exactly at a bound is not a clamp.
        assert_eq!(
            g.equivalent_age(g.rmst_from(18.0, 10.0), 10.0, 18.0, 85.0),
            (18.0, false)
        );
    }
}
