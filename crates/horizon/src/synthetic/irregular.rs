// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Irregular observation times that carry the information: the measured
//! values are noise, the visits are not.
//!
//! Swedish Embedded AB implements risk models that read informative
//! observation patterns - who was measured, and when - for its clients. If
//! your team needs expertise in informative visit processes you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! Each subject has a latent frailty `u ~ N(0, 1)` that sets its hazard,
//! `BASE * exp(BETA u)` (constant, so the cumulative incidence is closed
//! form). Before entry the subject is seen at the points of a Poisson process
//! of rate `VISIT_RATE * exp(GAMMA u)` over the last [`WINDOW`] time units;
//! every visit measures the variable `v`, which is standard normal noise
//! independent of everything. A model that ignores WHEN (and how often)
//! subjects were seen has nothing to go on; [`swap_visits`] makes that
//! ablation a pure transform of the data. Follow-up is cut by a uniform
//! censoring time.
//!
//! Given only the visit times the best possible prediction is exact: the
//! visit pattern's likelihood depends on `u` only through the number of
//! visits `n` (Poisson with mean `VISIT_RATE * exp(GAMMA u) * WINDOW`), so
//! [`Truth::oracle_cif`] integrates the hazard over the posterior of `u`
//! given `n`.

use serde::{Deserialize, Serialize};

use data::rng::Rng;

use super::sim::{exp1, simpson, subject, ENTRY};
use crate::timeline::{Observation, Subject, Value};

/// The outcome code.
pub const CODE: &str = "event";
/// The (uninformative) measured variable.
pub const VARIABLE: &str = "v";
/// How far before entry visits can fall.
pub const WINDOW: f64 = 5.0;
/// Visits per unit time at `u = 0`.
pub const VISIT_RATE: f64 = 0.8;
/// Log visit rate per unit of `u`.
pub const GAMMA: f64 = 0.7;
/// Hazard per unit time at `u = 0`.
pub const BASE: f64 = 0.10;
/// Log-hazard per unit of `u`.
pub const BETA: f64 = 0.8;
/// Censoring time, uniform in this range (time units since entry).
pub const CENSOR: (f64, f64) = (2.0, 8.0);

/// What generated one subject's outcome and visit pattern.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    /// The latent frailty.
    pub u: f64,
    /// How many visits fell in the window.
    pub visits: usize,
}

/// The probability of the event by `t` for frailty `u` (closed form).
fn cif_given(u: f64, t: f64) -> f64 {
    1.0 - (-BASE * (BETA * u).exp() * t).exp()
}

impl Truth {
    /// The true hazard per unit time.
    pub fn hazard(&self) -> f64 {
        BASE * (BETA * self.u).exp()
    }

    /// The true probability of the event by `t` after entry (closed form).
    pub fn cif(&self, t: f64) -> f64 {
        cif_given(self.u, t)
    }

    /// The best prediction from the visit times alone: the probability of the
    /// event by `t` averaged over the posterior of `u` given the number of
    /// visits (a fine quadrature on `u`).
    pub fn oracle_cif(&self, t: f64) -> f64 {
        let n = self.visits as f64;
        let ln_fact: f64 = (2..=self.visits).map(|i| (i as f64).ln()).sum();
        let weight = |u: f64| {
            let mean = VISIT_RATE * (GAMMA * u).exp() * WINDOW;
            (-0.5 * u * u + n * mean.ln() - mean - ln_fact).exp()
        };
        let norm = simpson(weight, -8.0, 8.0, 800);
        simpson(|u| weight(u) * cif_given(u, t), -8.0, 8.0, 800) / norm
    }
}

/// `n` subjects and their truths, deterministic in `seed`.
pub fn population(n: usize, seed: u64) -> (Vec<Subject>, Vec<Truth>) {
    let mut rng = Rng::new(seed);
    let (mut subjects, mut truths) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let u = rng.next_gaussian();
        let rate = VISIT_RATE * (GAMMA * u).exp();
        // Poisson process on the window by exponential gaps.
        let mut observations = Vec::new();
        let mut ago = exp1(rng.next_f64()) / rate;
        while ago < WINDOW {
            observations.push(Observation {
                t: ENTRY - ago,
                var: VARIABLE.into(),
                value: Value::Number(rng.next_gaussian()),
                unit: None,
            });
            ago += exp1(rng.next_f64()) / rate;
        }
        observations.reverse();
        let truth = Truth {
            u,
            visits: observations.len(),
        };
        let time = exp1(rng.next_f64()) / truth.hazard();
        let censor = rng.uniform(CENSOR.0, CENSOR.1);
        let event = (time < censor).then_some((time, CODE));
        subjects.push(subject(format!("irregular{i}"), observations, event, censor));
        truths.push(truth);
    }
    (subjects, truths)
}

/// The visit-blind ablation: every subject's observations (times and values)
/// are replaced by those of another subject, drawn by a seeded shuffle with no
/// subject keeping its own. Times are moved onto the receiving subject's
/// clock (`t - entry` is preserved). Outcomes and everything else are
/// untouched, so a model trained and tested on the result can learn nothing
/// from how, when or how often the subject was seen.
pub fn swap_visits(subjects: &[Subject], seed: u64) -> Vec<Subject> {
    let n = subjects.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed);
    for i in (1..n).rev() {
        order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
    let mut out = subjects.to_vec();
    for k in 0..n {
        let (to, from) = (order[k], order[(k + 1) % n]);
        out[to].observations = subjects[from]
            .observations
            .iter()
            .map(|o| Observation {
                t: o.t - subjects[from].entry + subjects[to].entry,
                ..o.clone()
            })
            .collect();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::sim::outcomes;
    use super::*;

    #[test]
    fn the_population_is_valid_and_the_oracle_matches_the_simulated_events() {
        let n = 30_000;
        let (subjects, truths) = population(n, 5);
        for s in &subjects[..200] {
            s.validate().unwrap();
        }
        assert_eq!(subjects, population(n, 5).0, "deterministic in the seed");
        let obs: Vec<survival::Obs> = outcomes(&subjects, &[CODE])
            .into_iter()
            .map(|(time, cause)| survival::Obs { time, cause, weight: 1.0 })
            .collect();
        let observed = survival::estimate::aalen_johansen(&obs, 0).at(4.0);
        let mean = |f: &dyn Fn(&Truth) -> f64| truths.iter().map(f).sum::<f64>() / n as f64;
        let (truth, oracle) = (mean(&|t| t.cif(4.0)), mean(&|t| t.oracle_cif(4.0)));
        assert!((observed - truth).abs() < 0.01, "{observed} vs {truth}");
        assert!((oracle - truth).abs() < 0.01, "{oracle} vs {truth}");
    }

    /// The visits say something about risk, the values nothing: the visit
    /// count correlates with the frailty, the measured values do not.
    #[test]
    fn the_gaps_carry_the_information_and_the_values_do_not() {
        let (subjects, truths) = population(5_000, 8);
        let corr = |x: &[f64], y: &[f64]| {
            let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (mx, my) = (m(x), m(y));
            let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let var = |v: &[f64], mv: f64| v.iter().map(|a| (a - mv).powi(2)).sum::<f64>();
            cov / (var(x, mx) * var(y, my)).sqrt()
        };
        let u: Vec<f64> = truths.iter().map(|t| t.u).collect();
        let count: Vec<f64> = truths.iter().map(|t| t.visits as f64).collect();
        assert!(corr(&count, &u) > 0.5, "count vs frailty: {}", corr(&count, &u));
        let mean_value: Vec<f64> = subjects
            .iter()
            .map(|s| {
                let v: Vec<f64> = s
                    .observations
                    .iter()
                    .filter_map(|o| match o.value {
                        Value::Number(x) => Some(x),
                        _ => None,
                    })
                    .collect();
                v.iter().sum::<f64>() / v.len().max(1) as f64
            })
            .collect();
        assert!(corr(&mean_value, &u).abs() < 0.05, "values vs frailty: {}", corr(&mean_value, &u));
    }

    #[test]
    fn swapping_visits_breaks_the_link_and_changes_nothing_else() {
        let (subjects, truths) = population(4_000, 9);
        let swapped = swap_visits(&subjects, 1);
        assert_eq!(swapped, swap_visits(&subjects, 1), "deterministic in the seed");
        for (a, b) in subjects.iter().zip(&swapped) {
            assert_eq!((&a.events, &a.at_risk, a.entry), (&b.events, &b.at_risk, b.entry));
            b.validate().unwrap();
        }
        let u: Vec<f64> = truths.iter().map(|t| t.u).collect();
        let kept: f64 = subjects.iter().map(|s| s.observations.len() as f64).sum();
        let moved: f64 = swapped.iter().map(|s| s.observations.len() as f64).sum();
        assert_eq!(kept, moved, "a permutation of the same visits");
        // The high-frailty half had more visits before, and not after.
        let n_high = |s: &[Subject]| {
            let hi: Vec<f64> = (0..s.len()).filter(|&i| u[i] > 0.0).map(|i| s[i].observations.len() as f64).collect();
            let lo: Vec<f64> = (0..s.len()).filter(|&i| u[i] <= 0.0).map(|i| s[i].observations.len() as f64).collect();
            hi.iter().sum::<f64>() / hi.len() as f64 - lo.iter().sum::<f64>() / lo.len() as f64
        };
        assert!(n_high(&subjects) > 1.5, "{}", n_high(&subjects));
        assert!(n_high(&swapped).abs() < 0.4, "{}", n_high(&swapped));
    }
}
