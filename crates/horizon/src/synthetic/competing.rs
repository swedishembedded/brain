// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two competing absorbing causes whose cause-specific hazards depend on
//! different covariates and change with time differently, so the cumulative
//! incidence of each cause is a genuine integral `CIF_k(t) = int h_k S`.
//!
//! Swedish Embedded AB implements validation of competing-risks models
//! against data with known ground truth for its clients. If your team needs
//! expertise in cause-specific risk prediction you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! Covariates `x0, x1, x2` are standard normal, measured at entry. Cause a
//! has hazard `A0 exp(0.8 x0 - 0.5 x1 + GA t)` (rising with time), cause b
//! `B0 exp(0.7 x2 + 0.4 x1 - GB t)` (falling). Both are Gompertz in `t`, so
//! the cumulative hazards are closed form, the event time is drawn exactly
//! by inverting the summed cumulative hazard, and [`Truth::cif`] integrates
//! `h_k S` by Simpson's rule (error far below sampling error). Follow-up is
//! cut by a uniform censoring time independent of the causes.

use serde::{Deserialize, Serialize};

use data::rng::Rng;

use super::sim::{exp1, invert, simpson, subject, ENTRY};
use crate::timeline::{Observation, Subject, Value};

/// The outcome codes: cause a, cause b. Both absorbing.
pub const CODES: [&str; 2] = ["cause_a", "cause_b"];
/// Which codes are absorbing.
pub const ABSORBING: [bool; 2] = [true, true];
/// Covariate variable names.
pub const COVARIATES: [&str; 3] = ["x0", "x1", "x2"];
/// Baseline hazards at all covariates 0 and `t = 0`.
pub const BASE: [f64; 2] = [0.04, 0.05];
/// Log-hazard slope in time of each cause.
pub const TIME_SLOPE: [f64; 2] = [0.15, -0.12];
/// Censoring time, uniform in this range (time units since entry).
pub const CENSOR: (f64, f64) = (3.0, 10.0);

/// What generated one subject's outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    /// The covariates.
    pub x: [f64; 3],
}

impl Truth {
    /// `A0`/`B0` times the covariate part: the hazard of cause `k` at `t = 0`.
    fn scale(&self, k: usize) -> f64 {
        let lin = match k {
            0 => 0.8 * self.x[0] - 0.5 * self.x[1],
            _ => 0.7 * self.x[2] + 0.4 * self.x[1],
        };
        BASE[k] * lin.exp()
    }

    /// The true cause-specific hazard of `k` at time `t` since entry.
    pub fn hazard(&self, k: usize, t: f64) -> f64 {
        self.scale(k) * (TIME_SLOPE[k] * t).exp()
    }

    fn cumulative(&self, k: usize, t: f64) -> f64 {
        self.scale(k) * (TIME_SLOPE[k] * t).exp_m1() / TIME_SLOPE[k]
    }

    /// The true probability of no event of either cause by `t`.
    pub fn survival(&self, t: f64) -> f64 {
        (-(self.cumulative(0, t) + self.cumulative(1, t))).exp()
    }

    /// The true cumulative incidence of cause `k` by `t`: `int_0^t h_k S`.
    pub fn cif(&self, k: usize, t: f64) -> f64 {
        simpson(|s| self.hazard(k, s) * self.survival(s), 0.0, t, 400)
    }
}

/// `n` subjects and their truths, deterministic in `seed`.
pub fn population(n: usize, seed: u64) -> (Vec<Subject>, Vec<Truth>) {
    let mut rng = Rng::new(seed);
    let (mut subjects, mut truths) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let x = [(); 3].map(|_| rng.next_gaussian());
        let truth = Truth { x };
        let target = exp1(rng.next_f64());
        let pick = rng.next_f64();
        let censor = rng.uniform(CENSOR.0, CENSOR.1);
        let total = |t: f64| truth.cumulative(0, t) + truth.cumulative(1, t);
        let event = invert(total, target, censor).map(|t| {
            let (ha, hb) = (truth.hazard(0, t), truth.hazard(1, t));
            (t, CODES[usize::from(pick * (ha + hb) >= ha)])
        });
        let observations = COVARIATES
            .iter()
            .zip(x)
            .map(|(name, v)| Observation {
                t: ENTRY,
                var: (*name).into(),
                value: Value::Number(v),
            })
            .collect();
        subjects.push(subject(format!("compete{i}"), observations, event, censor));
        truths.push(truth);
    }
    (subjects, truths)
}

#[cfg(test)]
mod tests {
    use super::super::sim::outcomes;
    use super::*;

    /// The analytic cumulative incidence, averaged over the population, equals
    /// the Aalen-Johansen estimate from the censored simulated data within
    /// sampling error (a few binomial standard errors).
    #[test]
    fn the_analytic_cif_matches_aalen_johansen_from_the_simulation() {
        let n = 30_000;
        let (subjects, truths) = population(n, 11);
        for s in &subjects[..200] {
            s.validate().unwrap();
        }
        assert_eq!(subjects, population(n, 11).0, "deterministic in the seed");
        let obs: Vec<survival::Obs> = outcomes(&subjects, &CODES)
            .into_iter()
            .map(|(time, cause)| survival::Obs { time, cause, weight: 1.0 })
            .collect();
        for k in 0..2 {
            let aj = survival::estimate::aalen_johansen(&obs, k);
            for t in [2.0, 4.0, 6.0] {
                let truth = truths.iter().map(|tr| tr.cif(k, t)).sum::<f64>() / n as f64;
                let se = (truth * (1.0 - truth) / n as f64).sqrt();
                let got = aj.at(t);
                assert!(
                    (got - truth).abs() < 4.0 * se + 0.002,
                    "cause {k} at {t}: Aalen-Johansen {got:.4} vs analytic {truth:.4} (se {se:.4})"
                );
            }
        }
    }

    #[test]
    fn the_causes_depend_on_different_covariates() {
        let at = |x: [f64; 3]| Truth { x };
        let (z, a, b) = (at([0.0; 3]), at([1.0, 0.0, 0.0]), at([0.0, 0.0, 1.0]));
        assert!(a.hazard(0, 1.0) > 2.0 * z.hazard(0, 1.0));
        assert_eq!(a.hazard(1, 1.0), z.hazard(1, 1.0), "x0 does not move cause b");
        assert!(b.hazard(1, 1.0) > 1.9 * z.hazard(1, 1.0));
        assert_eq!(b.hazard(0, 1.0), z.hazard(0, 1.0), "x2 does not move cause a");
    }
}
