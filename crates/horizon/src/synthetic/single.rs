// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One absorbing event with a log-linear hazard and uniform right censoring:
//! the plain survival problem, with every subject's true risk known.
//!
//! Swedish Embedded AB implements validation of time-to-event models against
//! data with known ground truth for its clients. If your team needs expertise
//! in survival model verification you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Each subject has four standard-normal covariates `x0..x3` (measured at
//! entry; `x3` is irrelevant) and a constant hazard
//! `BASE * exp(sum_j BETA[j] x_j)`, so the cumulative incidence is
//! `1 - exp(-h t)` in closed form. Follow-up is cut by a censoring time
//! uniform in [`CENSOR`], independent of everything else.

use serde::{Deserialize, Serialize};

use data::rng::Rng;

use super::sim::{exp1, subject, ENTRY};
use crate::timeline::{Observation, Subject, Value};

/// The outcome code.
pub const CODE: &str = "event";
/// Covariate variable names.
pub const COVARIATES: [&str; 4] = ["x0", "x1", "x2", "x3"];
/// Log-hazard per unit of each covariate.
pub const BETA: [f64; 4] = [0.9, -0.6, 0.4, 0.0];
/// Hazard per unit time at all covariates 0.
pub const BASE: f64 = 0.06;
/// Censoring time, uniform in this range (time units since entry).
pub const CENSOR: (f64, f64) = (2.0, 8.0);

/// What generated one subject's outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    /// The covariates.
    pub x: [f64; 4],
}

impl Truth {
    /// The true log-hazard (the risk score a perfect ranker would use).
    pub fn log_hazard(&self) -> f64 {
        BASE.ln() + self.x.iter().zip(BETA).map(|(x, b)| x * b).sum::<f64>()
    }

    /// The true hazard per unit time.
    pub fn hazard(&self) -> f64 {
        self.log_hazard().exp()
    }

    /// The true probability of the event by `t` after entry (closed form).
    pub fn cif(&self, t: f64) -> f64 {
        1.0 - (-self.hazard() * t).exp()
    }
}

/// `n` subjects and their truths, deterministic in `seed`.
pub fn population(n: usize, seed: u64) -> (Vec<Subject>, Vec<Truth>) {
    let mut rng = Rng::new(seed);
    let (mut subjects, mut truths) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let x = [(); 4].map(|_| rng.next_gaussian());
        let truth = Truth { x };
        let time = exp1(rng.next_f64()) / truth.hazard();
        let censor = rng.uniform(CENSOR.0, CENSOR.1);
        let observations = COVARIATES
            .iter()
            .zip(x)
            .map(|(name, v)| Observation {
                t: ENTRY,
                var: (*name).into(),
                value: Value::Number(v),
            })
            .collect();
        let event = (time < censor).then_some((time, CODE));
        subjects.push(subject(format!("single{i}"), observations, event, censor));
        truths.push(truth);
    }
    (subjects, truths)
}

#[cfg(test)]
mod tests {
    use super::super::sim::outcomes;
    use super::*;

    #[test]
    fn the_population_is_valid_deterministic_and_matches_its_closed_form_cif() {
        let (a, truths) = population(20_000, 3);
        assert_eq!(a, population(20_000, 3).0);
        for s in &a[..200] {
            s.validate().unwrap();
        }
        // Kaplan-Meier handles the censoring: its failure probability by 4
        // years against the mean closed-form CIF.
        let obs: Vec<survival::Obs> = outcomes(&a, &[CODE])
            .into_iter()
            .map(|(t, c)| survival::Obs { time: t, cause: c, weight: 1.0 })
            .collect();
        let observed = 1.0 - survival::estimate::kaplan_meier(&obs).at(4.0);
        let expected = truths.iter().map(|t| t.cif(4.0)).sum::<f64>() / truths.len() as f64;
        assert!((observed - expected).abs() < 0.01, "{observed} vs {expected}");
    }
}
