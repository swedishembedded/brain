// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A hidden state with known dynamics, measured noisily and partially, that
//! drives both future measurements and an event, under an intervention.
//!
//! Swedish Embedded AB implements models that infer a hidden state from noisy
//! partial measurements and predict both its future readings and the events
//! it causes, for its clients. If your team needs expertise in joint
//! longitudinal and time-to-event modelling you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Each subject has a hidden state `z0 ~ N(0, 1)` and an action `a` in
//! {`control`, `treated`} taken at entry (the variable `arm`). Time `t` is
//! counted from entry. Before entry the state is steady at `z0`; after entry
//! `z(t) = z0 exp(-RATE[a] t)` - noise-free given `z0`, so the true
//! cumulative incidence given `z0` is exact. The treated state decays faster.
//!
//! - Three channels measure it partially and noisily: `m_j = LOAD[j] z +
//!   OFFSET[j] + SIGMA N(0, 1)`, `j = 1..3`. Before entry the subject has
//!   2 to 4 visits within [`LOOKBACK`] time units, each measuring a random
//!   non-empty subset of the channels. After entry, while alive, channels
//!   `m1` and `m2` are measured at [`FOLLOW_UP_VISITS`]: the forecast
//!   targets, never inputs.
//! - The hazard is `BASE * exp(BETA z(t))`; [`Truth::cif`] integrates it
//!   (Simpson, error far below sampling error). Follow-up is cut by a
//!   uniform censoring time.
//!
//! Everything is Gaussian-linear before entry, so the best possible
//! inference from the visits is a conjugate update: [`Truth::posterior`] and
//! [`Truth::oracle_cif`].

use serde::{Deserialize, Serialize};

use data::rng::Rng;

use super::sim::{exp1, invert, simpson, subject, ENTRY};
use crate::timeline::{Observation, Subject, Value};

/// The outcome code.
pub const CODE: &str = "event";
/// The measured channels.
pub const CHANNELS: [&str; 3] = ["m1", "m2", "m3"];
/// Channel loading on the state.
pub const LOAD: [f64; 3] = [1.0, 0.6, -0.8];
/// Channel offset.
pub const OFFSET: [f64; 3] = [0.0, 1.0, -1.0];
/// Channel noise standard deviation.
pub const SIGMA: f64 = 0.5;
/// Decay rate of the state after entry, for `control` and `treated`.
pub const RATE: [f64; 2] = [0.05, 0.35];
/// The action levels of the variable [`ARM`].
pub const ARMS: [&str; 2] = ["control", "treated"];
/// The action variable.
pub const ARM: &str = "arm";
/// Hazard per unit time at `z = 0`.
pub const BASE: f64 = 0.08;
/// Log-hazard per unit of the state.
pub const BETA: f64 = 1.0;
/// Pre-entry visits fall within this long before entry.
pub const LOOKBACK: f64 = 3.0;
/// Times since entry of the post-entry measurements of `m1` and `m2`.
pub const FOLLOW_UP_VISITS: [f64; 2] = [1.5, 3.0];
/// Censoring time, uniform in this range (time units since entry).
pub const CENSOR: (f64, f64) = (3.0, 6.0);

/// What generated one subject.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    /// The hidden state at entry.
    pub z0: f64,
    /// Index into [`ARMS`].
    pub arm: usize,
    /// Posterior mean of `z0` given the pre-entry measurements.
    pub posterior_mean: f64,
    /// Posterior variance of `z0` given the pre-entry measurements.
    pub posterior_var: f64,
}

impl Truth {
    /// The hidden state `t` after entry.
    pub fn state(&self, t: f64) -> f64 {
        self.z0 * (-RATE[self.arm] * t.max(0.0)).exp()
    }

    /// The true hazard `t` after entry.
    pub fn hazard(&self, t: f64) -> f64 {
        BASE * (BETA * self.state(t)).exp()
    }

    /// The true probability of the event by `t` after entry, given `z0`.
    pub fn cif(&self, t: f64) -> f64 {
        1.0 - (-simpson(|s| self.hazard(s), 0.0, t, 200)).exp()
    }

    /// The mean of channel `j` (index into [`CHANNELS`]) `ahead` after entry.
    pub fn channel_mean(&self, j: usize, ahead: f64) -> f64 {
        LOAD[j] * self.state(ahead) + OFFSET[j]
    }

    /// The best prediction from the pre-entry measurements: the event
    /// probability by `t` averaged over the Gaussian posterior of `z0`.
    pub fn oracle_cif(&self, t: f64) -> f64 {
        let sd = self.posterior_var.sqrt();
        let draw = |z0: f64| Truth { z0, ..self.clone() }.cif(t);
        let w = |z: f64| (-0.5 * z * z).exp();
        let norm = simpson(w, -8.0, 8.0, 160);
        simpson(|z| w(z) * draw(self.posterior_mean + sd * z), -8.0, 8.0, 160) / norm
    }
}

/// `n` subjects and their truths, deterministic in `seed`.
pub fn population(n: usize, seed: u64) -> (Vec<Subject>, Vec<Truth>) {
    let mut rng = Rng::new(seed);
    let (mut subjects, mut truths) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let z0 = rng.next_gaussian();
        let arm = usize::from(rng.next_f64() < 0.5);
        let measure = |rng: &mut Rng, j: usize, z: f64| {
            LOAD[j] * z + OFFSET[j] + SIGMA * rng.next_gaussian()
        };
        let mut observations = vec![Observation {
            t: ENTRY,
            var: ARM.into(),
            value: Value::Category(ARMS[arm].into()),
        }];
        // Conjugate update of the N(0, 1) prior with each reading.
        let (mut precision, mut weighted) = (1.0, 0.0);
        let visits = 2 + (rng.next_u64() % 3) as usize;
        for _ in 0..visits {
            let t = ENTRY - rng.uniform(0.0, LOOKBACK);
            let mut any = false;
            let mut taken = [false; 3];
            for slot in taken.iter_mut() {
                *slot = rng.next_f64() < 0.6;
                any |= *slot;
            }
            if !any {
                taken[(rng.next_u64() % 3) as usize] = true;
            }
            for (j, _) in taken.iter().enumerate().filter(|(_, on)| **on) {
                let y = measure(&mut rng, j, z0);
                precision += LOAD[j] * LOAD[j] / (SIGMA * SIGMA);
                weighted += LOAD[j] * (y - OFFSET[j]) / (SIGMA * SIGMA);
                observations.push(Observation {
                    t,
                    var: CHANNELS[j].into(),
                    value: Value::Number(y),
                });
            }
        }
        let truth = Truth {
            z0,
            arm,
            posterior_mean: weighted / precision,
            posterior_var: 1.0 / precision,
        };
        let target = exp1(rng.next_f64());
        let censor = rng.uniform(CENSOR.0, CENSOR.1);
        let cumulative = |s: f64| simpson(|u| truth.hazard(u), 0.0, s, 200);
        let time = invert(cumulative, target, censor);
        let exit = time.unwrap_or(censor);
        // Post-entry readings while the subject is alive and followed.
        for ahead in FOLLOW_UP_VISITS {
            if ahead < exit {
                for (j, channel) in CHANNELS.iter().enumerate().take(2) {
                    observations.push(Observation {
                        t: ENTRY + ahead,
                        var: (*channel).into(),
                        value: Value::Number(measure(&mut rng, j, truth.state(ahead))),
                    });
                }
            }
        }
        subjects.push(subject(
            format!("longitudinal{i}"),
            observations,
            time.map(|t| (t, CODE)),
            censor,
        ));
        truths.push(truth);
    }
    (subjects, truths)
}

/// Only what a model needs to know without measurements: the action (the
/// baseline covariate), with every channel reading dropped. The ablation of
/// the longitudinal information as a pure transform.
pub fn baseline_only(subjects: &[Subject]) -> Vec<Subject> {
    subjects
        .iter()
        .map(|s| Subject {
            observations: s
                .observations
                .iter()
                .filter(|o| o.var == ARM)
                .cloned()
                .collect(),
            ..s.clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::sim::outcomes;
    use super::*;

    #[test]
    fn the_population_is_valid_and_the_truth_matches_the_simulation() {
        let n = 30_000;
        let (subjects, truths) = population(n, 4);
        for s in &subjects[..300] {
            s.validate().unwrap();
        }
        assert_eq!(subjects, population(n, 4).0, "deterministic in the seed");
        let obs: Vec<survival::Obs> = outcomes(&subjects, &[CODE])
            .into_iter()
            .map(|(time, cause)| survival::Obs { time, cause, weight: 1.0 })
            .collect();
        let km = survival::estimate::aalen_johansen(&obs, 0);
        let mean = |f: &dyn Fn(&Truth) -> f64| truths.iter().map(f).sum::<f64>() / n as f64;
        for t in [2.0, 4.0] {
            let (truth, oracle) = (mean(&|x| x.cif(t)), mean(&|x| x.oracle_cif(t)));
            assert!((km.at(t) - truth).abs() < 0.01, "{t}: {} vs {truth}", km.at(t));
            assert!((oracle - truth).abs() < 0.01, "{t}: oracle {oracle} vs {truth}");
        }
    }

    #[test]
    fn the_posterior_is_calibrated_and_the_action_changes_the_dynamics() {
        let (_, truths) = population(20_000, 6);
        // z0 standardised by its posterior is standard normal.
        let z: Vec<f64> = truths
            .iter()
            .map(|t| (t.z0 - t.posterior_mean) / t.posterior_var.sqrt())
            .collect();
        let m = z.iter().sum::<f64>() / z.len() as f64;
        let v = z.iter().map(|x| (x - m).powi(2)).sum::<f64>() / z.len() as f64;
        assert!(m.abs() < 0.03 && (v - 1.0).abs() < 0.05, "mean {m}, var {v}");
        let hi = Truth { z0: 1.5, arm: 0, posterior_mean: 0.0, posterior_var: 1.0 };
        let treated = Truth { arm: 1, ..hi.clone() };
        assert!(treated.cif(5.0) < hi.cif(5.0), "treatment lowers risk");
        assert_eq!(treated.cif(0.0), 0.0);
    }

    #[test]
    fn baseline_only_keeps_the_action_and_the_outcome() {
        let (subjects, _) = population(50, 2);
        for (a, b) in subjects.iter().zip(baseline_only(&subjects)) {
            assert!(b.observations.iter().all(|o| o.var == ARM));
            assert_eq!((&a.events, &a.at_risk), (&b.events, &b.at_risk));
            b.validate().unwrap();
        }
    }
}
