// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A population whose risk factor drifts in continuous time and is measured
//! only at irregular visits before entry, so how much an old measurement
//! still says is a property of the process, not of the data layout.
//!
//! - The risk factor `x(t)` is a stationary Ornstein-Uhlenbeck process with
//!   unit variance, reverting to 0 at [`THETA`] per year.
//! - Each visit measures it with Gaussian noise of sd [`NOISE`] (variable
//!   `x`); the visits and the gap from the last one to entry are drawn from
//!   [`Gaps`], so a test population can have gaps no training population had.
//! - After entry, death has hazard `BASE * exp(AGEING (age - 55) + BETA x(entry))`.
//!
//! Because the process is linear and Gaussian, the best possible prediction
//! from the visits is exact: a Kalman filter gives `x(entry)`'s posterior,
//! and [`Posterior::cif`] integrates the hazard over it.

use data::rng::Rng;

use crate::timeline::{AtRisk, Event, Observation, Subject, Value};

/// Reversion rate of the risk factor, per year.
pub const THETA: f64 = 0.3;
/// Measurement noise, as a standard deviation.
pub const NOISE: f64 = 0.3;
/// Baseline death hazard at 55, per year.
pub const BASE: f64 = 0.02;
/// Log-hazard per year of age.
pub const AGEING: f64 = 0.08;
/// Log-hazard per unit of the risk factor at entry.
pub const BETA: f64 = 0.9;
/// The outcome code.
pub const CODE: &str = "death";

/// How visits are spread before entry.
#[derive(Clone, Copy, Debug)]
pub struct Gaps {
    /// From the last visit to entry, uniform in this range (years).
    pub last: (f64, f64),
    /// Between consecutive earlier visits, uniform in this range.
    pub between: (f64, f64),
    /// How many visits, uniform in this inclusive range.
    pub visits: (usize, usize),
}

/// What the visits say about the risk factor at entry, and the age then.
#[derive(Clone, Copy, Debug)]
pub struct Posterior {
    /// Age at entry.
    pub age: f64,
    /// Posterior mean of `x(entry)`.
    pub mean: f64,
    /// Posterior variance of `x(entry)`.
    pub var: f64,
}

/// The cumulative hazard by `t` after entry for a risk factor of 0.
fn base_cumulative(age: f64, t: f64) -> f64 {
    BASE * (AGEING * (age - 55.0)).exp() * (AGEING * t).exp_m1() / AGEING
}

impl Posterior {
    /// The probability of death within `t` of entry given the visits: the
    /// hazard integrated over the posterior of `x(entry)` (a fine
    /// quadrature on the standard normal).
    pub fn cif(&self, t: f64) -> f64 {
        const POINTS: usize = 401;
        const SPAN: f64 = 8.0;
        let c = base_cumulative(self.age, t);
        let sd = self.var.sqrt();
        let h = 2.0 * SPAN / (POINTS - 1) as f64;
        let (mut sum, mut norm) = (0.0, 0.0);
        for i in 0..POINTS {
            let z = -SPAN + i as f64 * h;
            let w = (-0.5 * z * z).exp() * if i == 0 || i == POINTS - 1 { 0.5 } else { 1.0 };
            sum += w * (-c * (BETA * (self.mean + sd * z)).exp()).exp();
            norm += w;
        }
        1.0 - sum / norm
    }
}

/// `n` subjects and the best prediction from each one's visits, followed for
/// `follow` years after entry (administrative censoring), deterministic in
/// `seed`.
pub fn population(n: usize, seed: u64, gaps: &Gaps, follow: f64) -> (Vec<Subject>, Vec<Posterior>) {
    let mut rng = Rng::new(seed);
    let mut subjects = Vec::with_capacity(n);
    let mut posteriors = Vec::with_capacity(n);
    for i in 0..n {
        let age = rng.uniform(40.0, 75.0);
        let k = gaps.visits.0 + (rng.next_u64() as usize) % (gaps.visits.1 - gaps.visits.0 + 1);
        // Visit times, latest first, then chronological.
        let mut times = vec![age - rng.uniform(gaps.last.0, gaps.last.1)];
        for _ in 1..k {
            let prev = *times.last().expect("a visit");
            times.push(prev - rng.uniform(gaps.between.0, gaps.between.1));
        }
        times.reverse();
        // Simulate the process exactly and filter it alongside.
        let mut x = rng.next_gaussian();
        let (mut mean, mut var) = (0.0, 1.0);
        let mut at = times[0];
        let mut observations = Vec::with_capacity(k);
        for &t in &times {
            let decay = (-THETA * (t - at)).exp();
            x = x * decay + (1.0 - decay * decay).sqrt() * rng.next_gaussian();
            (mean, var) = (mean * decay, var * decay * decay + 1.0 - decay * decay);
            let y = x + NOISE * rng.next_gaussian();
            let gain = var / (var + NOISE * NOISE);
            (mean, var) = (mean + gain * (y - mean), (1.0 - gain) * var);
            observations.push(Observation {
                t,
                var: "x".into(),
                value: Value::Number(y),
                unit: None,
            });
            at = t;
        }
        let decay = (-THETA * (age - at)).exp();
        x = x * decay + (1.0 - decay * decay).sqrt() * rng.next_gaussian();
        (mean, var) = (mean * decay, var * decay * decay + 1.0 - decay * decay);
        // Death by inverting the cumulative hazard; censored at `follow`.
        let target = -rng.next_f64().max(f64::MIN_POSITIVE).ln() * (-BETA * x).exp();
        let scale = BASE * (AGEING * (age - 55.0)).exp() / AGEING;
        let death = (1.0 + target / scale).ln() / AGEING;
        let mut events = Vec::new();
        let exit = if death < follow {
            events.push(Event {
                t: age + death,
                code: CODE.into(),
            });
            age + death
        } else {
            age + follow
        };
        subjects.push(Subject {
            subject_id: format!("drift{i}"),
            group_id: None,
            weight: 1.0,
            source: "synthetic".into(),
            entry: age,
            calendar_at_entry: 2010.0,
            observations,
            events,
            at_risk: vec![AtRisk {
                code: "*".into(),
                from: age,
                to: exit,
            }],
        });
        posteriors.push(Posterior { age, mean, var });
    }
    (subjects, posteriors)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAPS: Gaps = Gaps {
        last: (0.0, 2.0),
        between: (0.5, 2.0),
        visits: (1, 4),
    };

    #[test]
    fn the_population_is_valid_and_the_posterior_matches_the_simulated_deaths() {
        let (subjects, posteriors) = population(20_000, 3, &GAPS, 10.0);
        for s in &subjects[..200] {
            s.validate().unwrap();
        }
        let died = |s: &Subject| s.events.iter().any(|e| e.t - s.entry <= 5.0);
        let observed = subjects.iter().filter(|s| died(s)).count() as f64 / subjects.len() as f64;
        let expected = posteriors.iter().map(|p| p.cif(5.0)).sum::<f64>() / posteriors.len() as f64;
        assert!(
            (observed - expected).abs() < 0.01,
            "observed {observed:.4} vs posterior {expected:.4}"
        );
        // Those whose visits point to a high risk factor die more, as predicted.
        let high: Vec<usize> = (0..subjects.len())
            .filter(|&i| posteriors[i].mean > 1.0)
            .collect();
        let obs_high =
            high.iter().filter(|&&i| died(&subjects[i])).count() as f64 / high.len() as f64;
        let exp_high =
            high.iter().map(|&i| posteriors[i].cif(5.0)).sum::<f64>() / high.len() as f64;
        assert!(
            (obs_high - exp_high).abs() < 0.03,
            "high risk: observed {obs_high:.4} vs {exp_high:.4}"
        );
    }

    #[test]
    fn an_old_measurement_says_less() {
        let long = Gaps {
            last: (8.0, 10.0),
            ..GAPS
        };
        let (_, near) = population(2000, 4, &GAPS, 10.0);
        let (_, far) = population(2000, 4, &long, 10.0);
        let mean_var = |p: &[Posterior]| p.iter().map(|x| x.var).sum::<f64>() / p.len() as f64;
        assert!(
            mean_var(&far) > 0.9 && mean_var(&near) < 0.6,
            "{} {}",
            mean_var(&near),
            mean_var(&far)
        );
    }
}
