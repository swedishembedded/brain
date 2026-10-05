// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A synthetic population whose true hazards are known, so a trained model
//! can be checked against the truth rather than against itself.
//!
//! Each subject enters at an age in `[40, 70)` and a calendar year in
//! `[2000, 2015)` with:
//! - `x1`, `x2` ~ N(0, 1); `x2` below -1 is reported only as "below -1" (a
//!   detection limit), so its information survives only through the Tobit
//!   state;
//! - `group` in {a, b, c};
//! - a history event `dx` with probability 0.3, 0-10 years before entry;
//! - an irrelevant variable `noise` ~ N(0, 1).
//!
//! Outcomes, followed for 5-15 years (uniform administrative censoring):
//! - `death:a` hazard `0.01 * exp(0.08 (age - 55) + 0.7 x1)` (absorbing);
//! - `death:b` hazard `0.005 * exp(0.08 (age - 55) + 0.5 x2 + 0.6 [dx])` (absorbing);
//! - `onset` hazard `0.03 * exp(0.8 [group = b])`, a first occurrence that
//!   death competes with.
//!
//! Hazards change with age, so events are simulated on a fine grid
//! ([`STEP`]); [`Truth::cif`] integrates the same hazards on the same grid.

use data::rng::Rng;

use crate::timeline::{AtRisk, Event, Observation, Subject, Value};

pub mod drifting;

/// Simulation and integration step, in years.
pub const STEP: f64 = 0.02;
/// The outcome codes, in the order the generator reports them.
pub const CODES: [&str; 3] = ["death:a", "death:b", "onset"];
/// Which codes are absorbing.
pub const ABSORBING: [bool; 3] = [true, true, false];

/// The true covariates of one synthetic subject.
#[derive(Clone, Debug)]
pub struct Truth {
    age: f64,
    x1: f64,
    x2: f64,
    dx: bool,
    group_b: bool,
}

impl Truth {
    fn hazards(&self, t: f64) -> [f64; 3] {
        let ageing = 0.08 * (self.age + t - 55.0);
        [
            0.01 * (ageing + 0.7 * self.x1).exp(),
            0.005 * (ageing + 0.5 * self.x2 + if self.dx { 0.6 } else { 0.0 }).exp(),
            0.03 * if self.group_b { 0.8f64.exp() } else { 1.0 },
        ]
    }

    /// The true cumulative incidence of code `k` by `t` years after entry,
    /// with the same competing structure the model uses.
    pub fn cif(&self, k: usize, t: f64) -> f64 {
        let (mut free, mut cif, mut s) = (1.0, 0.0, 0.0);
        while s < t {
            let dt = STEP.min(t - s);
            let h = self.hazards(s + 0.5 * dt);
            let mut lam = h[0] + h[1];
            if !ABSORBING[k] {
                lam += h[k];
            }
            let leave = 1.0 - (-lam * dt).exp();
            cif += h[k] / lam * free * leave;
            free *= 1.0 - leave;
            s += dt;
        }
        cif
    }
}

/// Standard deviation of the noise around `x1`'s trajectory after entry.
pub const X1_NOISE: f64 = 0.3;
/// Times after entry at which [`population_with_followup`] measures `x1`.
pub const FOLLOW_UP_VISITS: [f64; 3] = [1.0, 3.0, 5.0];

impl Truth {
    /// The true distribution of `x1` at `ahead` after entry, `(mean, sd)`:
    /// it drifts up by 0.2 a year in group b and down by 0.05 elsewhere.
    pub fn x1_at(&self, ahead: f64) -> (f64, f64) {
        let slope = if self.group_b { 0.2 } else { -0.05 };
        (self.x1 + slope * ahead, X1_NOISE)
    }
}

/// [`population`] with `x1` measured again at [`FOLLOW_UP_VISITS`] after
/// entry, at each visit the subject was alive and under follow-up for: the
/// longitudinal measurements a forecast is scored against.
pub fn population_with_followup(n: usize, seed: u64) -> (Vec<Subject>, Vec<Truth>) {
    let (mut subjects, truths) = population(n, seed);
    let mut rng = Rng::new(seed ^ 0x5EED_F011_0000_0001);
    for (s, tr) in subjects.iter_mut().zip(&truths) {
        let exit = s.at_risk[0].to;
        for ahead in FOLLOW_UP_VISITS {
            if s.entry + ahead < exit {
                let (mean, sd) = tr.x1_at(ahead);
                s.observations.push(Observation {
                    t: s.entry + ahead,
                    var: "x1".into(),
                    value: Value::Number(mean + sd * rng.next_gaussian()),
                });
            }
        }
    }
    (subjects, truths)
}

/// `n` subjects and their truths, deterministic in `seed`.
pub fn population(n: usize, seed: u64) -> (Vec<Subject>, Vec<Truth>) {
    let mut rng = Rng::new(seed);
    let mut subjects = Vec::with_capacity(n);
    let mut truths = Vec::with_capacity(n);
    for i in 0..n {
        let truth = Truth {
            age: rng.uniform(40.0, 70.0),
            x1: rng.next_gaussian(),
            x2: rng.next_gaussian(),
            dx: rng.next_f64() < 0.3,
            group_b: false,
        };
        let group = ["a", "b", "c"][(rng.next_u64() % 3) as usize];
        let truth = Truth {
            group_b: group == "b",
            ..truth
        };
        let entry = truth.age;
        let follow = rng.uniform(5.0, 15.0);
        let mut observations = vec![
            Observation {
                t: entry,
                var: "x1".into(),
                value: Value::Number(truth.x1),
            },
            Observation {
                t: entry,
                var: "x2".into(),
                value: if truth.x2 < -1.0 {
                    Value::Below { below: -1.0 }
                } else {
                    Value::Number(truth.x2)
                },
            },
            Observation {
                t: entry,
                var: "noise".into(),
                value: Value::Number(rng.next_gaussian()),
            },
            Observation {
                t: entry,
                var: "group".into(),
                value: Value::Category(group.into()),
            },
        ];
        observations.push(Observation {
            t: entry,
            var: "age".into(),
            value: Value::Number(entry),
        });
        let mut events = Vec::new();
        if truth.dx {
            events.push(Event {
                t: entry - rng.uniform(0.0, 10.0),
                code: "dx".into(),
            });
        }
        // Simulate the competing processes on the grid.
        let (mut s, mut onset_seen) = (0.0, false);
        let mut exit = entry + follow;
        while s < follow {
            let dt = STEP.min(follow - s);
            let h = truth.hazards(s + 0.5 * dt);
            let u = rng.next_f64();
            let total = h[0] + h[1] + if onset_seen { 0.0 } else { h[2] };
            if u < 1.0 - (-total * dt).exp() {
                let pick = rng.next_f64() * total;
                let t = entry + s + 0.5 * dt;
                if pick < h[0] {
                    events.push(Event {
                        t,
                        code: "death:a".into(),
                    });
                    exit = t; // the SAME number: follow-up ends at the death
                    break;
                } else if pick < h[0] + h[1] {
                    events.push(Event {
                        t,
                        code: "death:b".into(),
                    });
                    exit = t;
                    break;
                } else {
                    events.push(Event {
                        t,
                        code: "onset".into(),
                    });
                    onset_seen = true;
                }
            }
            s += dt;
        }
        subjects.push(Subject {
            subject_id: format!("syn{i}"),
            group_id: None,
            weight: 1.0,
            source: "synthetic".into(),
            entry,
            calendar_at_entry: rng.uniform(2000.0, 2015.0),
            observations,
            events,
            at_risk: vec![AtRisk {
                code: "*".into(),
                from: entry,
                to: exit,
            }],
        });
        truths.push(truth);
    }
    (subjects, truths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_population_is_valid_deterministic_and_has_every_outcome() {
        let (a, _) = population(400, 5);
        let (b, _) = population(400, 5);
        assert_eq!(a, b);
        for s in &a {
            s.validate().unwrap();
        }
        for code in CODES {
            let n = a
                .iter()
                .filter(|s| s.events.iter().any(|e| e.code == code))
                .count();
            assert!(n > 10, "{code}: {n} events in 400 subjects");
        }
        assert!(a.iter().any(|s| s
            .observations
            .iter()
            .any(|o| matches!(o.value, Value::Below { .. }))));
    }

    /// Every outcome event the generator wrote is an event in the encoded
    /// data: none is lost to its window boundary.
    #[test]
    fn every_generated_outcome_survives_encoding() {
        use crate::encode::encode;
        use crate::vocab::{FitOptions, Vocab};
        let (subjects, _) = population(3000, 4);
        let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
        let vocab = Vocab::fit(&subjects, &codes, &codes[..2], &FitOptions::default()).unwrap();
        let mut cfg = crate::HorizonConfig::tiny(vocab.len(), 3);
        cfg.knots = vec![0.0, 5.0, 10.0, 15.0, 20.0];
        for (k, code) in CODES.iter().enumerate() {
            let raw = subjects
                .iter()
                .filter(|s| s.events.iter().any(|e| e.code == *code && e.t > s.entry))
                .count();
            let enc = subjects
                .iter()
                .filter(|s| encode(s, &vocab, &cfg).outcomes[k].event_piece.is_some())
                .count();
            assert_eq!(enc, raw, "{code}");
        }
    }

    #[test]
    fn the_true_cif_matches_the_simulated_frequency() {
        let (subjects, truths) = population(4000, 9);
        // Among subjects followed at least 5 years, the share dying of cause a
        // within 5 years against the mean true CIF.
        let five: Vec<usize> = (0..subjects.len())
            .filter(|&i| {
                subjects[i].at_risk[0].to - subjects[i].entry >= 5.0
                    || subjects[i]
                        .events
                        .iter()
                        .any(|e| e.code.starts_with("death"))
            })
            .collect();
        let observed = five
            .iter()
            .filter(|&&i| {
                subjects[i]
                    .events
                    .iter()
                    .any(|e| e.code == "death:a" && e.t - subjects[i].entry <= 5.0)
            })
            .count() as f64
            / five.len() as f64;
        let expected: f64 =
            five.iter().map(|&i| truths[i].cif(0, 5.0)).sum::<f64>() / five.len() as f64;
        assert!(
            (observed - expected).abs() < 0.015,
            "observed {observed:.4} vs true {expected:.4}"
        );
    }
}
