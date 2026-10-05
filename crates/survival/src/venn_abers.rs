// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Venn-Abers intervals for a predicted probability of an event by a
//! horizon: a model's risk turned into an interval whose width says how much
//! calibration data stands behind it.
//!
//! For a new subject with score `s`, the calibration subjects and the new
//! one (labelled once as having had the event by `t` and once as not) are
//! fitted by isotonic regression of outcome on score; the two fitted values
//! at `s` are the interval `(p0, p1)` (Vovk and Petej's inductive Venn-Abers
//! predictor). One of the two is calibrated whatever the model; with few
//! calibration subjects near `s` they lie far apart. The usual single
//! probability is `p1 / (1 - p0 + p1)` ([`merged`]).
//!
//! Under censoring the outcome by `t` is unknown for a subject censored
//! before it. Those subjects drop out and the others are reweighted by the
//! inverse probability of having stayed uncensored that long, as the IPCW
//! Brier score does (a case at `T_i <= t` by `1 / G(T_i)`, a subject free at
//! `t` by `1 / G(t)`, a competing event first by `1 / G(T_i)`), times its
//! sampling weight. The validity guarantee is then asymptotic, through `G`.
//! The new subject enters with the weight its caller gives it and no
//! censoring factor: it stands for itself, where each labelled calibration
//! subject also stands for the censored ones like it. Under sampling weights
//! that vary widely, give it [`VennAbers::mean_weight`] - an average
//! calibration subject - rather than its own survey weight, which would let
//! one heavily weighted subject swing its own interval to the extremes.

use crate::estimate::Step;
use crate::Obs;

/// Calibration subjects for one cause at one horizon, ready to give
/// intervals.
#[derive(Clone, Debug)]
pub struct VennAbers {
    /// Distinct scores, ascending.
    scores: Vec<f64>,
    /// Per distinct score: total weight and weighted count of events.
    weight: Vec<f64>,
    events: Vec<f64>,
    /// Labelled calibration subjects.
    count: usize,
}

/// One block of the pool-adjacent-violators fit.
#[derive(Clone, Copy)]
struct Block {
    weight: f64,
    events: f64,
}

impl Block {
    fn mean(&self) -> f64 {
        self.events / self.weight
    }
}

/// The isotonic fit of `points` (in score order) at index `at`.
fn isotonic_at(points: impl Iterator<Item = Block>, at: usize) -> f64 {
    // Each block remembers the index of its last point.
    let mut stack: Vec<(Block, usize)> = Vec::new();
    for (i, p) in points.enumerate() {
        let mut cur = (p, i);
        while let Some(&(prev, _)) = stack.last() {
            if prev.mean() < cur.0.mean() {
                break;
            }
            stack.pop();
            cur.0 = Block {
                weight: prev.weight + cur.0.weight,
                events: prev.events + cur.0.events,
            };
        }
        stack.push(cur);
    }
    let k = stack.partition_point(|&(_, last)| last < at);
    stack[k].0.mean()
}

impl VennAbers {
    /// From calibration subjects: `cif[i]` is subject `i`'s predicted
    /// probability of `cause` by `t`, `g` the censoring distribution (from
    /// the training data). Subjects censored before `t`, or whose `G` is zero
    /// where it is needed, drop out.
    pub fn at_horizon(cif: &[f64], obs: &[Obs], cause: usize, t: f64, g: &Step) -> VennAbers {
        assert_eq!(cif.len(), obs.len(), "one prediction per subject");
        let mut labelled: Vec<(f64, f64, f64)> = Vec::with_capacity(obs.len());
        for (&f, o) in cif.iter().zip(obs) {
            let (y, gi) = if o.time <= t {
                match o.cause {
                    Some(c) => (f64::from(u8::from(c == cause)), g.at(o.time)),
                    None => continue,
                }
            } else {
                (0.0, g.at(t))
            };
            if gi > 0.0 && o.weight > 0.0 {
                labelled.push((f, y, o.weight / gi));
            }
        }
        labelled.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut va = VennAbers {
            scores: Vec::new(),
            weight: Vec::new(),
            events: Vec::new(),
            count: labelled.len(),
        };
        for (s, y, w) in labelled {
            if va.scores.last() == Some(&s) {
                *va.weight.last_mut().expect("a score has a weight") += w;
                *va.events.last_mut().expect("a score has a count") += w * y;
            } else {
                va.scores.push(s);
                va.weight.push(w);
                va.events.push(w * y);
            }
        }
        va
    }

    /// How many distinct calibration scores there are.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// The mean weight (sampling weight over the censoring probability) of a
    /// labelled calibration subject: the weight a new subject enters with as
    /// an average one. `None` without calibration subjects.
    pub fn mean_weight(&self) -> Option<f64> {
        let n = self.count;
        (n > 0).then(|| self.weight.iter().sum::<f64>() / n as f64)
    }

    /// Whether no calibration subject was labelled.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// The interval `(p0, p1)` for a new subject of predicted probability
    /// `score` and sampling weight `weight`.
    pub fn interval(&self, score: f64, weight: f64) -> (f64, f64) {
        let at = self.scores.partition_point(|&s| s < score);
        let tie = self.scores.get(at) == Some(&score);
        let fit = |y: f64| {
            let new = Block {
                weight,
                events: weight * y,
            };
            let cal = |i: usize| Block {
                weight: self.weight[i],
                events: self.events[i],
            };
            if tie {
                let merged = Block {
                    weight: self.weight[at] + new.weight,
                    events: self.events[at] + new.events,
                };
                let points = (0..self.len()).map(|i| if i == at { merged } else { cal(i) });
                isotonic_at(points, at)
            } else {
                let points = (0..at)
                    .map(cal)
                    .chain(std::iter::once(new))
                    .chain((at..self.len()).map(cal));
                isotonic_at(points, at)
            }
        };
        (fit(0.0), fit(1.0))
    }
}

/// The single probability of a Venn-Abers interval: `p1 / (1 - p0 + p1)`,
/// the minimiser of the worst-case log loss over the two.
pub fn merged((p0, p1): (f64, f64)) -> f64 {
    p1 / (1.0 - p0 + p1)
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

    fn logit(p: f64) -> f64 {
        (p / (1.0 - p)).ln()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn a_small_case_matches_the_isotonic_fit_by_hand() {
        // Uncensored at the horizon: an event by t = 1 is label 1, a subject
        // still free at t is label 0.
        let obs = [
            Obs::censored(2.0),
            Obs::event(0.5, 0),
            Obs::censored(2.0),
            Obs::event(0.5, 0),
        ];
        let g = censoring(&[]);
        let va = VennAbers::at_horizon(&[0.1, 0.2, 0.3, 0.4], &obs, 0, 1.0, &g);
        // Labels by score: 0, 1, [new], 0, 1. With the new one 0, the
        // violators 1, 0, 0 pool to 1/3; with it 1, 1, 1, 0 pool to 2/3.
        let (p0, p1) = va.interval(0.25, 1.0);
        assert!(
            (p0 - 1.0 / 3.0).abs() < 1e-12 && (p1 - 2.0 / 3.0).abs() < 1e-12,
            "{p0} {p1}"
        );
        assert!((merged((p0, p1)) - 0.5).abs() < 1e-12);
        // A score tied with a calibration subject pools with it.
        let (q0, q1) = va.interval(0.2, 1.0);
        assert!(
            (q0 - 1.0 / 3.0).abs() < 1e-12 && (q1 - 2.0 / 3.0).abs() < 1e-12,
            "{q0} {q1}"
        );
        // Below every calibration score, labelled 1 it stands alone.
        let (r0, r1) = va.interval(0.0, 1.0);
        assert!(r0 == 0.0 && (r1 - 0.5).abs() < 1e-12, "{r0} {r1}");
    }

    #[test]
    fn the_mean_weight_is_an_average_labelled_subject() {
        let obs = [
            Obs::event(0.5, 0).weighted(2.0),
            Obs::censored(2.0).weighted(4.0),
            Obs::censored(0.5),
        ];
        let va = VennAbers::at_horizon(&[0.1, 0.1, 0.3], &obs, 0, 1.0, &censoring(&[]));
        // The subject censored before the horizon is not labelled.
        assert_eq!(va.mean_weight(), Some(3.0));
        assert_eq!(
            VennAbers::at_horizon(&[], &[], 0, 1.0, &censoring(&[])).mean_weight(),
            None
        );
    }

    #[test]
    fn an_integer_weight_counts_as_that_many_copies() {
        let obs = [
            Obs::event(0.5, 0),
            Obs::censored(2.0),
            Obs::event(0.7, 0),
            Obs::censored(3.0),
        ];
        let cif = [0.15, 0.35, 0.55, 0.75];
        let g = censoring(&[]);
        let mut weighted = obs;
        weighted[1].weight = 3.0;
        let mut copies = obs.to_vec();
        let mut cif_copies = cif.to_vec();
        for _ in 0..2 {
            copies.push(obs[1]);
            cif_copies.push(cif[1]);
        }
        let a = VennAbers::at_horizon(&cif, &weighted, 0, 1.0, &g);
        let b = VennAbers::at_horizon(&cif_copies, &copies, 0, 1.0, &g);
        for s in [0.1, 0.35, 0.5, 0.9] {
            let (x, y) = (a.interval(s, 1.0), b.interval(s, 1.0));
            assert!(
                (x.0 - y.0).abs() < 1e-12 && (x.1 - y.1).abs() < 1e-12,
                "{s}: {x:?} {y:?}"
            );
        }
    }

    /// Subjects with rate `0.2 exp(x)`, censored at rate `censor_rate`; the
    /// score is the true risk by `t` with every log-odds doubled: too extreme.
    fn overconfident(n: usize, censor_rate: f64, t: f64, seed: u64) -> (Vec<f64>, Vec<Obs>) {
        let mut u = uniforms(seed);
        let (mut score, mut obs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let rate = 0.2 * (3.0 * u() - 1.5).exp();
            let time = -u().ln() / rate;
            let c = if censor_rate > 0.0 {
                -u().ln() / censor_rate
            } else {
                f64::INFINITY
            };
            obs.push(if time <= c {
                Obs::event(time, 0)
            } else {
                Obs::censored(c)
            });
            score.push(sigmoid(2.0 * logit(1.0 - (-rate * t).exp())));
        }
        (score, obs)
    }

    #[test]
    fn overconfident_scores_come_out_calibrated_with_and_without_censoring() {
        let t = 3.0;
        for censor_rate in [0.0, 0.1] {
            let (cal_score, cal_obs) = overconfident(4000, censor_rate, t, 5);
            let (test_score, test_obs) = overconfident(8000, censor_rate, t, 6);
            let g = censoring(&cal_obs);
            let g_test = censoring(&test_obs);
            let before = at_horizon(&test_score, &test_obs, 0, t, &g_test, 10);
            assert!((before.slope - 0.5).abs() < 0.06, "{before:?}");
            let va = VennAbers::at_horizon(&cal_score, &cal_obs, 0, t, &g);
            let after: Vec<f64> = test_score
                .iter()
                .map(|&s| merged(va.interval(s, 1.0)))
                .collect();
            let cal = at_horizon(&after, &test_obs, 0, t, &g_test, 10);
            assert!(
                (cal.slope - 1.0).abs() < 0.1 && cal.mean_abs_gap < 0.02,
                "censoring {censor_rate}: {cal:?}"
            );
        }
    }

    #[test]
    fn intervals_narrow_as_calibration_data_grows() {
        let t = 3.0;
        let width = |n: usize| {
            let (s, o) = overconfident(n, 0.1, t, 7);
            let va = VennAbers::at_horizon(&s, &o, 0, t, &censoring(&o));
            let (probe, _) = overconfident(2000, 0.1, t, 8);
            probe
                .iter()
                .map(|&x| {
                    let (p0, p1) = va.interval(x, 1.0);
                    p1 - p0
                })
                .sum::<f64>()
                / probe.len() as f64
        };
        let (small, large) = (width(200), width(5000));
        assert!(
            large < 0.25 * small,
            "mean width {small} with 200, {large} with 5000"
        );
    }
}
