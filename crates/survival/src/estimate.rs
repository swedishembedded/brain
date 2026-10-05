// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Non-parametric estimates, weighted: Kaplan-Meier survival, the censoring
//! distribution `G` that inverse-probability-of-censoring weights divide by,
//! and Aalen-Johansen cumulative incidence for one cause among competing ones.
//!
//! Ties follow the usual convention: at a shared time, events happen before
//! censorings, so a subject censored at `t` is still at risk of an event at
//! `t`, and a subject with an event at `t` is no longer at risk of censoring
//! at `t` (scikit-survival's `CensoringDistributionEstimator` convention,
//! which the tests hold this to).

use crate::Obs;

/// A right-continuous step function: `value(t)` is the value after the last
/// jump at or before `t`, `1.0` (or `start`) before the first.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    start: f64,
    times: Vec<f64>,
    values: Vec<f64>,
}

impl Step {
    /// The function at `t`.
    pub fn at(&self, t: f64) -> f64 {
        match self.times.partition_point(|&x| x <= t) {
            0 => self.start,
            i => self.values[i - 1],
        }
    }

    /// The left limit at `t` (the value just before any jump at `t`).
    pub fn before(&self, t: f64) -> f64 {
        match self.times.partition_point(|&x| x < t) {
            0 => self.start,
            i => self.values[i - 1],
        }
    }

    /// Jump times and values after each jump.
    pub fn points(&self) -> (&[f64], &[f64]) {
        (&self.times, &self.values)
    }
}

/// Distinct times with, per time: weight of events by cause, weight censored,
/// and weight at risk (time >= t).
struct Table {
    times: Vec<f64>,
    events: Vec<Vec<f64>>,
    censored: Vec<f64>,
    at_risk: Vec<f64>,
}

fn table(obs: &[Obs], n_causes: usize) -> Table {
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by(|&a, &b| obs[a].time.partial_cmp(&obs[b].time).expect("finite times"));
    let total: f64 = obs.iter().map(|o| o.weight).sum();
    let mut t = Table {
        times: vec![],
        events: vec![],
        censored: vec![],
        at_risk: vec![],
    };
    let mut gone = 0.0;
    let mut i = 0;
    while i < order.len() {
        let time = obs[order[i]].time;
        let (mut ev, mut ce) = (vec![0.0; n_causes], 0.0);
        let at_risk = total - gone;
        while i < order.len() && obs[order[i]].time == time {
            let o = &obs[order[i]];
            match o.cause {
                Some(c) => ev[c] += o.weight,
                None => ce += o.weight,
            }
            gone += o.weight;
            i += 1;
        }
        t.times.push(time);
        t.events.push(ev);
        t.censored.push(ce);
        t.at_risk.push(at_risk);
    }
    t
}

fn n_causes(obs: &[Obs]) -> usize {
    obs.iter()
        .filter_map(|o| o.cause)
        .max()
        .map_or(1, |c| c + 1)
}

/// Kaplan-Meier probability of no event of ANY cause.
pub fn kaplan_meier(obs: &[Obs]) -> Step {
    let t = table(obs, n_causes(obs));
    let mut s = 1.0;
    let mut values = Vec::with_capacity(t.times.len());
    for i in 0..t.times.len() {
        let d: f64 = t.events[i].iter().sum();
        if d > 0.0 {
            s *= 1.0 - d / t.at_risk[i];
        }
        values.push(s);
    }
    Step {
        start: 1.0,
        times: t.times,
        values,
    }
}

/// The censoring distribution `G(t) = P(C > t)`: Kaplan-Meier with censoring
/// as the event and the subjects with an event at `t` removed from the risk
/// set at `t` (events precede censorings at a tie).
pub fn censoring(obs: &[Obs]) -> Step {
    let t = table(obs, n_causes(obs));
    let mut g = 1.0;
    let mut values = Vec::with_capacity(t.times.len());
    for i in 0..t.times.len() {
        let d: f64 = t.events[i].iter().sum();
        let risk = t.at_risk[i] - d;
        if t.censored[i] > 0.0 && risk > 0.0 {
            g *= 1.0 - t.censored[i] / risk;
        }
        values.push(g);
    }
    Step {
        start: 1.0,
        times: t.times,
        values,
    }
}

/// Aalen-Johansen cumulative incidence of `cause`, every other cause competing:
/// `F(t) = sum_{s <= t} S(s-) d_cause(s) / n(s)`.
pub fn aalen_johansen(obs: &[Obs], cause: usize) -> Step {
    let t = table(obs, n_causes(obs).max(cause + 1));
    let (mut s, mut f) = (1.0, 0.0);
    let mut values = Vec::with_capacity(t.times.len());
    for i in 0..t.times.len() {
        let n = t.at_risk[i];
        if n > 0.0 {
            f += s * t.events[i][cause] / n;
            s *= 1.0 - t.events[i].iter().sum::<f64>() / n;
        }
        values.push(f);
    }
    Step {
        start: 0.0,
        times: t.times,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<Obs> {
        vec![
            Obs::event(1.0, 0),
            Obs::censored(2.0),
            Obs::event(2.0, 1),
            Obs::event(3.0, 0),
            Obs::censored(4.0),
            Obs::event(5.0, 0),
        ]
    }

    #[test]
    fn kaplan_meier_by_hand() {
        let s = kaplan_meier(&data());
        // 6 at risk at 1: 5/6; at 2: 5 at risk, 1 event -> 4/5; at 3: 3 at risk -> 2/3; at 5: 1 at risk -> 0.
        assert!((s.at(1.0) - 5.0 / 6.0).abs() < 1e-12);
        assert!((s.at(2.5) - 5.0 / 6.0 * 4.0 / 5.0).abs() < 1e-12);
        assert!((s.at(3.0) - 5.0 / 6.0 * 4.0 / 5.0 * 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(s.at(5.0), 0.0);
        assert_eq!(s.at(0.5), 1.0);
        assert!((s.before(3.0) - 5.0 / 6.0 * 4.0 / 5.0).abs() < 1e-12);
    }

    #[test]
    fn censoring_puts_events_first_at_a_tie() {
        let g = censoring(&data());
        // at 2: 5 at risk minus the event at 2 = 4, one censored -> 3/4; at 4: 2 at risk, 1 censored -> 1/2.
        assert!((g.at(2.0) - 0.75).abs() < 1e-12);
        assert!((g.at(4.0) - 0.375).abs() < 1e-12);
    }

    #[test]
    fn aalen_johansen_partitions_with_survival() {
        let obs = data();
        let (s, f0, f1) = (
            kaplan_meier(&obs),
            aalen_johansen(&obs, 0),
            aalen_johansen(&obs, 1),
        );
        for t in [0.5, 1.0, 2.0, 3.5, 5.0] {
            assert!(
                (s.at(t) + f0.at(t) + f1.at(t) - 1.0).abs() < 1e-12,
                "t = {t}"
            );
        }
    }

    #[test]
    fn integer_weights_equal_duplicated_rows() {
        let obs = data();
        let weighted: Vec<Obs> = obs
            .iter()
            .enumerate()
            .map(|(i, o)| o.weighted(1.0 + (i % 3) as f64))
            .collect();
        let dup: Vec<Obs> = obs
            .iter()
            .enumerate()
            .flat_map(|(i, o)| std::iter::repeat_n(*o, 1 + i % 3))
            .collect();
        for t in [0.5, 1.0, 2.0, 3.0, 4.5, 6.0] {
            assert!((kaplan_meier(&weighted).at(t) - kaplan_meier(&dup).at(t)).abs() < 1e-12);
            assert!((censoring(&weighted).at(t) - censoring(&dup).at(t)).abs() < 1e-12);
            assert!(
                (aalen_johansen(&weighted, 0).at(t) - aalen_johansen(&dup, 0).at(t)).abs() < 1e-12
            );
        }
    }
}
