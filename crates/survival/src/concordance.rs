// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Concordance: of the pairs where one subject had the event first, the share
//! the model ranked that subject higher.
//!
//! A pair `(i, j)` is comparable when `i` had the event of interest at `T_i`
//! and `j` was still free of it after `T_i` (`T_j > T_i`, or censored at
//! exactly `T_i`). A subject whose event was a DIFFERENT cause never has the
//! event of interest: it stays comparable at every later time (the
//! competing-risks concordance of Wolbers et al.). Ties in the prediction
//! (within [`TIED_TOL`]) count one half.
//!
//! Uno's C restricts cases to `T_i < tau` and weights each by `1 / G(T_i)^2`,
//! which removes the dependence on the censoring distribution that Harrell's
//! C has. Sampling weights multiply both sides of a pair.
//!
//! `O(n log n)`: subjects in descending time feed a Fenwick tree over the
//! ranks of the predictions.

use crate::estimate::Step;
use crate::Obs;

/// Predictions within this distance of each other are a tie.
pub const TIED_TOL: f64 = 1e-8;

struct Fenwick {
    tree: Vec<f64>,
}

impl Fenwick {
    fn new(n: usize) -> Fenwick {
        Fenwick {
            tree: vec![0.0; n + 1],
        }
    }
    fn add(&mut self, i: usize, w: f64) {
        let mut i = i + 1;
        while i < self.tree.len() {
            self.tree[i] += w;
            i += i.isolate_lowest_one();
        }
    }
    /// Sum over ranks `< i`.
    fn below(&self, i: usize) -> f64 {
        let (mut i, mut s) = (i, 0.0);
        while i > 0 {
            s += self.tree[i];
            i -= i.isolate_lowest_one();
        }
        s
    }
}

/// `(concordant + tied / 2, comparable)` weighted sums, with each case
/// carrying `case_weight(i)` on top of its sampling weight.
fn counts(
    risk: &[f64],
    obs: &[Obs],
    cause: usize,
    is_case: impl Fn(usize) -> bool,
    case_weight: impl Fn(usize) -> f64,
) -> (f64, f64) {
    assert_eq!(risk.len(), obs.len(), "one prediction per subject");
    // Ties within TIED_TOL share a rank: quantise, then rank the distinct keys.
    let keys: Vec<i64> = risk.iter().map(|r| (r / TIED_TOL).round() as i64).collect();
    let mut distinct = keys.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let rank = |k: i64| distinct.binary_search(&k).expect("key present");
    // Event time for this cause: a competing event never reaches it.
    let time = |i: usize| match obs[i].cause {
        Some(c) if c != cause => f64::INFINITY,
        _ => obs[i].time,
    };
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by(|&a, &b| {
        time(b)
            .partial_cmp(&time(a))
            .expect("finite or infinite times")
    });
    let mut tree = Fenwick::new(distinct.len());
    let (mut conc, mut comp, mut seen) = (0.0, 0.0, 0.0);
    let mut i = 0;
    while i < order.len() {
        let t = time(order[i]);
        let block_end = order[i..]
            .iter()
            .position(|&j| time(j) != t)
            .map_or(order.len(), |p| i + p);
        let block = &order[i..block_end];
        // Censored at t (and anything that is not a case here) is comparable
        // with a case at t; cases at the same time are not with each other.
        for &j in block {
            if obs[j].cause != Some(cause) {
                tree.add(rank(keys[j]), obs[j].weight);
                seen += obs[j].weight;
            }
        }
        for &c in block {
            if obs[c].cause == Some(cause) && is_case(c) {
                let r = rank(keys[c]);
                let (lt, le) = (tree.below(r), tree.below(r + 1));
                let w = obs[c].weight * case_weight(c);
                conc += w * (lt + 0.5 * (le - lt));
                comp += w * seen;
            }
        }
        for &j in block {
            if obs[j].cause == Some(cause) {
                tree.add(rank(keys[j]), obs[j].weight);
                seen += obs[j].weight;
            }
        }
        i = block_end;
    }
    (conc, comp)
}

/// Harrell's C for `cause`: higher `risk` should mean an earlier event.
/// `None` when no pair is comparable.
pub fn harrell(risk: &[f64], obs: &[Obs], cause: usize) -> Option<f64> {
    let (conc, comp) = counts(risk, obs, cause, |_| true, |_| 1.0);
    (comp > 0.0).then(|| conc / comp)
}

/// Uno's C for `cause`, truncated at `tau`: cases with `T < tau`, each
/// weighted by `1 / G(T)^2` with `g` the censoring distribution (estimated on
/// the TRAINING data, [`crate::estimate::censoring`]). `None` when no pair is
/// comparable or `G` reaches zero at a case.
pub fn uno(risk: &[f64], obs: &[Obs], cause: usize, tau: f64, g: &Step) -> Option<f64> {
    if obs
        .iter()
        .any(|o| o.cause == Some(cause) && o.time < tau && g.at(o.time) <= 0.0)
    {
        return None;
    }
    let (conc, comp) = counts(
        risk,
        obs,
        cause,
        |i| obs[i].time < tau,
        |i| 1.0 / g.at(obs[i].time).powi(2),
    );
    (comp > 0.0).then(|| conc / comp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::censoring;

    /// The definition, pair by pair, for checking the fast path.
    fn brute(risk: &[f64], obs: &[Obs], cause: usize, tau: f64, g: Option<&Step>) -> f64 {
        let time = |i: usize| match obs[i].cause {
            Some(c) if c != cause => f64::INFINITY,
            _ => obs[i].time,
        };
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..obs.len() {
            if obs[i].cause != Some(cause) || obs[i].time >= tau {
                continue;
            }
            let wi = obs[i].weight * g.map_or(1.0, |g| 1.0 / g.at(obs[i].time).powi(2));
            for j in 0..obs.len() {
                let comparable =
                    time(j) > time(i) || (time(j) == time(i) && obs[j].cause != Some(cause));
                if i == j || !comparable {
                    continue;
                }
                let w = wi * obs[j].weight;
                den += w;
                num += w * if (risk[i] - risk[j]).abs() <= TIED_TOL {
                    0.5
                } else if risk[i] > risk[j] {
                    1.0
                } else {
                    0.0
                };
            }
        }
        num / den
    }

    fn sample(n: usize, seed: u64) -> (Vec<f64>, Vec<Obs>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut risk = vec![];
        let mut obs = vec![];
        for _ in 0..n {
            let r = (next() * 5.0).floor() / 5.0; // coarse: plenty of prediction ties
            let t = ((next() * 10.0) as u32) as f64; // coarse: plenty of time ties
            let kind = next();
            let o = if kind < 0.3 {
                Obs::censored(t)
            } else if kind < 0.8 {
                Obs::event(t, 0)
            } else {
                Obs::event(t, 1)
            };
            risk.push(r);
            obs.push(o.weighted(1.0 + (next() * 3.0).floor()));
        }
        (risk, obs)
    }

    #[test]
    fn the_fast_path_equals_the_definition_with_ties_weights_and_competing_events() {
        for seed in 1..6 {
            let (risk, obs) = sample(80, seed);
            let g = censoring(&obs);
            let h = harrell(&risk, &obs, 0).unwrap();
            assert!(
                (h - brute(&risk, &obs, 0, f64::INFINITY, None)).abs() < 1e-12,
                "harrell seed {seed}"
            );
            let u = uno(&risk, &obs, 0, 7.0, &g).unwrap();
            assert!(
                (u - brute(&risk, &obs, 0, 7.0, Some(&g))).abs() < 1e-12,
                "uno seed {seed}"
            );
        }
    }

    #[test]
    fn perfect_and_reversed_rankings() {
        let obs: Vec<Obs> = (1..=5).map(|t| Obs::event(t as f64, 0)).collect();
        let risk: Vec<f64> = (1..=5).map(|t| -(t as f64)).collect();
        assert_eq!(harrell(&risk, &obs, 0), Some(1.0));
        let rev: Vec<f64> = risk.iter().map(|r| -r).collect();
        assert_eq!(harrell(&rev, &obs, 0), Some(0.0));
        assert_eq!(harrell(&[0.0; 5], &obs, 0), Some(0.5));
        assert_eq!(harrell(&[1.0], &[Obs::censored(1.0)], 0), None);
    }
}
