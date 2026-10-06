// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Time-dependent AUROC at a horizon: cumulative cases against dynamic
//! controls (Uno et al. 2007; scikit-survival's `cumulative_dynamic_auc`).
//!
//! At horizon `t` a CASE is a subject with the cause of interest at `T_i <= t`,
//! weighted by `1 / G(T_i)` with `g` the censoring distribution estimated on
//! the training data; a CONTROL is a subject still free at `t` (`T_j > t`). The
//! AUC is the weighted share of case-control pairs the model ranked in the
//! right order, a tie in the prediction (within [`TIED_TOL`]) counting one
//! half. A subject censored by `t` is neither (its weight is carried by the
//! cases), and so is a subject whose COMPETING event came by `t`: its outcome
//! by `t` is known to be a non-case but it was not at risk afterwards, so it
//! is left out of the controls (Blanche et al.'s "type 1" controls) rather
//! than counted on a side the model was never asked to rank. Sampling
//! weights multiply both sides of a pair.
//!
//! Like concordance this is a rank statistic: report it beside a Brier score
//! and calibration. `O(n log n)`: controls are sorted once by prediction and
//! each case reads the weight below it from a prefix sum.

use crate::concordance::TIED_TOL;
use crate::estimate::Step;
use crate::Obs;

fn key(r: f64) -> i64 {
    (r / TIED_TOL).round() as i64
}

/// The AUC at `t` of `risk` (each subject's predicted risk of `cause` by `t`;
/// higher should mean the event came by `t`). `None` when there is no case or
/// no control, `G` is zero at a case, or a prediction is not finite.
pub fn at(risk: &[f64], obs: &[Obs], cause: usize, t: f64, g: &Step) -> Option<f64> {
    assert_eq!(risk.len(), obs.len(), "one prediction per subject");
    if risk.iter().any(|r| !r.is_finite()) {
        return None;
    }
    // Controls by prediction key, with the running weight at or below each.
    let mut controls: Vec<(i64, f64)> = obs
        .iter()
        .zip(risk)
        .filter(|(o, _)| o.time > t)
        .map(|(o, &r)| (key(r), o.weight))
        .collect();
    controls.sort_by_key(|c| c.0);
    let keys: Vec<i64> = controls.iter().map(|c| c.0).collect();
    let mut prefix = Vec::with_capacity(controls.len() + 1);
    prefix.push(0.0);
    for c in &controls {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + c.1);
    }
    let control_weight = prefix[controls.len()];
    let (mut num, mut cases) = (0.0, 0.0);
    for (o, &r) in obs.iter().zip(risk) {
        if o.cause != Some(cause) || o.time > t {
            continue;
        }
        let gi = g.at(o.time);
        if gi <= 0.0 {
            return None;
        }
        let k = key(r);
        let below = prefix[keys.partition_point(|&x| x < k)];
        let through = prefix[keys.partition_point(|&x| x <= k)];
        let w = o.weight / gi;
        num += w * (below + 0.5 * (through - below));
        cases += w;
    }
    (cases > 0.0 && control_weight > 0.0).then(|| num / (cases * control_weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::censoring;

    /// The definition, pair by pair.
    fn brute(risk: &[f64], obs: &[Obs], cause: usize, t: f64, g: &Step) -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (i, oi) in obs.iter().enumerate() {
            if oi.cause != Some(cause) || oi.time > t {
                continue;
            }
            for (j, oj) in obs.iter().enumerate() {
                if oj.time <= t {
                    continue;
                }
                let w = oi.weight / g.at(oi.time) * oj.weight;
                den += w;
                num += w * if key(risk[i]) == key(risk[j]) {
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

    #[test]
    fn a_hand_computed_example() {
        // No censoring, so G = 1. At t = 2: cases are subjects 0 (risk .9) and
        // 1 (risk .4); controls are 2 (risk .6) and 3 (risk .1). Pairs: .9>.6,
        // .9>.1, .4<.6, .4>.1 -> 3 of 4.
        let obs = [
            Obs::event(1.0, 0),
            Obs::event(2.0, 0),
            Obs::event(5.0, 0),
            Obs::event(6.0, 0),
        ];
        let risk = [0.9, 0.4, 0.6, 0.1];
        let g = censoring(&obs);
        assert_eq!(at(&risk, &obs, 0, 2.0, &g), Some(0.75));
        // A competing event by t is neither case nor control: subject 1 as a
        // competing death leaves one case against the same two controls.
        let mut other = obs;
        other[1] = Obs::event(2.0, 1);
        assert_eq!(at(&risk, &other, 0, 2.0, &censoring(&other)), Some(1.0));
        // Ties count one half; no controls or no cases is not measured.
        assert_eq!(at(&[0.5; 4], &obs, 0, 2.0, &g), Some(0.5));
        assert_eq!(at(&risk, &obs, 0, 0.5, &g), None);
        assert_eq!(at(&risk, &obs, 0, 9.0, &g), None);
        assert_eq!(at(&[f64::NAN, 0.0, 0.0, 0.0], &obs, 0, 2.0, &g), None);
    }

    #[test]
    fn the_fast_path_equals_the_definition_with_censoring_ties_weights_and_competing_events() {
        for seed in 1..6u64 {
            let mut s = seed;
            let mut next = || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            };
            let (mut risk, mut obs) = (vec![], vec![]);
            for _ in 0..90 {
                risk.push((next() * 5.0).floor() / 5.0);
                let t = ((next() * 10.0) as u32) as f64;
                let kind = next();
                let o = if kind < 0.3 {
                    Obs::censored(t)
                } else if kind < 0.8 {
                    Obs::event(t, 0)
                } else {
                    Obs::event(t, 1)
                };
                obs.push(o.weighted(1.0 + (next() * 3.0).floor()));
            }
            let g = censoring(&obs);
            for t in [2.0, 5.0, 8.0] {
                let got = at(&risk, &obs, 0, t, &g).unwrap();
                assert!((got - brute(&risk, &obs, 0, t, &g)).abs() < 1e-12, "seed {seed} t {t}");
            }
        }
    }
}
