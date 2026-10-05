// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Brier score at a horizon under censoring, and its integral over a
//! window: the mean squared distance between a predicted probability of the
//! event by `t` and what happened, each observed subject reweighted by the
//! inverse probability of having stayed uncensored that long.
//!
//! With `F_i` the predicted cumulative incidence of the cause by `t`:
//! - a case (that cause, at `T_i <= t`) contributes `(1 - F_i)^2 / G(T_i)`;
//! - a subject still free at `t` (`T_i > t`) contributes `F_i^2 / G(t)`;
//! - a subject whose competing event came first (`T_i <= t`, another cause)
//!   contributes `F_i^2 / G(T_i)`: its outcome by `t` is known;
//! - a subject censored by `t` contributes nothing; the weights stand in.
//!
//! The mean is over ALL subjects (weighted), which is what makes the
//! estimator consistent (Graf et al.; Schoop et al. for competing risks). With
//! one cause it equals scikit-survival's `brier_score` on `1 - F`.

use crate::estimate::Step;
use crate::Obs;

/// Each subject's own term of the IPCW Brier score at `t` (before its
/// sampling weight): what [`brier`] averages, and what a paired comparison of
/// two models differences subject by subject. `None` when `G` is zero where
/// it is needed.
pub fn brier_terms(cif: &[f64], obs: &[Obs], cause: usize, t: f64, g: &Step) -> Option<Vec<f64>> {
    assert_eq!(cif.len(), obs.len(), "one prediction per subject");
    let g_t = g.at(t);
    cif.iter()
        .zip(obs)
        .map(|(f, o)| {
            if o.time <= t {
                match o.cause {
                    Some(c) => {
                        let gi = g.at(o.time);
                        let y = if c == cause { 1.0 } else { 0.0 };
                        (gi > 0.0).then(|| (y - f).powi(2) / gi)
                    }
                    None => Some(0.0),
                }
            } else {
                (g_t > 0.0).then(|| f * f / g_t)
            }
        })
        .collect()
}

/// The IPCW Brier score of `cif` (one prediction per subject: the
/// probability of `cause` by `t`). `g` is the censoring distribution,
/// estimated on the training data. `None` when `G` is zero where it is needed.
pub fn brier(cif: &[f64], obs: &[Obs], cause: usize, t: f64, g: &Step) -> Option<f64> {
    let terms = brier_terms(cif, obs, cause, t, g)?;
    let weight: f64 = obs.iter().map(|o| o.weight).sum();
    (weight > 0.0).then(|| {
        terms
            .iter()
            .zip(obs)
            .map(|(x, o)| x * o.weight)
            .sum::<f64>()
            / weight
    })
}

/// The Brier score integrated over `times` by the trapezoid rule and divided
/// by the window's length (scikit-survival's `integrated_brier_score`).
/// `cif_at(k)` gives every subject's prediction at `times[k]`.
pub fn integrated_brier(
    times: &[f64],
    cif_at: impl Fn(usize) -> Vec<f64>,
    obs: &[Obs],
    cause: usize,
    g: &Step,
) -> Option<f64> {
    if times.len() < 2 || times.windows(2).any(|w| w[1] <= w[0]) {
        return None;
    }
    let scores: Option<Vec<f64>> = (0..times.len())
        .map(|k| brier(&cif_at(k), obs, cause, times[k], g))
        .collect();
    let scores = scores?;
    let area: f64 = times
        .windows(2)
        .zip(scores.windows(2))
        .map(|(t, s)| 0.5 * (s[0] + s[1]) * (t[1] - t[0]))
        .sum();
    Some(area / (times[times.len() - 1] - times[0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::censoring;

    #[test]
    fn a_perfect_prediction_scores_zero_and_weights_equal_duplicates() {
        let obs = vec![
            Obs::event(1.0, 0),
            Obs::event(2.0, 1),
            Obs::censored(1.5),
            Obs::event(4.0, 0),
            Obs::censored(5.0),
        ];
        let g = censoring(&obs);
        // At t = 3: subject 0 is a case, 1 a competing non-case, 2 censored, 3 and 4 free.
        let perfect = [1.0, 0.0, 0.3, 0.0, 0.0];
        assert_eq!(brier(&perfect, &obs, 0, 3.0, &g), Some(0.0));
        let pred = [0.6, 0.2, 0.3, 0.4, 0.1];
        let w: Vec<Obs> = obs
            .iter()
            .enumerate()
            .map(|(i, o)| o.weighted(1.0 + i as f64))
            .collect();
        let (dp, dobs): (Vec<f64>, Vec<Obs>) = obs
            .iter()
            .zip(pred)
            .enumerate()
            .flat_map(|(i, (o, p))| std::iter::repeat_n((p, *o), 1 + i))
            .unzip();
        let a = brier(&pred, &w, 0, 3.0, &censoring(&w)).unwrap();
        let b = brier(&dp, &dobs, 0, 3.0, &censoring(&dobs)).unwrap();
        assert!((a - b).abs() < 1e-12);
    }
}
