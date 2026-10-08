// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Summaries of the whole calibration curve at a horizon.
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in checking that a predicted risk means what it says across
//! the whole range, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! [`calibration::at_horizon`](crate::calibration::at_horizon) reports an
//! intercept, a slope and a table by risk group, none of which says how far
//! the predictions are from the truth at a typical subject. Here the observed
//! risk is smoothed as a function of the predicted risk and the absolute gap
//! `|predicted - smoothed observed|` is summarised over the subjects: its
//! weighted mean (the integrated calibration index, ICI), median (E50), 90th
//! percentile (E90) and maximum over the evaluation grid (Emax).
//!
//! The observed side is the censoring-aware outcome of
//! [`calibration`](crate::calibration) (events of the cause by `t`, inverse
//! probability of censoring weights, subjects censored before `t` dropped),
//! smoothed by local linear regression with tricube weights on the weighted
//! rank of the predicted risk: a window holding a `span` share of the weight,
//! evaluated on a grid of weighted quantiles and interpolated in the predicted
//! risk. Austin et al. (2020) smooth with a hazard regression instead; this
//! needs no model of the hazard, only the censoring distribution the other
//! calibration summaries use.

use crate::calibration::{labelled, sigmoid};
use crate::estimate::Step;
use crate::Obs;

/// Evaluation points of the smoother, spread over the weighted quantiles.
const GRID: usize = 101;

/// Gaps between predicted and smoothed observed risk.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveErrors {
    /// Weighted mean absolute gap: the integrated calibration index.
    pub ici: f64,
    /// Weighted median gap.
    pub e50: f64,
    /// Weighted 90th percentile of the gap.
    pub e90: f64,
    /// Largest gap on the evaluation grid.
    pub emax: f64,
}

/// Calibration-curve errors of `cif` (each subject's predicted probability of
/// `cause` by `t`) against the outcomes. `span` is the share of the weight in
/// the smoothing window (0.1 to 0.5 is usual). `None` when too few subjects
/// have a known outcome to smooth.
pub fn errors_at_horizon(
    cif: &[f64],
    obs: &[Obs],
    cause: usize,
    t: f64,
    g: &Step,
    span: f64,
) -> Option<CurveErrors> {
    assert_eq!(cif.len(), obs.len(), "one prediction per subject");
    assert!(span > 0.0 && span <= 1.0, "span is a share of the weight");
    let l = labelled(cif, obs, cause, t, g);
    if l.x.len() < 20 {
        return None;
    }
    let mut order: Vec<usize> = (0..l.x.len()).collect();
    order.sort_by(|&a, &b| l.x[a].partial_cmp(&l.x[b]).expect("finite predictions"));
    let p: Vec<f64> = order.iter().map(|&i| sigmoid(l.x[i])).collect();
    let y: Vec<f64> = order.iter().map(|&i| l.y[i]).collect();
    let w: Vec<f64> = order.iter().map(|&i| l.v[i]).collect();
    let total: f64 = w.iter().sum();
    // Weighted rank of each subject: the midpoint of its weight in the order.
    let mut u = Vec::with_capacity(p.len());
    let mut acc = 0.0;
    for wi in &w {
        u.push((acc + wi / 2.0) / total);
        acc += wi;
    }
    let half = span / 2.0;
    let mut grid_p = Vec::with_capacity(GRID);
    let mut grid_f = Vec::with_capacity(GRID);
    for k in 0..GRID {
        let u0 = k as f64 / (GRID - 1) as f64;
        // A window of constant width in rank, shifted to stay inside [0, 1].
        let centre = u0.clamp(half, 1.0 - half);
        let (lo, hi) = (centre - half, centre + half);
        let a = u.partition_point(|&x| x < lo);
        let b = u.partition_point(|&x| x <= hi);
        if b <= a {
            continue;
        }
        // The predicted risk at this quantile.
        let x0 = quantile(&p, &u, u0).unwrap_or(p[a]);
        let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in a..b {
            let d = ((u[i] - centre).abs() / half).min(1.0);
            let k = (1.0 - d * d * d).powi(3);
            let wi = w[i] * k;
            let dx = p[i] - x0;
            sw += wi;
            swx += wi * dx;
            swy += wi * y[i];
            swxx += wi * dx * dx;
            swxy += wi * dx * y[i];
        }
        let det = sw * swxx - swx * swx;
        let f = if det.abs() > 1e-18 * (sw * swxx).abs().max(1e-300) {
            (swxx * swy - swx * swxy) / det
        } else {
            swy / sw
        };
        grid_p.push(x0);
        grid_f.push(f.clamp(0.0, 1.0));
    }
    if grid_p.len() < 2 {
        return None;
    }
    let smooth = |x: f64| -> f64 {
        let j = grid_p.partition_point(|&g| g <= x);
        if j == 0 {
            return grid_f[0];
        }
        if j == grid_p.len() {
            return grid_f[grid_f.len() - 1];
        }
        let (x0, x1) = (grid_p[j - 1], grid_p[j]);
        if x1 <= x0 {
            return grid_f[j];
        }
        grid_f[j - 1] + (grid_f[j] - grid_f[j - 1]) * (x - x0) / (x1 - x0)
    };
    let mut gaps: Vec<(f64, f64)> = p
        .iter()
        .zip(&w)
        .map(|(&pi, &wi)| ((pi - smooth(pi)).abs(), wi))
        .collect();
    let ici = gaps.iter().map(|(e, wi)| e * wi).sum::<f64>() / total;
    gaps.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite gaps"));
    let weighted_quantile = |q: f64| {
        let mut acc = 0.0;
        for (e, wi) in &gaps {
            acc += wi;
            if acc >= q * total {
                return *e;
            }
        }
        gaps[gaps.len() - 1].0
    };
    let emax = grid_p
        .iter()
        .zip(&grid_f)
        .map(|(&x, &f)| (x - f).abs())
        .fold(0.0, f64::max);
    Some(CurveErrors {
        ici,
        e50: weighted_quantile(0.5),
        e90: weighted_quantile(0.9),
        emax,
    })
}

/// The value of `p` (sorted, with weighted ranks `u`) at rank `q`.
fn quantile(p: &[f64], u: &[f64], q: f64) -> Option<f64> {
    let j = u.partition_point(|&x| x < q);
    p.get(j.min(p.len().checked_sub(1)?)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::censoring;

    /// Deterministic generator: no random-number dependency in the test.
    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Rate `0.04 exp(2 (x - 0.5))`, followed 12 to 20 years; the true risk by
    /// ten years and the observations.
    fn cohort(n: usize, seed: u64) -> (Vec<f64>, Vec<Obs>) {
        let mut s = seed;
        let (mut truth, mut obs) = (vec![], vec![]);
        for _ in 0..n {
            let x = lcg(&mut s);
            let rate = 0.04 * (2.0 * (x - 0.5)).exp();
            let time = -lcg(&mut s).ln() / rate;
            let censor = 12.0 + 8.0 * lcg(&mut s);
            obs.push(if time <= censor {
                Obs::event(time, 0)
            } else {
                Obs::censored(censor)
            });
            truth.push(1.0 - (-rate * 10.0).exp());
        }
        (truth, obs)
    }

    fn errors(cif: &[f64], obs: &[Obs]) -> CurveErrors {
        errors_at_horizon(cif, obs, 0, 10.0, &censoring(obs), 0.3).unwrap()
    }

    #[test]
    fn a_calibrated_prediction_has_small_errors_and_a_distorted_one_large() {
        let (truth, obs) = cohort(20_000, 1);
        let good = errors(&truth, &obs);
        assert!(good.ici < 0.01 && good.e90 < 0.02, "{good:?}");
        // Twice the log-odds: over-confident at both ends.
        let bent: Vec<f64> = truth
            .iter()
            .map(|p| sigmoid(2.0 * (p / (1.0 - p)).ln()))
            .collect();
        let bad = errors(&bent, &obs);
        assert!(
            bad.ici > 3.0 * good.ici && bad.ici > 0.03,
            "{bad:?} against {good:?}"
        );
        assert!(bad.e50 <= bad.e90 && bad.e90 <= bad.emax + 1e-12);
        // A constant offset in risk is a constant gap.
        let shifted: Vec<f64> = truth.iter().map(|p| (p + 0.05).min(1.0)).collect();
        let s = errors(&shifted, &obs);
        assert!((s.ici - 0.05).abs() < 0.015, "{s:?}");
    }

    #[test]
    fn integer_weights_equal_copies() {
        let (truth, obs) = cohort(1500, 2);
        let weighted: Vec<Obs> = obs
            .iter()
            .enumerate()
            .map(|(i, o)| o.weighted(if i % 2 == 0 { 2.0 } else { 1.0 }))
            .collect();
        let (mut cif2, mut obs2) = (vec![], vec![]);
        for (i, o) in obs.iter().enumerate() {
            for _ in 0..(if i % 2 == 0 { 2 } else { 1 }) {
                cif2.push(truth[i]);
                obs2.push(*o);
            }
        }
        let a = errors(&truth, &weighted);
        let b = errors(&cif2, &obs2);
        assert!(
            (a.ici - b.ici).abs() < 2e-3 && (a.e90 - b.e90).abs() < 5e-3,
            "{a:?} {b:?}"
        );
    }

    #[test]
    fn too_few_subjects_have_no_curve() {
        let obs = vec![Obs::event(1.0, 0); 5];
        assert!(errors_at_horizon(&[0.5; 5], &obs, 0, 10.0, &censoring(&obs), 0.3).is_none());
    }
}
