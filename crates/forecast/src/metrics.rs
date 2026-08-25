// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Float-valued, probabilistic forecasting metrics — pure `&[f32] -> f32`
//! functions with no model dependency, so any benchmark, baseline, or
//! backtester can compute and report them the same way.
//!
//! They live here (in the light `forecast` crate) rather than in `bench`
//! because the backtester and the served baselines need them without pulling in
//! the training/model stack.
//!
//! Definitions:
//! - **pinball / quantile loss** — the proper scoring rule for a single
//!   quantile: `max(tau*(y-q), (tau-1)*(y-q))`. Asymmetric: at `tau=0.9` an
//!   under-forecast costs nine times an over-forecast.
//! - **weighted quantile loss (wQL)** — `2 * sum(pinball)` over the `[H, Q]`
//!   grid, normalised by `sum|actual|`; the GIFT-Eval / Chronos scale-free
//!   probabilistic headline. The factor of 2 is the gluonts convention (median
//!   wQL then equals the magnitude-normalised MAE).
//! - **CRPS** — Continuous Ranked Probability Score. `crps_gaussian` is the
//!   closed form (Gneiting et al. 2004 eq. 5, matching `properscoring`);
//!   `crps_ensemble` is the sample-based energy form `E|X-x| - 0.5*E|X-X'|`.
//! - **MASE** — Mean Absolute Scaled Error: forecast MAE over the in-sample
//!   seasonal-naive MAE. `1.0` = as good as seasonal naive.
//! - **directional accuracy** — fraction of horizon steps whose predicted
//!   step-over-step sign matches the actual sign (the first step compares
//!   against `origin`).
//! - **coverage** — fraction of actuals inside a predicted interval; compare
//!   against the nominal level to read calibration.
//! - **rank IC** — Spearman rank correlation between predicted and realised
//!   values; the finance-native cross-sectional signal metric.
//! - **skill score** — `clamp(1 - error/baseline_error, 0, 1)`; converts a
//!   lower-is-better error into a 0..1 higher-is-better headline.
//!
//! sMAPE is deliberately absent - N-BEATS ships two definitions that differ by
//! a factor of two;
//! MASE is our scale-free point metric of record to avoid that ambiguity.

/// Pinball (quantile) loss for one predicted quantile `q` at level `tau`
/// against actual `y`: `max(tau*(y-q), (tau-1)*(y-q))`. Zero iff `q == y`.
pub fn pinball(q: f32, y: f32, tau: f32) -> f32 {
    let e = y - q;
    (tau * e).max((tau - 1.0) * e)
}

/// Gradient of [`pinball`] with respect to the prediction `q` — the backward the
/// Chronos-2 (and any quantile-head) trainer needs. Piecewise-constant: the loss
/// is `tau*(y-q)` when `q < y` (slope `-tau`) and `(1-tau)*(q-y)` when `q > y`
/// (slope `1-tau`). At the non-differentiable kink `q == y` we return the
/// subgradient midpoint `tau - 0.5` (any value in `[-tau, 1-tau]` is valid).
pub fn pinball_grad(q: f32, y: f32, tau: f32) -> f32 {
    if q < y {
        -tau
    } else if q > y {
        1.0 - tau
    } else {
        tau - 0.5
    }
}

/// Gradient of [`mean_pinball`] w.r.t. each predicted quantile: a row-major
/// `[H, Q]` matrix matching `quantiles`, scaled by `1/(H*Q)` to match the mean.
pub fn mean_pinball_grad(quantiles: &[f32], levels: &[f32], actual: &[f32]) -> Vec<f32> {
    let (h, qn) = (actual.len(), levels.len());
    let mut g = vec![0.0f32; quantiles.len()];
    if h == 0 || qn == 0 {
        return g;
    }
    let scale = 1.0 / (h * qn) as f32;
    for (t, &y) in actual.iter().enumerate() {
        for (k, &tau) in levels.iter().enumerate() {
            let idx = t * qn + k;
            if idx < quantiles.len() {
                g[idx] = pinball_grad(quantiles[idx], y, tau) * scale;
            }
        }
    }
    g
}

/// Mean pinball loss over a horizon and quantile grid. `quantiles` is a
/// row-major `[H, Q]` matrix (step-major), `levels` the `Q` tau values,
/// `actual` length `H`.
pub fn mean_pinball(quantiles: &[f32], levels: &[f32], actual: &[f32]) -> f32 {
    let (h, qn) = (actual.len(), levels.len());
    if h == 0 || qn == 0 {
        return 0.0;
    }
    assert_eq!(quantiles.len(), h * qn, "mean_pinball: quantiles must be [H, Q]");
    let mut total = 0.0f32;
    for t in 0..h {
        for (j, &tau) in levels.iter().enumerate() {
            total += pinball(quantiles[t * qn + j], actual[t], tau);
        }
    }
    total / (h * qn) as f32
}

/// Weighted quantile loss: `2 * sum(pinball)` over the `[H, Q]` grid normalised
/// by `sum|actual|` (gluonts / GIFT-Eval convention). Returns 0 when the actuals
/// are all zero.
pub fn weighted_quantile_loss(quantiles: &[f32], levels: &[f32], actual: &[f32]) -> f32 {
    let (h, qn) = (actual.len(), levels.len());
    if h == 0 || qn == 0 {
        return 0.0;
    }
    assert_eq!(quantiles.len(), h * qn, "weighted_quantile_loss: quantiles must be [H, Q]");
    let mut total = 0.0f32;
    for t in 0..h {
        for (j, &tau) in levels.iter().enumerate() {
            total += pinball(quantiles[t * qn + j], actual[t], tau);
        }
    }
    let denom: f32 = actual.iter().map(|v| v.abs()).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    2.0 * total / denom
}

/// CRPS of observation `x` under a Gaussian forecast `N(mu, sig^2)`, closed
/// form: `sig * ( sx*(2*Phi(sx)-1) + 2*phi(sx) - 1/sqrt(pi) )`, `sx=(x-mu)/sig`.
pub fn crps_gaussian(x: f32, mu: f32, sig: f32) -> f32 {
    if sig <= 0.0 {
        return (x - mu).abs();
    }
    let sx = (x - mu) / sig;
    let pdf = (1.0 / (2.0 * std::f32::consts::PI).sqrt()) * (-(sx * sx) / 2.0).exp();
    let cdf = 0.5 * (1.0 + erf(sx / std::f32::consts::SQRT_2));
    let pi_inv = 1.0 / std::f32::consts::PI.sqrt();
    sig * (sx * (2.0 * cdf - 1.0) + 2.0 * pdf - pi_inv)
}

/// Sample-based CRPS via `CRPS = E|X - x| - 0.5*E|X - X'|` over all pairs of the
/// `samples` ensemble. O(n^2).
pub fn crps_ensemble(samples: &[f32], x: f32) -> f32 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let e_abs: f32 = samples.iter().map(|&s| (s - x).abs()).sum::<f32>() / n as f32;
    let mut e_diff = 0.0f32;
    for &a in samples {
        for &b in samples {
            e_diff += (a - b).abs();
        }
    }
    e_diff /= (n * n) as f32;
    e_abs - 0.5 * e_diff
}

/// Mean absolute error. Its own function because [`mase`] and every caller
/// that wants a raw, unscaled number were each computing it inline.
pub fn mae(pred: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(pred.len(), actual.len(), "mae: pred/actual length mismatch");
    if pred.is_empty() {
        return 0.0;
    }
    pred.iter().zip(actual).map(|(p, a)| (p - a).abs()).sum::<f32>() / pred.len() as f32
}

/// Root mean squared error - the companion to [`mae`], reported alongside it
/// because the two disagree exactly when a forecast is occasionally badly
/// wrong rather than consistently slightly wrong.
pub fn rmse(pred: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(pred.len(), actual.len(), "rmse: pred/actual length mismatch");
    if pred.is_empty() {
        return 0.0;
    }
    (pred.iter().zip(actual).map(|(p, a)| (p - a) * (p - a)).sum::<f32>() / pred.len() as f32).sqrt()
}

/// Mean absolute PERCENTAGE error, in percent. Scale-free, so a reader can
/// judge a price forecast without knowing the price level. Steps whose actual
/// is zero are skipped rather than yielding an infinity.
pub fn mape(pred: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(pred.len(), actual.len(), "mape: pred/actual length mismatch");
    let mut acc = 0.0f32;
    let mut n = 0usize;
    for (p, a) in pred.iter().zip(actual) {
        if *a != 0.0 {
            acc += ((p - a) / a).abs();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        acc / n as f32 * 100.0
    }
}

/// Mean Absolute Scaled Error: forecast MAE divided by the in-sample
/// seasonal-naive MAE (`|y_t - y_{t-season}|`) on `insample`. `season = 1` is
/// plain naive. Returns unscaled MAE if the naive scale is degenerate.
pub fn mase(pred: &[f32], actual: &[f32], insample: &[f32], season: usize) -> f32 {
    assert_eq!(pred.len(), actual.len(), "mase: pred/actual length mismatch");
    if pred.is_empty() {
        return 0.0;
    }
    let mae = mae(pred, actual);
    let s = season.max(1);
    let scale = if insample.len() > s {
        let mut acc = 0.0f32;
        for t in s..insample.len() {
            acc += (insample[t] - insample[t - s]).abs();
        }
        acc / (insample.len() - s) as f32
    } else {
        0.0
    };
    if scale <= 0.0 {
        return mae;
    }
    mae / scale
}

/// Directional accuracy: fraction of horizon steps whose predicted
/// step-over-step sign matches the actual sign. The first step compares against
/// `origin`; step `t>0` against `actual[t-1]`.
pub fn directional_accuracy(pred: &[f32], actual: &[f32], origin: f32) -> f32 {
    assert_eq!(pred.len(), actual.len(), "directional_accuracy: length mismatch");
    if pred.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    for t in 0..pred.len() {
        let prev = if t == 0 { origin } else { actual[t - 1] };
        if (pred[t] - prev).signum() == (actual[t] - prev).signum() {
            hits += 1;
        }
    }
    hits as f32 / pred.len() as f32
}

/// Interval coverage: fraction of `actual[t]` within `[lo[t], hi[t]]`
/// (endpoints inclusive). Compare against the nominal level to read calibration.
pub fn coverage(lo: &[f32], hi: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(lo.len(), hi.len(), "coverage: lo/hi length mismatch");
    assert_eq!(lo.len(), actual.len(), "coverage: bounds/actual length mismatch");
    if actual.is_empty() {
        return 0.0;
    }
    let hits = (0..actual.len()).filter(|&t| actual[t] >= lo[t] && actual[t] <= hi[t]).count();
    hits as f32 / actual.len() as f32
}

/// Rank information coefficient: Spearman rank correlation (ties averaged).
pub fn rank_ic(pred: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(pred.len(), actual.len(), "rank_ic: length mismatch");
    let n = pred.len();
    if n < 2 {
        return 0.0;
    }
    let pr = average_ranks(pred);
    let ar = average_ranks(actual);
    pearson(&pr, &ar)
}

/// Skill score: `clamp(1 - error/baseline_error, 0, 1)`. Turns a lower-is-better
/// error into a 0..1 higher-is-better headline (1 = perfect, 0 = no better than
/// baseline). Clamped so worse-than-baseline reads as 0.
pub fn skill_score(error: f32, baseline_error: f32) -> f32 {
    if baseline_error <= 0.0 {
        return if error <= 0.0 { 1.0 } else { 0.0 };
    }
    (1.0 - error / baseline_error).clamp(0.0, 1.0)
}

/// Average (tie-corrected) ranks of `xs`, 1-based.
fn average_ranks(xs: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f32) / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation of two equal-length series.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Error function via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).
/// Dependency-free, adequate for fp32.
pub fn erf(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinball_is_asymmetric_about_the_quantile_level() {
        assert!((pinball(1.0, 3.0, 0.9) - 1.8).abs() < 1e-6);
        assert!((pinball(3.0, 1.0, 0.9) - 0.2).abs() < 1e-6);
        assert!((pinball(1.0, 3.0, 0.5) - 1.0).abs() < 1e-6);
        assert!(pinball(2.0, 2.0, 0.1).abs() < 1e-6);
    }

    #[test]
    fn mean_pinball_averages_over_horizon_and_levels() {
        let q = [1.0f32, 5.0];
        assert!((mean_pinball(&q, &[0.5], &[3.0, 4.0]) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn pinball_grad_matches_finite_difference() {
        // Away from the kink (q != y), the analytic slope must match a central
        // finite difference of `pinball`.
        let eps = 1e-3f32;
        for &(q, y, tau) in &[(1.0f32, 3.0, 0.1), (5.0, 2.0, 0.9), (0.3, -0.4, 0.5), (2.0, 2.5, 0.25)] {
            let numeric = (pinball(q + eps, y, tau) - pinball(q - eps, y, tau)) / (2.0 * eps);
            let analytic = pinball_grad(q, y, tau);
            assert!((numeric - analytic).abs() < 1e-3, "q={q} y={y} tau={tau}: {numeric} vs {analytic}");
        }
        // At the kink the subgradient is the midpoint tau-0.5, in [-tau, 1-tau].
        let g = pinball_grad(2.0, 2.0, 0.3);
        assert!((-0.3..=0.7).contains(&g));
    }

    #[test]
    fn mean_pinball_grad_matches_finite_difference_elementwise() {
        // The gradient the quantile-head trainer backprops must be the exact
        // derivative of `mean_pinball` w.r.t. every predicted quantile.
        let levels = [0.1f32, 0.5, 0.9];
        let actual = [1.0f32, -0.5, 2.0, 0.25];
        // [H=4, Q=3] grid, none coinciding with an actual (avoid the kink).
        let mut q: Vec<f32> = (0..12).map(|i| 0.37 * i as f32 - 1.1).collect();
        for (i, &a) in actual.iter().enumerate() {
            for k in 0..levels.len() {
                if (q[i * levels.len() + k] - a).abs() < 1e-2 {
                    q[i * levels.len() + k] += 0.1;
                }
            }
        }
        let analytic = mean_pinball_grad(&q, &levels, &actual);
        let eps = 1e-3f32;
        for i in 0..q.len() {
            let mut qp = q.clone();
            qp[i] += eps;
            let mut qm = q.clone();
            qm[i] -= eps;
            let numeric =
                (mean_pinball(&qp, &levels, &actual) - mean_pinball(&qm, &levels, &actual)) / (2.0 * eps);
            assert!((numeric - analytic[i]).abs() < 1e-3, "elem {i}: {numeric} vs {}", analytic[i]);
        }
    }

    #[test]
    fn weighted_quantile_loss_uses_the_factor_two_convention() {
        // 2*(1.0+0.5) / (3+4) = 3/7. Median wQL == normalized MAE.
        let q = [1.0f32, 5.0];
        assert!((weighted_quantile_loss(&q, &[0.5], &[3.0, 4.0]) - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn crps_gaussian_matches_the_closed_form() {
        assert!((crps_gaussian(0.0, 0.0, 1.0) - 0.233_694_98).abs() < 1e-5);
        assert!((crps_gaussian(0.0, 0.0, 3.0) - 3.0 * 0.233_694_98).abs() < 1e-4);
        assert!(crps_gaussian(5.0, 0.0, 1.0) > crps_gaussian(0.0, 0.0, 1.0));
    }

    #[test]
    fn crps_ensemble_matches_the_energy_form() {
        assert!((crps_ensemble(&[0.0, 2.0], 1.0) - 0.5).abs() < 1e-6);
        assert!((crps_ensemble(&[4.0], 1.0) - 3.0).abs() < 1e-6);
        assert!(crps_ensemble(&[2.0, 2.0], 2.0).abs() < 1e-6);
    }

    #[test]
    fn mase_is_one_when_as_good_as_seasonal_naive() {
        assert!((mase(&[3.0, 4.0], &[4.0, 5.0], &[1.0, 2.0, 3.0], 1) - 1.0).abs() < 1e-6);
        assert!(mase(&[4.0, 5.0], &[4.0, 5.0], &[1.0, 2.0, 3.0], 1).abs() < 1e-6);
    }

    #[test]
    fn directional_accuracy_compares_step_over_step_signs() {
        assert!((directional_accuracy(&[12.0, 13.0], &[11.0, 10.5], 10.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn coverage_counts_actuals_inside_the_interval() {
        assert!((coverage(&[0.0, 0.0], &[2.0, 2.0], &[1.0, 3.0]) - 0.5).abs() < 1e-6);
        assert!((coverage(&[0.0], &[2.0], &[2.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rank_ic_is_spearman_correlation() {
        assert!((rank_ic(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]) - 1.0).abs() < 1e-6);
        assert!((rank_ic(&[1.0, 2.0, 3.0], &[30.0, 20.0, 10.0]) + 1.0).abs() < 1e-6);
        assert!(rank_ic(&[1.0, 1.0, 1.0], &[10.0, 20.0, 30.0]).abs() < 1e-6);
    }

    #[test]
    fn skill_score_maps_error_ratio_to_zero_one() {
        assert!(skill_score(1.0, 1.0).abs() < 1e-6);
        assert!((skill_score(0.5, 1.0) - 0.5).abs() < 1e-6);
        assert!((skill_score(0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!(skill_score(4.0, 1.0).abs() < 1e-6);
    }
}
