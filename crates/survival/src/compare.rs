// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Comparing two models evaluated on the same data.
//!
//! - [`corrected_resampled_t`]: the paired t-test across cross-validation
//!   folds with Nadeau and Bengio's variance correction. Folds share most of
//!   their training data, so their results are positively correlated and the
//!   naive standard error is too small; the correction inflates the variance
//!   of the mean difference from `s^2 / J` to `(1/J + n_test/n_train) s^2`.
//! - [`cluster_bootstrap`]: a percentile interval for a weighted mean of
//!   per-unit values (for example per-subject differences in Brier terms on a
//!   locked test), resampling whole clusters (survey primary sampling units,
//!   households) with replacement so within-cluster correlation is respected.
//!
//! The Student t distribution is computed from the regularized incomplete
//! beta function (continued fraction, Numerical Recipes `betai`).

use crate::special::ln_gamma;

/// `I_x(a, b)`, the regularized incomplete beta function.
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    assert!(
        a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x),
        "incomplete_beta({a}, {b}, {x})"
    );
    if x == 0.0 || x == 1.0 {
        return x;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    // The continued fraction converges fast for x < (a + 1) / (a + b + 2).
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, 1.0 - x) / b
    }
}

fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..1000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < tiny { tiny } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < tiny { tiny } else { c };
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < tiny { tiny } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < tiny { tiny } else { c };
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// `P(T <= t)` for Student's t with `df` degrees of freedom.
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
    let tail = 0.5 * incomplete_beta(0.5 * df, 0.5, df / (df + t * t));
    if t >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// The `p` quantile of Student's t with `df` degrees of freedom (bisection
/// on the CDF to 1e-12).
pub fn student_t_quantile(p: f64, df: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "quantile of {p}");
    let (mut lo, mut hi) = (-1e3, 1e3);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// A paired comparison's result.
#[derive(Clone, Debug, PartialEq)]
pub struct TTest {
    /// Mean difference.
    pub mean: f64,
    /// Corrected standard error of the mean difference.
    pub se: f64,
    /// `mean / se`.
    pub t: f64,
    /// Degrees of freedom (`J - 1`).
    pub df: f64,
    /// Two-sided p-value.
    pub p_two_sided: f64,
    /// 95% confidence interval of the mean difference.
    pub ci95: (f64, f64),
}

/// The corrected resampled t-test over per-fold differences `diffs` (model A
/// minus model B on the same folds), where every fold tested on
/// `test_over_train` times as many units as it trained on. `None` with fewer
/// than two folds or no variance.
pub fn corrected_resampled_t(diffs: &[f64], test_over_train: f64) -> Option<TTest> {
    let j = diffs.len();
    if j < 2 {
        return None;
    }
    let mean = diffs.iter().sum::<f64>() / j as f64;
    let s2 = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (j - 1) as f64;
    let se = ((1.0 / j as f64 + test_over_train) * s2).sqrt();
    if se <= 0.0 || !se.is_finite() {
        return None;
    }
    let df = (j - 1) as f64;
    let t = mean / se;
    let q = student_t_quantile(0.975, df);
    Some(TTest {
        mean,
        se,
        t,
        df,
        p_two_sided: 2.0 * (1.0 - student_t_cdf(t.abs(), df)),
        ci95: (mean - q * se, mean + q * se),
    })
}

/// A bootstrap interval.
#[derive(Clone, Debug, PartialEq)]
pub struct Interval {
    /// The statistic on the full sample.
    pub estimate: f64,
    /// Lower percentile bound.
    pub lo: f64,
    /// Upper percentile bound.
    pub hi: f64,
}

/// A percentile interval at `level` (e.g. 0.95) for the weighted mean of
/// `values`, resampling whole `clusters` with replacement `reps` times,
/// deterministic in `seed`.
pub fn cluster_bootstrap(
    values: &[f64],
    weights: &[f64],
    clusters: &[u64],
    reps: usize,
    level: f64,
    seed: u64,
) -> Option<Interval> {
    assert!(
        values.len() == weights.len() && values.len() == clusters.len(),
        "one value, weight and cluster per unit"
    );
    let mut by: std::collections::BTreeMap<u64, (f64, f64)> = std::collections::BTreeMap::new();
    for ((v, w), c) in values.iter().zip(weights).zip(clusters) {
        let e = by.entry(*c).or_default();
        e.0 += v * w;
        e.1 += w;
    }
    let sums: Vec<(f64, f64)> = by.into_values().collect();
    if sums.len() < 2 {
        return None;
    }
    let total = |pick: &mut dyn FnMut() -> usize| {
        let (mut num, mut den) = (0.0, 0.0);
        for _ in 0..sums.len() {
            let (a, b) = sums[pick()];
            num += a;
            den += b;
        }
        num / den
    };
    let estimate = sums.iter().map(|s| s.0).sum::<f64>() / sums.iter().map(|s| s.1).sum::<f64>();
    let mut state = seed;
    let n = sums.len() as u64;
    let mut pick = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n) as usize
    };
    let mut stats: Vec<f64> = (0..reps).map(|_| total(&mut pick)).collect();
    stats.sort_by(f64::total_cmp);
    let at = |q: f64| stats[((q * (reps - 1) as f64).round() as usize).min(reps - 1)];
    let alpha = (1.0 - level) / 2.0;
    Some(Interval {
        estimate,
        lo: at(alpha),
        hi: at(1.0 - alpha),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against scipy.stats.t and scipy.special.betainc.
    #[test]
    fn student_t_and_incomplete_beta() {
        for (x, df, want) in [
            (0.5, 3.0, 0.674_276_017_575_924_5),
            (2.0, 24.0, 0.971_530_075_031_704_2),
            (-1.3, 10.0, 0.111_382_908_603_422_36),
            (4.5, 4.0, 0.994_588_724_768_695_9),
            (1.96, 1000.0, 0.974_863_407_522_125_6),
        ] {
            assert!(
                (student_t_cdf(x, df) - want).abs() < 1e-12,
                "cdf({x}, {df})"
            );
        }
        for (p, df, want) in [
            (0.975, 24.0, 2.063_898_561_628_024),
            (0.975, 4.0, 2.776_445_105_197_793_4),
            (0.95, 10.0, 1.812_461_122_811_676_3),
        ] {
            assert!(
                (student_t_quantile(p, df) - want).abs() < 1e-9,
                "quantile({p}, {df})"
            );
        }
        assert!((incomplete_beta(2.5, 0.5, 0.3) - 0.018_927_124_071_945_65).abs() < 1e-13);
        assert!((incomplete_beta(12.0, 0.5, 0.9) - 0.115_522_854_266_832_15).abs() < 1e-13);
    }

    /// The corrected test against the same arithmetic in numpy/scipy.
    #[test]
    fn corrected_resampled_t_matches_the_reference() {
        let d = [
            0.012, -0.004, 0.020, 0.008, 0.015, 0.001, 0.010, -0.002, 0.018, 0.006, 0.009, 0.011,
            0.004, 0.016, 0.007, 0.013, -0.001, 0.012, 0.005, 0.010, 0.014, 0.003, 0.008, 0.017,
            0.006,
        ];
        let r = corrected_resampled_t(&d, 0.25).unwrap();
        assert!((r.mean - 0.00872).abs() < 1e-15);
        assert!((r.se - 0.003_421_875_703_957_309_5).abs() < 1e-15);
        assert!((r.p_two_sided - 0.017_642_979_874_486_753).abs() < 1e-10);
        assert!((r.ci95.0 - 0.001_657_595_656_532_626_4).abs() < 1e-10);
        assert!((r.ci95.1 - 0.015_782_404_343_467_372).abs() < 1e-10);
        assert!(corrected_resampled_t(&[1.0], 0.25).is_none());
        assert!(
            corrected_resampled_t(&[1.0, 1.0], 0.25).is_none(),
            "no variance, no test"
        );
    }

    #[test]
    fn the_cluster_bootstrap_covers_the_mean_and_respects_clusters() {
        // Twenty clusters of five identical units: resampling units would
        // understate the spread; resampling clusters does not.
        let values: Vec<f64> = (0..100).map(|i| ((i / 5) as f64 - 9.5) * 0.1).collect();
        let weights = vec![1.0; 100];
        let clusters: Vec<u64> = (0..100).map(|i| (i / 5) as u64).collect();
        let ci = cluster_bootstrap(&values, &weights, &clusters, 2000, 0.95, 3).unwrap();
        assert!(
            ci.estimate.abs() < 1e-12 && ci.lo < 0.0 && ci.hi > 0.0,
            "{ci:?}"
        );
        let units: Vec<u64> = (0..100).collect();
        let narrow = cluster_bootstrap(&values, &weights, &units, 2000, 0.95, 3).unwrap();
        assert!(
            ci.hi - ci.lo > 1.5 * (narrow.hi - narrow.lo),
            "clusters {ci:?} vs units {narrow:?}"
        );
        assert_eq!(
            ci,
            cluster_bootstrap(&values, &weights, &clusters, 2000, 0.95, 3).unwrap(),
            "deterministic in the seed"
        );
    }
}
