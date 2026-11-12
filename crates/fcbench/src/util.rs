// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Small dependency-free numeric helpers shared by the baselines and the
//! backtester: summary statistics, differencing, and a tiny dense linear solver
//! for AR fitting.

/// Sample mean.
pub fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f32>() / x.len() as f32
}

/// Sample standard deviation (population, ddof=0). Returns 0 for < 2 points.
pub fn std(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(x);
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / n as f32;
    var.max(0.0).sqrt()
}

/// First difference: `y[t] = x[t] - x[t-1]`, length `x.len() - 1`.
pub fn diff(x: &[f32]) -> Vec<f32> {
    if x.len() < 2 {
        return Vec::new();
    }
    (1..x.len()).map(|t| x[t] - x[t - 1]).collect()
}

/// `d`-fold difference.
pub fn diff_n(x: &[f32], d: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    for _ in 0..d {
        y = diff(&y);
    }
    y
}

/// Mean of `values` and the Newey-West standard error of that mean: the
/// Bartlett-weighted sum of autocovariances up to `lag`, which accounts for
/// values that are correlated because their windows overlap. `lag = 0` is the
/// plain `sigma / sqrt(n)` of independent values (population variance). A
/// single value has no spread and gets `0`.
pub fn mean_and_se(values: &[f32], lag: usize) -> (f32, f32) {
    let n = values.len();
    if n < 2 {
        return (mean(values), 0.0);
    }
    let m = mean(values);
    let dev: Vec<f64> = values.iter().map(|&v| (v - m) as f64).collect();
    let gamma = |k: usize| dev[k..].iter().zip(&dev).map(|(a, b)| a * b).sum::<f64>() / n as f64;
    let lag = lag.min(n - 1);
    let mut var = gamma(0);
    for k in 1..=lag {
        var += 2.0 * (1.0 - k as f64 / (lag as f64 + 1.0)) * gamma(k);
    }
    (m, (var.max(0.0) / n as f64).sqrt() as f32)
}

/// A paired difference of two score series with its standard error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Paired {
    pub mean: f32,
    pub std_err: f32,
    /// Origins or windows that both series scored.
    pub n: usize,
}

/// `a - b` over the positions where both series have a score, as a
/// [`Paired`] with its Newey-West standard error at `lag`. `None` when no
/// position is shared.
pub fn paired_diff(a: &[Option<f32>], b: &[Option<f32>], lag: usize) -> Option<Paired> {
    let diffs: Vec<f32> = a.iter().zip(b).filter_map(|(x, y)| Some((*x)? - (*y)?)).collect();
    if diffs.is_empty() {
        return None;
    }
    let (mean, std_err) = mean_and_se(&diffs, lag);
    Some(Paired { mean, std_err, n: diffs.len() })
}

/// Standard deviation of the first differences — the 1-step innovation scale of
/// a random walk.
pub fn diff_std(x: &[f32]) -> f32 {
    std(&diff(x))
}

/// Solve a small dense linear system `A x = b` by Gaussian elimination with
/// partial pivoting. `a` is row-major `n x n`. Returns `None` if singular.
pub fn solve(a: &[f32], b: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut m = a.to_vec();
    let mut y = b.to_vec();
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            y.swap(col, piv);
        }
        // eliminate
        let d = m[col * n + col];
        for r in (col + 1)..n {
            let f = m[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                m[r * n + c] -= f * m[col * n + c];
            }
            y[r] -= f * y[col];
        }
    }
    // back-substitute
    let mut x = vec![0.0f32; n];
    for row in (0..n).rev() {
        let mut s = y[row];
        for c in (row + 1)..n {
            s -= m[row * n + c] * x[c];
        }
        x[row] = s / m[row * n + row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_basic() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-6);
        assert!((std(&[1.0, 1.0, 1.0])).abs() < 1e-6);
        assert_eq!(diff(&[1.0, 3.0, 6.0]), vec![2.0, 3.0]);
        assert_eq!(diff_n(&[1.0, 3.0, 6.0, 10.0], 2), vec![1.0, 1.0]);
    }

    #[test]
    fn newey_west_at_lag_zero_is_the_plain_standard_error() {
        // population variance of 1..5 is 2, so se of the mean = sqrt(2/5)
        let (m, se) = mean_and_se(&[1.0, 2.0, 3.0, 4.0, 5.0], 0);
        assert!((m - 3.0).abs() < 1e-6 && (se - (2.0f32 / 5.0).sqrt()).abs() < 1e-6);
        assert_eq!(mean_and_se(&[7.0], 3), (7.0, 0.0));
    }

    #[test]
    fn newey_west_widens_the_error_of_positively_correlated_values() {
        // runs of equal values: neighbours move together, so the mean is
        // less certain than independence would claim.
        let v: Vec<f32> = (0..24).map(|i| if (i / 6) % 2 == 0 { 0.0 } else { 4.0 }).collect();
        assert!(mean_and_se(&v, 3).1 > 1.4 * mean_and_se(&v, 0).1);
    }

    #[test]
    fn paired_difference_uses_only_origins_both_models_scored() {
        let a = [Some(1.0), None, Some(3.0), Some(5.0)];
        let b = [Some(2.0), Some(9.0), Some(3.5), None];
        let d = paired_diff(&a, &b, 0).unwrap();
        assert_eq!(d.n, 2);
        assert!((d.mean - (-0.75)).abs() < 1e-6); // mean of -1.0 and -0.5
        assert!(paired_diff(&[None], &[Some(1.0)], 0).is_none());
    }

    #[test]
    fn solve_recovers_known_system() {
        // [[2,1],[1,3]] x = [3,5] -> x = [0.8, 1.4]
        let x = solve(&[2.0, 1.0, 1.0, 3.0], &[3.0, 5.0], 2).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-5);
        assert!((x[1] - 1.4).abs() < 1e-5);
    }

    #[test]
    fn solve_reports_singular() {
        assert!(solve(&[1.0, 1.0, 1.0, 1.0], &[2.0, 3.0], 2).is_none());
    }
}
