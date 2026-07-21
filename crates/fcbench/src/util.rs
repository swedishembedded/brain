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
