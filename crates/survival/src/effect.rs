// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The effect of an assigned treatment in a randomised trial, optionally
//! adjusted by a prognostic score (PROCOVA).
//!
//! The estimate is the treatment coefficient of the least-squares fit of
//! the outcome on `[1, treated]`, or on `[1, treated, score]` when a score
//! is given - a model's prediction of the outcome from baseline data alone.
//! Randomisation makes the estimate unbiased whatever the score is; a score
//! correlated with the outcome removes that share of its variance and
//! narrows the interval, which is the whole point. A wrong or useless score
//! costs at most a degree of freedom. The standard error is the
//! heteroskedasticity-robust HC3 estimator (arms may differ in spread) and
//! the interval and p-value use Student's t on the residual degrees of
//! freedom.

use crate::compare::{student_t_cdf, student_t_quantile};

/// A treatment effect with its uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Effect {
    /// Treated minus control, adjusted when a score was given.
    pub estimate: f64,
    /// Its HC3 standard error.
    pub se: f64,
    /// The 95% interval's lower end.
    pub lo: f64,
    /// The 95% interval's upper end.
    pub hi: f64,
    /// The two-sided p-value of no effect.
    pub p_value: f64,
    /// Subjects.
    pub n: usize,
}

/// Solve `a x = b` for a small symmetric positive-definite `a` (Gaussian
/// elimination with partial pivoting); `None` when it is singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let p = a.len();
    for col in 0..p {
        let pivot = (col..p).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let (pivot_a, pivot_b) = (a[col].clone(), b[col].clone());
        for (row, (ar, br)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            if row != col {
                let f = ar[col] / pivot_a[col];
                ar.iter_mut().zip(&pivot_a).for_each(|(x, y)| *x -= f * y);
                br.iter_mut().zip(&pivot_b).for_each(|(x, y)| *x -= f * y);
            }
        }
    }
    Some(
        (0..p)
            .map(|i| b[i].iter().map(|x| x / a[i][i]).collect())
            .collect(),
    )
}

/// The effect of `treated` on `outcome`, adjusted by `score` when given.
/// `None` when it cannot be estimated: fewer subjects than the fit needs,
/// one arm empty, or a score collinear with the assignment.
pub fn ancova(outcome: &[f64], treated: &[bool], score: Option<&[f64]>) -> Option<Effect> {
    let n = outcome.len();
    assert_eq!(treated.len(), n, "one assignment per subject");
    if let Some(s) = score {
        assert_eq!(s.len(), n, "one score per subject");
    }
    let row = |i: usize| -> Vec<f64> {
        let mut r = vec![1.0, f64::from(u8::from(treated[i]))];
        if let Some(s) = score {
            r.push(s[i]);
        }
        r
    };
    let p = if score.is_some() { 3 } else { 2 };
    if n <= p {
        return None;
    }
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![vec![0.0]; p];
    for (i, &y) in outcome.iter().enumerate() {
        let x = row(i);
        for a in 0..p {
            xty[a][0] += x[a] * y;
            for b in 0..p {
                xtx[a][b] += x[a] * x[b];
            }
        }
    }
    let identity: Vec<Vec<f64>> = (0..p)
        .map(|a| (0..p).map(|b| f64::from(u8::from(a == b))).collect())
        .collect();
    let inv = solve(xtx, identity)?;
    let beta: Vec<f64> = (0..p)
        .map(|a| (0..p).map(|b| inv[a][b] * xty[b][0]).sum())
        .collect();
    // HC3: (X'X)^-1 X' diag(e_i^2 / (1 - h_ii)^2) X (X'X)^-1.
    let mut meat = vec![vec![0.0; p]; p];
    for (i, &y) in outcome.iter().enumerate() {
        let x = row(i);
        let fitted: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum();
        let e = y - fitted;
        let leverage: f64 = (0..p)
            .map(|a| (0..p).map(|b| x[a] * inv[a][b] * x[b]).sum::<f64>())
            .sum();
        if leverage >= 1.0 - 1e-12 {
            return None;
        }
        let w = e * e / ((1.0 - leverage) * (1.0 - leverage));
        for a in 0..p {
            for b in 0..p {
                meat[a][b] += w * x[a] * x[b];
            }
        }
    }
    let var: f64 = (0..p)
        .map(|a| {
            (0..p)
                .map(|b| inv[1][a] * meat[a][b] * inv[b][1])
                .sum::<f64>()
        })
        .sum();
    let se = var.sqrt();
    let df = (n - p) as f64;
    let t = student_t_quantile(0.975, df);
    let estimate = beta[1];
    Some(Effect {
        estimate,
        se,
        lo: estimate - t * se,
        hi: estimate + t * se,
        p_value: 2.0 * (1.0 - student_t_cdf((estimate / se).abs(), df)),
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussians(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        let mut uniform = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        move || (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()
    }

    /// Trials of `n` with no effect and a score correlated `rho` with the outcome.
    fn trials(count: usize, n: usize, rho: f64, effect: f64, seed: u64) -> Vec<(Effect, Effect)> {
        let mut g = gaussians(seed);
        (0..count)
            .map(|_| {
                let mut outcome = Vec::with_capacity(n);
                let (mut treated, mut score) = (Vec::with_capacity(n), Vec::with_capacity(n));
                for i in 0..n {
                    let s = g();
                    let arm = i % 2 == 1;
                    outcome.push(
                        rho * s + (1.0 - rho * rho).sqrt() * g() + if arm { effect } else { 0.0 },
                    );
                    treated.push(arm);
                    score.push(s);
                }
                (
                    ancova(&outcome, &treated, None).unwrap(),
                    ancova(&outcome, &treated, Some(&score)).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn the_score_keeps_the_error_rate_and_narrows_the_interval() {
        let rho = 0.7;
        let runs = trials(2000, 60, rho, 0.0, 9);
        let rejected = |pick: fn(&(Effect, Effect)) -> &Effect| {
            runs.iter().filter(|r| pick(r).p_value < 0.05).count() as f64 / runs.len() as f64
        };
        let (plain, adjusted) = (rejected(|r| &r.0), rejected(|r| &r.1));
        assert!(
            (plain - 0.05).abs() < 0.015 && (adjusted - 0.05).abs() < 0.015,
            "type-I error {plain} unadjusted, {adjusted} adjusted"
        );
        let mean_se = |pick: fn(&(Effect, Effect)) -> &Effect| {
            runs.iter().map(|r| pick(r).se).sum::<f64>() / runs.len() as f64
        };
        let ratio = mean_se(|r| &r.1) / mean_se(|r| &r.0);
        let expected = (1.0 - rho * rho).sqrt();
        assert!(
            (ratio - expected).abs() < 0.05,
            "standard error ratio {ratio}, expected about {expected}"
        );
        // With an effect, the adjusted intervals still cover it.
        let effect = 0.4;
        let with = trials(2000, 60, rho, effect, 10);
        let covered = with
            .iter()
            .filter(|r| r.1.lo <= effect && effect <= r.1.hi)
            .count() as f64
            / with.len() as f64;
        assert!((covered - 0.95).abs() < 0.015, "coverage {covered}");
    }

    #[test]
    fn it_refuses_what_it_cannot_estimate() {
        assert!(ancova(&[1.0, 2.0], &[true, false], None).is_none());
        assert!(
            ancova(&[1.0, 2.0, 3.0, 4.0], &[true; 4], None).is_none(),
            "one arm empty"
        );
        let arm = [true, false, true, false, true];
        let collinear: Vec<f64> = arm.iter().map(|&a| f64::from(u8::from(a))).collect();
        assert!(ancova(&[1.0, 2.0, 3.0, 4.0, 5.0], &arm, Some(&collinear)).is_none());
    }
}
