// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one statistic a promote/reject decision rests on: an exact one-sided
//! paired binomial sign test over discordant pairs.
//!
//! It lived in `bench::metrics`, on the argument that "`bench` is how this
//! repo scores models". That argument still holds for `bench`'s other
//! statistics and they stay there - but `brain-bench` links every model crate
//! it benchmarks, and this particular one is the significance bar
//! [`crate::gate::gate`] computes, so keeping it there made the gate
//! unreachable from a model crate (see the crate doc). Sixty lines of
//! binomial arithmetic depend on nothing, so they move to where the layering
//! allows them to be used; `bench::metrics` re-exports both items and every
//! existing caller of `bench::metrics::sign_test` is unchanged.
//!
//! No external statistics dependency: the binomial coefficient is spelled out
//! in the multiplicative running-product form, same stance the rest of this
//! workspace's small numerical helpers take.

/// Result of [`sign_test`]: the discordant-pair count, how many of those
/// favored the candidate, and the resulting one-sided p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignTest {
    /// Number of discordant pairs (`candidate != baseline`); concordant
    /// (tied) pairs carry no directional information and are excluded.
    pub n: usize,
    /// Of those `n` discordant pairs, how many favored the candidate
    /// (`candidate > baseline`).
    pub k: usize,
    /// One-sided `P(Binomial(n, 0.5) >= k)` - the chance of seeing at least
    /// this many candidate wins among the discordant pairs if the true win
    /// probability were 0.5 (candidate no better than baseline).
    pub p_value: f64,
}

/// Exact one-sided paired sign test, conditioned on discordant pairs only
/// (self-improve roadmap P16's promotion gate; hoisted from an inlined
/// version this fixes two defects in: summing the binomial tail over *every*
/// pair rather than just the discordant ones - wrong-shaped for a binary
/// 0/1 outcome, since a tied pair is not evidence either way - and a
/// `n - k` `usize` subtraction that underflowed whenever `k > n`).
///
/// `candidate` and `baseline` must be the same length, one paired score per
/// task, same tasks in the same order for both arms.
pub fn sign_test(candidate: &[f64], baseline: &[f64]) -> SignTest {
    assert_eq!(candidate.len(), baseline.len(), "sign_test: paired arrays must have equal length");
    let mut n = 0usize;
    let mut k = 0usize;
    for (c, b) in candidate.iter().zip(baseline) {
        if c > b {
            n += 1;
            k += 1;
        } else if c < b {
            n += 1;
        }
    }
    SignTest { n, k, p_value: binom_sf(n, k) }
}

/// `P(Binomial(n, 0.5) >= k)`.
fn binom_sf(n: usize, k: usize) -> f64 {
    let mut p = 0.0f64;
    for i in k..=n {
        p += binom_coeff(n, i) * 0.5f64.powi(n as i32);
    }
    p
}

/// `n choose k`, computed via the multiplicative running-product form (no
/// factorials, so it never overflows for realistic `n`). Returns `0.0` for
/// the out-of-range `k > n` rather than underflowing: the historical bug
/// this hoist fixes computed `n - k` on `usize` before checking `k <= n`,
/// which panicked in debug builds and wrapped to a huge value in release.
fn binom_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut c = 1.0f64;
    for i in 0..k {
        c = c * (n - i) as f64 / (i + 1) as f64;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binom_coeff_matches_known_values_and_never_underflows() {
        assert_eq!(binom_coeff(5, 2), 10.0);
        assert_eq!(binom_coeff(5, 0), 1.0);
        assert_eq!(binom_coeff(5, 5), 1.0);
        // The historical bug: `n - k` on `usize` before checking `k <= n`
        // underflowed here. Must be exactly 0, not a panic or a huge number.
        assert_eq!(binom_coeff(3, 5), 0.0);
    }

    #[test]
    fn sign_test_all_discordant_pairs_favor_candidate() {
        let candidate = [1.0, 1.0, 1.0, 1.0, 1.0];
        let baseline = [0.0, 0.0, 0.0, 0.0, 0.0];
        let r = sign_test(&candidate, &baseline);
        assert_eq!(r.n, 5);
        assert_eq!(r.k, 5);
        assert!((r.p_value - 0.5f64.powi(5)).abs() < 1e-9, "p={}", r.p_value);
    }

    #[test]
    fn sign_test_conditions_on_discordant_pairs_only() {
        // Two tied pairs carry no directional evidence and must not count
        // toward `n` - this is the "sums over ALL pairs" defect the hoist
        // fixes: with the old (buggy) shape, tied pairs would each
        // contribute as a "loss," diluting a real, significant win.
        let candidate = [1.0, 0.0, 1.0];
        let baseline = [1.0, 0.0, 0.0];
        let r = sign_test(&candidate, &baseline);
        assert_eq!(r.n, 1);
        assert_eq!(r.k, 1);
        assert!((r.p_value - 0.5).abs() < 1e-9);
    }

    #[test]
    fn sign_test_no_discordant_pairs_is_not_significant() {
        let candidate = [1.0, 0.0, 1.0];
        let baseline = [1.0, 0.0, 1.0];
        let r = sign_test(&candidate, &baseline);
        assert_eq!(r.n, 0);
        assert_eq!(r.k, 0);
        assert_eq!(r.p_value, 1.0);
    }

    #[test]
    fn sign_test_mixed_discordant_pairs_matches_hand_binomial_tail() {
        // 4 discordant pairs, 3 favor the candidate:
        // P(Binomial(4,0.5) >= 3) = (C(4,3) + C(4,4)) / 16 = 5/16.
        let candidate = [1.0, 1.0, 1.0, 0.0, 5.0];
        let baseline = [0.0, 0.0, 0.0, 1.0, 5.0];
        let r = sign_test(&candidate, &baseline);
        assert_eq!(r.n, 4);
        assert_eq!(r.k, 3);
        assert!((r.p_value - 5.0 / 16.0).abs() < 1e-9, "p={}", r.p_value);
    }
}
