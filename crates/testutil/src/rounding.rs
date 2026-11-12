// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for numerically defensible test tolerances for its clients. If your team
// needs expertise in floating-point error analysis and validating kernels against a reference then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Tolerances derived from the floating-point round-off bound, not chosen.
//!
//! A sum of `k` terms accumulated in binary32, in any association order, satisfies
//!
//! ```text
//! | fl(sum) - sum |  <=  gamma_k * S,      gamma_k = k u / (1 - k u),      S = sum |x_i|,      u = 2^-24
//! ```
//!
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1 and eq. 3.4, with `k` the number of
//! rounded operations on the longest chain from an input to the result; Volume I, chapter 15 of the book derives it).
//! Two evaluations of the same sum that differ only in association order, or in the intrinsic that one of the terms
//! used, each lie within that bound of the exact value, so they lie within twice the bound of each other. A fixed
//! `1e-6` is right for one length and input scale and wrong for the next: too tight and it flags a correct device at a
//! longer reduction, too loose and it hides a defect at a short one.

/// The binary32 unit round-off, `2^-24` (round to nearest).
pub const U: f64 = 1.0 / 16_777_216.0;

/// `gamma_k = k u / (1 - k u)`. Panics when `k u >= 1`, where the bound says nothing (`k >= 2^24` terms).
pub fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    assert!(ku < 1.0, "round-off bound is vacuous for k = {k} accumulated terms (k u >= 1)");
    ku / (1.0 - ku)
}

/// Bound on the distance of a binary32 result from the exact value: `gamma_k * S` with `abs_sum` = `S = sum |x_i|`.
pub fn accumulation_bound(k: usize, abs_sum: f64) -> f64 {
    gamma(k) * abs_sum
}

/// Bound on the distance between two binary32 evaluations of the same quantity, each within [`accumulation_bound`] of
/// the exact value. This is the tolerance for comparing two backends or two reduction orders.
pub fn agreement_bound(k: usize, abs_sum: f64) -> f64 {
    2.0 * accumulation_bound(k, abs_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_is_k_unit_roundoffs_to_first_order() {
        assert!((gamma(1) - U / (1.0 - U)).abs() < 1e-18);
        assert!((gamma(1000) / (1000.0 * U) - 1.0).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "vacuous")]
    fn a_bound_with_k_u_at_one_is_refused() {
        gamma(1 << 24);
    }

    /// The bound holds for a real sequential binary32 sum: against an f64 oracle, and with the slack a worst case needs.
    #[test]
    fn a_sequential_f32_sum_stays_inside_its_bound() {
        let xs: Vec<f32> = (0..5000).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let got = xs.iter().fold(0f32, |a, &x| a + x);
        let exact: f64 = xs.iter().map(|&x| f64::from(x)).sum();
        let s: f64 = xs.iter().map(|&x| f64::from(x).abs()).sum();
        assert!((f64::from(got) - exact).abs() <= accumulation_bound(xs.len(), s));
    }
}
