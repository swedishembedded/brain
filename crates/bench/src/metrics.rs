// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-agnostic evaluation metrics for the benchmark suite.
//!
//! These operate on plain values (cross-entropy in nats, token-id sequences,
//! per-position predictions) — never on a particular model type — so any
//! benchmark, and eventually any architecture, can produce and report them the
//! same way. A [`Metrics`] is a small bag of named scalars plus the headline
//! `score` that the runner thresholds.
//!
//! Definitions:
//! - **token cross-entropy** — mean next-token negative log-likelihood. In
//!   *nats* (natural log) or *bits* (`/ ln 2`).
//! - **bits-per-byte** — bits of CE per source byte; the corpus-size-independent
//!   compression number. `bits_per_byte = (total_nats / ln 2) / n_bytes`.
//! - **exact-match accuracy** — fraction of items whose full predicted sequence
//!   equals the reference.
//! - **associative-recall accuracy** — fraction of *queried answer positions*
//!   predicted correctly (the MQAR headline metric); chance is `1/vocab`.
//! - **distinct-n** — `unique n-grams / total n-grams`, a diversity proxy.
//! - **repetition-rate** — fraction of adjacent token pairs that are identical.
//!
//! Float-valued probabilistic forecasting metrics (pinball, CRPS, MASE, …) live
//! in [`forecast::metrics`] — the light crate the backtester and served
//! baselines can reach without the training stack.

use std::collections::HashMap;

/// Natural log of 2 — the nats→bits conversion factor.
pub const LN_2: f32 = std::f32::consts::LN_2;

/// A bag of named scalar metrics plus a headline `score` the runner thresholds.
///
/// Construct via [`Metrics::new`] then attach extra fields with [`Metrics::with`];
/// the `score` is whatever the benchmark considers its pass/fail quantity (for
/// MQAR: associative-recall accuracy). Keep field names stable — they become the
/// columns of the comparison table.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    /// Headline quantity the runner compares against a threshold.
    pub score: f32,
    /// Additional named scalars (cross-entropy, bpb, chance level, …).
    pub fields: HashMap<String, f32>,
}

impl Metrics {
    /// A metrics bag with the given headline `score` and no extra fields.
    pub fn new(score: f32) -> Self {
        Metrics { score, fields: HashMap::new() }
    }

    /// Attach (or overwrite) a named field; chainable.
    pub fn with(mut self, name: &str, value: f32) -> Self {
        self.fields.insert(name.to_string(), value);
        self
    }

    /// Look up an extra field.
    pub fn get(&self, name: &str) -> Option<f32> {
        self.fields.get(name).copied()
    }

    /// Serialize to a JSON object `{ "score": …, "fields": { … } }`. Field order
    /// within `fields` is sorted for stable, diffable artifacts. Used by the
    /// architecture-eval harness when writing results under `results/`.
    pub fn to_json(&self) -> serde_json::Value {
        let mut fields: Vec<(&String, &f32)> = self.fields.iter().collect();
        fields.sort_by(|a, b| a.0.cmp(b.0));
        let obj: serde_json::Map<String, serde_json::Value> = fields
            .into_iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(*v)))
            .collect();
        serde_json::json!({ "score": self.score, "fields": obj })
    }

    /// Reconstruct from a JSON object produced by [`Metrics::to_json`]. Missing /
    /// malformed fields are skipped; a missing `score` defaults to `0.0`.
    pub fn from_json(v: &serde_json::Value) -> Self {
        let score = v.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
        let mut fields = HashMap::new();
        if let Some(obj) = v.get("fields").and_then(|f| f.as_object()) {
            for (k, val) in obj {
                if let Some(x) = val.as_f64() {
                    fields.insert(k.clone(), x as f32);
                }
            }
        }
        Metrics { score, fields }
    }
}

/// Mean cross-entropy in **nats** from a total nats sum and token count.
pub fn cross_entropy_nats(total_nats: f32, n_tokens: usize) -> f32 {
    total_nats / n_tokens.max(1) as f32
}

/// Mean cross-entropy in **bits** (nats / ln 2).
pub fn cross_entropy_bits(total_nats: f32, n_tokens: usize) -> f32 {
    cross_entropy_nats(total_nats, n_tokens) / LN_2
}

/// Bits-per-byte: total CE (nats) expressed in bits, per source byte.
pub fn bits_per_byte(total_nats: f32, n_bytes: usize) -> f32 {
    (total_nats / LN_2) / n_bytes.max(1) as f32
}

/// Exact-match accuracy over `(prediction, reference)` token-sequence pairs.
pub fn exact_match<T: PartialEq>(pairs: &[(Vec<T>, Vec<T>)]) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let hits = pairs.iter().filter(|(p, r)| p == r).count();
    hits as f32 / pairs.len() as f32
}

/// Associative-recall accuracy: fraction of `(predicted, expected)` answer
/// positions that match. This is the MQAR headline metric. Chance is `1/vocab`.
pub fn associative_recall(predicted: &[u32], expected: &[u32]) -> f32 {
    assert_eq!(predicted.len(), expected.len(), "recall: length mismatch");
    if expected.is_empty() {
        return 0.0;
    }
    let hits = predicted.iter().zip(expected).filter(|(p, e)| p == e).count();
    hits as f32 / expected.len() as f32
}

/// Distinct-n: `unique n-grams / total n-grams` over a token sequence. A
/// diversity proxy (1.0 = no repeated n-grams, →0 = highly repetitive).
pub fn distinct_ngrams(tokens: &[u32], n: usize) -> f32 {
    if n == 0 || tokens.len() < n {
        return 0.0;
    }
    let total = tokens.len() - n + 1;
    let unique: std::collections::HashSet<&[u32]> = tokens.windows(n).collect();
    unique.len() as f32 / total as f32
}

/// Repetition-rate: fraction of adjacent token pairs that are identical.
pub fn repetition_rate(tokens: &[u32]) -> f32 {
    if tokens.len() < 2 {
        return 0.0;
    }
    let reps = tokens.windows(2).filter(|w| w[0] == w[1]).count();
    reps as f32 / (tokens.len() - 1) as f32
}

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

/// Result of [`ols_slope_ci`]: the fitted line plus a two-sided 95 %
/// confidence interval on its slope.
///
/// Reported, not asserted on, by the continual-learning study in `crates/rl`:
/// a 12-point single-seed series cannot power a slope test, so the CI is a
/// diagnostic that says how much the series could support, never a pass/fail
/// quantity. A CI that contains 0 means "no trend distinguishable at this
/// resolution", which is NOT the same claim as "there is no trend".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slope {
    pub slope: f64,
    pub intercept: f64,
    /// Standard error of `slope`.
    pub stderr: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    /// Number of points the fit ran over.
    pub n: usize,
}

/// Two-sided 95 % critical value of the t distribution at `df` degrees of
/// freedom. Small hard-coded table for `df` 1..=30 (the only regime a
/// 12-point series can reach), 1.960 (the normal limit) beyond - the same
/// "no external stats dependency, spell the constants out" stance
/// [`binom_coeff`] takes for the sign test.
fn t_crit_95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080,
        2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    ];
    if df == 0 {
        return f64::INFINITY;
    }
    if df <= 30 {
        T[df - 1]
    } else {
        1.960
    }
}

/// Ordinary-least-squares fit of `ys` on `xs` with a two-sided 95 %
/// confidence interval on the slope from the t distribution at `n - 2`
/// degrees of freedom. `None` for fewer than 3 points (no residual degrees of
/// freedom, so no interval) or a degenerate `xs` (zero variance).
///
/// Lives next to [`sign_test`] for the same reason that one was hoisted here:
/// `bench` is how this repo scores models, and a trend-over-cycles statistic
/// is a scoring statistic, not a training-loop detail.
pub fn ols_slope_ci(xs: &[f64], ys: &[f64]) -> Option<Slope> {
    assert_eq!(xs.len(), ys.len(), "ols_slope_ci: xs and ys must have equal length");
    let n = xs.len();
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    if sxx <= f64::EPSILON {
        return None;
    }
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let sse: f64 = xs.iter().zip(ys).map(|(x, y)| {
        let r = y - (intercept + slope * x);
        r * r
    }).sum();
    let df = n - 2;
    let stderr = (sse / df as f64 / sxx).sqrt();
    let t = t_crit_95(df);
    Some(Slope { slope, intercept, stderr, ci_lo: slope - t * stderr, ci_hi: slope + t * stderr, n })
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
    fn ce_nats_bits_bpb() {
        // total 4 nats over 2 tokens -> 2 nats/tok -> 2/ln2 bits/tok.
        assert!((cross_entropy_nats(4.0, 2) - 2.0).abs() < 1e-6);
        assert!((cross_entropy_bits(4.0, 2) - 2.0 / LN_2).abs() < 1e-5);
        // 4 nats over 8 bytes -> (4/ln2)/8 bits/byte.
        assert!((bits_per_byte(4.0, 8) - (4.0 / LN_2) / 8.0).abs() < 1e-5);
    }

    #[test]
    fn exact_match_counts_full_sequence_equality() {
        let pairs = vec![(vec![1u32, 2], vec![1u32, 2]), (vec![1u32, 2], vec![1u32, 3])];
        assert!((exact_match(&pairs) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recall_fraction_of_matching_positions() {
        let pred = [3u32, 1, 4, 1];
        let exp = [3u32, 2, 4, 1];
        assert!((associative_recall(&pred, &exp) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn distinct_and_repetition() {
        // [1,1,2,2]: bigrams (1,1)(1,2)(2,2) all unique -> 1.0; reps (1,1)&(2,2) -> 2/3.
        let t = [1u32, 1, 2, 2];
        assert!((distinct_ngrams(&t, 2) - 1.0).abs() < 1e-6);
        assert!((repetition_rate(&t) - 2.0 / 3.0).abs() < 1e-6);
    }

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
    fn ols_slope_ci_recovers_an_exact_line_with_a_ci_containing_it() {
        // y = 3 - 0.5x exactly: zero residual -> zero stderr -> a degenerate
        // CI that still contains the true slope.
        let xs: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 - 0.5 * x).collect();
        let s = ols_slope_ci(&xs, &ys).expect("12 points is enough");
        assert!((s.slope - -0.5).abs() < 1e-9, "{s:?}");
        assert!((s.intercept - 3.0).abs() < 1e-9, "{s:?}");
        assert_eq!(s.n, 12);
        assert!(s.ci_lo <= -0.5 && s.ci_hi >= -0.5, "the CI must contain the slope it recovered: {s:?}");
    }

    #[test]
    fn ols_slope_ci_on_a_flat_noisy_series_has_a_ci_containing_zero() {
        // Deterministic zig-zag around a constant: no real trend, so a
        // 95 % CI that EXCLUDED 0 would be the statistic lying about a trend.
        let xs: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..12).map(|i| 0.5 + if i % 2 == 0 { 0.05 } else { -0.05 }).collect();
        let s = ols_slope_ci(&xs, &ys).expect("12 points is enough");
        assert!(s.ci_lo <= 0.0 && s.ci_hi >= 0.0, "a flat noisy series' slope CI must contain 0: {s:?}");
    }

    #[test]
    fn ols_slope_ci_is_none_without_residual_degrees_of_freedom() {
        assert!(ols_slope_ci(&[1.0, 2.0], &[1.0, 2.0]).is_none(), "n = 2 has no residual df, so no interval");
        assert!(ols_slope_ci(&[], &[]).is_none());
        // Degenerate x (every point at the same cycle index) has no slope.
        assert!(ols_slope_ci(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
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
