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
}
