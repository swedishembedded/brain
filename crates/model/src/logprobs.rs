// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-token log-probability access over [`crate::Model`] - the primitive
//! DPO, GRPO, and distillation are all built from (see
//! `crate::training::TokenObjectives::batch_token_logprobs`).
//!
//! Swedish Embedded AB builds training engines where a new objective composes
//! from existing primitives instead of duplicating them. If your team needs
//! expertise in preference optimization, RL-from-verifiable-rewards, or
//! distillation on top of a from-scratch training stack, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::Model;

/// Numerically-stable `log(softmax(row))[idx]` - the same three-line
/// reduction previously written independently in `qwen3::eval::
/// score_chat`/`score_chat_paged`; both now call this instead of carrying
/// their own copy.
pub fn row_logprob(row: &[f32], idx: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    row[idx] - max - sum_exp.ln()
}

/// `log(softmax(row / temperature))[idx]`: the log-probability the model's full vocabulary distribution gives `idx`
/// at `temperature`, untruncated (no top-k, top-p or min-p). The one definition of the "full distribution" figure a
/// policy trace reports, so a trainer scoring the same tokens with `1 / temperature` as its logit scale
/// ([`token_logprobs_scaled`]) and the server agree by construction. `temperature` must be finite and positive.
pub fn row_logprob_at(row: &[f32], idx: usize, temperature: f32) -> f32 {
    assert!(temperature.is_finite() && temperature > 0.0, "the temperature must be finite and positive, got {temperature}");
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = row.iter().map(|&v| ((v - max) / temperature).exp()).sum();
    (row[idx] - max) / temperature - sum_exp.ln()
}

/// The `k` largest entries of `row`'s log-softmax as `(index, logprob)`, most likely first (ties toward the lower
/// index). O(vocab) for the normaliser plus O(vocab log k) for the selection.
pub fn row_top_logprobs(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    let k = k.min(row.len());
    if k == 0 {
        return Vec::new();
    }
    // A min-heap of the k best seen so far: (value, Reverse(index)) orders equal values toward the lower index.
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(OrdF32, std::cmp::Reverse<u32>)>> = std::collections::BinaryHeap::with_capacity(k + 1);
    for (i, &v) in row.iter().enumerate() {
        heap.push(std::cmp::Reverse((OrdF32(v), std::cmp::Reverse(i as u32))));
        if heap.len() > k {
            heap.pop();
        }
    }
    let mut top: Vec<(u32, f32)> = heap.into_iter().map(|std::cmp::Reverse((OrdF32(v), std::cmp::Reverse(i)))| (i, v - lse)).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    top
}

/// An `f32` ordered by [`f32::total_cmp`], for heaps.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Per-position `log p(targets[i])` for one sequence, computed via
/// [`Model::logits_all`] plus a host log-softmax. Correct for any
/// token-classification model, at the cost of a full `[len * vocab]`
/// re-prefill and O(len·vocab) host work - the always-correct oracle, not
/// the fast path (see [`crate::training::TokenObjectives::batch_token_logprobs`] for that).
///
/// `targets[i] == crate::IGNORE` produces `0.0`, matching
/// [`crate::training::TokenObjectives::batch_token_logprobs`]'s documented convention exactly, so the
/// two are directly comparable on a batch that mixes real and masked
/// positions. `tokens` and `targets` must be the same length; returns `None`
/// if the model has no token-classification head (`logits_all` returns
/// `None`).
///
/// WARNING: on a model whose `logits_all` re-sets its current batch as a
/// side effect (e.g. `qwen3::Qwen`), calling this mid-training-step destroys
/// whatever batch the caller had set. Use [`crate::training::TokenObjectives::batch_token_logprobs`]
/// instead inside a training step; this function is for scoring outside one.
pub fn token_logprobs<M: Model>(m: &M, tokens: &[u32], targets: &[u32]) -> Option<Vec<f32>> {
    token_logprobs_scaled(m, tokens, targets, 1.0)
}

/// [`token_logprobs`] under logits multiplied by `logit_scale` (`1/T` for a
/// sampler that drew at temperature `T`), the distribution
/// `TokenObjectives::set_logit_scale` makes a model train against. The scale
/// is applied here on the host to the model's unscaled logits: a model whose
/// own scale is set would be scaled twice, so score with it at `1.0`.
pub fn token_logprobs_scaled<M: Model>(m: &M, tokens: &[u32], targets: &[u32], logit_scale: f32) -> Option<Vec<f32>> {
    assert_eq!(tokens.len(), targets.len(), "logprobs::token_logprobs: tokens/targets length mismatch");
    assert!(logit_scale.is_finite() && logit_scale > 0.0, "the logit scale must be finite and positive, got {logit_scale}");
    let logits = m.logits_all(tokens)?;
    let vocab = logits.len() / tokens.len();
    let mut scaled = vec![0.0f32; vocab];
    Some(
        (0..tokens.len())
            .map(|i| {
                if targets[i] == crate::IGNORE {
                    return 0.0;
                }
                for (s, z) in scaled.iter_mut().zip(&logits[i * vocab..(i + 1) * vocab]) {
                    *s = z * logit_scale;
                }
                row_logprob(&scaled, targets[i] as usize)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::{row_logprob, row_logprob_at, row_top_logprobs};

    #[test]
    fn row_logprob_at_is_the_log_softmax_of_the_scaled_row_and_row_logprob_at_one() {
        let row = [1.0f32, 3.0, 2.0, -1.0, 0.25];
        for idx in 0..row.len() {
            assert!((row_logprob_at(&row, idx, 1.0) - row_logprob(&row, idx)).abs() < 1e-6);
            let scaled: Vec<f32> = row.iter().map(|v| v / 0.7).collect();
            assert!((row_logprob_at(&row, idx, 0.7) - row_logprob(&scaled, idx)).abs() < 1e-5, "temperature 0.7, index {idx}");
        }
        let total: f32 = (0..row.len()).map(|i| row_logprob_at(&row, i, 2.5).exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "a distribution sums to one, got {total}");
    }

    #[test]
    fn row_top_logprobs_are_the_k_largest_log_softmax_entries_most_likely_first() {
        let row = [1.0f32, 3.0, 2.0, 3.0, -1.0];
        let top = row_top_logprobs(&row, 3);
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3, 2], "ties go to the lower index");
        for &(i, lp) in &top {
            assert!((lp - row_logprob(&row, i as usize)).abs() < 1e-6);
        }
        assert_eq!(row_top_logprobs(&row, 9).len(), 5, "k is capped at the row");
        assert!(row_top_logprobs(&row, 0).is_empty());
    }

    #[test]
    fn row_logprob_matches_definition() {
        let row = [1.0f32, 2.0, 0.5, -1.0];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&v| (v - max).exp()).sum();
        for (idx, &v) in row.iter().enumerate() {
            let expected = v - max - sum_exp.ln();
            assert!((row_logprob(&row, idx) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn row_logprob_is_never_positive() {
        // log(softmax(x)) <= 0 for every element, since softmax outputs sum to 1.
        let row = [10.0f32, -3.0, 0.0, 7.5, 100.0];
        for idx in 0..row.len() {
            assert!(row_logprob(&row, idx) <= 1e-5, "row_logprob({idx}) = {}", row_logprob(&row, idx));
        }
    }
}
