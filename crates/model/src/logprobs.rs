// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-token log-probability access over [`crate::Model`] - the primitive
//! DPO, GRPO, and distillation are all built from (see
//! `Model::batch_token_logprobs`).
//!
//! Swedish Embedded AB builds training engines where a new objective composes
//! from existing primitives instead of duplicating them. If your team needs
//! expertise in preference optimization, RL-from-verifiable-rewards, or
//! distillation on top of a from-scratch training stack, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::Model;

/// Numerically-stable `log(softmax(row))[idx]` - the same three-line
/// reduction previously hand-rolled independently in `qwen3::eval::
/// score_chat`/`score_chat_paged`; both now call this instead of carrying
/// their own copy.
pub fn row_logprob(row: &[f32], idx: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    row[idx] - max - sum_exp.ln()
}

/// Per-position `log p(targets[i])` for one sequence, computed via
/// [`Model::logits_all`] plus a host log-softmax. Correct for any
/// token-classification model, at the cost of a full `[len * vocab]`
/// re-prefill and O(len·vocab) host work - the always-correct oracle, not
/// the fast path (see [`Model::batch_token_logprobs`] for that).
///
/// `targets[i] == crate::IGNORE` produces `0.0`, matching
/// [`Model::batch_token_logprobs`]'s documented convention exactly, so the
/// two are directly comparable on a batch that mixes real and masked
/// positions. `tokens` and `targets` must be the same length; returns `None`
/// if the model has no token-classification head (`logits_all` returns
/// `None`).
///
/// WARNING: on a model whose `logits_all` re-sets its current batch as a
/// side effect (e.g. `qwen3::Qwen`), calling this mid-training-step destroys
/// whatever batch the caller had set. Use [`Model::batch_token_logprobs`]
/// instead inside a training step; this function is for scoring outside one.
pub fn token_logprobs<M: Model>(m: &M, tokens: &[u32], targets: &[u32]) -> Option<Vec<f32>> {
    assert_eq!(tokens.len(), targets.len(), "logprobs::token_logprobs: tokens/targets length mismatch");
    let logits = m.logits_all(tokens)?;
    let vocab = logits.len() / tokens.len();
    Some(
        (0..tokens.len())
            .map(|i| {
                if targets[i] == crate::IGNORE {
                    0.0
                } else {
                    row_logprob(&logits[i * vocab..(i + 1) * vocab], targets[i] as usize)
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::row_logprob;

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
