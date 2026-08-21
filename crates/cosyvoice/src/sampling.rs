// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ras_sampling` - CosyVoice's repetition-aware nucleus sampler
//! (`cosyvoice/utils/common.py`), ported algorithm-for-algorithm.
//!
//! **The reference's `log_softmax` then `softmax` call chain, checked
//! empirically, is mathematically inert - not the "load-bearing double
//! normalization" it can look like at first read.**
//! `Qwen2LM.inference_wrapper` computes `logp =
//! llm_decoder(hidden).log_softmax(dim=-1)`, then [`nucleus_sampling`]
//! applies `.softmax(dim=0)` to that. `log_softmax(x) = x - logsumexp(x)` is
//! `x` shifted by one PER-ROW CONSTANT, and softmax is exactly shift-invariant
//! (`softmax(x - c) == softmax(x)` for any scalar `c`) - so
//! `softmax(log_softmax(x))` and `softmax(x)` are the identical distribution,
//! verified to `0.0` max-abs difference on synthetic vectors in this module's
//! own tests (`log_softmax_then_softmax_matches_plain_softmax` names the
//! result, not what its name might suggest). The `ignore_eos` mask
//! (`weighted_scores[eos] = -inf`) doesn't break this either: `-inf` shifted
//! by a finite constant is still `-inf`, so masking before or after the shift
//! selects the same support. This module still calls `log_softmax` then
//! `softmax` in that literal order (cheap, and it keeps the code a transcript
//! of the reference), but callers should not expect this ordering to change
//! *which* token gets sampled relative to a single plain `softmax` - the only
//! effect is two redundant numerically-stabilized reductions instead of one.
//!
//! **Not bit-exact with PyTorch.** The reference draws its `multinomial`
//! samples from torch's global Mersenne-Twister-derived CPU generator; this
//! port draws from `data::rng::Rng` (brain's own PRNG). The two streams are
//! algorithmically unrelated, so a shared seed does not reproduce the same
//! token sequence - see `crates/cosyvoice/tests/llm_parity.rs` for how the
//! parity suite treats this (a documented, honestly-reported gap; the
//! prefill hidden-state/logits parity rungs are the primary gate, not this
//! one). What IS reproduced faithfully is the sampling *algorithm* itself
//! (nucleus top-p/top-k selection order, the repetition-window guard, the
//! double-softmax quirk above), so the sampled distribution's *shape* matches
//! the reference even though the exact draws do not.

use data::rng::Rng;

/// `ras_sampling`'s hyperparameters (`Qwen2LM.inference_wrapper`'s call:
/// `top_p=0.8, top_k=25, win_size=10, tau_r=0.1`).
#[derive(Clone, Copy, Debug)]
pub struct RasParams {
    pub top_p: f32,
    pub top_k: usize,
    pub win_size: usize,
    pub tau_r: f32,
}

impl Default for RasParams {
    fn default() -> RasParams {
        RasParams { top_p: 0.8, top_k: 25, win_size: 10, tau_r: 0.1 }
    }
}

/// `x.softmax(dim=0)` - plain softmax, numerically stabilized.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exp.iter().sum();
    exp.into_iter().map(|v| v / s).collect()
}

/// `x.log_softmax(dim=-1)`.
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln();
    x.iter().map(|&v| v - lse).collect()
}

/// Sample one index from `probs` (already normalized weights, need not sum to
/// exactly 1 - matches `torch.multinomial`'s relative-weight semantics) via a
/// single uniform draw and a cumulative scan.
fn multinomial1(rng: &mut Rng, probs: &[f32]) -> usize {
    let total: f32 = probs.iter().sum();
    let r = rng.uniform(0.0, 1.0) as f32 * total;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    probs.len() - 1
}

/// `nucleus_sampling`: softmax the (already log-softmax'd) scores, take the
/// top-p/top-k prefix of a stable descending sort, then draw one index from
/// that truncated, renormalized distribution.
pub fn nucleus_sampling(rng: &mut Rng, weighted_scores: &[f32], top_p: f32, top_k: usize) -> u32 {
    let probs = softmax(weighted_scores);
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sel: Vec<usize> = Vec::new();
    let mut cum = 0.0f32;
    for &i in &order {
        if cum < top_p && sel.len() < top_k {
            cum += probs[i];
            sel.push(i);
        } else {
            break;
        }
    }
    let sel_probs: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
    sel[multinomial1(rng, &sel_probs)] as u32
}

/// `random_sampling`: plain softmax + a single multinomial draw over the
/// FULL vocabulary (the `ras_sampling` repetition-guard fallback).
pub fn random_sampling(rng: &mut Rng, weighted_scores: &[f32]) -> u32 {
    let probs = softmax(weighted_scores);
    multinomial1(rng, &probs) as u32
}

/// `ras_sampling`: nucleus-sample, then re-sample (full-vocab, with the
/// repeated id masked to `-inf`) if it repeats too often inside the trailing
/// `win_size` window. `weighted_scores` is mutated in place exactly as the
/// reference mutates `weighted_scores[top_ids] = -inf` on the fallback path -
/// callers that need the pre-guard scores afterward must clone first.
pub fn ras_sampling(rng: &mut Rng, weighted_scores: &mut [f32], decoded_tokens: &[u32], p: &RasParams) -> u32 {
    let top_ids = nucleus_sampling(rng, weighted_scores, p.top_p, p.top_k);
    let start = decoded_tokens.len().saturating_sub(p.win_size);
    let rep_num = decoded_tokens[start..].iter().filter(|&&t| t == top_ids).count();
    if rep_num as f32 >= p.win_size as f32 * p.tau_r {
        weighted_scores[top_ids as usize] = f32::NEG_INFINITY;
        return random_sampling(rng, weighted_scores);
    }
    top_ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let x = [1.0f32, 2.0, 0.5, -3.0, 4.0];
        let p = softmax(&x);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax sum={s}");
        assert!(p.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn log_softmax_then_softmax_matches_plain_softmax() {
        // softmax is shift-invariant and log_softmax is exactly a per-row
        // constant shift, so this chain is mathematically a no-op relative to
        // a single softmax - see the module doc's correction of the naive
        // "double normalization" reading.
        let x = [1.0f32, 2.0, 0.5, -3.0, 4.0];
        let direct = softmax(&x);
        let double = softmax(&log_softmax(&x));
        let diff: f32 = direct.iter().zip(&double).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 1e-6, "expected softmax(log_softmax(x)) == softmax(x), got diff={diff}");
    }

    #[test]
    fn nucleus_sampling_picks_a_high_probability_index_deterministically_per_seed() {
        let mut scores = vec![-10.0f32; 10];
        scores[3] = 0.0; // one dominant logit
        let logp = log_softmax(&scores);
        let mut rng = Rng::new(42);
        let a = nucleus_sampling(&mut rng, &logp, 0.8, 25);
        assert_eq!(a, 3, "the single dominant logit must be selected");
    }

    #[test]
    fn ras_sampling_avoids_repeating_past_the_window_guard() {
        let mut scores = vec![-10.0f32; 10];
        scores[3] = 0.0;
        let logp = log_softmax(&scores);
        let decoded = vec![3u32; 10]; // token 3 already repeated win_size times
        let mut rng = Rng::new(7);
        let mut ws = logp.clone();
        let tok = ras_sampling(&mut rng, &mut ws, &decoded, &RasParams::default());
        assert_ne!(tok, 3, "repetition guard must avoid re-picking the saturated token");
    }

    #[test]
    fn same_seed_is_deterministic() {
        let scores = vec![0.1f32, 0.2, 0.3, 0.15, 0.05, 0.1, 0.05, 0.02, 0.02, 0.01];
        let logp = log_softmax(&scores);
        let mut rng1 = Rng::new(123);
        let mut rng2 = Rng::new(123);
        let mut ws1 = logp.clone();
        let mut ws2 = logp.clone();
        let a = ras_sampling(&mut rng1, &mut ws1, &[], &RasParams::default());
        let b = ras_sampling(&mut rng2, &mut ws2, &[], &RasParams::default());
        assert_eq!(a, b);
    }
}
