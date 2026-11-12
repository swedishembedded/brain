// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sampling that stays exact at the full
// vocabulary for its clients. If your team needs expertise in decoding
// strategies for serving language models then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Sampling from a decoder's whole `[vocab]` logits row.
//!
//! A candidate list of the best K tokens cannot answer a request exactly once
//! the distribution it truncates is wider than K: the softmax normaliser needs
//! every token's mass, a nucleus (`top_p`) can span thousands of tokens of a
//! flat row, and a penalty can raise a token from anywhere. This module takes
//! the row itself, applies the logit processors to the tokens the context
//! touches, and draws from the exact distribution of the request's `top_k`,
//! `top_p` and `min_p`:
//!
//! * `top_k` (below the vocabulary) keeps exactly k tokens, lowest id first among
//!   equal logits, and the softmax runs over those k: the same set the
//!   candidate-list form keeps.
//! * Without `top_k` the softmax normaliser `Z` is summed over the whole row. A
//!   `top_p` nucleus is found on a sorted prefix that **grows** (64 tokens, then
//!   four times wider each round) until its mass reaches `top_p * Z`; the bound
//!   is the row itself, so the worst case is one full sort of the vocabulary and
//!   the result is exact whatever the width. `min_p` needs the tokens whose
//!   probability is at least `min_p` times the best one's, counted in one pass.
//!   With neither, the draw is the inverse CDF over the row in id order, no
//!   sort at all.

use data::rng::Rng;

use super::sampler::{apply_processors_to_row, ProcessorState, SampleContext};
use super::{argmax, draw_truncated, truncated_weights_of, SampleParams};

/// The first width of the sorted prefix a nucleus is searched in.
const FIRST_PREFIX: usize = 64;

/// A sorted prefix grows by this factor each round until it holds the nucleus.
const PREFIX_GROWTH: usize = 4;

/// `a` before `b` when its logit is higher; equal logits rank lowest id first; NaN ranks last.
fn rank(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan())).then(a.0.cmp(&b.0))
}

/// The `width` best tokens of `entries`, best first. Reorders `entries`.
fn best_first(entries: &mut [(u32, f32)], width: usize) -> Vec<(u32, f32)> {
    let width = width.clamp(1, entries.len());
    if width < entries.len() {
        entries.select_nth_unstable_by(width - 1, rank);
    }
    let mut head = entries[..width].to_vec();
    head.sort_by(rank);
    head
}

/// The `k` best `(token, logit)` pairs of a logits `row`, best first, equal logits lowest id first: the candidate
/// list a decoder without an on-device top-k answers [`PagedDecoder::forward_batched_topk`](super::PagedDecoder::forward_batched_topk)
/// with from a row it read back. `k` is at least 1.
pub fn best_candidates(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut entries: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    best_first(&mut entries, k)
}

/// Draw one token from a decoder's logits `row` under `params`, after the request's logit processors, and return it
/// with its log-probability under the exact distribution it was drawn from (`0.0` for a greedy pick). `row` is
/// consumed as scratch: the processors are applied to it in place.
///
/// Exact over the whole row for every `top_k`, `top_p`, `min_p` and processor setting; see the module doc.
pub fn sample_logits_row(row: &mut [f32], params: SampleParams, ctx: &SampleContext<'_>, state: &mut ProcessorState, rng: &mut Rng) -> (u32, f32) {
    assert!(!row.is_empty(), "sample_logits_row: empty logits row");
    apply_processors_to_row(row, &params, ctx, state);
    if params.is_greedy() {
        return (argmax(row), 0.0);
    }
    let mut entries: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    if params.top_k > 0 && params.top_k < row.len() {
        let list = best_first(&mut entries, params.top_k);
        return draw_truncated(&list, params, None, rng);
    }
    draw_over_row(&mut entries, params, rng)
}

/// The softmax weight of a logit relative to the row's best, as the candidate-list form computes it. NaN weighs nothing.
fn weight_of(l: f32, temp: f32, best: f32) -> f32 {
    if l.is_nan() {
        0.0
    } else {
        (l / temp - best / temp).exp()
    }
}

/// The best logit of `entries`, ignoring NaN; `None` when none is finite-or-infinite (an all-NaN or all `-inf` row).
fn best_logit(entries: &[(u32, f32)]) -> Option<f32> {
    let best = entries.iter().map(|e| e.1).filter(|l| !l.is_nan()).fold(f32::NEG_INFINITY, f32::max);
    (best != f32::NEG_INFINITY).then_some(best)
}

/// The best-first prefix of `entries` that holds every token a `top_p` or `min_p` request can draw, and the softmax
/// normaliser `total` of the whole row: min-p's survivors are counted in one pass, a nucleus prefix grows until its
/// mass reaches `top_p` of `total`. Only for a request with one of the two.
fn survivor_prefix(entries: &mut [(u32, f32)], params: SampleParams, total: f64, weight: impl Fn(f32) -> f32) -> Vec<(u32, f32)> {
    let nucleus = params.top_p > 0.0 && params.top_p < 1.0;
    let mut width = if nucleus { FIRST_PREFIX } else { entries.iter().filter(|e| weight(e.1) >= params.min_p).count() };
    loop {
        let list = best_first(entries, width);
        let reached = !nucleus || list.len() >= entries.len() || list.iter().map(|e| weight(e.1) as f64).sum::<f64>() >= params.top_p as f64 * total;
        if reached {
            return list;
        }
        width = (width * PREFIX_GROWTH).min(entries.len());
    }
}

/// The no-`top_k` draw: the softmax is over every token of `entries`.
fn draw_over_row(entries: &mut [(u32, f32)], params: SampleParams, rng: &mut Rng) -> (u32, f32) {
    let temp = params.temp.max(1e-6);
    let Some(best) = best_logit(entries) else {
        return (argmax(&entries.iter().map(|e| e.1).collect::<Vec<_>>()), 0.0);
    };
    let weight = |l: f32| weight_of(l, temp, best);
    let total: f64 = entries.iter().map(|e| weight(e.1) as f64).sum();
    let nucleus = params.top_p > 0.0 && params.top_p < 1.0;
    let min_p = params.min_p > 0.0;

    if !nucleus && !min_p {
        // Inverse CDF over the row in id order: no sort, and every token's mass is in the normaliser.
        let target = rng.next_f32() as f64 * total;
        let mut acc = 0.0f64;
        let mut last = 0usize;
        for (i, e) in entries.iter().enumerate() {
            let w = weight(e.1) as f64;
            if w > 0.0 {
                last = i;
            }
            acc += w;
            if acc >= target && w > 0.0 {
                return (e.0, (w.ln() - total.ln()) as f32);
            }
        }
        let w = weight(entries[last].1) as f64;
        return (entries[last].0, (w.ln() - total.ln()) as f32);
    }

    let list = survivor_prefix(entries, params, total, weight);
    draw_truncated(&list, params, Some(total as f32), rng)
}

/// The tokens a non-greedy request can draw, with their unnormalised softmax weights: the exact distribution
/// speculative verification tests a proposal against and corrects from. `weights[i]` belongs to `tokens[i]`, and
/// `kept_sum` is their sum, so a token's probability is its weight over it. A greedy request is the one token it picks.
pub(super) struct Survivors {
    tokens: Vec<u32>,
    weights: Vec<f32>,
    kept_sum: f32,
}

impl Survivors {
    /// The survivors of a best-first candidate `list`: exact within the list ([`truncated_weights_of`]).
    pub(super) fn of_list(list: &[(u32, f32)], params: SampleParams) -> Survivors {
        if params.is_greedy() {
            return Survivors { tokens: vec![list[0].0], weights: vec![1.0], kept_sum: 1.0 };
        }
        let (weights, kept_sum) = truncated_weights_of(list, params, None);
        Survivors { tokens: list[..weights.len()].iter().map(|c| c.0).collect(), weights, kept_sum }
    }

    /// The best token: the greedy pick, and the first one a best-first list holds.
    pub(super) fn best(&self) -> u32 {
        self.tokens[0]
    }

    /// The weight of `token`, `0` for a token the request cannot draw.
    pub(super) fn weight(&self, token: u32) -> f32 {
        self.tokens.iter().position(|&t| t == token).map_or(0.0, |i| self.weights[i])
    }

    pub(super) fn kept_sum(&self) -> f32 {
        self.kept_sum
    }

    /// The log-probability of `token`: `0` for a greedy pick, `-inf` for a token outside the survivors.
    pub(super) fn logprob(&self, token: u32, params: SampleParams) -> f32 {
        if params.is_greedy() {
            return 0.0;
        }
        match self.weight(token) {
            w if w > 0.0 => w.ln() - self.kept_sum.ln(),
            _ => f32::NEG_INFINITY,
        }
    }

    /// The survivor an inverse-CDF walk reaches at cumulative weight `r`, not counting `skip`'s weight (a rejected
    /// proposal's, whose mass the caller took out of the range `r` is drawn from). Rounding that leaves `r` past the
    /// last weight lands on the last survivor; `None` only when there is no survivor but `skip`.
    pub(super) fn draw_at(&self, r: f32, skip: Option<u32>) -> Option<u32> {
        let mut acc = 0.0f32;
        let mut last = None;
        for (&t, &w) in self.tokens.iter().zip(&self.weights) {
            if Some(t) == skip {
                continue;
            }
            last = Some(t);
            acc += w;
            if acc >= r {
                return Some(t);
            }
        }
        last
    }
}

/// The survivors of a decoder's logits `row` under `params` (no logit processor: a speculative verify excludes
/// them), exact over the whole row. The same set and weights [`sample_logits_row`] draws from; the order is the
/// ranking of the sorted prefix, or id order when nothing truncates.
pub(super) fn survivors_of_row(row: &[f32], params: SampleParams) -> Survivors {
    assert!(!row.is_empty(), "survivors_of_row: empty logits row");
    if params.is_greedy() {
        return Survivors { tokens: vec![argmax(row)], weights: vec![1.0], kept_sum: 1.0 };
    }
    let mut entries: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    if params.top_k > 0 && params.top_k < row.len() {
        return Survivors::of_list(&best_first(&mut entries, params.top_k), params);
    }
    let temp = params.temp.max(1e-6);
    let Some(best) = best_logit(&entries) else {
        return Survivors { tokens: vec![argmax(row)], weights: vec![1.0], kept_sum: 1.0 };
    };
    let weight = |l: f32| weight_of(l, temp, best);
    let total: f64 = entries.iter().map(|e| weight(e.1) as f64).sum();
    if params.min_p > 0.0 || (params.top_p > 0.0 && params.top_p < 1.0) {
        let list = survivor_prefix(&mut entries, params, total, weight);
        let (weights, kept_sum) = truncated_weights_of(&list, params, Some(total as f32));
        return Survivors { tokens: list[..weights.len()].iter().map(|c| c.0).collect(), weights, kept_sum };
    }
    let weights: Vec<f32> = row.iter().map(|&l| weight(l)).collect();
    let kept_sum = weights.iter().map(|&w| w as f64).sum::<f64>() as f32;
    Survivors { tokens: (0..row.len() as u32).collect(), weights, kept_sum }
}
