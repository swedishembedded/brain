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
use super::{argmax, draw_truncated, SampleParams};

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

/// The no-`top_k` draw: the softmax is over every token of `entries`.
fn draw_over_row(entries: &mut [(u32, f32)], params: SampleParams, rng: &mut Rng) -> (u32, f32) {
    let temp = params.temp.max(1e-6);
    let best = entries.iter().map(|e| e.1).filter(|l| !l.is_nan()).fold(f32::NEG_INFINITY, f32::max);
    if best == f32::NEG_INFINITY {
        return (argmax(&entries.iter().map(|e| e.1).collect::<Vec<_>>()), 0.0);
    }
    // The weight of a logit relative to the best one's, computed as the candidate-list form does.
    let weight = |l: f32| if l.is_nan() { 0.0 } else { (l / temp - best / temp).exp() };
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

    // The width that certainly holds the survivors: min-p's survivors are the tokens at least `min_p` of the best,
    // counted in one pass; a nucleus grows until its mass reaches `top_p` of the whole row's.
    let mut width = if nucleus { FIRST_PREFIX } else { entries.iter().filter(|e| weight(e.1) >= params.min_p).count() };
    loop {
        let list = best_first(entries, width);
        let reached = !nucleus || list.len() >= entries.len() || list.iter().map(|e| weight(e.1) as f64).sum::<f64>() >= params.top_p as f64 * total;
        if reached {
            return draw_truncated(&list, params, Some(total as f32), rng);
        }
        width = (width * PREFIX_GROWTH).min(entries.len());
    }
}
