// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sampled decoding that keeps the accelerator,
// not one host core, the bound of a served model for its clients. If your team
// needs expertise in exact, reproducible sampling at serving batch sizes then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Keyed whole-row draws: a token drawn from a decoder's `[vocab]` logits row by
//! the Gumbel-max rule, with noise that is a pure function of the request's seed,
//! the position being drawn and the token, so the same draw can run where the
//! logits are (`model::readout::RowDrawer`) or here on the host, and answers
//! the same either way.
//!
//! For a request with no logit processor ([`SampleParams::row_drawable`]) the distribution
//! [`super::sample_logits_row`] defines is the softmax of `logit / temp` over the survivors: every token of finite
//! weight, cut to the best `top_k`, to the `top_p` nucleus and to the `min_p` floor. The Gumbel-max rule draws from
//! exactly that distribution: with `g_i` independent standard Gumbel noise, `argmax_i (logit_i / temp + g_i)` over
//! the survivors is token `i` with probability `w_i / sum w`. Without a `top_k` or a nucleus that is one pass over
//! the row, no sort, no prefix sum; with one it is the same rule over the best-first list that holds the survivors
//! ([`draw_list`]), which a device answers from its top candidates whenever they hold them.
//!
//! The noise of token `t` is `-ln(-ln(u))` for `u = token_uniform(key, t)`, and
//! `key = draw_key(seed, position, draw)` depends on the request alone, never
//! on the batch slot it decodes in: a seed draws the same tokens in any batch.
//! The returned log-probability is that of the drawn token under the exact
//! survivor distribution, as [`super::sample_logits_row`] reports it.

use super::SampleParams;

/// One row's keyed draw: what a decoder needs to draw that row's token on its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowDraw {
    /// Temperature (positive; the host floor `1e-6` applies as in the row sampler).
    pub temp: f32,
    /// The min-p floor relative to the best token's weight, in `[0, 1]`; `0` for none.
    pub min_p: f32,
    /// The nucleus; `1` (or `0`) for none.
    pub top_p: f32,
    /// The best tokens kept; `0` for every token.
    pub top_k: usize,
    /// The row's noise key ([`draw_key`]).
    pub key: u32,
}

impl RowDraw {
    /// The keyed draw of a request sampled with `params` at `position` (the number of tokens before the one drawn)
    /// under its `seed`, or `None` when `params` is not drawable this way ([`SampleParams::row_drawable`]).
    pub fn of(params: &SampleParams, seed: u64, position: usize) -> Option<RowDraw> {
        params.row_drawable().then(|| RowDraw { temp: params.temp.max(1e-6), min_p: params.min_p.clamp(0.0, 1.0), top_p: params.top_p, top_k: params.top_k, key: draw_key(seed, position as u64, 0) })
    }

    /// Whether the draw keeps the best `top_k` of a `vocab`-wide row.
    pub fn keeps_top_k(&self, vocab: usize) -> bool {
        self.top_k > 0 && self.top_k < vocab
    }

    /// Whether the draw cuts a nucleus.
    pub fn has_nucleus(&self) -> bool {
        self.top_p > 0.0 && self.top_p < 1.0
    }

    /// Whether the survivors of a `vocab`-wide row are a best-first list rather than a threshold over the row.
    pub fn truncates(&self, vocab: usize) -> bool {
        self.keeps_top_k(vocab) || self.has_nucleus()
    }

    fn params(&self) -> SampleParams {
        SampleParams { temp: self.temp, top_k: self.top_k, top_p: self.top_p, min_p: self.min_p, ..SampleParams::greedy() }
    }
}

/// What a decoder answers one row of a keyed step with: the token it drew there, or the row itself for a request
/// the keyed draw does not cover (a processor, a nucleus, a `top_k` inside the vocabulary).
#[derive(Clone, Debug, PartialEq)]
pub enum Drawn {
    /// The drawn token, its log-probability under the distribution it was drawn from (the survivors of the cut), and
    /// its log-probability under the whole row at the request temperature, `log_softmax(logits / temp)[token]`: the
    /// behaviour policy's own figure for a draw that is cut (see [`draw_row_whole`]).
    Token(u32, f32, f32),
    /// The row's whole `[vocab]` logits, for the host sampler.
    Row(Vec<f32>),
}

/// The 32-bit finaliser the keyed noise is built from: every input bit reaches every output bit.
#[inline]
pub fn mix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// The noise key of the `draw`-th draw at `position` of the request seeded `seed`.
pub fn draw_key(seed: u64, position: u64, draw: u32) -> u32 {
    let h = mix32(seed as u32 ^ 0x5bd1_e995);
    let h = mix32(h ^ (seed >> 32) as u32);
    let h = mix32(h ^ position as u32);
    let h = mix32(h ^ (position >> 32) as u32);
    mix32(h ^ draw.wrapping_mul(0x27d4_eb2d))
}

/// The uniform in `(0, 1)` that token `token` of a row keyed `key` draws its noise from: 23 bits, centred in its
/// cell, so it is never 0 or 1 and its Gumbel transform is finite. The device kernels compute the same expression.
#[inline]
pub fn token_uniform(key: u32, token: u32) -> f32 {
    let h = mix32(key ^ mix32(token.wrapping_add(0x9e37_79b9)));
    ((h >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

/// Standard Gumbel noise of token `token` of a row keyed `key`.
#[inline]
pub fn token_gumbel(key: u32, token: u32) -> f32 {
    -(-token_uniform(key, token).ln()).ln()
}

/// The token `d` draws from `row` and its log-probability: the host reference of the device draw. A row with no
/// token of positive weight (all NaN or `-inf`) answers its first best index with log-probability `0`, as the row
/// sampler does.
pub fn draw_row(row: &[f32], d: RowDraw) -> (u32, f32) {
    assert!(!row.is_empty(), "draw_row: empty logits row");
    if d.truncates(row.len()) {
        return draw_truncated_row(row, d);
    }
    let temp = d.temp.max(1e-6);
    let best = row.iter().filter(|l| !l.is_nan()).map(|&l| l / temp).fold(f32::NEG_INFINITY, f32::max);
    if best == f32::NEG_INFINITY {
        return (super::argmax(row), 0.0);
    }
    let mut kept = 0.0f64;
    let (mut token, mut top, mut scaled) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &l) in row.iter().enumerate() {
        let s = l / temp;
        let w = (s - best).exp();
        // NaN fails both comparisons, as does a weight that underflowed: neither can be drawn.
        if !(w > 0.0 && w >= d.min_p) {
            continue;
        }
        kept += w as f64;
        let k = s + token_gumbel(d.key, i as u32);
        if k > top {
            (token, top, scaled) = (i as u32, k, s);
        }
    }
    (token, (scaled - best) - (kept as f32).ln())
}

/// [`draw_row`], and the token's log-probability under the whole row at the draw's temperature: the figure a policy
/// trace needs of a draw that cut the row, from the same logits the draw read.
pub fn draw_row_whole(row: &[f32], d: RowDraw) -> (u32, f32, f32) {
    let (token, lp) = draw_row(row, d);
    let whole = crate::logprobs::row_logprob_at(row, token as usize, d.temp.max(1e-6));
    (token, lp, if whole.is_nan() { lp } else { whole })
}

/// [`draw_row`] for a draw whose survivors are a best-first list: the row's best `top_k`, or the shortest sorted
/// prefix that holds the nucleus (the row sampler's own prefix search, [`super::fullrow`]).
fn draw_truncated_row(row: &[f32], d: RowDraw) -> (u32, f32) {
    let mut entries: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    let Some(best) = super::fullrow::best_logit(&entries) else {
        return (super::argmax(row), 0.0);
    };
    if d.keeps_top_k(row.len()) {
        let list = super::fullrow::best_first(&mut entries, d.top_k);
        return draw_list(&list, d, None);
    }
    let temp = d.temp.max(1e-6);
    let weight = |l: f32| super::fullrow::weight_of(l, temp, best);
    let total: f64 = entries.iter().map(|e| weight(e.1) as f64).sum();
    let list = super::fullrow::survivor_prefix(&mut entries, d.params(), total, weight);
    draw_list(&list, d, Some(total as f32))
}

/// The keyed draw over a best-first `list` (`(token, logit)`, equal logits lowest id first) that holds every
/// survivor of `d`: the row sampler's truncation of the list ([`super::truncated_weights_of`]; `full_mass` is the
/// softmax normaliser of the whole row when the list is only its prefix and the nucleus is measured against the row),
/// then the Gumbel-max rule over what survives, with the drawn token's log-probability.
pub fn draw_list(list: &[(u32, f32)], d: RowDraw, full_mass: Option<f32>) -> (u32, f32) {
    assert!(!list.is_empty(), "draw_list: empty candidate list");
    let (weights, kept) = super::truncated_weights_of(list, d.params(), full_mass);
    let (mut pick, mut top) = (0usize, f32::NEG_INFINITY);
    for (i, &w) in weights.iter().enumerate() {
        if w.is_nan() || w <= 0.0 {
            continue;
        }
        let k = w.ln() + token_gumbel(d.key, list[i].0);
        if k > top {
            (pick, top) = (i, k);
        }
    }
    (list[pick].0, weights[pick].ln() - kept.ln())
}

/// The keyed draw of a row answered from its best-first `list` of candidates and its statistics (`best`, the row's
/// largest `logit / temp`, and `mass`, the sum of `exp(logit / temp - best)` over the row), when the list holds every
/// survivor: the `top_k` best are in it, or its mass reaches the nucleus. `None` when it does not, and the row itself
/// must be drawn ([`draw_row`]). Only for a draw that [`RowDraw::truncates`] a `vocab`-wide row. The third figure is the
/// token's log-probability under the whole row, `logit / temp - best - ln(mass)`: the drawn token is in the list, and
/// `best` and `mass` are the row's own, so a cut draw's whole-distribution figure needs nothing more from the device.
pub fn draw_from_candidates(list: &[(u32, f32)], best: f32, mass: f32, d: RowDraw, vocab: usize) -> Option<(u32, f32, f32)> {
    if list.is_empty() || mass.is_nan() || mass <= 0.0 {
        return None;
    }
    let temp = d.temp.max(1e-6);
    let drawn = if d.keeps_top_k(vocab) {
        (list.len() >= d.top_k).then(|| draw_list(&list[..d.top_k], d, None))?
    } else {
        let held: f64 = list.iter().map(|&(_, l)| if l.is_nan() { 0.0 } else { ((l / temp - best).exp()) as f64 }).sum();
        (held >= d.top_p as f64 * mass as f64).then(|| draw_list(list, d, Some(mass)))?
    };
    let logit = list.iter().find(|&&(t, _)| t == drawn.0).map(|&(_, l)| l)?;
    Some((drawn.0, drawn.1, logit / temp - best - mass.ln()))
}

/// Every row's answer to a keyed step whose rows were read back: `draws[i]` drawn from `rows[i]` where it is
/// `Some`, the row handed back where it is `None`. The rows are drawn in parallel on the CPU pool.
pub fn draw_rows_host(rows: Vec<Vec<f32>>, draws: &[Option<RowDraw>]) -> Vec<Drawn> {
    assert_eq!(rows.len(), draws.len(), "draw_rows_host: {} rows for {} draws", rows.len(), draws.len());
    let drawn = backend_cpu::par::map(rows.len(), |i| draws[i].map(|d| draw_row_whole(&rows[i], d)));
    rows.into_iter().zip(drawn).map(|(row, t)| t.map_or(Drawn::Row(row), |(t, lp, whole)| Drawn::Token(t, lp, whole))).collect()
}
