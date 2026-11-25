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
    /// How many of the row's most probable tokens to report with the draw, as whole-row log-probabilities at the
    /// draw's temperature ([`Drawn::Traced`]); `0` for none. Only a draw over the whole row (no `top_k`, nucleus or
    /// min-p cut) carries one: a cut draw's list would be of a distribution it did not draw from.
    pub top: usize,
    /// The most tokens of a cut draw's survivors (`top_k`, nucleus, min-p) the draw reports as the support it drew
    /// from ([`Drawn::Supported`]); `0` for none. A draw that keeps more survivors reports that it overflowed, never a
    /// shortened set; a draw that cuts nothing has the whole row as its support and reports none.
    pub support: usize,
}

impl RowDraw {
    /// The keyed draw of a request sampled with `params` at `position` (the number of tokens before the one drawn)
    /// under its `seed`, or `None` when `params` is not drawable this way ([`SampleParams::row_drawable`]).
    pub fn of(params: &SampleParams, seed: u64, position: usize) -> Option<RowDraw> {
        params.row_drawable().then(|| RowDraw {
            temp: params.temp.max(1e-6),
            min_p: params.min_p.clamp(0.0, 1.0),
            top_p: params.top_p,
            top_k: params.top_k,
            key: draw_key(seed, position as u64, 0),
            top: if params.draws_from_whole_distribution() || params.trace_support > 0 { params.trace_top } else { 0 },
            support: params.trace_support,
        })
    }

    /// Whether the draw reports the row's top list: it asks for one ([`Self::top`]), draws over the whole row, and the
    /// list fits what a device keeps ([`crate::readout::MAX_DRAW_CANDIDATES`]) and the row.
    pub fn carries_top_list(&self, vocab: usize) -> bool {
        self.top > 0 && self.top <= vocab && self.top <= crate::readout::MAX_DRAW_CANDIDATES as usize && !self.truncates(vocab) && self.min_p <= 0.0
    }

    /// Whether the draw reports its support: it asks for one ([`Self::support`]) and cuts the row (a `top_k` inside the
    /// vocabulary, a nucleus or a min-p floor).
    pub fn reports_support(&self, vocab: usize) -> bool {
        self.support > 0 && (self.truncates(vocab) || self.min_p > 0.0)
    }

    /// Whether the draw is made from its best-first list of [`Self::support_list`] candidates against the whole row's
    /// mass: a draw that reports its support and has no `top_k` (a nucleus or a min-p floor), so that the list's cut
    /// is the support it drew from and a survivor past the cap shows in the list.
    pub fn list_drawn(&self, vocab: usize) -> bool {
        self.reports_support(vocab) && !self.keeps_top_k(vocab)
    }

    /// The best-first candidates a [`Self::list_drawn`] draw is made from: one more than its cap, so a draw whose
    /// survivors overflow the cap keeps the extra one (or the whole row, when that is shorter).
    pub fn support_list(&self, vocab: usize) -> usize {
        (self.support + 1).min(vocab)
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
    /// [`Self::Token`] for a row that asked for its top list ([`RowDraw::top`]): the token, its two log-probabilities,
    /// and the row's `top` most probable `(token, whole-row log-probability)` pairs, most probable first, from the
    /// arithmetic of the draw. The third figure is the drawn token's entry in that list whenever it is among them (the
    /// same number, bit for bit).
    Traced(u32, f32, f32, Vec<(u32, f32)>),
    /// [`Self::Token`] for a cut draw that reports its support ([`RowDraw::support`]).
    Supported(Supported),
    /// The row's whole `[vocab]` logits, for the host sampler.
    Row(Vec<f32>),
}

/// The answer of a cut draw that reports its support ([`Drawn::Supported`]): the arithmetic that drew the token, kept
/// for a trainer that replays the sampler's support.
#[derive(Clone, Debug, PartialEq)]
pub struct Supported {
    /// The drawn token.
    pub token: u32,
    /// Its log-probability under the distribution it was drawn from: the softmax of `logit / temp` over the support.
    pub sampler: f32,
    /// Its log-probability under the whole row at the draw's temperature.
    pub whole: f32,
    /// The survivors of the cut the token was drawn from, best first (equal logits lowest id first), the token among
    /// them; `None` when they are more than the draw's cap ([`RowDraw::support`]).
    pub support: Option<Vec<u32>>,
    /// The row's [`RowDraw::top`] most probable `(token, whole-row log-probability)` pairs, from the same candidate list,
    /// when the draw asked for them and the list holds them.
    pub top: Option<Vec<(u32, f32)>>,
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

/// [`draw_row_whole`] for a row that asks for its top list ([`RowDraw::top`]): the answer as [`Drawn::Traced`]. The row's
/// figures come from [`crate::logprobs::scored_position`], the arithmetic a scoring pass uses, so a decode-time list and a
/// scored one are the same numbers for the same row.
pub fn draw_row_traced(row: &[f32], d: RowDraw) -> Drawn {
    let (token, lp) = draw_row(row, d);
    let scored = crate::logprobs::scored_position(row, token, d.top, d.temp.max(1e-6));
    Drawn::Traced(token, lp, if scored.log_prob.is_nan() { lp } else { scored.log_prob }, scored.top)
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
/// survivor of `d`: the row sampler's truncation of the list (`super::truncated_weights_of`; `full_mass` is the
/// softmax normaliser of the whole row when the list is only its prefix and the nucleus is measured against the row),
/// then the Gumbel-max rule over what survives, with the drawn token's log-probability.
pub fn draw_list(list: &[(u32, f32)], d: RowDraw, full_mass: Option<f32>) -> (u32, f32) {
    let (token, lp, _) = draw_list_cut(list, d, full_mass);
    (token, lp)
}

/// [`draw_list`], and how many of the list's first entries survived the cut: the support the token was drawn from is
/// `list[..cut]` (the device draw writes the same count, `draw_list.wgsl`).
pub fn draw_list_cut(list: &[(u32, f32)], d: RowDraw, full_mass: Option<f32>) -> (u32, f32, usize) {
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
    (list[pick].0, weights[pick].ln() - kept.ln(), weights.len())
}

/// The support a cut draw reports from its best-first `list` and the cut it drew from: the first `cut` entries of
/// finite logit, or `None` when the survivors are more than `cap`.
pub fn support_of_cut(list: &[(u32, f32)], cut: usize, cap: usize) -> Option<Vec<u32>> {
    (cut <= cap).then(|| list[..cut.min(list.len())].iter().filter(|e| e.1.is_finite()).map(|e| e.0).collect())
}

/// [`draw_row_whole`] for a cut draw that reports its support ([`RowDraw::reports_support`]): the host reference of the
/// device's answer. A `top_k` row is drawn from its best `top_k`, a nucleus or min-p row from its best
/// [`RowDraw::support_list`] against the whole row's mass; the support is the list's cut. A row of no finite logit
/// answers its first best index, as [`draw_row`] does, with that index as its support.
pub fn draw_row_supported(row: &[f32], d: RowDraw) -> Drawn {
    assert!(!row.is_empty(), "draw_row_supported: empty logits row");
    let temp = d.temp.max(1e-6);
    let mut entries: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    let Some(best) = super::fullrow::best_logit(&entries) else {
        let token = super::argmax(row);
        return Drawn::Supported(Supported { token, sampler: 0.0, whole: 0.0, support: Some(vec![token]), top: None });
    };
    let drawn_from = if d.keeps_top_k(row.len()) { d.top_k } else { d.support_list(row.len()) };
    let list = super::fullrow::best_first(&mut entries, drawn_from.max(d.top));
    let full_mass = (!d.keeps_top_k(row.len())).then(|| row.iter().map(|&l| super::fullrow::weight_of(l, temp, best) as f64).sum::<f64>() as f32);
    let (token, sampler, cut) = draw_list_cut(&list[..drawn_from.min(list.len())], d, full_mass);
    let scored = crate::logprobs::scored_position(row, token, d.top, temp);
    let whole = if scored.log_prob.is_nan() { sampler } else { scored.log_prob };
    let top = (d.top > 0).then_some(scored.top);
    Drawn::Supported(Supported { token, sampler, whole, support: support_of_cut(&list, cut, d.support), top })
}

/// Every row's answer to a keyed step whose rows were read back: `draws[i]` drawn from `rows[i]` where it is
/// `Some`, the row handed back where it is `None`. The rows are drawn in parallel on the CPU pool.
pub fn draw_rows_host(rows: Vec<Vec<f32>>, draws: &[Option<RowDraw>]) -> Vec<Drawn> {
    assert_eq!(rows.len(), draws.len(), "draw_rows_host: {} rows for {} draws", rows.len(), draws.len());
    let drawn = backend_cpu::par::map(rows.len(), |i| {
        draws[i].map(|d| match d.top {
            _ if d.reports_support(rows[i].len()) => draw_row_supported(&rows[i], d),
            0 => {
                let (t, lp, whole) = draw_row_whole(&rows[i], d);
                Drawn::Token(t, lp, whole)
            }
            _ => draw_row_traced(&rows[i], d),
        })
    });
    rows.into_iter().zip(drawn).map(|(row, t)| t.unwrap_or(Drawn::Row(row))).collect()
}
