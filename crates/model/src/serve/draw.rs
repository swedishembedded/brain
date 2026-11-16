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
//! For a request with no `top_p` nucleus, no `top_k` below the vocabulary and no
//! logit processor ([`SampleParams::row_drawable`]), the distribution
//! [`super::sample_logits_row`] defines is the softmax of `logit / temp` over the
//! survivors (every token of finite weight, or with `min_p` the tokens whose
//! weight relative to the best is at least `min_p`). The Gumbel-max rule draws
//! from exactly that distribution: with `g_i` independent standard Gumbel noise,
//! `argmax_i (logit_i / temp + g_i)` over the survivors is token `i` with
//! probability `w_i / sum w`. One pass over the row, no sort, no prefix sum, and
//! each row is independent of every other, so a batch of rows reduces in
//! parallel and only `(token, logprob)` per row leaves the device.
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
    /// The row's noise key ([`draw_key`]).
    pub key: u32,
}

impl RowDraw {
    /// The keyed draw of a request sampled with `params` at `position` (the number of tokens before the one drawn)
    /// under its `seed`, or `None` when `params` is not drawable this way over a `vocab`-wide row
    /// ([`SampleParams::row_drawable`]).
    pub fn of(params: &SampleParams, vocab: usize, seed: u64, position: usize) -> Option<RowDraw> {
        params.row_drawable(vocab).then(|| RowDraw { temp: params.temp.max(1e-6), min_p: params.min_p.clamp(0.0, 1.0), key: draw_key(seed, position as u64, 0) })
    }
}

/// What a decoder answers one row of a keyed step with: the token it drew there, or the row itself for a request
/// the keyed draw does not cover (a processor, a nucleus, a `top_k` inside the vocabulary).
#[derive(Clone, Debug, PartialEq)]
pub enum Drawn {
    /// The drawn token and its log-probability.
    Token(u32, f32),
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

/// Every row's answer to a keyed step whose rows were read back: `draws[i]` drawn from `rows[i]` where it is
/// `Some`, the row handed back where it is `None`. The rows are drawn in parallel on the CPU pool.
pub fn draw_rows_host(rows: Vec<Vec<f32>>, draws: &[Option<RowDraw>]) -> Vec<Drawn> {
    assert_eq!(rows.len(), draws.len(), "draw_rows_host: {} rows for {} draws", rows.len(), draws.len());
    let drawn = backend_cpu::par::map(rows.len(), |i| draws[i].map(|d| draw_row(&rows[i], d)));
    rows.into_iter().zip(drawn).map(|(row, t)| t.map_or(Drawn::Row(row), |(t, lp)| Drawn::Token(t, lp))).collect()
}
