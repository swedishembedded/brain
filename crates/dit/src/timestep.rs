// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PixArt-style scalar timestep conditioning: `model::hostmath::
//! timestep_embedding`'s sinusoid through a `Linear -> SiLU -> Linear` MLP.
//!
//! This is the `TimestepEmbedder`/`time_embedding` pattern every diffusion
//! transformer in this repo that conditions on a scalar timestep implements -
//! Z-Image's `t_embedder` and Wan's `time_embedding` (the first two of its
//! three linears; Wan's third linear, `time_projection`, maps THIS function's
//! output into the per-block `[6·dim]` modulation and is not part of the
//! shared shape, so callers apply it themselves on the returned vector - see
//! `wan::model::timestep_cond`). Before this existed, both crates hand-wrote
//! the same `linear -> silu -> linear` wrapping around the sinusoid.

use model::hostmath;

/// `out[o] = b[o] + Σ_i x[i]·w[o·in_dim+i]`, `in_dim = x.len()`. A single-row
/// (batch-of-one) linear: every real caller here embeds ONE scalar timestep
/// per call, so there is no row axis to parallelise and this is a plain
/// sequential accumulation - the same order `wan::model::linear`'s row-
/// parallel form computes for a single row, which is what keeps this
/// bit-identical to the two ad hoc copies it replaces.
fn linear1(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out_dim * in_dim, "timestep::linear1: w is {}, need {}", w.len(), out_dim * in_dim);
    debug_assert_eq!(b.len(), out_dim, "timestep::linear1: b is {}, need {out_dim}", b.len());
    let mut out = vec![0f32; out_dim];
    for (o, slot) in out.iter_mut().enumerate() {
        let wr = &w[o * in_dim..o * in_dim + in_dim];
        let mut acc = b[o];
        for (xi, wi) in x.iter().zip(wr) {
            acc += xi * wi;
        }
        *slot = acc;
    }
    out
}

/// `t -> sinusoid(t, freq_dim) -> Linear(freq_dim,hidden_dim) -> SiLU ->
/// Linear(hidden_dim,out_dim)`.
///
/// The sinusoid is always `flip_sin_to_cos = true`, `downscale_freq_shift =
/// 0.0` (`[cos ‖ sin]`, both this crate's current callers' convention - see
/// `model::hostmath::timestep_embedding`'s own doc before assuming a new
/// caller wants the same halves order). `max_period` and `freq_dim` are
/// exposed because they differ between callers (Wan: `freq_dim = cfg.
/// freq_dim`, `max_period = 10000.0`; Z-Image: `freq_dim = 256`, `max_period =
/// 10000.0` but scaled `t` first).
#[allow(clippy::too_many_arguments)]
pub fn pixart_timestep_embed(
    t: f32,
    freq_dim: usize,
    w0: &[f32],
    b0: &[f32],
    hidden_dim: usize,
    w2: &[f32],
    b2: &[f32],
    out_dim: usize,
    max_period: f64,
) -> Vec<f32> {
    let te = hostmath::timestep_embedding(t, freq_dim, true, 0.0, max_period);
    let h0 = hostmath::silu_slice(&linear1(&te, w0, b0, hidden_dim));
    linear1(&h0, w2, b2, out_dim)
}
