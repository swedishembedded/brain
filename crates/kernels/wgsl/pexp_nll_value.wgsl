// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Piecewise-exponential (counting-process) survival negative log-likelihood, per (row, code) term
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The time axis is cut into pieces; inside piece p the hazard of event code k
// for subject i is the constant lambda = exp(loglam). A subject contributes,
// for every (piece, code), the log-likelihood of a Poisson count observed over
// its exposure:
//   ll = event * loglam - lambda * expo
// where `expo` is the time the subject spent at risk of code k inside piece p
// (zero outside its at-risk window: left truncation and right censoring are
// both exact this way) and `event` is 1 for the piece in which the event
// happened. This kernel writes the WEIGHTED NEGATIVE term
//   out[e] = w[i] * inv_wsum * (exp(loglam[e]) * expo[e] - event[e] * loglam[e])
// so the host loss is a plain sum of `out` (the mse_value convention).
//
// Layout: loglam/event/expo/out are [rows, k] row-major with rows = n_subj *
// pieces, the pieces of one subject contiguous; w is [n_subj], the per-subject
// sampling weight; inv_wsum = 1 / sum(w) over the batch, passed as f32 bits.
// No clamping of loglam: a clamp inside a loss silently zeroes the gradient
// of exactly the rows that need it. Gradient: pexp_nll_grad.

struct Params {
    rows: u32,
    k: u32,
    pieces: u32,
    inv_wsum: u32, // f32 bits
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       loglam: array<f32>;
@group(0) @binding(2) var<storage, read>       event:  array<f32>;
@group(0) @binding(3) var<storage, read>       expo:   array<f32>;
@group(0) @binding(4) var<storage, read>       w:      array<f32>;
@group(0) @binding(5) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let e = gid.y * (nwg.x * 64u) + gid.x;
    if (e >= p.rows * p.k) { return; }
    let subj = (e / p.k) / p.pieces;
    let l = loglam[e];
    let scale = w[subj] * bitcast<f32>(p.inv_wsum);
    out[e] = scale * (exp(l) * expo[e] - event[e] * l);
}
