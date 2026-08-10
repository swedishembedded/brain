// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of gdn_decay_gate.wgsl w.r.t. its a_proj input
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of `gdn_decay_gate.wgsl`'s
//   g[row,h] = -exp(A_log[h]) * softplus(a_proj[row,h] + dt_bias[h])
// w.r.t. its `a_proj` input (the only INPUT this kernel produces a gradient
// for -- `A_log`/`dt_bias`'s own gradients are a plain reduction over this
// kernel's output, computed OUTSIDE it, see below).
//
// Let `x = a_proj[row,h] + dt_bias[h]`. Since `d(softplus(x))/dx =
// sigmoid(x)`, `dg/dx = -exp(A_log[h]) * sigmoid(x)`. The `+dt_bias` term is a
// plain additive shift, so `d_a_proj[row,h] = d_x[row,h] = d_g[row,h] *
// dg/dx` directly -- this kernel's only output.
//
// `a_proj`/`dt_bias` are recomputed into `x`/`sigmoid(x)` here since neither
// was saved by the forward kernel (`gdn_decay_gate.wgsl` never materializes
// them, only the final `g`).
//
// The two other gradients `gdn_decay_gate` feeds are NOT computed here, per
// this repo's "one kernel, one job" convention (reuse an existing reduction
// rather than duplicate it) -- both derive from THIS kernel's own
// `d_a_proj` (== `d_x`), reduced over rows by the caller:
//   d_dt_bias[h] = sum_row d_a_proj[row,h]                  (bias_grad.wgsl on d_a_proj)
//   d_A_log[h]   = sum_row d_g[row,h] * g[row,h]             (mul.wgsl(d_g,g), then bias_grad.wgsl)
// (the second holds because `g = -exp(A_log)*softplus(x)`, so
// `dg/d(A_log) = -exp(A_log)*softplus(x) = g` exactly).

struct Params {
    rows: u32,
    num_v_heads: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a_proj:   array<f32>;
@group(0) @binding(2) var<storage, read>       a_log:    array<f32>;
@group(0) @binding(3) var<storage, read>       dt_bias:  array<f32>;
@group(0) @binding(4) var<storage, read>       d_g:      array<f32>;
@group(0) @binding(5) var<storage, read_write> d_a_proj: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.num_v_heads;
    if (idx >= total) { return; }
    let h = idx % p.num_v_heads;
    let x = a_proj[idx] + dt_bias[h];
    let s = 1.0 / (1.0 + exp(-x));
    d_a_proj[idx] = d_g[idx] * (-exp(a_log[h]) * s);
}
