// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of gdn_gap_gate: gradients of the write-strength pre-activation and of each element's share of the raw rate
// @how   one thread per (row, head) element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Given d_g and d_beta (the gradients of the raw log-decay and of the write
// strength of `gdn_gap_gate`):
//   d_b_pre[row,h]  = d_beta * s * (1 - s),  s = sigmoid(b_pre)
//   d_rate[row,h]   = d_g * (-dt[row]) * sigmoid(rate[h])   (d softplus / d rate)
// both zero on a padding row (dt < 0). `d_rate` is this element's share of
// the per-head rate gradient: the caller sums it over rows (a column sum,
// `bias_grad`), since every row of a head reads the same rate.

struct Params {
    rows: u32,
    heads: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       b_pre:  array<f32>;
@group(0) @binding(2) var<storage, read>       dt:     array<f32>;
@group(0) @binding(3) var<storage, read>       rate:   array<f32>;
@group(0) @binding(4) var<storage, read>       d_g:    array<f32>;
@group(0) @binding(5) var<storage, read>       d_beta: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_b_pre: array<f32>;
@group(0) @binding(7) var<storage, read_write> d_rate: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.heads) { return; }
    let row = idx / p.heads;
    let h = idx % p.heads;
    let gap = dt[row];
    if (gap < 0.0) {
        d_b_pre[idx] = 0.0;
        d_rate[idx] = 0.0;
        return;
    }
    let s = 1.0 / (1.0 + exp(-b_pre[idx]));
    d_b_pre[idx] = d_beta[idx] * s * (1.0 - s);
    d_rate[idx] = d_g[idx] * (-gap) / (1.0 + exp(-rate[h]));
}
