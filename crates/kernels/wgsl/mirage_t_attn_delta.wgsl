// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention row statistic: delta[qi, h] = sum_e dout[qi, h, e] * out[qi, h, e]
// @how   one thread per (query, head) row, looping over the value width
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Masked grouped-query attention: q [tq, hq*qk], k [tk, hk*qk], v [tk, hk*vh], head h reads key/value head h / (hq / hk),
// query qi sees keys 0..tk (or 0..=qi when causal). No probability matrix is stored: the forward keeps one log-sum-exp
// per (query, head) row, and every backward kernel rebuilds the probabilities of the keys it works on as
// exp(scale * q . k - lse). Params: tq, tk, hq, hk, qk, vh, causal, scale (f32 bits).
// delta equals sum_t P dP, the softmax-backward term, so the backward never needs the probabilities of a whole row at
// once. Dispatch: tq*hq threads.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  go: array<f32>;
@group(0) @binding(2) var<storage, read>  out: array<f32>;
@group(0) @binding(3) var<storage, read_write> delta: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.tq * p.hq) { return; }
    var acc = 0.0;
    for (var e = 0u; e < p.vh; e = e + 1u) {
        acc = acc + go[row * p.vh + e] * out[row * p.vh + e];
    }
    delta[row] = acc;
}
