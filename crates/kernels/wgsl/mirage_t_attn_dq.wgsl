// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention query gradient: dq[qi, h, e] += sum_t dS[qi, h, t] k[t, g, e]
// @how   one thread per query element, looping over the visible keys; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Masked grouped-query attention: q [tq, hq*qk], k [tk, hk*qk], v [tk, hk*vh], head h reads key/value head h / (hq / hk),
// query qi sees keys 0..tk (or 0..=qi when causal). Probabilities P are [tq*hq, tk] and only the visible prefix of a row is
// written. Params: tq, tk, hq, hk, qk, vh, causal, scale (f32 bits).
// Dispatch: tq*hq*qk threads.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  ds: array<f32>;
@group(0) @binding(2) var<storage, read>  k: array<f32>;
@group(0) @binding(3) var<storage, read_write> dq: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.tq * p.hq * p.qk) { return; }
    let e = idx % p.qk;
    let row = idx / p.qk;
    let h = row % p.hq;
    let qi = row / p.hq;
    let g = h / (p.hq / p.hk);
    let kw = p.hk * p.qk;
    var vis = p.tk;
    if (p.causal != 0u) { vis = qi + 1u; }
    var acc = 0.0;
    for (var t = 0u; t < vis; t = t + 1u) {
        acc = acc + ds[row * p.tk + t] * k[t * kw + g * p.qk + e];
    }
    dq[idx] = dq[idx] + acc;
}
