// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention score gradient: dS = P (dP - sum P dP) scale with dP = dout . v
// @how   64-thread workgroup per (query, head) row, keys strided over the threads, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Masked grouped-query attention: q [tq, hq*qk], k [tk, hk*qk], v [tk, hk*vh], head h reads key/value head h / (hq / hk),
// query qi sees keys 0..tk (or 0..=qi when causal). Probabilities P are [tq*hq, tk] and only the visible prefix of a row is
// written. Params: tq, tk, hq, hk, qk, vh, causal, scale (f32 bits).
// Dispatch: tq*hq workgroups.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  pr: array<f32>;
@group(0) @binding(2) var<storage, read>  go: array<f32>;
@group(0) @binding(3) var<storage, read>  v: array<f32>;
@group(0) @binding(4) var<storage, read_write> ds: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.tq * p.hq) { return; }
    let h = row % p.hq;
    let qi = row / p.hq;
    let g = h / (p.hq / p.hk);
    let vw = p.hk * p.vh;
    let vis = select(p.tk, qi + 1u, p.causal != 0u);
    let pb = row * p.tk;
    var dot = 0.0;
    for (var kt = t; kt < vis; kt = kt + 64u) {
        var dp = 0.0;
        for (var e = 0u; e < p.vh; e = e + 1u) {
            dp = dp + go[row * p.vh + e] * v[kt * vw + g * p.vh + e];
        }
        ds[pb + kt] = dp;
        dot = dot + pr[pb + kt] * dp;
    }
    partial[t] = dot;
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        total = total + partial[i];
    }
    for (var kt = t; kt < vis; kt = kt + 64u) {
        ds[pb + kt] = pr[pb + kt] * (ds[pb + kt] - total) * p.scale;
    }
}
