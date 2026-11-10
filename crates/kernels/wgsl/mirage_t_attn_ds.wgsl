// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention score gradient: dS = P (dP - sum P dP) scale with dP = dout . v
// @how   one thread per (query, head) row; two passes over the visible keys
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
// Dispatch: tq*hq threads.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  pr: array<f32>;
@group(0) @binding(2) var<storage, read>  go: array<f32>;
@group(0) @binding(3) var<storage, read>  v: array<f32>;
@group(0) @binding(4) var<storage, read_write> ds: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.tq * p.hq) { return; }
    let h = idx % p.hq;
    let qi = idx / p.hq;
    let g = h / (p.hq / p.hk);
    let vw = p.hk * p.vh;
    var vis = p.tk;
    if (p.causal != 0u) { vis = qi + 1u; }
    let pb = idx * p.tk;
    var dot = 0.0;
    for (var t = 0u; t < vis; t = t + 1u) {
        var dp = 0.0;
        for (var e = 0u; e < p.vh; e = e + 1u) {
            dp = dp + go[idx * p.vh + e] * v[t * vw + g * p.vh + e];
        }
        ds[pb + t] = dp;
        dot = dot + pr[pb + t] * dp;
    }
    for (var t = 0u; t < vis; t = t + 1u) {
        ds[pb + t] = pr[pb + t] * (ds[pb + t] - dot) * p.scale;
    }
}
