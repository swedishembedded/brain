// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention query gradient: dq[qi, h, e] += scale * sum_t P (dP - delta) k[t, g, e], P rebuilt from lse
// @how   one thread per (query, head) row, looping over the visible keys; accumulates
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
// dP = dout . v. Dispatch: tq*hq threads.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  q: array<f32>;
@group(0) @binding(2) var<storage, read>  k: array<f32>;
@group(0) @binding(3) var<storage, read>  v: array<f32>;
@group(0) @binding(4) var<storage, read>  go: array<f32>;
@group(0) @binding(5) var<storage, read>  lse: array<f32>;
@group(0) @binding(6) var<storage, read>  delta: array<f32>;
@group(0) @binding(7) var<storage, read_write> dq: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.tq * p.hq) { return; }
    let h = row % p.hq;
    let qi = row / p.hq;
    let g = h / (p.hq / p.hk);
    let kw = p.hk * p.qk;
    let vw = p.hk * p.vh;
    let vis = select(p.tk, qi + 1u, p.causal != 0u);
    for (var kt = 0u; kt < vis; kt = kt + 1u) {
        var sc = 0.0;
        for (var e = 0u; e < p.qk; e = e + 1u) {
            sc = sc + q[row * p.qk + e] * k[kt * kw + g * p.qk + e];
        }
        var dp = 0.0;
        for (var e = 0u; e < p.vh; e = e + 1u) {
            dp = dp + go[row * p.vh + e] * v[kt * vw + g * p.vh + e];
        }
        let d = exp(sc * p.scale - lse[row]) * (dp - delta[row]) * p.scale;
        for (var e = 0u; e < p.qk; e = e + 1u) {
            dq[row * p.qk + e] = dq[row * p.qk + e] + d * k[kt * kw + g * p.qk + e];
        }
    }
}
