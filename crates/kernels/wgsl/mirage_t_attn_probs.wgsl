// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention probabilities: P[qi, h, t] = softmax over the visible keys of scale * q . k
// @how   64-thread workgroup per (query, head) row, keys strided over the threads, online (max, sum), 1 barrier
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
@group(0) @binding(1) var<storage, read>  q: array<f32>;
@group(0) @binding(2) var<storage, read>  k: array<f32>;
@group(0) @binding(3) var<storage, read_write> pr: array<f32>;

var<workgroup> p_max: array<f32, 64>;
var<workgroup> p_sum: array<f32, 64>;

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
    let kw = p.hk * p.qk;
    let vis = select(p.tk, qi + 1u, p.causal != 0u);
    let qb = row * p.qk;
    let pb = row * p.tk;
    var m = -3.0e38;
    var s = 0.0;
    for (var kt = t; kt < vis; kt = kt + 64u) {
        var sc = 0.0;
        for (var e = 0u; e < p.qk; e = e + 1u) {
            sc = sc + q[qb + e] * k[kt * kw + g * p.qk + e];
        }
        sc = sc * p.scale;
        pr[pb + kt] = sc;
        if (sc > m) {
            s = s * exp(m - sc) + 1.0;
            m = sc;
        } else {
            s = s + exp(sc - m);
        }
    }
    p_max[t] = m;
    p_sum[t] = s;
    workgroupBarrier();
    var big = -3.0e38;
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (p_max[i] > big) { big = p_max[i]; }
    }
    var total = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        total = total + p_sum[i] * exp(p_max[i] - big);
    }
    for (var kt = t; kt < vis; kt = kt + 64u) {
        pr[pb + kt] = exp(pr[pb + kt] - big) / total;
    }
}
