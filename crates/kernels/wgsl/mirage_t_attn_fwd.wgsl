// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention forward: out[qi, h, e] = sum_t softmax(scale * q . k)[t] v[t, g, e], and lse[qi, h]
// @how   64-thread workgroup per (query, head) row: keys strided over the threads with a running (max, sum, output), 1 barrier merge
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Masked grouped-query attention: q [tq, hq*qk], k [tk, hk*qk], v [tk, hk*vh], head h reads key/value head h / (hq / hk),
// query qi sees keys 0..tk (or 0..=qi when causal). No probability matrix is stored: the forward keeps one log-sum-exp
// per (query, head) row, and every backward kernel rebuilds the probabilities of the keys it works on as
// exp(scale * q . k - lse). The head widths qk and vh are at most 64 (a thread holds one partial result row in
// registers; the launch checks it). Params: tq, tk, hq, hk, qk, vh, causal, scale (f32 bits).
// lse is the log of the softmax denominator (max + ln sum). Dispatch: tq*hq workgroups.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  q: array<f32>;
@group(0) @binding(2) var<storage, read>  k: array<f32>;
@group(0) @binding(3) var<storage, read>  v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> lse: array<f32>;

var<workgroup> p_max: array<f32, 64>;
var<workgroup> p_sum: array<f32, 64>;
var<workgroup> p_acc: array<f32, 4096>;

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
    let vw = p.hk * p.vh;
    let vis = select(p.tk, qi + 1u, p.causal != 0u);
    let qb = row * p.qk;
    var acc: array<f32, 64>;
    for (var e = 0u; e < p.vh; e = e + 1u) { acc[e] = 0.0; }
    var m = -3.0e38;
    var s = 0.0;
    for (var kt = t; kt < vis; kt = kt + 64u) {
        var sc = 0.0;
        for (var e = 0u; e < p.qk; e = e + 1u) {
            sc = sc + q[qb + e] * k[kt * kw + g * p.qk + e];
        }
        sc = sc * p.scale;
        var w = 1.0;
        var r = 1.0;
        if (sc > m) {
            r = exp(m - sc);
            m = sc;
        } else {
            w = exp(sc - m);
        }
        s = s * r + w;
        for (var e = 0u; e < p.vh; e = e + 1u) {
            acc[e] = acc[e] * r + w * v[kt * vw + g * p.vh + e];
        }
    }
    p_max[t] = m;
    p_sum[t] = s;
    for (var e = 0u; e < p.vh; e = e + 1u) { p_acc[t * 64u + e] = acc[e]; }
    workgroupBarrier();
    var big = -3.0e38;
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (p_max[i] > big) { big = p_max[i]; }
    }
    var total = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        total = total + p_sum[i] * exp(p_max[i] - big);
    }
    if (t == 0u) { lse[row] = big + log(total); }
    for (var e = t; e < p.vh; e = e + 64u) {
        var o = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            o = o + p_acc[i * 64u + e] * exp(p_max[i] - big);
        }
        out[row * p.vh + e] = o / total;
    }
}
