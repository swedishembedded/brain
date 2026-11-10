// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention key and value gradients: dk[t, g, e] += scale * sum P (dP - delta) q, dv[t, g, e] += sum P dout
// @how   64-thread workgroup per (key, key head): the queries and heads of its group strided over the threads into partial rows, 1 barrier merge; accumulates
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
// dP = dout . v. A workgroup owns one key row of one key head, so there are no atomics. Dispatch: tk*hk workgroups.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  q: array<f32>;
@group(0) @binding(2) var<storage, read>  k: array<f32>;
@group(0) @binding(3) var<storage, read>  v: array<f32>;
@group(0) @binding(4) var<storage, read>  go: array<f32>;
@group(0) @binding(5) var<storage, read>  lse: array<f32>;
@group(0) @binding(6) var<storage, read>  delta: array<f32>;
@group(0) @binding(7) var<storage, read_write> dk: array<f32>;
@group(0) @binding(8) var<storage, read_write> dv: array<f32>;

var<workgroup> p_dk: array<f32, 4096>;
var<workgroup> p_dv: array<f32, 4096>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let unit = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (unit >= p.tk * p.hk) { return; }
    let key = unit / p.hk;
    let g = unit % p.hk;
    let group = p.hq / p.hk;
    let kw = p.hk * p.qk;
    let vw = p.hk * p.vh;
    var first = 0u;
    if (p.causal != 0u) { first = key; }
    let pairs = (p.tq - first) * group;
    var ak: array<f32, 64>;
    var av: array<f32, 64>;
    for (var e = 0u; e < p.qk; e = e + 1u) { ak[e] = 0.0; }
    for (var e = 0u; e < p.vh; e = e + 1u) { av[e] = 0.0; }
    for (var pi = t; pi < pairs; pi = pi + 64u) {
        let rw = (first + pi / group) * p.hq + g * group + pi % group;
        var sc = 0.0;
        for (var e = 0u; e < p.qk; e = e + 1u) {
            sc = sc + q[rw * p.qk + e] * k[key * kw + g * p.qk + e];
        }
        var dp = 0.0;
        for (var e = 0u; e < p.vh; e = e + 1u) {
            dp = dp + go[rw * p.vh + e] * v[key * vw + g * p.vh + e];
        }
        let pv = exp(sc * p.scale - lse[rw]);
        let d = pv * (dp - delta[rw]) * p.scale;
        for (var e = 0u; e < p.qk; e = e + 1u) {
            ak[e] = ak[e] + d * q[rw * p.qk + e];
        }
        for (var e = 0u; e < p.vh; e = e + 1u) {
            av[e] = av[e] + pv * go[rw * p.vh + e];
        }
    }
    for (var e = 0u; e < p.qk; e = e + 1u) { p_dk[t * 64u + e] = ak[e]; }
    for (var e = 0u; e < p.vh; e = e + 1u) { p_dv[t * 64u + e] = av[e]; }
    workgroupBarrier();
    for (var e = t; e < p.qk; e = e + 64u) {
        var total = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            total = total + p_dk[i * 64u + e];
        }
        dk[key * kw + g * p.qk + e] = dk[key * kw + g * p.qk + e] + total;
    }
    for (var e = t; e < p.vh; e = e + 64u) {
        var total = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            total = total + p_dv[i * 64u + e];
        }
        dv[key * vw + g * p.vh + e] = dv[key * vw + g * p.vh + e] + total;
    }
}
