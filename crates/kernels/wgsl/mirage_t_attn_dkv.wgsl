// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention key and value gradients: dk[t, g, e] += scale * sum P (dP - delta) q, dv[t, g, e] += sum P dout
// @how   one thread per (key, key head), gathering over the queries and heads of its group; accumulates
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
// dP = dout . v. A thread owns one key row of one key head, so there are no atomics. Dispatch: tk*hk threads.

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let unit = gid.y * (nwg.x * 64u) + gid.x;
    if (unit >= p.tk * p.hk) { return; }
    let key = unit / p.hk;
    let g = unit % p.hk;
    let group = p.hq / p.hk;
    let kw = p.hk * p.qk;
    let vw = p.hk * p.vh;
    var first = 0u;
    if (p.causal != 0u) { first = key; }
    for (var qi = first; qi < p.tq; qi = qi + 1u) {
        for (var hh = 0u; hh < group; hh = hh + 1u) {
            let rw = qi * p.hq + g * group + hh;
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
                dk[key * kw + g * p.qk + e] = dk[key * kw + g * p.qk + e] + d * q[rw * p.qk + e];
            }
            for (var e = 0u; e < p.vh; e = e + 1u) {
                dv[key * vw + g * p.vh + e] = dv[key * vw + g * p.vh + e] + pv * go[rw * p.vh + e];
            }
        }
    }
}
