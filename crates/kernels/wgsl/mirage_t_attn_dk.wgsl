// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training attention key gradient: dk[t, g, e] += sum over the queries seeing key t of dS q
// @how   one thread per key element, gathering over the queries and heads of its group; accumulates
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
// Dispatch: tk*hk*qk threads.

struct Params { tq: u32, tk: u32, hq: u32, hk: u32, qk: u32, vh: u32, causal: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  ds: array<f32>;
@group(0) @binding(2) var<storage, read>  q: array<f32>;
@group(0) @binding(3) var<storage, read_write> dk: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.tk * p.hk * p.qk) { return; }
    let e = idx % p.qk;
    let g = (idx / p.qk) % p.hk;
    let t = idx / (p.qk * p.hk);
    let group = p.hq / p.hk;
    var first = 0u;
    if (p.causal != 0u) { first = t; }
    var acc = 0.0;
    for (var qi = first; qi < p.tq; qi = qi + 1u) {
        for (var hh = 0u; hh < group; hh = hh + 1u) {
            let row = qi * p.hq + g * group + hh;
            acc = acc + ds[row * p.tk + t] * q[row * p.qk + e];
        }
    }
    dk[idx] = dk[idx] + acc;
}
