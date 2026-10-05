// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Continuous-time state across irregular visits: mean reversion over elapsed time, then a gated update towards each visit
// @how   one thread per (subject, channel), a serial loop over that subject's visits
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per subject b and channel c, with r = softplus(state[c]) and m = state[D + c]
// (the population state), visits v = 0..V-1 in time order:
//   h = m
//   for each present visit (dt[b*V + v] >= 0, the time since the previous one):
//     a = exp(-r * dt);  q = m + a * (h - m)            reversion over the gap
//     i = sigmoid(xg[s, D + c]);  h = q + i * (xg[s, c] - q)   gated delta update
//   hs[s, c] = h                                       (s = b*V + v; absent visits carry h)
// then to the prediction time, dt[S + b] after the last visit:
//   z[b, c] = m + exp(-r * dt[S + b]) * (h - m)
// A subject with no visits is the population state.

struct Params {
    subjects: u32,
    visits: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xg:    array<f32>;
@group(0) @binding(2) var<storage, read>       dt:    array<f32>;
@group(0) @binding(3) var<storage, read>       state: array<f32>;
@group(0) @binding(4) var<storage, read_write> hs:    array<f32>;
@group(0) @binding(5) var<storage, read_write> z:     array<f32>;

fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let d = p.d;
    if (idx >= p.subjects * d) { return; }
    let b = idx / d;
    let c = idx % d;
    let r = softplus(state[c]);
    let m = state[d + c];
    var h = m;
    for (var v = 0u; v < p.visits; v = v + 1u) {
        let s = b * p.visits + v;
        let gap = dt[s];
        if (gap >= 0.0) {
            let q = m + exp(-r * gap) * (h - m);
            let i = 1.0 / (1.0 + exp(-xg[s * 2u * d + d + c]));
            h = q + i * (xg[s * 2u * d + c] - q);
        }
        hs[s * d + c] = h;
    }
    let tail = dt[p.subjects * p.visits + b];
    z[idx] = m + exp(-r * tail) * (h - m);
}
