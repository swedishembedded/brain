// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row L2 normalization with a learnable per-dim scale - the QK-norm used by GenieRedux attention (applied to each head slice of q and k, over head_dim, before the scores kernel; the scores kernel then uses a constant scale of 8)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-row L2 normalization with a learnable per-dim scale — the QK-norm used by
// GenieRedux attention (applied to each head slice of q and k, over head_dim,
// before the scores kernel; the scores kernel then uses a constant scale of 8):
//   y[n,d] = x[n,d] * rsqrt(sum_k x[n,k]^2 + eps) * g[d]
// View q (or k) as [N, D] with N = tokens*heads and D = head_dim; g is [D],
// shared across rows. One invocation per (n,d).

struct Params {
    n: u32,
    d: u32,
    eps: u32,   // f32 bits
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       g: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n * p.d;
    if (gidx >= total) { return; }

    let dd = gidx % p.d;
    let n = gidx / p.d;
    let base = n * p.d;

    var s = 0.0;
    for (var k: u32 = 0u; k < p.d; k = k + 1u) {
        let v = x[base + k];
        s = s + v * v;
    }
    let r = inverseSqrt(s + bitcast<f32>(p.eps));
    y[gidx] = x[gidx] * r * g[dd];
}
