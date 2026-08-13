// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward w.r.t. the per-dim scale g for l2norm_scale
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward w.r.t. the per-dim scale g for l2norm_scale. g is shared across rows,
// so its gradient is the column sum of dy * normalized-x:
//   d_g[d] = sum_n dy[n,d] * x[n,d] * rsqrt(sum_k x[n,k]^2 + eps)
// One invocation per d; loops over the rows, recomputing each row's r.

struct Params {
    n: u32,
    d: u32,
    eps: u32,   // f32 bits
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dg: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.d) { return; }
    let dd = gidx;

    var acc = 0.0;
    for (var n: u32 = 0u; n < p.n; n = n + 1u) {
        let base = n * p.d;
        var s = 0.0;
        for (var k: u32 = 0u; k < p.d; k = k + 1u) {
            let v = x[base + k];
            s = s + v * v;
        }
        let r = inverseSqrt(s + bitcast<f32>(p.eps));
        acc = acc + dy[base + dd] * x[base + dd] * r;
    }
    dg[dd] = acc;
}
