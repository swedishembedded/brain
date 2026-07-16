// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Backward w.r.t. x for l2norm_scale. With r = rsqrt(sum_k x_k^2 + eps),
// y_d = x_d * r * g_d, the input gradient for row n is:
//   a_d   = dy_d * g_d
//   d_x_e = r * ( a_e - x_e * r^2 * sum_d a_d * x_d )
// One invocation per (n,e); each recomputes the row's r and the a·x dot.

struct Params {
    n: u32,
    d: u32,
    eps: u32,   // f32 bits
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       g:  array<f32>;
@group(0) @binding(3) var<storage, read>       dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n * p.d;
    if (gidx >= total) { return; }

    let e = gidx % p.d;
    let n = gidx / p.d;
    let base = n * p.d;

    var s = 0.0;
    var adotx = 0.0;
    for (var k: u32 = 0u; k < p.d; k = k + 1u) {
        let xv = x[base + k];
        s = s + xv * xv;
        adotx = adotx + dy[base + k] * g[k] * xv;
    }
    let r = inverseSqrt(s + bitcast<f32>(p.eps));
    let ae = dy[gidx] * g[e];
    dx[gidx] = r * (ae - x[gidx] * r * r * adotx);
}
