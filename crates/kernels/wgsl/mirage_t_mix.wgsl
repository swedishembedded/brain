// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training lane mixing forward: out[r, i, e] = sum_j h[r, i, j] x[r, j, e]
// @how   one thread per output element, looping over the sources
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// out[r, i, :] = sum_j h[r, i b + j] x[r, j, :]; h [t, a b], x [t, b d], out [t, a d].
// Params: t, a, b, d. Dispatch: t*a*d threads.

struct Params { t: u32, a: u32, b: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  h: array<f32>;
@group(0) @binding(2) var<storage, read>  x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.t * p.a * p.d) { return; }
    let e = idx % p.d;
    let i = (idx / p.d) % p.a;
    let r = idx / (p.a * p.d);
    var acc = 0.0;
    for (var j = 0u; j < p.b; j = j + 1u) {
        acc = acc + h[r * p.a * p.b + i * p.b + j] * x[(r * p.b + j) * p.d + e];
    }
    y[idx] = acc;
}
