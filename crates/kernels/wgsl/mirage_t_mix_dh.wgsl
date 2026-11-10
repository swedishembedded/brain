// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training lane mixing weight gradient: dh[r, i, j] += sum_e gy[r, i, e] x[r, j, e]
// @how   one thread per weight, looping over the width; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// out[r, i, :] = sum_j h[r, i b + j] x[r, j, :]; h [t, a b], x [t, b d], out [t, a d].
// Params: t, a, b, d. Dispatch: t*a*b threads.

struct Params { t: u32, a: u32, b: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  gy: array<f32>;
@group(0) @binding(2) var<storage, read>  x: array<f32>;
@group(0) @binding(3) var<storage, read_write> dh: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.t * p.a * p.b) { return; }
    let j = idx % p.b;
    let i = (idx / p.b) % p.a;
    let r = idx / (p.a * p.b);
    var acc = 0.0;
    for (var e = 0u; e < p.d; e = e + 1u) {
        acc = acc + gy[(r * p.a + i) * p.d + e] * x[(r * p.b + j) * p.d + e];
    }
    dh[idx] = dh[idx] + acc;
}
