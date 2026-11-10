// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training causal dilated taps tap gradient: dt[j, c] += sum_r gy[r, c] x[r - j dil, c]
// @how   one thread per tap element, looping over the rows; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// y[r, c] = sum_j taps[j, c] x[r - j dil, c] over the rows r - j dil >= 0; x [t, w], taps [kt, w].
// Params: t, w, kt, dil. Dispatch: kt*w threads.

struct Params { t: u32, w: u32, kt: u32, dil: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  x: array<f32>;
@group(0) @binding(2) var<storage, read>  gy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dt: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.kt * p.w) { return; }
    let c = idx % p.w;
    let j = idx / p.w;
    var acc = 0.0;
    for (var r = j * p.dil; r < p.t; r = r + 1u) {
        acc = acc + gy[r * p.w + c] * x[(r - j * p.dil) * p.w + c];
    }
    dt[idx] = dt[idx] + acc;
}
