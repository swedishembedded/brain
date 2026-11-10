// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training causal dilated taps input gradient: dx[s, c] += sum_j gy[s + j dil, c] taps[j, c]
// @how   one thread per input element, looping over the taps; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// y[r, c] = sum_j taps[j, c] x[r - j dil, c] over the rows r - j dil >= 0; x [t, w], taps [kt, w].
// Params: t, w, kt, dil. Dispatch: t*w threads.

struct Params { t: u32, w: u32, kt: u32, dil: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  gy: array<f32>;
@group(0) @binding(2) var<storage, read>  taps: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.t * p.w) { return; }
    let c = idx % p.w;
    let s = idx / p.w;
    var acc = 0.0;
    for (var j = 0u; j < p.kt; j = j + 1u) {
        let r = s + j * p.dil;
        if (r < p.t) {
            acc = acc + gy[r * p.w + c] * taps[j * p.w + c];
        }
    }
    dx[idx] = dx[idx] + acc;
}
