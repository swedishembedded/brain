// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training Kronecker rotation left-factor gradient: da[i, k] += sum_{r, l} tmp[r, i, l] g[r, k, l]
// @how   one thread per factor element, looping over rows and the width; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// tmp = Z B. Params: rows, fa, fb. Dispatch: fa*fa threads.

struct Params { rows: u32, fa: u32, fb: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  tmp: array<f32>;
@group(0) @binding(2) var<storage, read>  g: array<f32>;
@group(0) @binding(3) var<storage, read_write> da: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.fa * p.fa) { return; }
    let k = idx % p.fa;
    let i = idx / p.fa;
    var acc = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        for (var l = 0u; l < p.fb; l = l + 1u) {
            acc = acc + tmp[(r * p.fa + i) * p.fb + l] * g[(r * p.fa + k) * p.fb + l];
        }
    }
    da[idx] = da[idx] + acc;
}
