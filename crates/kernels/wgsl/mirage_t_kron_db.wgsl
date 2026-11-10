// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training Kronecker rotation right-factor gradient: db[j, l] += sum_{r, i} z[r, i, j] dtmp[r, i, l]
// @how   one thread per factor element, looping over rows and the left index; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// dtmp = A dOut. Params: rows, fa, fb. Dispatch: fb*fb threads.

struct Params { rows: u32, fa: u32, fb: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  z: array<f32>;
@group(0) @binding(2) var<storage, read>  dtmp: array<f32>;
@group(0) @binding(3) var<storage, read_write> db: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.fb * p.fb) { return; }
    let l = idx % p.fb;
    let j = idx / p.fb;
    var acc = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        for (var i = 0u; i < p.fa; i = i + 1u) {
            acc = acc + z[(r * p.fa + i) * p.fb + j] * dtmp[(r * p.fa + i) * p.fb + l];
        }
    }
    db[idx] = db[idx] + acc;
}
