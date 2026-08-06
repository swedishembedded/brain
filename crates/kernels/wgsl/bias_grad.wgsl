// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bias gradient:  dbias[n] += sum_m dy[m,n]
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Bias gradient:  dbias[n] += sum_m dy[m,n].
// One invocation per output feature n. Accumulates into the pre-zeroed buffer.

struct Params {
    m: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dbias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let col = gidx;
    if (col >= p.n) { return; }
    var acc = 0.0;
    for (var mm: u32 = 0u; mm < p.m; mm = mm + 1u) {
        acc = acc + dy[mm * p.n + col];
    }
    dbias[col] = dbias[col] + acc;
}
