// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  LayerNorm backward w.r.t. beta
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// LayerNorm backward w.r.t. beta:  dbeta[c] += sum_n dy[n,c].
// One invocation per channel. Accumulates into the (pre-zeroed) grad buffer.

struct Params {
    d_model: u32,
    n_rows: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dbeta: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let c = gidx;
    if (c >= p.d_model) { return; }
    let d = p.d_model;
    var acc = 0.0;
    for (var n: u32 = 0u; n < p.n_rows; n = n + 1u) {
        acc = acc + dy[n * d + c];
    }
    dbeta[c] = dbeta[c] + acc;
}
