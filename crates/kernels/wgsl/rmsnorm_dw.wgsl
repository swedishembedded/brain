// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm backward w.r.t. the gain weight
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// RMSNorm backward w.r.t. the gain weight:
//   dW[c] += sum_n dY[n,c] * x[n,c] * inv[n]
// One invocation per channel; uses the precomputed per-row inv (rms_inv).
// Accumulates into the (pre-zeroed) weight-grad buffer.

struct Params {
    d_model: u32,
    n_rows: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:  array<f32>;
@group(0) @binding(2) var<storage, read>       x:   array<f32>;
@group(0) @binding(3) var<storage, read>       inv: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let c = gidx;
    if (c >= p.d_model) { return; }
    var acc = 0.0;
    for (var n: u32 = 0u; n < p.n_rows; n = n + 1u) {
        acc = acc + dy[n * p.d_model + c] * x[n * p.d_model + c] * inv[n];
    }
    dw[c] = dw[c] + acc;
}
