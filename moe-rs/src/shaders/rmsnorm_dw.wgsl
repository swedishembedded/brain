// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.d_model) { return; }
    var acc = 0.0;
    for (var n: u32 = 0u; n < p.n_rows; n = n + 1u) {
        acc = acc + dy[n * p.d_model + c] * x[n * p.d_model + c] * inv[n];
    }
    dw[c] = dw[c] + acc;
}
