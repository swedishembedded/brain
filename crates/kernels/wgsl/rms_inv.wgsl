// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Helper: per-row inverse RMS,  inv[n] = 1/sqrt(mean_c(x[n,c]^2) + eps)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Helper: per-row inverse RMS,  inv[n] = 1/sqrt(mean_c(x[n,c]^2) + eps).
// One invocation per row. Used by rmsnorm_dw.

struct Params {
    d_model: u32,
    n_rows: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> inv: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.n_rows) { return; }
    let base = n * p.d_model;
    var ss = 0.0;
    for (var c: u32 = 0u; c < p.d_model; c = c + 1u) {
        let v = x[base + c];
        ss = ss + v * v;
    }
    inv[n] = inverseSqrt(ss / f32(p.d_model) + 1e-6);
}
