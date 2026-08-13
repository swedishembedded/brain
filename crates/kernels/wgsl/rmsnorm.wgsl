// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm:  out[t, c] = weight[c] * x[t, c] / sqrt(mean_c(x[t, c]^2) + eps) One invocation per row (token)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// RMSNorm:  out[t, c] = weight[c] * x[t, c] / sqrt(mean_c(x[t, c]^2) + eps)
// One invocation per row (token). d_model is small, so the per-row loop is fine.

struct Params {
    d_model: u32,
    seq_len: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let t = gidx;
    if (t >= p.seq_len) { return; }
    let base = t * p.d_model;
    var ss = 0.0;
    for (var c: u32 = 0u; c < p.d_model; c = c + 1u) {
        let v = x[base + c];
        ss = ss + v * v;
    }
    let inv = inverseSqrt(ss / f32(p.d_model) + 1e-6);
    for (var c: u32 = 0u; c < p.d_model; c = c + 1u) {
        out[base + c] = weight[c] * x[base + c] * inv;
    }
}
