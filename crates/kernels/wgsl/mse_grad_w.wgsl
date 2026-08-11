// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-sample weighted MSE gradient — spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Per-sample weighted MSE gradient. Exact gradient of
// L = scale * Σ_n out[n] (out from mse_value_w) w.r.t. pred. [N, M]
// row-major, total = N*M, one invocation per ELEMENT:
//   n = i / m
//   dpred[i] = w[n] * 2.0 * (pred[i] - tgt[i]) / f32(m) * bitcast<f32>(scale)
// `scale` is the upstream scale (e.g. 1/N batch mean) as f32 bits in params
// (gpu_core::f convention) — it appears in THIS kernel only. Divides by the
// SAME m as mse_value_w (mirrored reduction convention). Exactly 4 storage
// buffers.
//

struct Params {
    total: u32,
    m: u32,
    scale: u32,  // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:  array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:   array<f32>;
@group(0) @binding(3) var<storage, read>       w:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dpred: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let n = i / p.m;
    dpred[i] = w[n] * 2.0 * (pred[i] - tgt[i]) / f32(p.m) * bitcast<f32>(p.scale);
}
