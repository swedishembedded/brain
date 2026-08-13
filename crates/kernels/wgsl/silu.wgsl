// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  SiLU (a.k.a. swish) activation
// @how   one thread per output element
// @opt   3
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// SiLU (a.k.a. swish) activation:  y = x * sigmoid(x) = x / (1 + exp(-x)).
// Plain (non-gated) elementwise activation. The matching derivative is in
// silu_bwd.wgsl.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let v = x[idx];
    out[idx] = v / (1.0 + exp(-v));
}
