// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  SiLU backward - gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// SiLU backward — gradient w.r.t. the pre-activation `x`.
//   s    = sigmoid(x) = 1 / (1 + exp(-x))
//   y    = x * s
//   y'(x) = s + x * s * (1 - s)
//   dx[i] = dy[i] * y'(x[i])
// Elementwise; must stay consistent with silu.wgsl for the gradient check.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;   // pre-activation
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let v = x[idx];
    let s = 1.0 / (1.0 + exp(-v));
    dx[idx] = dy[idx] * (s + v * s * (1.0 - s));
}
