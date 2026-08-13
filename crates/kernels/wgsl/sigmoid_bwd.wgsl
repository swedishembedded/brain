// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sigmoid backward:  dx = dy * s * (1 - s),  s = sigmoid(x)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Sigmoid backward:  dx = dy * s * (1 - s),  s = sigmoid(x).
// Takes the PRE-activation `x` (not the output `s`), matching the convention of
// every other *_bwd kernel here (silu_bwd, gelu_bwd, leaky_relu_bwd) so blocks
// can cache one activation per stage under the SSA discipline.

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
    let s = 1.0 / (1.0 + exp(-x[idx]));
    dx[idx] = dy[idx] * s * (1.0 - s);
}
