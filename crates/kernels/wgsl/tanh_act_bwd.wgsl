// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tanh backward: dx = dy * (1 - tanh(x)^2)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Tanh backward: dx = dy * (1 - tanh(x)^2). Takes `x` (not `y`) so it
// composes with `tanh_act.wgsl`'s own signature without requiring the
// forward's output to be kept around separately.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let t = tanh(x[idx]);
    dx[idx] = dy[idx] * (1.0 - t * t);
}
