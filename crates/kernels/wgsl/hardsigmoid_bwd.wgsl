// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Hard sigmoid backward: gradient w.r.t. the pre-activation
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Gradient of hardsigmoid.wgsl w.r.t. its input: 1/6 on [-3, 3], 0 outside.
//   dx[i] = dout[i] * hardsigmoid'(x[i])

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       dout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let v = x[idx];
    var g = 0.0;
    if (v >= -3.0 && v <= 3.0) {
        g = 1.0 / 6.0;
    }
    dx[idx] = dout[idx] * g;
}
