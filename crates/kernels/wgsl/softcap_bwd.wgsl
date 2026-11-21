// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Soft cap backward: dx = dy * (1 - tanh(x / c)^2)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Soft cap backward for y = c * tanh(x / c): dx = dy * (1 - tanh(x / c)^2).
// Takes the forward input `x` (see softcap.wgsl); `cap` is the f32 bit pattern
// of c. Same tanh clamp as the forward so the two agree at saturation.

struct Params {
    total: u32,
    cap: u32,   // f32 bits
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
    let c = bitcast<f32>(p.cap);
    let t = tanh(clamp(x[idx] / c, -15.0, 15.0));
    dx[idx] = dy[idx] * (1.0 - t * t);
}
