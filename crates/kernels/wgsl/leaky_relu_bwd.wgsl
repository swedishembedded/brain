// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Leaky ReLU backward — gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Leaky ReLU backward — gradient w.r.t. the pre-activation `x`.
//   dx = dy           if x >= 0
//   dx = slope*dy     otherwise.

struct Params {
    total: u32,
    slope: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    if (x[idx] >= 0.0) { dx[idx] = dy[idx]; } else { dx[idx] = p.slope * dy[idx]; }
}
