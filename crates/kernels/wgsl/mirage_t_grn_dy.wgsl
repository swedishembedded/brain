// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training global-response-norm input gradient: dy[r, c] += valid[r] * de[c] * y[r, c] / E[c]
// @how   one thread per element; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: total, d. Dispatch: total threads.

struct Params { total: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  y: array<f32>;
@group(0) @binding(2) var<storage, read>  mask: array<f32>;
@group(0) @binding(3) var<storage, read>  energy: array<f32>;
@group(0) @binding(4) var<storage, read>  de: array<f32>;
@group(0) @binding(5) var<storage, read_write> dy: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let c = idx % p.d;
    let r = idx / p.d;
    if (mask[r] != 0.0) {
        dy[idx] = dy[idx] + de[c] * y[idx] / energy[c];
    }
}
