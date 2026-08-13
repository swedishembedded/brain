// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Scale a gradient buffer in place by a coefficient that lives in a GPU buffer (written by clip_coef)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Scale a gradient buffer in place by a coefficient that lives in a GPU buffer
// (written by clip_coef): grad[i] *= coef[0]. Lets the whole optimizer step run
// on-device with no host readback. One invocation per element.

struct Params {
    numel: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> grad: array<f32>;
@group(0) @binding(2) var<storage, read>       coef: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.numel) { return; }
    grad[gidx] = grad[gidx] * coef[0];
}
