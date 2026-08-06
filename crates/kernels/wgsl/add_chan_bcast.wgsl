// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Add a per-(image, channel) scalar to a full map, NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Add a per-(image, channel) scalar to a full map, NCHW.
//   x : [N, C, H, W]
//   v : [N, C]
//   y : [N, C, H, W]   y[n,c,h,w] = x[n,c,h,w] + v[n,c]
//
// GlobalContextBlock's residual: `x + transform(context)` where context is
// [B,C,1,1]. bias_add does NOT fit — its bias is a [C] vector SHARED across the
// batch, whereas this context is computed per image, so at N>1 bias_add would add
// the wrong image's context.

struct Params {
    N: u32,
    C: u32,
    HW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.HW;
    if (idx >= total) { return; }
    let nc = idx / p.HW;
    y[idx] = x[idx] + v[nc];
}
