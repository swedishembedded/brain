// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Broadcast-add a row-strip and a column-strip into a full map, NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Broadcast-add a row-strip and a column-strip into a full map, NCHW.
//   a : [N, C, H, 1]   (a strip pooled over W)
//   b : [N, C, 1, W]   (a strip pooled over H)
//   y : [N, C, H, W]   y[n,c,h,w] = a[n,c,h] + b[n,c,w]
//
// ZipDepth's StripPoolingAttention does exactly `h_strip + w_strip` and relies on
// torch broadcasting to materialize the full map. There is no existing kernel for
// this shape: add2 is elementwise same-shape, and bias_add broadcasts a [C]
// vector (shared across N and space), not two orthogonal strips.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }
    let w  = idx % p.W;
    let t1 = idx / p.W;
    let h  = t1 % p.H;
    let nc = t1 / p.H;
    y[idx] = a[nc * p.H + h] + b[nc * p.W + w];
}
