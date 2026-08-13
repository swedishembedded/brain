// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Strided-region copy between same-layout buffers
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Strided-region copy between same-layout buffers: for each row, copy
// `width` values at `off` within a `row_stride` row (e.g. the v region of a
// fused qkv-gradient buffer, where qk-norm backward rewrites q/k but v
// passes through). One invocation per element.

struct Params {
    rows: u32,
    width: u32,
    row_stride: u32,
    off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.width) { return; }
    let row = idx / p.width;
    let i = row * p.row_stride + p.off + (idx % p.width);
    dst[i] = src[i];
}
