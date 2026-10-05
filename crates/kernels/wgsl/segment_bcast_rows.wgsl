// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of segment_sum_rows: each segment's gradient broadcast to its masked rows
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// dx[(s*len + j)*d + c] = mask[s*len + j] * dz[s*d + c]
// The exact adjoint of segment_sum_rows: rows outside the sum get zero.
// Writes (does not accumulate).

struct Params {
    segments: u32,
    len: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       mask: array<u32>;
@group(0) @binding(2) var<storage, read>       dz:   array<f32>;
@group(0) @binding(3) var<storage, read_write> dx:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.segments * p.len * p.d) { return; }
    let row = i / p.d;
    let c = i % p.d;
    let s = row / p.len;
    if (mask[row] != 0u) {
        dx[i] = dz[s * p.d + c];
    } else {
        dx[i] = 0.0;
    }
}
