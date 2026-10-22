// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Quantize + pack an [M, K] f32 activation into [M, K/4] u32 (4 int8 per u32, little-endian along K) using its per-(row, 32-group) scales sx
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Quantize + pack an [M, K] f32 activation into [M, K/4] u32 (4 int8 per u32,
// little-endian along K) with the per-(row, 32-group) scales `max_abs_row`
// wrote (sx: [M, K/32]):
//   q = clamp(round(x / sx[m, k/32]), -127, 127)
// One invocation per OUTPUT u32 (M*K/4). K must be a multiple of 32. Matches
// the packing matmul_i8 expects (x_q[m, g] = bytes for x[m, 4g .. 4g+4]).

struct Params { m: u32, k: u32 };  // k = full K (multiple of 32)

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       sx: array<f32>;
@group(0) @binding(3) var<storage, read_write> xq: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let kg = p.k / 4u;
    let total = p.m * kg;
    if (idx >= total) { return; }
    let m = idx / kg;
    let g = idx % kg;
    let base = m * p.k + g * 4u;
    let inv = 1.0 / sx[m * (p.k / 32u) + g / 8u]; // the word's group scale
    var w: u32 = 0u;
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        let q = clamp(round(x[base + b] * inv), -127.0, 127.0);
        // int8 → low byte of a u32, placed at byte b (LE).
        let byte = u32(i32(q) & 0xff);
        w = w | (byte << (8u * b));
    }
    xq[idx] = w;
}
