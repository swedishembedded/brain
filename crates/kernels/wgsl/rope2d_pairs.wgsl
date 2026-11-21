// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Table-driven 2D RoPE with interleaved channel pairs (2d, 2d+1), in place on the q or k region of a fused [rows, row_stride] buffer
// @how   one thread per channel pair
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The interleaved-pair twin of `rope2d.wgsl` and a drop-in for its `VitKernelIds::rope2d` slot: same Params, same
// bindings, same table `[tmod, half]` (one angle per channel PAIR) and the same `sign` (1 forward, -1 the exact
// inverse rotation, which is the backward). It differs only in which channels form a pair: `rope2d` rotates
// (d, d + half) (the half-split convention), this rotates the adjacent pair (2d, 2d + 1), the convention of
// `rotate_pairwise` in axial vision rotary embeddings whose table is `repeat_interleave(2)` of a per-pair angle.
// `rope_interleave_table.wgsl` has the same pairing but runs out of place over a [seq, heads*head_dim] buffer, with no
// row stride, region offset or sign, so it cannot rotate the q and k regions of a fused qkv in place.
//   y1 = x1*cos - sign*x2*sin ;  y2 = x2*cos + sign*x1*sin     with x1 = x[2d], x2 = x[2d+1]

struct Params {
    rows: u32,
    heads: u32,
    half: u32,       // head_dim / 2: pairs per head and table width
    row_stride: u32, // e.g. 3*d_model for fused qkv
    off: u32,        // region offset: 0 = q, d_model = k
    tmod: u32,       // table rows; token row -> table row = row % tmod
    sign: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@group(0) @binding(2) var<storage, read>       cos_t: array<f32>; // [tmod, half]
@group(0) @binding(3) var<storage, read>       sin_t: array<f32>; // [tmod, half]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.heads * p.half;
    if (idx >= total) { return; }
    let d = idx % p.half;
    let r1 = idx / p.half;
    let h = r1 % p.heads;
    let row = r1 / p.heads;

    let t = (row % p.tmod) * p.half + d;
    let c = cos_t[t];
    let s = sin_t[t] * p.sign;
    let base = row * p.row_stride + p.off + h * (2u * p.half) + 2u * d;
    let x1 = buf[base];
    let x2 = buf[base + 1u];
    buf[base] = x1 * c - x2 * s;
    buf[base + 1u] = x2 * c + x1 * s;
}
