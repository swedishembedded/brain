// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Table-driven 2D RoPE (DINOv3/WorldMirror "normalized" variant), in place on
// the q or k region of a fused [rows, row_stride] buffer. The host precomputes
// per-token-position cos/sin tables [tmod, half] (half = head_dim/2; the
// reference duplicates the half-length angle vector, so rotate-half pairs
// (d, d+half) share angle index d). Rows map to table rows modulo `tmod`
// (per-frame positions repeat across frames). `sign` = 1 forward, -1 = the
// exact inverse rotation (backward).
//   y1 = x1*cos - sign*x2*sin ;  y2 = x2*cos + sign*x1*sin

struct Params {
    rows: u32,
    heads: u32,
    half: u32,       // head_dim / 2
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
    let base = row * p.row_stride + p.off + h * (2u * p.half);
    let x1 = buf[base + d];
    let x2 = buf[base + d + p.half];
    buf[base + d] = x1 * c - x2 * s;
    buf[base + d + p.half] = x2 * c + x1 * s;
}
