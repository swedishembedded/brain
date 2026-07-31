// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Inverse of head_pack: scatter per-head [rows, hd] context blocks back into
// the row-major [rows, d_model] stream the output projection consumes:
//   out[i*dst_stride + dst_off + ho*hd + d] = src[ho*head_stride + i*hd + d]
// One invocation per (ho, i, d); total = heads * rows * hd. Pure copy.

struct Params {
    rows: u32,
    heads: u32,
    hd: u32,
    dst_stride: u32,
    dst_off: u32,
    head_stride: u32, // matches the padded stride head_pack wrote ctx_pack with
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.heads * p.rows * p.hd;
    if (idx >= total) { return; }
    let d = idx % p.hd;
    let r1 = idx / p.hd;
    let i = r1 % p.rows;
    let ho = r1 / p.rows;
    out[i * p.dst_stride + p.dst_off + ho * p.hd + d] = src[ho * p.head_stride + i * p.hd + d];
}
