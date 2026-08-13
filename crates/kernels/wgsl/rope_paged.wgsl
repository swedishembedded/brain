// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RoPE (half-split, base theta) applied to a batch of single-token rows, each at its OWN absolute position `positions[row]` - the batched-decode twin of rope_at (which assumes pos_base+row)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// RoPE (half-split, base theta) applied to a batch of single-token rows, each at
// its OWN absolute position `positions[row]` — the batched-decode twin of rope_at
// (which assumes pos_base+row). One invocation per (row, head, m<half).

struct Params {
    n_rows: u32,     // batch (one new token per sequence)
    n_heads: u32,
    head_dim: u32,
    row_stride: u32, // n_heads*head_dim
    rope_base: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@group(0) @binding(2) var<storage, read>       positions: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    if (gidx >= p.n_rows * p.n_heads * half) { return; }
    let m = gidx % half;
    let tmp = gidx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = positions[row];
    let hbase = row * p.row_stride + h * p.head_dim;
    let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let x0 = buf[hbase + m];
    let x1 = buf[hbase + m + half];
    buf[hbase + m]        = x0 * c - x1 * s;
    buf[hbase + m + half] = x1 * c + x0 * s;
}
