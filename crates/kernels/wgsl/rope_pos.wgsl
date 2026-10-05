// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RoPE at real-valued per-row positions (times), forward or its adjoint, in place on the q or k region of a fused qkv buffer
// @how   one thread per (row, head, channel-pair)
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// angle = pos[row] * theta^(-2j / head_dim) * dir. With dir = 1 this rotates
// each channel pair by the row's angle; dir = -1 rotates back, which is the
// transpose of the forward and so its gradient (a rotation is orthogonal).
// Positions are any real numbers - time in a dataset's unit, say - not
// integer token indices.

struct Params {
    n_rows: u32,
    n_heads: u32,
    head_dim: u32,
    row_stride: u32, // 3*d_model
    base_off: u32,   // 0 for q, d_model for k
    theta: f32,
    dir: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    if (idx >= p.n_rows * p.n_heads * half) { return; }
    let j = idx % half;
    let tmp = idx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let base = row * p.row_stride + p.base_off + h * p.head_dim + 2u * j;
    let angle = p.dir * pos[row] * pow(p.theta, -f32(2u * j) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let e = buf[base];
    let o = buf[base + 1u];
    buf[base]      = e * c - o * s;
    buf[base + 1u] = e * s + o * c;
}
