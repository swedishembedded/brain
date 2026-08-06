// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Rotary position embedding, applied in place to either the q or k region of the fused qkv buffer (select via base_off)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Rotary position embedding, applied in place to either the q or k region of
// the fused qkv buffer (select via base_off). Matches the Python apply_rope:
// channel pair (2j, 2j+1) within a head is rotated by angle = t * base^(-2j/hd).
//
// Buffer layout per token row: [ q(d_model) | k(d_model) | v(d_model) ], so
// row_stride = 3*d_model and a head occupies head_dim contiguous channels.
// One invocation per (token, head, channel-pair); pairs never overlap.

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    row_stride: u32,
    base_off: u32,   // 0 for q, d_model for k
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    let total = p.seq_len * p.n_heads * half;
    let idx = gidx;
    if (idx >= total) { return; }

    let j = idx % half;
    let tmp = idx / half;
    let h = tmp % p.n_heads;
    let t = tmp / p.n_heads;

    let base = t * p.row_stride + p.base_off + h * p.head_dim + 2u * j;
    let angle = f32(t) * pow(10000.0, -f32(2u * j) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let e = buf[base];
    let o = buf[base + 1u];
    buf[base]      = e * c - o * s;
    buf[base + 1u] = e * s + o * c;
}
