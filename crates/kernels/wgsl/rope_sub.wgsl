// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Interleaved RoPE (forward, in place) on the FIRST `rope_dim` channels of each head (a sub-slice of a `head_dim`-wide head), for the DSA indexer where each head is laid out [rope / pass]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Interleaved RoPE (forward, in place) on the FIRST `rope_dim` channels of each
// head (a sub-slice of a `head_dim`-wide head), for the DSA indexer where each
// head is laid out [rope | pass]. Angle uses within-sequence position row % T and
// the interleaved-pair convention (rope_train). One invocation per
// (row, head, pair). `base` = 10000 (matches rope_train).

struct Params {
    n_rows: u32,     // B*T
    n_heads: u32,
    head_dim: u32,   // full per-head width (rope + pass)
    rope_dim: u32,   // channels rotated (from offset 0 within each head)
    row_stride: u32, // per-row width (n_heads * head_dim)
    tcols: u32,      // T
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let halfr = p.rope_dim / 2u;
    let total = p.n_rows * p.n_heads * halfr;
    if (gidx >= total) { return; }

    let j = gidx % halfr;
    let tmp = gidx / halfr;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = row % p.tcols;

    let base = row * p.row_stride + h * p.head_dim + 2u * j;
    let angle = f32(pos) * pow(10000.0, -f32(2u * j) / f32(p.rope_dim));
    let c = cos(angle);
    let s = sin(angle);
    let e = buf[base];
    let o = buf[base + 1u];
    buf[base]      = e * c - o * s;
    buf[base + 1u] = e * s + o * c;
}
