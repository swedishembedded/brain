// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched RoPE (forward). Rows are flattened [B*T, ...]; the rotation angle uses the WITHIN-sequence position = row % T
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Batched RoPE (forward). Rows are flattened [B*T, ...]; the rotation angle uses
// the WITHIN-sequence position = row % T. Applied in place to the q or k region
// of the fused qkv buffer (select via base_off). One invocation per
// (row, head, channel-pair).

struct Params {
    n_rows: u32,    // B*T
    n_heads: u32,
    head_dim: u32,
    row_stride: u32, // 3*d_model
    base_off: u32,   // 0 for q, d_model for k
    tcols: u32,      // T
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    let total = p.n_rows * p.n_heads * half;
    let idx = gidx;
    if (idx >= total) { return; }

    let j = idx % half;
    let tmp = idx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = row % p.tcols;

    let base = row * p.row_stride + p.base_off + h * p.head_dim + 2u * j;
    let angle = f32(pos) * pow(10000.0, -f32(2u * j) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let e = buf[base];
    let o = buf[base + 1u];
    buf[base]      = e * c - o * s;
    buf[base + 1u] = e * s + o * c;
}
