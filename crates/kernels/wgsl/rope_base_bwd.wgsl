// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RoPE backward for the half-split (HF/Qwen) convention (see `rope_base.wgsl`)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// RoPE backward for the half-split (HF/Qwen) convention (see `rope_base.wgsl`).
// The gradient is the transpose (inverse) rotation, i.e. rotate by -θ. Given
// d_out in the q/k buffer, overwrite it with d_in in place.
//   forward:  o0 = x0*c - x1*s ;  o1 = x1*c + x0*s
//   backward: d_x0 = dO0*c + dO1*s ;  d_x1 = -dO0*s + dO1*c
// One invocation per (row, head, m).

struct Params {
    n_rows: u32,
    n_heads: u32,
    head_dim: u32,
    row_stride: u32,
    base_off: u32,
    tcols: u32,
    rope_base: f32,
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

    let m = idx % half;
    let tmp = idx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = row % p.tcols;

    let hbase = row * p.row_stride + p.base_off + h * p.head_dim;
    let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let d0 = buf[hbase + m];
    let d1 = buf[hbase + m + half];
    buf[hbase + m]        = d0 * c + d1 * s;
    buf[hbase + m + half] = -d0 * s + d1 * c;
}
