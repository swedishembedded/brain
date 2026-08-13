// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RoPE backward: gradient is the inverse (transpose) rotation, i.e
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// RoPE backward: gradient is the inverse (transpose) rotation, i.e. rotate by
// -angle. Given d_out in the q/k region of d_qkv, overwrite it with d_in.
//   forward:  out_e = e*c - o*s ;  out_o = e*s + o*c
//   backward: d_e   = dE*c + dO*s ;  d_o   = -dE*s + dO*c

struct Params {
    n_rows: u32,
    n_heads: u32,
    head_dim: u32,
    row_stride: u32,
    base_off: u32,
    tcols: u32,
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
    let de = buf[base];
    let do_ = buf[base + 1u];
    buf[base]      = de * c + do_ * s;
    buf[base + 1u] = -de * s + do_ * c;
}
