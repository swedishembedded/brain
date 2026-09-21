// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RoPE backward for `rope_base_yarn.wgsl` (half-split convention, YaRN inv_freq table)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// RoPE backward for `rope_base_yarn.wgsl` (see that kernel's own doc). The
// gradient is the transpose (inverse) rotation, i.e. rotate by -angle, with
// the same `attention_factor` scale applied to both cos and sin - the exact
// twin `rope_base_bwd.wgsl` is to `rope_base.wgsl`, over the table-based
// frequency instead of the analytic one.
//   forward:  o0 = c*x0 - s*x1 ;  o1 = c*x1 + s*x0   (c,s already carry attention_factor)
//   backward: d_x0 = c*dO0 + s*dO1 ;  d_x1 = -s*dO0 + c*dO1
// One invocation per (row, head, m).

struct Params {
    n_rows: u32,
    n_heads: u32,
    head_dim: u32,
    row_stride: u32,
    base_off: u32,
    tcols: u32,
    attention_factor: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@group(0) @binding(2) var<storage, read>       inv_freq: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
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
    let angle = f32(pos) * inv_freq[m];
    let c = cos(angle) * p.attention_factor;
    let s = sin(angle) * p.attention_factor;
    let d0 = buf[hbase + m];
    let d1 = buf[hbase + m + half];
    buf[hbase + m]        = d0 * c + d1 * s;
    buf[hbase + m + half] = -d0 * s + d1 * c;
}
