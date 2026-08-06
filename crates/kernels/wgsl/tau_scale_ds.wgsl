// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  tau_scale backward w.r.t. the per-(head,token) scale `s`
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// tau_scale backward w.r.t. the per-(head,token) scale `s`:
//   ds[h, row] = sum_d d_out[row, h, d] * in[row, h, d]
// `d_out`/`in` are [rows, heads*head_dim]; `ds` is [heads, rows]. One invocation
// per (head, row); loops head_dim (no atomics).

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_out: array<f32>;
@group(0) @binding(2) var<storage, read>       inp:   array<f32>;
@group(0) @binding(3) var<storage, read_write> ds:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.heads * p.rows) { return; }
    let h = idx / p.rows;
    let row = idx % p.rows;
    let hd_total = p.heads * p.head_dim;
    let base = row * hd_total + h * p.head_dim;
    var acc = 0.0;
    for (var d = 0u; d < p.head_dim; d = d + 1u) {
        acc = acc + d_out[base + d] * inp[base + d];
    }
    ds[idx] = acc;
}
