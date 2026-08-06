// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Moondream per-(head, token) attention-temperature scale, broadcast over head_dim
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Moondream per-(head, token) attention-temperature scale, broadcast over
// head_dim:  out[row, h, d] = in[row, h, d] * s[h, row]
// `in`/`out` are [rows, heads*head_dim] (a q or v buffer); `s` is [heads, rows].
// Also serves the input-gradient backward (d_in = d_out * s → same kernel). One
// invocation per element.

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read>       s:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let hd_total = p.heads * p.head_dim;
    if (idx >= p.rows * hd_total) { return; }
    let row = idx / hd_total;
    let h = (idx % hd_total) / p.head_dim;
    out[idx] = inp[idx] * s[h * p.rows + row];
}
