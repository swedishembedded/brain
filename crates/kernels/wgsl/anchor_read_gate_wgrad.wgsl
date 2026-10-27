// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read gate parameter gradients: weight rows, bias and scale, accumulated over the rows
// @how   one thread per (head, column), serial sum over the rows
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Columns `0..head_dim` of head `h` accumulate `dweight[h, d] += sum_row dz[row, h] * q[row, h, d] / sqrt(head_dim)`;
// column `head_dim` accumulates `dbias[h] += sum_row dz[row, h]` and column
// `head_dim + 1` `dscale[h] += sum_row dsc[row, h]`. All three are added to
// pre-zeroed gradient buffers, like every parameter gradient.
//
//   q, dz, dsc : as in `anchor_read_gate_bwd`

struct Params {
    rows: u32,
    n_heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:       array<f32>;
@group(0) @binding(2) var<storage, read>       dz:      array<f32>;
@group(0) @binding(3) var<storage, read>       dsc:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dweight: array<f32>;
@group(0) @binding(5) var<storage, read_write> dbias:   array<f32>;
@group(0) @binding(6) var<storage, read_write> dscale:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let hd = p.head_dim;
    let cols = hd + 2u;
    if (gidx >= p.n_heads * cols) { return; }

    let h = gidx / cols;
    let col = gidx % cols;
    var acc = 0.0;
    for (var row = 0u; row < p.rows; row = row + 1u) {
        let i = row * p.n_heads + h;
        if (col < hd) {
            acc = acc + dz[i] * q[i * hd + col];
        } else if (col == hd) {
            acc = acc + dz[i];
        } else {
            acc = acc + dsc[i];
        }
    }
    if (col < hd) {
        dweight[h * hd + col] = dweight[h * hd + col] + acc * inverseSqrt(f32(hd));
    } else if (col == hd) {
        dbias[h] = dbias[h] + acc;
    } else {
        dscale[h] = dscale[h] + acc;
    }
}
