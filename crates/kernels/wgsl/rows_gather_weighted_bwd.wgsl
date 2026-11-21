// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Table gradient of the weighted row gather, accumulated: dtable[i] += sum over the taps that read row i of w * dout[r]
// @how   one thread per table element, serial sum over the row's incoming edges (CSR), no atomics
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The adjoint of `rows_gather_weighted.wgsl` with respect to the table. Scattering `w * dout[r]` into the rows the
// taps named needs atomics (several outputs read one table row), which core-compute WGSL does not have, so the host
// transposes the tap list once into CSR form: for table row i, `ptr[i] .. ptr[i + 1]` index the `edge_row` (the
// output row r that read it) and `edge_w` (the tap weight) lists. For i < table_rows and c < width:
//
//   dtable[i * width + c] += sum_{e in ptr[i] .. ptr[i + 1]} edge_w[e] * dout[edge_row[e] * width + c]
//
// Edges are summed in list order. Gradients accumulate (`+=`): zero the buffer before the backward.
//
//   params: table_rows, width

struct Params {
    table_rows: u32,
    width: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ptr:      array<u32>;
@group(0) @binding(2) var<storage, read>       edge_row: array<u32>;
@group(0) @binding(3) var<storage, read>       edge_w:   array<f32>;
@group(0) @binding(4) var<storage, read>       dout:     array<f32>;
@group(0) @binding(5) var<storage, read_write> dtable:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= p.table_rows * p.width) { return; }
    let i = t / p.width;
    let c = t % p.width;
    var acc = 0.0;
    for (var e = ptr[i]; e < ptr[i + 1u]; e = e + 1u) {
        acc = acc + edge_w[e] * dout[edge_row[e] * p.width + c];
    }
    dtable[t] = dtable[t] + acc;
}
