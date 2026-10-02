// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gather rows of two row pools into two contiguous batch slabs, one dispatch for the whole batch
// @how   one thread per output element, row picked by an index buffer
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// A serving engine keeps each resident sequence's recurrent state as one ROW of
// a per-layer pool, and the stateful decode kernels address the batch through
// one flat axis, so a step needs the rows of ITS sequences contiguous. This
// copies them in - `b` rows of two pools (the Gated-DeltaNet state and the
// causal-conv history) - in one dispatch, where a buffer per sequence costs
// one dispatch per sequence per layer.
//
//   out_a[i, :] = pool_a[ids[i], :]    (len_a words per row)
//   out_b[i, :] = pool_b[ids[i], :]    (len_b words per row)
//
//   params : u32 [b, len_a, len_b]
//   pool_a : [rows, len_a], pool_b : [rows, len_b]
//   ids    : [b] u32     pool row of each batch row
//   out_a  : [b, len_a],  out_b  : [b, len_b]
//
// `pool_scatter2` is the inverse. Row ids must be distinct for the scatter.

struct Params {
    b:     u32,
    len_a: u32,
    len_b: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pool_a: array<f32>;
@group(0) @binding(2) var<storage, read>       pool_b: array<f32>;
@group(0) @binding(3) var<storage, read>       ids:    array<u32>;
@group(0) @binding(4) var<storage, read_write> out_a:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out_b:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let width = p.len_a + p.len_b;
    if (idx >= p.b * width) { return; }
    let row = idx / width;
    let c = idx % width;
    let src = ids[row];
    if (c < p.len_a) {
        out_a[row * p.len_a + c] = pool_a[src * p.len_a + c];
    } else {
        let cb = c - p.len_a;
        out_b[row * p.len_b + cb] = pool_b[src * p.len_b + cb];
    }
}
