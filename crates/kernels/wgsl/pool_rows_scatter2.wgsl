// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Scatter two contiguous batch slabs back into rows of two row pools, one dispatch for the whole batch
// @how   one thread per input element, destination row picked by an index buffer
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The inverse of `pool_rows_gather2.wgsl`: after a decode step has advanced the
// staged state of `b` sequences, return each one to its pool row.
//
//   pool_a[ids[i], :] = in_a[i, :]
//   pool_b[ids[i], :] = in_b[i, :]
//
//   params : u32 [b, len_a, len_b]
//   in_a   : [b, len_a],  in_b : [b, len_b]
//   ids    : [b] u32     pool row of each batch row - DISTINCT, or two threads
//                        write one element and the survivor is unspecified
//   pool_a : [rows, len_a], pool_b : [rows, len_b]
//
// With `b = 1` and a zero slab it is also how a pool row is cleared.

struct Params {
    b:     u32,
    len_a: u32,
    len_b: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       in_a:   array<f32>;
@group(0) @binding(2) var<storage, read>       in_b:   array<f32>;
@group(0) @binding(3) var<storage, read>       ids:    array<u32>;
@group(0) @binding(4) var<storage, read_write> pool_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> pool_b: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let width = p.len_a + p.len_b;
    if (idx >= p.b * width) { return; }
    let row = idx / width;
    let c = idx % width;
    let dst = ids[row];
    if (c < p.len_a) {
        pool_a[dst * p.len_a + c] = in_a[row * p.len_a + c];
    } else {
        let cb = c - p.len_a;
        pool_b[dst * p.len_b + cb] = in_b[row * p.len_b + cb];
    }
}
