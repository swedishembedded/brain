// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Stage one of the LayerNorm gamma gradient: per row chunk and column, the partial sum of dy * xhat
// @how   one thread per (chunk, column), rows strided by the chunk count
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// part[c, col] = sum over rows r = c, c + P, c + 2P, ... of
//                dy[r, col] * (x[r, col] - mean[r]) * inv[r]
// `bias_grad_final` folds the P partial rows into dgamma (accumulating). The
// one-stage `layernorm_dgamma` walks every row with one thread per column:
// on a batch of tens of thousands of rows and a few dozen columns that is a
// few dozen threads doing all the work. Rows are strided so that the threads
// of one chunk read consecutive columns of the same row.

struct Params {
    n_rows: u32,
    d: u32,
    P: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       x:    array<f32>;
@group(0) @binding(3) var<storage, read>       mean: array<f32>;
@group(0) @binding(4) var<storage, read>       inv:  array<f32>;
@group(0) @binding(5) var<storage, read_write> part: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.d * p.P) { return; }
    let col = gidx % p.d;
    let chunk = gidx / p.d;
    var acc = 0.0;
    for (var r = chunk; r < p.n_rows; r = r + p.P) {
        acc = acc + dy[r * p.d + col] * (x[r * p.d + col] - mean[r]) * inv[r];
    }
    part[chunk * p.d + col] = acc;
}
