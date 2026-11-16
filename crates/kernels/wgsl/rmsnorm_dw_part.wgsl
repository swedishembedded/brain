// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm gain-weight gradient, stage 1: per-chunk partial sums
// @how   one thread per (row chunk, channel), serial over the chunk's rows
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The first stage of the two-stage gain gradient for many rows:
//   part[j, c] = sum_{n in chunk j} dY[n,c] * x[n,c] * inv[n]
// over chunks of `chunk` rows. `rmsnorm_dw_sum` adds the partials into dW. One
// thread per channel (`rmsnorm_dw`) walks every row serially, which leaves a
// card idle at a training batch's row count; chunking spreads the rows over
// `chunks * d_model` threads with the same reads.

struct Params {
    d_model: u32,
    n_rows: u32,
    chunk: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       x:    array<f32>;
@group(0) @binding(3) var<storage, read>       inv:  array<f32>;
@group(0) @binding(4) var<storage, read_write> part: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let chunks = (p.n_rows + p.chunk - 1u) / p.chunk;
    if (idx >= chunks * p.d_model) { return; }
    let j = idx / p.d_model;
    let c = idx % p.d_model;
    let n0 = j * p.chunk;
    let n1 = min(n0 + p.chunk, p.n_rows);
    var acc = 0.0;
    for (var n: u32 = n0; n < n1; n = n + 1u) {
        acc = acc + dy[n * p.d_model + c] * x[n * p.d_model + c] * inv[n];
    }
    part[idx] = acc;
}
