// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row-wise argmax, PARTIAL stage
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Row-wise argmax, PARTIAL stage: each thread reduces one chunk of one row.
//
//   x   : [M, N] row-major
//   part: [M, P, 2]  — per (row, chunk): [best value, best index (as f32)]
//
// Splits each row into P chunks so a 32k-vocab row is reduced by P threads
// instead of one — the single-thread argmax_row was 10% of decode time. The
// two-dispatch split (this + argmax_final) is the engine's standard reduction
// idiom (max_abs_part -> max_abs_final): no atomics, no barriers, both
// backends trivially correct. Ties break to the LOWEST index at both stages.
//
// params: m, n, p (chunks per row), chunk (elements per chunk)

struct Params {
    m: u32,
    n: u32,
    p: u32,
    chunk: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    let total = pr.m * pr.p;
    if (t >= total) { return; }
    let row = t / pr.p;
    let c = t % pr.p;
    let start = c * pr.chunk;
    let end = min(start + pr.chunk, pr.n);
    var best_n = start;
    var best_v = -3.402823e38;
    for (var i = start; i < end; i = i + 1u) {
        let v = x[row * pr.n + i];
        if (v > best_v) {
            best_v = v;
            best_n = i;
        }
    }
    part[2u * t] = best_v;
    part[2u * t + 1u] = f32(best_n);
}
