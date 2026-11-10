// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training softmax backward: dx[r, c] += p[r, c] * (dy[r, c] - sum_j p[r, j] dy[r, j])
// @how   64-thread workgroup per row, strided partials, 1 barrier; accumulates
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// p is the softmax output. Params: rows, d. Dispatch: rows workgroups.

struct Params { rows: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pr: array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.d;
    var acc = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        acc = acc + pr[base + c] * dy[base + c];
    }
    partial[t] = acc;
    workgroupBarrier();
    var dot = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        dot = dot + partial[i];
    }
    for (var c = t; c < p.d; c = c + 64u) {
        dx[base + c] = dx[base + c] + pr[base + c] * (dy[base + c] - dot);
    }
}
