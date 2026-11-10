// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training scalar reduction: out[0] = c * sum of x[0..n]
// @how   one 64-thread workgroup, strided partials, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The loss reduction (`sum_weighted`). Params: n, c (f32 bits). Dispatch: 1 workgroup.

struct Params { n: u32, c: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    var acc = 0.0;
    for (var i = t; i < p.n; i = i + 64u) {
        acc = acc + x[i];
    }
    partial[t] = acc;
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[i];
        }
        out[0] = s * p.c;
    }
}
