// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training row dot product: dst[r] (+)= sum over c of a[r, c] * b[r, c]
// @how   64-thread workgroup per row, strided partials, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Forward of `row_dot` (acc 0 assigns) and the scale gradient of `row_mul` (acc 1 accumulates). The 64 partial
// sums are folded in a fixed order, so the result is deterministic.
// Params: rows, d, acc. Dispatch: rows workgroups.

struct Params { rows: u32, d: u32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

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
        acc = acc + a[base + c] * b[base + c];
    }
    partial[t] = acc;
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[i];
        }
        if (p.acc != 0u) {
            dst[row] = dst[row] + s;
        } else {
            dst[row] = s;
        }
    }
}
