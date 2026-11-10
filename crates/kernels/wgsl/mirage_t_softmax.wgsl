// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training row-wise softmax: out[r, c] = exp(x[r, c] - max) / sum
// @how   64-thread workgroup per row, online (max, sum) partials, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Each thread keeps a running (max, scaled sum) pair over its strided slice; the 64 pairs are merged in a fixed order.
// (The shared `softmax_rows` needs two barriers and does not JIT on the CPU backend.)
// Params: rows, d. Dispatch: rows workgroups.

struct Params { rows: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> p_max: array<f32, 64>;
var<workgroup> p_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.d;
    var m = -3.0e38;
    var s = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        let v = x[base + c];
        if (v > m) {
            s = s * exp(m - v) + 1.0;
            m = v;
        } else {
            s = s + exp(v - m);
        }
    }
    p_max[t] = m;
    p_sum[t] = s;
    workgroupBarrier();
    var big = -3.0e38;
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (p_max[i] > big) { big = p_max[i]; }
    }
    var total = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        total = total + p_sum[i] * exp(p_max[i] - big);
    }
    for (var c = t; c < p.d; c = c + 64u) {
        out[base + c] = exp(x[base + c] - big) / total;
    }
}
