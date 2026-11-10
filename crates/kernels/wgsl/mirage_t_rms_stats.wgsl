// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training RMS-norm row statistics: inv_rms and the correction factor k of the input gradient
// @how   64-thread workgroup per row, strided partials, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// For each row of x [rows, d] with the norm weight w [d] (1 + scale, or ones) and the output gradient g:
//   inv = 1 / sqrt(mean(x^2) + eps),  k = inv^3 * sum_j(w_j g_j x_j) / d.
// stats[2 row] = inv, stats[2 row + 1] = k. Params: d, rows, eps (f32 bits). Dispatch: rows workgroups.

struct Params { d: u32, rows: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       g: array<f32>;
@group(0) @binding(3) var<storage, read>       w: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;

var<workgroup> p_ss: array<f32, 64>;
var<workgroup> p_dot: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.d;
    var ss = 0.0;
    var dot = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        let v = x[base + c];
        ss = ss + v * v;
        dot = dot + w[c] * g[base + c] * v;
    }
    p_ss[t] = ss;
    p_dot[t] = dot;
    workgroupBarrier();
    if (t == 0u) {
        var s_ss = 0.0;
        var s_dot = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s_ss = s_ss + p_ss[i];
            s_dot = s_dot + p_dot[i];
        }
        let inv = 1.0 / sqrt(s_ss / f32(p.d) + p.eps);
        stats[row * 2u] = inv;
        stats[row * 2u + 1u] = inv * inv * inv * s_dot / f32(p.d);
    }
}
