// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of rmsnorm_nogain w.r.t. x, one workgroup per row
// @how   64-thread workgroup per row, strided partial sums of x^2 and dy*x, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of rmsnorm_nogain.wgsl. With r = 1/sqrt(mean(x^2) + eps) and A = sum_c dy_c * x_c:
//   dx_i = r * dy_i - (r^3 * x_i / d) * A
// (rmsnorm_dx_rows with a gain of one.) Takes the forward input `x`; there is no parameter gradient.
//
//   x, dy, dx: [rows, d]   params: d, rows, eps (f32 bits)
// Dispatch: rows * 64 invocations (one workgroup per row).

struct Params {
    d: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

var<workgroup> p_ss: array<f32, 64>;
var<workgroup> p_a: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = row < p.rows;
    let base = row * p.d;
    var ss = 0.0;
    var a = 0.0;
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            let xv = x[base + c];
            ss = ss + xv * xv;
            a = a + dy[base + c] * xv;
        }
    }
    p_ss[t] = ss;
    p_a[t] = a;
    workgroupBarrier();
    var tss = 0.0;
    var ta = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        tss = tss + p_ss[i];
        ta = ta + p_a[i];
    }
    let r = inverseSqrt(tss / f32(p.d) + p.eps);
    let coef = r * r * r * ta / f32(p.d);
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            dx[base + c] = r * dy[base + c] - coef * x[base + c];
        }
    }
}
