// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated residual update from a normalised branch fused with the next norm: y += a * rmsnorm(x) * w1, h = rmsnorm(y) * w2
// @how   64-thread workgroup tile, 1 barrier: the second norm's sum of squares is expanded over the first pass's partials
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// mirage_rms_axpy followed by rmsnorm_rows on the updated row, in one dispatch. The updated row's sum of squares needs
// the first norm's scale before it can be formed, which would cost a second barrier (the CPU interpreter takes one
// per kernel); but with u = x * w1 and inv1 the first norm's scale,
//   sum (y + a inv1 u)^2 = sum y^2 + 2 a inv1 sum (y u) + a^2 inv1^2 sum u^2,
// so one strided pass accumulates sum x^2, sum y^2, sum y u and sum u^2 together, ONE barrier folds all four, and
// every thread then updates its columns and normalises them. The expansion is exact algebra; its fp32 result
// differs from summing the updated row by rounding only (the terms are sums of squares and one cross term of
// the same magnitude).
//
//   x : [rows, d] branch output   y: [rows, d] residual (read-modify-write)   h: [rows, d] normalised updated residual
//   w1, w2: [d] (1 + scale)       params: d, rows, eps (f32 bits), a (f32 bits, the branch gate)
// Dispatch: rows * 64 invocations (one workgroup per row).

struct Params {
    d: u32,
    rows: u32,
    eps: f32,
    a: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       w1: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:  array<f32>;
@group(0) @binding(4) var<storage, read>       w2: array<f32>;
@group(0) @binding(5) var<storage, read_write> h:  array<f32>;

var<workgroup> pxx: array<f32, 64>;
var<workgroup> pyy: array<f32, 64>;
var<workgroup> pyu: array<f32, 64>;
var<workgroup> puu: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.d;
    var axx = 0.0;
    var ayy = 0.0;
    var ayu = 0.0;
    var auu = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        let xv = x[base + c];
        let yv = y[base + c];
        let u = xv * w1[c];
        axx = axx + xv * xv;
        ayy = ayy + yv * yv;
        ayu = ayu + yv * u;
        auu = auu + u * u;
    }
    pxx[t] = axx;
    pyy[t] = ayy;
    pyu[t] = ayu;
    puu[t] = auu;
    workgroupBarrier();
    var sxx = 0.0;
    var syy = 0.0;
    var syu = 0.0;
    var suu = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        sxx = sxx + pxx[i];
        syy = syy + pyy[i];
        syu = syu + pyu[i];
        suu = suu + puu[i];
    }
    let inv1 = 1.0 / sqrt(sxx / f32(p.d) + p.eps);
    let ss = syy + 2.0 * p.a * inv1 * syu + p.a * p.a * inv1 * inv1 * suu;
    let inv2 = 1.0 / sqrt(max(ss, 0.0) / f32(p.d) + p.eps);
    for (var c = t; c < p.d; c = c + 64u) {
        let yn = y[base + c] + p.a * (x[base + c] * inv1 * w1[c]);
        y[base + c] = yn;
        h[base + c] = yn * inv2 * w2[c];
    }
}
