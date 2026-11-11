// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  mirage_rms_axpy_norm with a tree reduction over a 128-thread workgroup, for a row of at most 512 columns
// @how   each thread owns four adjacent columns held in registers; the four partial sums fold in a log2(128)-step tree
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The same operator as mirage_rms_axpy_norm (y += a * rmsnorm(x) * w1, h = rmsnorm(y) * w2, the second norm's sum of
// squares expanded over the first pass's four sums), built for a device whose workgroups execute any number of
// barriers. The one-barrier original folds 64 partials of each sum serially in every thread, which at one row is a
// chain of a few hundred dependent shared-memory reads on a single multiprocessor; here every column is read once
// into registers, the sums fold in seven steps, and the update and both stores reuse the registers.
// Contract: d <= 512 and a multiple of 4 (the caller falls back to mirage_rms_axpy_norm otherwise); same params,
// bindings and result as the original up to the reduction order.
// Dispatch: rows * 128 invocations (one workgroup per row).

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

var<workgroup> red: array<vec4<f32>, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let c = 4u * t;
    let live = row < p.rows && c < p.d;
    let at = row * p.d + c;
    var xv = vec4<f32>(0.0);
    var yv = vec4<f32>(0.0);
    var wv = vec4<f32>(0.0);
    var w2v = vec4<f32>(0.0);
    if (live) {
        xv = vec4<f32>(x[at], x[at + 1u], x[at + 2u], x[at + 3u]);
        yv = vec4<f32>(y[at], y[at + 1u], y[at + 2u], y[at + 3u]);
        wv = vec4<f32>(w1[c], w1[c + 1u], w1[c + 2u], w1[c + 3u]);
        w2v = vec4<f32>(w2[c], w2[c + 1u], w2[c + 2u], w2[c + 3u]);
    }
    let u = xv * wv;
    // sum x^2, sum y^2, sum y u, sum u^2
    red[t] = vec4<f32>(dot(xv, xv), dot(yv, yv), dot(yv, u), dot(u, u));
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (t < s) {
            red[t] = red[t] + red[t + s];
        }
        workgroupBarrier();
    }
    let sums = red[0];
    let inv1 = 1.0 / sqrt(sums.x / f32(p.d) + p.eps);
    let ss = sums.y + 2.0 * p.a * inv1 * sums.z + p.a * p.a * inv1 * inv1 * sums.w;
    let inv2 = 1.0 / sqrt(max(ss, 0.0) / f32(p.d) + p.eps);
    if (live) {
        let yn = yv + p.a * (xv * inv1 * wv);
        let hn = yn * inv2 * w2v;
        y[at] = yn.x;
        y[at + 1u] = yn.y;
        y[at + 2u] = yn.z;
        y[at + 3u] = yn.w;
        h[at] = hn.x;
        h[at + 1u] = hn.y;
        h[at + 2u] = hn.z;
        h[at + 3u] = hn.w;
    }
}
