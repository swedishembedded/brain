// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm, one workgroup per row, for rows of at most 2048 columns, with a tree reduction
// @how   128 threads, four adjacent columns per load, up to four loads per thread held in registers, log2(128)-step fold
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The operator of rmsnorm_rows (x * rsqrt(mean(x^2) + eps) * w) for a device whose workgroups execute any number of
// barriers. rmsnorm_rows has every thread fold the 64 partials serially after its one barrier and reads the row twice;
// here a thread reads its columns once into registers (the weights are requested together with them), the sum of
// squares folds in seven steps, and the same registers are scaled and stored.
// Contract: d a multiple of 4 and at most 2048; the caller keeps rmsnorm_rows for anything else. Same params and
// bindings; the result differs from rmsnorm_rows by the order of the reduction only.
//
//   x : [rows, d]   out: [rows, d]   w: [d]   params: d, rows, eps (f32 bits)
// Dispatch: rows * 128 invocations (one workgroup per row).

struct Params {
    d: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = row < p.rows;
    let base = row * p.d;
    let d4 = p.d / 4u;
    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0);
    var w0 = vec4<f32>(0.0);
    var w1 = vec4<f32>(0.0);
    var w2 = vec4<f32>(0.0);
    var w3 = vec4<f32>(0.0);
    if (live) {
        var c = 4u * t;
        if (t < d4) {
            v0 = vec4<f32>(x[base + c], x[base + c + 1u], x[base + c + 2u], x[base + c + 3u]);
            w0 = vec4<f32>(w[c], w[c + 1u], w[c + 2u], w[c + 3u]);
        }
        c = 4u * (t + 128u);
        if (t + 128u < d4) {
            v1 = vec4<f32>(x[base + c], x[base + c + 1u], x[base + c + 2u], x[base + c + 3u]);
            w1 = vec4<f32>(w[c], w[c + 1u], w[c + 2u], w[c + 3u]);
        }
        c = 4u * (t + 256u);
        if (t + 256u < d4) {
            v2 = vec4<f32>(x[base + c], x[base + c + 1u], x[base + c + 2u], x[base + c + 3u]);
            w2 = vec4<f32>(w[c], w[c + 1u], w[c + 2u], w[c + 3u]);
        }
        c = 4u * (t + 384u);
        if (t + 384u < d4) {
            v3 = vec4<f32>(x[base + c], x[base + c + 1u], x[base + c + 2u], x[base + c + 3u]);
            w3 = vec4<f32>(w[c], w[c + 1u], w[c + 2u], w[c + 3u]);
        }
    }
    red[t] = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3);
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (t < s) {
            red[t] = red[t] + red[t + s];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] / f32(p.d) + p.eps);
    if (live) {
        var c = 4u * t;
        if (t < d4) {
            let o = v0 * inv * w0;
            out[base + c] = o.x; out[base + c + 1u] = o.y; out[base + c + 2u] = o.z; out[base + c + 3u] = o.w;
        }
        c = 4u * (t + 128u);
        if (t + 128u < d4) {
            let o = v1 * inv * w1;
            out[base + c] = o.x; out[base + c + 1u] = o.y; out[base + c + 2u] = o.z; out[base + c + 3u] = o.w;
        }
        c = 4u * (t + 256u);
        if (t + 256u < d4) {
            let o = v2 * inv * w2;
            out[base + c] = o.x; out[base + c + 1u] = o.y; out[base + c + 2u] = o.z; out[base + c + 3u] = o.w;
        }
        c = 4u * (t + 384u);
        if (t + 384u < d4) {
            let o = v3 * inv * w3;
            out[base + c] = o.x; out[base + c + 1u] = o.y; out[base + c + 2u] = o.z; out[base + c + 3u] = o.w;
        }
    }
}
