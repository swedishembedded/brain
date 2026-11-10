// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Residual update from a normalised branch: y += a * rmsnorm(x) * w, one workgroup per row
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The decode step's post-norm of an attention or cross-attention branch and the gated residual add that follows
// it ran as two dispatches around a row-sized round trip through global memory; at decode batch sizes both are
// latency bound, so folding the add into the norm removes a launch and a buffer per use.
//
//   x  : [rows, d]   y: [rows, d] (read-modify-write)   w: [d] (the norm weight, 1 + scale)
//   params: d, rows, eps (f32 bits), a (f32 bits, the branch gate)
//
// Reduction as rmsnorm_rows: strided partial sums of squares, ONE barrier, every thread folds the 64 partials.
// Dispatch: rows * 64 invocations (one workgroup per row).

struct Params {
    d: u32,
    rows: u32,
    eps: f32,
    a: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

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
        let v = x[base + c];
        acc = acc + v * v;
    }
    partial[t] = acc;
    workgroupBarrier();
    var ss = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        ss = ss + partial[i];
    }
    let inv = 1.0 / sqrt(ss / f32(p.d) + p.eps);
    for (var c = t; c < p.d; c = c + 64u) {
        y[base + c] = y[base + c] + p.a * (x[base + c] * inv * w[c]);
    }
}
