// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// RMSNorm, one WORKGROUP per row — the decode-regime variant.
//
//   x  : [rows, d]   out: [rows, d]   w: [d]
//   params: d, rows
//
// The per-element kernel (rmsnorm.wgsl) assigns one THREAD per row: at decode
// batch sizes that is 8 threads on a 3840-core card (measured at 16.6% of
// decode time). Here 64 threads cooperate on one row: each accumulates a
// strided partial sum of squares into workgroup memory, ONE barrier, then
// every thread redundantly folds the 64 partials (64 adds — cheaper than a
// second barrier, and the CPU JIT supports exactly one top-level barrier) and
// scales its strided slice of the row.
//
// Dispatch: rows * 64 invocations (one workgroup per row).

struct Params {
    d: u32,
    rows: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
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
    let inv = 1.0 / sqrt(ss / f32(p.d) + 1e-6);
    for (var c = t; c < p.d; c = c + 64u) {
        out[base + c] = x[base + c] * inv * w[c];
    }
}
