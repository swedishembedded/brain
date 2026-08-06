// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm, one WORKGROUP per row — the decode-regime variant
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// RMSNorm, one WORKGROUP per row — the decode-regime variant.
//
//   x  : [rows, d]   out: [rows, d]   w: [d]
//   params: d, rows, eps (f32 bits — the runtime epsilon `rmsnorm_eps` takes)
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
//
// This is the coalescing fix, not just a decode-regime fix. The per-element
// kernel gives thread t row t, so a warp's 32 loads are `d` floats apart: each
// 32-byte sector fetched serves ONE useful float. Here the 64 threads of a
// workgroup walk one row with stride 64, so every fetch is fully used.
// Measured on a Tesla P40 (fp32, this kernel vs rmsnorm_eps, same total
// elements) — the cooperative variant wins at EVERY row width, not only at
// decode row counts:
//
//     rows      d    per-element   this kernel   speedup
//    36864    128       3.85 ms       0.20 ms      19.4x
//    18432    256       4.64 ms       0.21 ms      22.6x
//     4608   1024       0.73 ms       0.29 ms       2.5x
//     1536   3072       1.27 ms       0.30 ms       4.2x
//      512   9216       3.02 ms       0.27 ms      11.2x
//
// (agreement: max_abs 3.3e-6 — the reduction order differs, the math does not.)

struct Params {
    d: u32,
    rows: u32,
    eps: f32,
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
    let inv = 1.0 / sqrt(ss / f32(p.d) + p.eps);
    for (var c = t; c < p.d; c = c + 64u) {
        out[base + c] = x[base + c] * inv * w[c];
    }
}
