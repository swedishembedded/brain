// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  LayerNorm forward, one WORKGROUP per row - the coalesced variant
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// LayerNorm forward, one WORKGROUP per row — the coalesced variant.
//
//   x: [rows, d]   gamma,beta: [d]   out: [rows, d]
//   params: d_model, n_rows, eps (f32 bits)   dispatch: n_rows * 64 invocations
//
// Same math as `layernorm.wgsl` (torch.nn.LayerNorm, biased/population
// variance); only the thread mapping changes.
//
// WHY: the per-element kernel gives thread t row t, so a warp's 32 loads are
// `d` floats apart and each 32-byte sector fetched serves ONE useful float:
// eight-way read and write amplification, and the kernel runs at a small fraction of
// memory bandwidth no matter how many rows there are. That is the same
// coalescing bug `rmsnorm_rows.wgsl` fixes (measured an order of magnitude for
// QK-norm); here the 64 threads of a workgroup walk one row with stride 64, so
// every fetch is fully used.
//
// ONE barrier, deliberately: the CPU JIT (`wgsl-cpu`) splits a kernel body at
// exactly one top-level `workgroupBarrier()`, so the textbook two-pass
// (mean, then sum of squared deviations) is not available. Instead this uses
// the **shifted** one-pass form, which is numerically sound in a way that the
// naive `E[x^2] - E[x]^2` is not: with K = x[row, 0] as the shift,
//
//     S1 = sum(x - K),  S2 = sum((x - K)^2)
//     mean = K + S1/d,  var = S2/d - (S1/d)^2
//
// the cancellation in `S2/d - (S1/d)^2` is bounded by how far K sits from the
// mean (a fraction of a standard deviation for activations), not by the row's
// absolute magnitude. Agreement with the two-pass kernel is checked in
// `brain-gpu-core`'s `bench_layernorm`.
//
// @workgroup_size(64) — the engine's rule, and enough here: a row-reduction
// only needs its threads to cooperate, not to fill a 256-wide tile. 64 is at
// or below every device's `DeviceCaps::max_workgroup_size` (256 is the WebGPU
// floor), so selecting this kernel needs no size gate at all — only
// `caps.workgroup_reductions`, which is what `backend_api::select` checks.

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       beta:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;

var<workgroup> psum:  array<f32, 64>;
var<workgroup> psumsq: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let n = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;
    let df = f32(d);

    // Shift by the row's first element (a broadcast read: same address for
    // every thread in the workgroup, so it costs one sector).
    let k = x[base];
    var s1 = 0.0;
    var s2 = 0.0;
    for (var c = t; c < d; c = c + 64u) {
        let v = x[base + c] - k;
        s1 = s1 + v;
        s2 = s2 + v * v;
    }
    psum[t] = s1;
    psumsq[t] = s2;
    workgroupBarrier();
    // Every thread redundantly folds the 64 partials (128 adds — cheaper than
    // a second barrier, and the CPU JIT allows only the one above).
    var t1 = 0.0;
    var t2 = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        t1 = t1 + psum[i];
        t2 = t2 + psumsq[i];
    }
    let moff = t1 / df;              // mean - k
    let va = max(t2 / df - moff * moff, 0.0);
    let inv = inverseSqrt(va + p.eps);
    for (var c = t; c < d; c = c + 64u) {
        // (x - mean) == (x - k) - moff: keeps the subtraction in the shifted
        // frame, where it is exact.
        out[base + c] = ((x[base + c] - k) - moff) * inv * gamma[c] + beta[c];
    }
}
