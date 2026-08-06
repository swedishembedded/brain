// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  LayerNorm per-row mean + inverse-std, one WORKGROUP per row
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// LayerNorm per-row mean + inverse-std, one WORKGROUP per row.
//
//   x: [rows, d]  ->  mean: [rows], inv: [rows]
//   params: d_model, n_rows, eps (f32 bits)   dispatch: n_rows * 64 invocations
//
// The coalesced counterpart of `ln_stats.wgsl` (one thread per row, two serial
// passes over the row). Same reason as `layernorm_rows.wgsl`: one thread per
// row means a warp's loads are `d` floats apart and each fetched sector serves
// one useful float. Feeds `layernorm_dgamma` unchanged — the outputs are the
// same two arrays.
//
// One barrier (the CPU JIT's limit), so mean and variance come from the same
// shifted one-pass reduction `layernorm_rows.wgsl` documents:
// with K = x[row, 0], mean = K + S1/d and var = S2/d - (S1/d)^2.

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> inv:  array<f32>;

var<workgroup> psum:   array<f32, 64>;
var<workgroup> psumsq: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;
    let df = f32(d);

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
    var t1 = 0.0;
    var t2 = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        t1 = t1 + psum[i];
        t2 = t2 + psumsq[i];
    }
    let moff = t1 / df;
    let va = max(t2 / df - moff * moff, 0.0);
    // One thread stores the row's two scalars; the rest are done.
    if (t == 0u) {
        mean[n] = k + moff;
        inv[n] = inverseSqrt(va + p.eps);
    }
}
