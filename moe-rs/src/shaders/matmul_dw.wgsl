// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Backward of  out = x @ W^T  w.r.t. W:
//   dW[n, k] += sum_m dY[m, n] * X[m, k]
// Accumulates (the weight-grad buffer is zeroed once before the backward pass),
// which also lets the tied embedding collect both the lm_head and embedding
// contributions. One invocation per (n, k).

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.n * p.k;
    if (idx >= total) { return; }
    let nn = idx / p.k;   // n
    let col = idx % p.k;  // k
    var acc = 0.0;
    for (var mm: u32 = 0u; mm < p.m; mm = mm + 1u) {
        acc = acc + dy[mm * p.n + nn] * x[mm * p.k + col];
    }
    dw[idx] = dw[idx] + acc;
}
