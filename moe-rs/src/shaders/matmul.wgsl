// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Generic matmul matching PyTorch nn.Linear (no bias):  out = x @ W^T
//   x : [M, K]   row-major
//   W : [N, K]   row-major  (W[n, k] is weight row n = output feature n)
//   out:[M, N]   row-major,  out[m, n] = sum_k x[m, k] * W[n, k]
//
// Used for every linear stage: qkv, attention-out, router, expert
// gate/up/down, and the (weight-tied) lm_head. One invocation per output
// element; the K-loop is plain fp32 so it runs on any compute capability.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.m * p.n;
    if (idx >= total) { return; }
    let row = idx / p.n;     // m
    let col = idx % p.n;     // n
    let x_base = row * p.k;
    let w_base = col * p.k;
    var acc = 0.0;
    for (var i: u32 = 0u; i < p.k; i = i + 1u) {
        acc = acc + x[x_base + i] * w[w_base + i];
    }
    out[idx] = acc;
}
