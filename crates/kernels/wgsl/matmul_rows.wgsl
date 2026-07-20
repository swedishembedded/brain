// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Row-blocked matmul, same contract as matmul.wgsl:  out = x @ W^T
//   x : [M, K],  W : [N, K],  out : [M, N]   (all row-major)
//
// One thread computes out[r0 .. r0+8, col] — the weight row `col` is loaded
// ONCE per 8 output rows instead of once per row. The naive kernel streams
// the whole [N, K] weight for every row of x (e.g. 16 MB × 1376 rows ≈ 22 GB
// per ViT MLP layer); this cuts that memory traffic 8×, which is what
// dominates large-transformer forwards on the CPU backend. Per-output
// accumulation order (sequential k) is unchanged → bit-identical to matmul.
//
// Dispatch ceil(M/8) * N threads.

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let blocks = (p.m + 7u) / 8u;
    if (gidx >= blocks * p.n) { return; }
    let col = gidx % p.n;
    let r0 = (gidx / p.n) * 8u;
    let rows = min(8u, p.m - r0);
    var acc: array<f32, 8>;
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
        acc[r] = 0.0;
    }
    let w_base = col * p.k;
    for (var i: u32 = 0u; i < p.k; i = i + 1u) {
        let wv = w[w_base + i];
        for (var r: u32 = 0u; r < rows; r = r + 1u) {
            acc[r] = acc[r] + x[(r0 + r) * p.k + i] * wv;
        }
    }
    for (var r: u32 = 0u; r < rows; r = r + 1u) {
        out[(r0 + r) * p.n + col] = acc[r];
    }
}
