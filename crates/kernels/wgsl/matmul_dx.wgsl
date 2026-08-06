// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of  out = x @ W^T  w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
//
// Backward of  out = x @ W^T  w.r.t. x:
//   dX[m, k] = sum_n dY[m, n] * W[n, k]
// W is [N, K] row-major (same layout as the forward weight). One invocation per
// (m, k). `accumulate` selects overwrite (0) or add (1) so dX can collect
// contributions from several matmuls (e.g. the shared MoE input).

struct Params {
    m: u32,
    k: u32,
    n: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.m * p.k;
    if (idx >= total) { return; }
    let row = idx / p.k;   // m
    let col = idx % p.k;   // k
    var acc = 0.0;
    for (var nn: u32 = 0u; nn < p.n; nn = nn + 1u) {
        acc = acc + dy[row * p.n + nn] * w[nn * p.k + col];
    }
    if (p.accumulate == 0u) { dx[idx] = acc; }
    else                    { dx[idx] = dx[idx] + acc; }
}
