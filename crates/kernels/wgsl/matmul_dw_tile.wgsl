// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of a column-tiled matmul w.r.t. its weight tile
// @how   one thread per weight-gradient element, serial reduction over rows
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Backward of `matmul_tile` (out[:, n_off : n_off+n_tile] = x · Wᵀ) w.r.t. the
// weight rows of the tile, for the tile's own columns of the output gradient:
//   dW[j, k] += sum_m dY[m, n_off + j] * X[m, k]
// dW is bound to the tile's rows only ([n_tile, K], a sub-range of the
// [N_full, K] weight gradient) and dY is the full `[M, N_full]` gradient.
// Accumulates, like `matmul_dw` (the gradient buffer is zeroed once before
// the backward pass, which also lets a tied embedding collect both the head's
// and the embedding's contributions).

struct Params {
    m: u32,
    k: u32,
    n_full: u32,   // the row stride of dY
    n_off: u32,    // first output feature of this tile
    n_tile: u32,   // output features in this tile
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;  // tile rows only

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n_tile * p.k) { return; }
    let j = idx / p.k;
    let col = idx % p.k;
    var acc = 0.0;
    for (var mm: u32 = 0u; mm < p.m; mm = mm + 1u) {
        acc = acc + dy[mm * p.n_full + p.n_off + j] * x[mm * p.k + col];
    }
    dw[idx] = dw[idx] + acc;
}
