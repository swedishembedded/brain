// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Matmul into a COLUMN TILE of the output:  out[:, n_off : n_off+n_tile] = x · Wᵀ
// where W is bound to a sub-range covering output features [n_off, n_off+n_tile)
// — i.e. rows [n_off, n_off+n_tile) of the [N_full, K] weight. This lets a
// >128MB weight (e.g. the Qwen lm_head over a 151936 vocab) be applied in
// several passes, each binding only its tile while writing the strided column
// slice of the full `out` (which stays small for short sequences). One
// invocation per (row m, tile-column).
//
//   out[row*n_full + n_off + col] = sum_k x[row*K + kk] * w[col*K + kk]

struct Params {
    m: u32,        // rows (B*T)
    k: u32,        // input features
    n_full: u32,   // full output width (vocab) — the row stride of `out`
    n_off: u32,    // first output feature of this tile
    n_tile: u32,   // output features in this tile
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;  // tile rows only
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.m * p.n_tile;
    if (idx >= total) { return; }
    let row = idx / p.n_tile;
    let col = idx % p.n_tile;
    var acc = 0.0;
    let xb = row * p.k;
    let wb = col * p.k;
    for (var kk: u32 = 0u; kk < p.k; kk = kk + 1u) {
        acc = acc + x[xb + kk] * w[wb + kk];
    }
    out[row * p.n_full + p.n_off + col] = acc;
}
