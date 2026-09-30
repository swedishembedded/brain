// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of a column-tiled matmul w.r.t. its input
// @how   one thread per input element, serial reduction over the tile
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Backward of `matmul_tile` (out[:, n_off : n_off+n_tile] = x · Wᵀ) w.r.t. x,
// for the tile's own columns of the output gradient:
//   dX[m, k] (+)= sum_{j < n_tile} dY[m, n_off + j] * W[j, k]
// W is bound to the tile's rows only ([n_tile, K], a sub-range of the
// [N_full, K] weight) and dY is the full `[M, N_full]` gradient, so a weight
// beyond one storage binding (a 152k-token vocabulary head at d = 3584) is
// differentiated in several passes, each binding only its tile.
// `accumulate` selects overwrite (0, the first tile) or add (1, the rest).

struct Params {
    m: u32,
    k: u32,
    n_full: u32,   // the row stride of dY
    n_off: u32,    // first output feature of this tile
    n_tile: u32,   // output features in this tile
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;  // tile rows only
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.m * p.k) { return; }
    let row = idx / p.k;
    let col = idx % p.k;
    var acc = 0.0;
    let db = row * p.n_full + p.n_off;
    for (var j: u32 = 0u; j < p.n_tile; j = j + 1u) {
        acc = acc + dy[db + j] * w[j * p.k + col];
    }
    if (p.accumulate != 0u) {
        dx[idx] = dx[idx] + acc;
    } else {
        dx[idx] = acc;
    }
}
