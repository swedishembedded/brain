// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// adaLN gated residual, branch gradient — spec:
// docs/world-models/specs/P1.film.md §4.8. One invocation per element
// (R*D threads), r = idx/D, d = idx%D, k = r/rows_per_cond:
//   dh[r,d] = dy[r,d] * g[k,d]
// Gate buffer g[NC,D]. OVERWRITES dh (=, SSA fresh buffer).
// (dx of gate_row is the identity dy — no kernel; see gate_row.wgsl.)
//

struct Params {
    R: u32,
    D: u32,
    rows_per_cond: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       g:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dh: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.R * p.D) { return; }
    let r = idx / p.D;
    let d = idx % p.D;
    let k = r / p.rows_per_cond;
    dh[idx] = dy[idx] * g[k * p.D + d];
}
