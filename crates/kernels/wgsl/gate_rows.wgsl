// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  adaLN gated residual merge from an indexed gate table: y = x + g[idx[r]] * h
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// gate_row's merge for a gate that varies per row through a table of
// distinct rows: row r reads gate row idx[r]. One invocation per element
// (R*D threads), r = i/D, d = i%D:
//   y[r,d] = x[r,d] + g[idx[r]*D + d] * h[r,d]
// The per-token gate never exists as an [R,D] buffer.
// OVERWRITES y (=, SSA fresh buffer).

struct Params {
    R: u32,
    D: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       g:   array<f32>;
@group(0) @binding(3) var<storage, read>       idx: array<u32>;
@group(0) @binding(4) var<storage, read>       h:   array<f32>;
@group(0) @binding(5) var<storage, read_write> y:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.R * p.D) { return; }
    let r = i / p.D;
    let d = i % p.D;
    y[i] = x[i] + g[idx[r] * p.D + d] * h[i];
}
