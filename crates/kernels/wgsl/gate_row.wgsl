// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  adaLN gated residual merge (forward) for [R,D] rows - spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// adaLN gated residual merge (forward) for [R,D] rows. One invocation per element
// (R*D threads), r = idx/D, d = idx%D, condition group k = r/rows_per_cond:
//   y[r,d] = x[r,d] + g[k,d] * h[r,d]
// Gate buffer g[NC,D] (NC = R/rows_per_cond): g[k,d] = g[k*D + d].
// Exactly 4 storage buffers — at the family limit.
// OVERWRITES y (=, SSA fresh buffer).
//
// BACKWARD CONTRACT: dx of this op is the IDENTITY
// (dx = dy element-wise). There is deliberately NO gate_row_dx kernel —
// callers reuse dy directly, or the existing add kernel to accumulate the
// residual gradient. dh/dg are gate_row_dh / gate_row_dg.
//

struct Params {
    R: u32,
    D: u32,
    rows_per_cond: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       g: array<f32>;
@group(0) @binding(3) var<storage, read>       h: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.R * p.D) { return; }
    let r = idx / p.D;
    let d = idx % p.D;
    let k = r / p.rows_per_cond;
    y[idx] = x[idx] + g[k * p.D + d] * h[idx];
}
