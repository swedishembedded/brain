// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  FiLM per-row-group modulation (forward) for [R,D] rows — spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// FiLM per-row-group modulation (forward) for [R,D] rows — spec:
// docs/world-models/specs/P1.film.md §4.4. One invocation per element
// (R*D threads), r = idx/D, d = idx%D, condition group k = r/rows_per_cond:
//   y[r,d] = x[r,d] * (1 + s[k,d]) + b[k,d]
// s,b packed in ONE buffer sb[NC,2D] (NC = R/rows_per_cond, divisibility
// assumed — enforced by the host FilmRowDims::new), scale first per group:
//   s[k,d] = sb[k*2D + d],  b[k,d] = sb[k*2D + D + d].
// rows_per_cond = tokens-per-frame gives the per-frame diffusion-forcing
// conditioning path; rows_per_cond = R is whole-sequence adaLN.
// OVERWRITES y (=, SSA fresh buffer).
//

struct Params {
    R: u32,
    D: u32,
    rows_per_cond: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       sb: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.R * p.D) { return; }
    let r = idx / p.D;
    let d = idx % p.D;
    let k = r / p.rows_per_cond;
    let s = sb[k * 2u * p.D + d];
    let b = sb[k * 2u * p.D + p.D + d];
    y[idx] = x[idx] * (1.0 + s) + b;
}
