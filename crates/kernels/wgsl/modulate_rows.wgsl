// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  adaLN modulation from an indexed table: y = x * (1 + scale[idx[r]]) + shift[idx[r]]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// adaLN modulation (forward) for [R,D] rows whose scale and shift come from a
// small table of distinct rows: row r reads table row idx[r]. One invocation
// per element (R*D threads), r = i/D, d = i%D:
//   y[r,d] = x[r,d] * (1 + scale[idx[r]*D + d]) + shift[idx[r]*D + d]
// The tables hold one row per distinct (timestep, modality), so the per-token
// scale/shift never exist as [R,D] buffers - the gather, the multiply and the
// two adds of the unfused form are one pass over x.
// OVERWRITES y (=, SSA fresh buffer).

struct Params {
    R: u32,
    D: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       scale: array<f32>;
@group(0) @binding(3) var<storage, read>       shift: array<f32>;
@group(0) @binding(4) var<storage, read>       idx:   array<u32>;
@group(0) @binding(5) var<storage, read_write> y:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.R * p.D) { return; }
    let r = i / p.D;
    let d = i % p.D;
    let t = idx[r] * p.D + d;
    y[i] = x[i] * (1.0 + scale[t]) + shift[t];
}
