// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-pixel weighted accumulate: acc[c,h,w] += x[c,h,w] * weight[h,w]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The blended-overlap tile compositor: acc[i] += x[i] * weight[i % hw], where
// weight holds one value per PIXEL (length hw) broadcast across every channel
// of a row-major [C, H, W] tensor - imaging::tiling's blended TilePlan variant
// accumulates each tile's masked contribution into a zeroed canvas this way,
// then divides by the summed weight once every tile has landed (a separate
// pass; this kernel never divides). total = C*hw.
//   i = c*hw + p ;  acc[i] += x[i] * weight[p]
//

struct Params {
    total: u32,
    hw: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    acc[i] += x[i] * weight[i % p.hw];
}
