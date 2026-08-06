// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS backward, stage 4: per-gaussian segmented reduction over the id-sorted gradient records
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// 3DGS backward, stage 4: per-gaussian segmented reduction over the
// id-sorted gradient records. `ranges` comes from splat_tile_ranges run with
// depth_bits=0 over the sorted keys (segment = gaussian id). Outputs the
// per-gaussian 2D gradient bundle pgrad[N*9] = {v_xy(2), v_conic(3), v_op,
// v_rgb(3)} and the color grads directly. One invocation per gaussian.

struct Params {
    n_gauss: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       recs:     array<f32>; // n*10
@group(0) @binding(2) var<storage, read>       vals:     array<u32>; // sorted record idx
@group(0) @binding(3) var<storage, read>       ranges:   array<u32>; // n_gauss*2
@group(0) @binding(4) var<storage, read_write> pgrad:    array<f32>; // N*9
@group(0) @binding(5) var<storage, read_write> d_colors: array<f32>; // N*3 (+=)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = gid.y * (nwg.x * 64u) + gid.x;
    if (g >= p.n_gauss) { return; }
    var acc: array<f32, 9>;
    for (var k = 0u; k < 9u; k = k + 1u) { acc[k] = 0.0; }
    let start = ranges[g * 2u];
    let end = ranges[g * 2u + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let r = vals[j] * 10u;
        for (var k = 0u; k < 9u; k = k + 1u) {
            acc[k] = acc[k] + recs[r + k];
        }
    }
    for (var k = 0u; k < 9u; k = k + 1u) {
        pgrad[g * 9u + k] = acc[k];
    }
    d_colors[g * 3u] = d_colors[g * 3u] + acc[6];
    d_colors[g * 3u + 1u] = d_colors[g * 3u + 1u] + acc[7];
    d_colors[g * 3u + 2u] = d_colors[g * 3u + 2u] + acc[8];
}
