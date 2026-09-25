// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS backward, stage 4: per-gaussian segmented reduction over the id-sorted gradient records
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// 3DGS backward, stage 4: per-gaussian segmented reduction over the
// id-sorted gradient records. `ranges` comes from splat_tile_ranges run with
// depth_bits=0 over the sorted keys (segment = gaussian id). One invocation
// per gaussian sums its records channel by channel and hands them on:
// channels [0, npg) become the per-gaussian bundle pgrad[N*npg] the
// projection backward consumes, [col, col+3) are added to the colour
// gradient, `abs` to absgrad, and channels 0 and 1 to sumgrad. The layouts:
//   EWA (splat_bwd_slots, stride 12): npg 10 = {v_xy(2), v_conic(3), v_op,
//     v_rgb(3), v_depth}, col 6, abs 10 - v_depth is dL/d(this gaussian's
//     camera-space z), which splat_project_bwd turns into a world-space mean
//     gradient along the viewing ray;
//   ray (splat_ray_bwd_slots, stride 18): npg 13 = {dL/dm(3), dL/dA(6),
//     dL/dopacity', dL/dn(3)}, col 13, abs 16.
//
// Also accumulates `absgrad[N]`: the sum over pixels of the MAGNITUDE of each
// pixel's 2D position gradient, as opposed to the magnitude of their sum.
// Density control needs the former. A gaussian spanning an edge gets pushed
// one way by the pixels on one side and the other way by the pixels on the
// other; those cancel in the sum, so the usual criterion reads "well fitted"
// for exactly the gaussians that are too big and blurring a detail. Summing
// magnitudes cannot cancel, and the sum also grows with the area a gaussian
// covers. Records are per (tile, gaussian) instance, so the per-pixel
// magnitudes arrive pre-summed in channel `abs`.

struct Params {
    n_gauss: u32,
    stride: u32, // words per record
    npg: u32,    // channels handed on in pgrad, at most 13
    col: u32,    // first colour channel
    abs: u32,    // the AbsGS channel
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       recs:     array<f32>; // n*stride
@group(0) @binding(2) var<storage, read>       vals:     array<u32>; // sorted record idx
@group(0) @binding(3) var<storage, read>       ranges:   array<u32>; // n_gauss*2
@group(0) @binding(4) var<storage, read_write> pgrad:    array<f32>; // N*npg
@group(0) @binding(5) var<storage, read_write> d_colors: array<f32>; // N*3 (+=)
@group(0) @binding(6) var<storage, read_write> absgrad:  array<f32>; // N (+=)
@group(0) @binding(7) var<storage, read_write> sumgrad:  array<f32>; // N*2 (+=)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = gid.y * (nwg.x * 64u) + gid.x;
    if (g >= p.n_gauss) { return; }
    var acc: array<f32, 17>;
    let nch = p.stride - 1u;
    for (var k = 0u; k < 17u; k = k + 1u) { acc[k] = 0.0; }
    let start = ranges[g * 2u];
    let end = ranges[g * 2u + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let r = vals[j] * p.stride;
        for (var k = 0u; k < nch; k = k + 1u) {
            acc[k] = acc[k] + recs[r + k];
        }
    }
    absgrad[g] = absgrad[g] + acc[p.abs];
    // and the summed 2D gradient across views, which is the reference
    // criterion - kept so the two can be compared rather than assumed.
    sumgrad[g * 2u] = sumgrad[g * 2u] + acc[0];
    sumgrad[g * 2u + 1u] = sumgrad[g * 2u + 1u] + acc[1];
    for (var k = 0u; k < p.npg; k = k + 1u) {
        pgrad[g * p.npg + k] = acc[k];
    }
    d_colors[g * 3u] = d_colors[g * 3u] + acc[p.col];
    d_colors[g * 3u + 1u] = d_colors[g * 3u + 1u] + acc[p.col + 1u];
    d_colors[g * 3u + 2u] = d_colors[g * 3u + 2u] + acc[p.col + 2u];
}
