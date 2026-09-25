// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated tiled splat compositing in each pixel's own range order (colour, depth, normal, distortion)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
// @import splat_ray_pair
// @import splat_ray_window
//
// Ray-evaluated splatting, compositing. One invocation per pixel; the 256
// pixels of a 16x16 tile are four consecutive 64-invocation workgroups
// walking the same sorted list, so the loads are coherent - no workgroup
// memory, no barriers. Each pixel composites the list in its own order of
// range along its ray, refined from the list's through a window of its
// RAY_WINDOW nearest pending pairs (`splat_ray_window`).
//
// Each pixel casts its own ray through the lens (`pixel_ray`) and evaluates
// every gaussian at the point of maximum response along it: for mean m (in
// the mid-frame camera), precision A and a ray o + t d,
//
//   t* = d^T A (m - o) / d^T A d,   e = (m - o) - t* d,   q = e^T A e,
//   alpha = min(0.99, opacity' exp(-q / 2)),
//
// computed through the closest-point offset e rather than as
// (m-o)^T A (m-o) - (d^T A (m-o))^2 / d^T A d: the two are equal, but the
// second subtracts two numbers of order (range / sigma)^2 to get one of order
// 1 and loses the gaussian in f32.
//
// Composited per pixel, front to back in that order, with w_i = alpha_i T_i:
//   img   = sum w c + T bg, alpha out = 1 - T;
//   aux   = {expected range sum w t / (1 - T),
//            normal sum w n (unnormalized),
//            depth distortion sum_{j<i} 2 w_i w_j (t_i - t_j)} (2DGS, Huang et
//            al. 2024, the absolute form - exact when the pairs are in range
//            order along the ray, which the window makes them);
//   joined = how many pairs each pixel's window took in before its walk
//            stopped, with a zero after the last pixel - scanned, where each
//            pixel's pairs sit in the backward's log (`splat_ray_bwd_walk`);
//   totals = the walk's sums as it ended {sum w c (3), sum w n (3), sum w t,
//            sum w, distortion, T}, where the backward's second walk starts.

@group(0) @binding(0) var<uniform> p: View;
@group(0) @binding(1) var<storage, read>       ray:    array<vec4<f32>>; // N*4
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted gaussian ids
@group(0) @binding(3) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(4) var<storage, read_write> img:    array<f32>; // W*H*4
@group(0) @binding(5) var<storage, read_write> aux:    array<f32>; // W*H*5
@group(0) @binding(6) var<storage, read_write> joined: array<u32>; // W*H + 1
@group(0) @binding(7) var<storage, read_write> totals: array<f32>; // W*H*10

fn ray_record(g: u32) -> RayRec {
    return ray_rec4(ray[g * 4u], ray[g * 4u + 1u], ray[g * 4u + 2u], ray[g * 4u + 3u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let tiles_x = (p.width + 15u) / 16u;
    let tiles_y = (p.height + 15u) / 16u;
    let tile = idx / 256u;
    if (tile >= tiles_x * tiles_y) { return; }
    let local = idx % 256u;
    let px = (tile % tiles_x) * 16u + local % 16u;
    let py = (tile / tiles_x) * 16u + local / 16u;
    if (px >= p.width || py >= p.height) { return; }
    let pr = pixel_ray(p, vec2<f32>(f32(px) + 0.5, f32(py) + 0.5));
    var w = ray_walk(pr.o, pr.d, pr.ok != 0.0);
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    for (var j = start; j < end && w.live; j = j + 1u) {
        let g = vals[j];
        w = ray_walk_join(w, g, ray_record(g));
    }
    w = ray_walk_drain(w);
    let pix = py * p.width + px;
    joined[pix] = w.win.joined;
    if (pix == 0u) { joined[p.width * p.height] = 0u; }
    let a = 1.0 - w.tr;
    img[pix * 4u] = w.col.x + w.tr * p.bg.x;
    img[pix * 4u + 1u] = w.col.y + w.tr * p.bg.y;
    img[pix * 4u + 2u] = w.col.z + w.tr * p.bg.z;
    img[pix * 4u + 3u] = a;
    var dexp = 0.0;
    if (a > 1e-6) { dexp = w.dsum / a; }
    aux[pix * 5u] = dexp;
    aux[pix * 5u + 1u] = w.nsum.x;
    aux[pix * 5u + 2u] = w.nsum.y;
    aux[pix * 5u + 3u] = w.nsum.z;
    aux[pix * 5u + 4u] = w.dist;
    let f = pix * 10u;
    totals[f] = w.col.x;
    totals[f + 1u] = w.col.y;
    totals[f + 2u] = w.col.z;
    totals[f + 3u] = w.nsum.x;
    totals[f + 4u] = w.nsum.y;
    totals[f + 5u] = w.nsum.z;
    totals[f + 6u] = w.dsum;
    totals[f + 7u] = w.wsum;
    totals[f + 8u] = w.dist;
    totals[f + 9u] = w.tr;
}
