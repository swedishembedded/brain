// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated tiled splat compositing (colour, depth, normal, distortion)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
//
// Ray-evaluated splatting, compositing. One invocation per pixel; the 256
// pixels of a 16x16 tile are four consecutive 64-invocation workgroups
// walking the same depth-sorted list, so the loads are coherent - no
// workgroup memory, no barriers.
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
// Composited per pixel, front to back in list order, with w_i = alpha_i T_i:
//   img   = sum w c + T bg, alpha out = 1 - T;
//   aux   = {expected range sum w t / (1 - T),
//            normal sum w n (unnormalized),
//            depth distortion sum_{j<i} 2 w_i w_j (t_i - t_j)} (2DGS, Huang et
//            al. 2024, the absolute form - exact when the list is in range
//            order along the ray, which the sort keys make it).

@group(0) @binding(0) var<uniform> p: View;
@group(0) @binding(1) var<storage, read>       ray:    array<f32>; // N*16
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted gaussian ids
@group(0) @binding(3) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(4) var<storage, read_write> img:    array<f32>; // W*H*4
@group(0) @binding(5) var<storage, read_write> aux:    array<f32>; // W*H*5

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
    let d = pr.d;

    var tr = 1.0;
    var col = vec3<f32>(0.0, 0.0, 0.0);
    var nsum = vec3<f32>(0.0, 0.0, 0.0);
    var dsum = 0.0;
    var wsum = 0.0;
    var dist = 0.0;
    if (pr.ok != 0.0) {
        let start = ranges[tile * 2u];
        let end = ranges[tile * 2u + 1u];
        for (var j = start; j < end; j = j + 1u) {
            let g = vals[j];
            let r = g * 16u;
            let dm = vec3<f32>(ray[r], ray[r + 1u], ray[r + 2u]) - pr.o;
            let a00 = ray[r + 3u];
            let a11 = ray[r + 4u];
            let a22 = ray[r + 5u];
            let a01 = ray[r + 6u];
            let a02 = ray[r + 7u];
            let a12 = ray[r + 8u];
            let ad = vec3<f32>(a00 * d.x + a01 * d.y + a02 * d.z,
                               a01 * d.x + a11 * d.y + a12 * d.z,
                               a02 * d.x + a12 * d.y + a22 * d.z);
            let vv = dot(d, ad);
            if (vv <= 0.0) { continue; }
            let t = dot(dm, ad) / vv;
            if (t <= 0.0) { continue; }
            let e = dm - t * d;
            let ae = vec3<f32>(a00 * e.x + a01 * e.y + a02 * e.z,
                               a01 * e.x + a11 * e.y + a12 * e.z,
                               a02 * e.x + a12 * e.y + a22 * e.z);
            let q = dot(e, ae);
            let alpha = min(0.99, ray[r + 12u] * exp(-0.5 * q));
            if (alpha < 1.0 / 255.0) { continue; }
            let next_t = tr * (1.0 - alpha);
            if (next_t <= 1e-4) {
                // the terminating gaussian is excluded (gsplat semantics)
                break;
            }
            let w = alpha * tr;
            col = col + w * vec3<f32>(ray[r + 13u], ray[r + 14u], ray[r + 15u]);
            nsum = nsum + w * vec3<f32>(ray[r + 9u], ray[r + 10u], ray[r + 11u]);
            dist = dist + 2.0 * w * (t * wsum - dsum);
            dsum = dsum + w * t;
            wsum = wsum + w;
            tr = next_t;
        }
    }
    let pix = py * p.width + px;
    let a = 1.0 - tr;
    img[pix * 4u] = col.x + tr * p.bg.x;
    img[pix * 4u + 1u] = col.y + tr * p.bg.y;
    img[pix * 4u + 2u] = col.z + tr * p.bg.z;
    img[pix * 4u + 3u] = a;
    var dexp = 0.0;
    if (a > 1e-6) { dexp = dsum / a; }
    aux[pix * 5u] = dexp;
    aux[pix * 5u + 1u] = nsum.x;
    aux[pix * 5u + 2u] = nsum.y;
    aux[pix * 5u + 3u] = nsum.z;
    aux[pix * 5u + 4u] = dist;
}
