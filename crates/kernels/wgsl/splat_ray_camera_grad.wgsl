// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated splat backward: camera pose, rolling shutter and lens gradients
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
//
// Turns the per-pixel ray gradients `splat_ray_bwd_slots` wrote (dray =
// dL/d origin, dL/d direction, in the mid-frame camera) into gradients of
// everything that decides where the rays are, as per-workgroup partial sums
// the host adds up (28 floats per workgroup):
//
//   [0..3)   dL/d omega, the camera's rotation in its own frame
//   [3..6)   dL/d tau, its translation in its own frame
//   [6..9)   dL/d rs_w, the rolling-shutter angular velocity
//   [9..12)  dL/d rs_v, the rolling-shutter linear velocity
//   [12..28) dL/d calibration, in `camera::Intrinsics::params` order
//
// Moving the camera by (omega, tau) moves every ray the same way the scene
// would move the opposite way: o -> o + omega x o + tau, d -> d + omega x d,
// so dL/d omega = sum o x g_o + d x g_d and dL/d tau = sum g_o. Colour,
// alpha, range and distortion are scalars and that is all of them; the
// composited normal is a vector IN the camera frame, which turns by -omega
// when the camera turns by omega, adding sum g_N x N. The lens
// enters through each pixel's unit ray, whose motion under the calibration at
// a FIXED pixel is the implicit derivative `lens_ray_param_col`.
//
// The view-dependent filters hang off the camera centre and the lens's local
// sampling rate rather than any one ray; `splat_ray_project_bwd` left what
// they need in pgrad (see there): a translation moves every camera-frame mean
// by -tau, a rotation turns it across the lens, and the calibration moves the
// sampling rate at it - differentiated here by a central difference of
// `lens_pixels_per_radian` in each parameter, which, like the one in
// `splat_ray_project_bwd`, spares the lens's second derivatives.
//
// One invocation per index in [0, max(pixels, gaussians)): a pixel's rays
// and a gaussian's filter term are both folded into the same partials.

struct Params {
    v: View,
    n_pix: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dray:    array<f32>; // W*H*6
@group(0) @binding(2) var<storage, read>       pgrad:   array<f32>; // N*13 (see splat_ray_project_bwd)
@group(0) @binding(3) var<storage, read>       aux:     array<f32>; // W*H*5 forward geometry
@group(0) @binding(4) var<storage, read>       daux:    array<f32>; // W*H*5 its upstream gradient
@group(0) @binding(5) var<storage, read_write> partial: array<f32>; // n_wg*28

var<workgroup> acc: array<f32, 1792>; // 28 x 64

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let t = lid.x;
    var g: array<f32, 28>;
    for (var k = 0u; k < 28u; k = k + 1u) { g[k] = 0.0; }
    if (idx < p.n_pix) {
        let px = idx % p.v.width;
        let py = idx / p.v.width;
        let go = vec3<f32>(dray[idx * 6u], dray[idx * 6u + 1u], dray[idx * 6u + 2u]);
        let gd = vec3<f32>(dray[idx * 6u + 3u], dray[idx * 6u + 4u], dray[idx * 6u + 5u]);
        let nacc = vec3<f32>(aux[idx * 5u + 1u], aux[idx * 5u + 2u], aux[idx * 5u + 3u]);
        let gnacc = vec3<f32>(daux[idx * 5u + 1u], daux[idx * 5u + 2u], daux[idx * 5u + 3u]);
        if (dot(go, go) + dot(gd, gd) + abs(dot(nacc, gnacc)) + dot(gnacc, gnacc) * dot(nacc, nacc) > 0.0) {
            let uv = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
            let pr = pixel_ray(p.v, uv);
            if (pr.ok != 0.0) {
                let gw = cross(pr.o, go) + cross(pr.d, gd) + cross(gnacc, nacc);
                g[0] = gw.x;
                g[1] = gw.y;
                g[2] = gw.z;
                g[3] = go.x;
                g[4] = go.y;
                g[5] = go.z;
                if (view_rs(p.v)) {
                    let rw = pr.tau * cross(pr.d, gd);
                    g[6] = rw.x;
                    g[7] = rw.y;
                    g[8] = rw.z;
                    g[9] = pr.tau * go.x;
                    g[10] = pr.tau * go.y;
                    g[11] = pr.tau * go.z;
                }
                // the lens's own ray, before the shutter turned it
                let dl = lens_unproject(p.v.lens, uv).xyz;
                let lp = lens_project(p.v.lens, dl);
                if (lp.ok != 0.0) {
                    let jp = lens_param_jac(p.v.lens, dl);
                    for (var k = 0u; k < 16u; k = k + 1u) {
                        if (jp[k].x != 0.0 || jp[k].y != 0.0) {
                            var col = lens_ray_param_col(lp, dl, jp[k]);
                            if (view_rs(p.v)) { col = rot_exp_apply(pr.tau * p.v.rs_w.xyz, col); }
                            g[12u + k] = dot(col, gd);
                        }
                    }
                }
            }
        }
    }
    if (idx < p.v.n) {
        let pg = idx * 13u;
        g[0] = g[0] + pgrad[pg + 3u];
        g[1] = g[1] + pgrad[pg + 4u];
        g[2] = g[2] + pgrad[pg + 5u];
        g[3] = g[3] - pgrad[pg];
        g[4] = g[4] - pgrad[pg + 1u];
        g[5] = g[5] - pgrad[pg + 2u];
        let gppr = pgrad[pg + 6u];
        if (gppr != 0.0) {
            let m = vec3<f32>(pgrad[pg + 7u], pgrad[pg + 8u], pgrad[pg + 9u]);
            for (var k = 0u; k < 16u; k = k + 1u) {
                if (k == 2u || k == 3u) { continue; } // the principal point moves no sampling rate
                var a = p.v.lens;
                var b = p.v.lens;
                var h = 1e-3;
                if (k < 2u) {
                    h = 1e-3 * a.f[k];
                    a.f[k] = a.f[k] + h;
                    b.f[k] = b.f[k] - h;
                } else if (k < 8u) {
                    a.ka[k - 4u] = a.ka[k - 4u] + h;
                    b.ka[k - 4u] = b.ka[k - 4u] - h;
                } else if (k < 12u) {
                    a.kb[k - 8u] = a.kb[k - 8u] + h;
                    b.kb[k - 8u] = b.kb[k - 8u] - h;
                } else {
                    a.kc[k - 12u] = a.kc[k - 12u] + h;
                    b.kc[k - 12u] = b.kc[k - 12u] - h;
                }
                let dppr = (lens_pixels_per_radian(a, m) - lens_pixels_per_radian(b, m)) / (2.0 * h);
                g[12u + k] = g[12u + k] + gppr * dppr;
            }
        }
    }
    for (var k = 0u; k < 28u; k = k + 1u) { acc[k * 64u + t] = g[k]; }
    workgroupBarrier();
    if (t < 28u) {
        var s = 0.0;
        for (var j = 0u; j < 64u; j = j + 1u) { s = s + acc[t * 64u + j]; }
        partial[(wid.y * nwg.x + wid.x) * 28u + t] = s;
    }
}
