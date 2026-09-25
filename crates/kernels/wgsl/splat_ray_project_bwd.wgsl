// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated splat backward: projection VJP to world parameters
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
// The ray renderer's backward, last stage: one invocation per gaussian takes
// the reduced camera-frame gradients pgrad = {dL/dm (3), dL/dA (6, unique
// entries: a00 a11 a22 a01 a02 a12, off-diagonals counted twice),
// dL/dopacity', dL/dn (3)} and carries them through `splat_ray_project.wgsl`
// back to the world-space parameters:
//   d_gauss[N*10] += {d mean (3), d scale (3, linear), d quat (4, raw)},
//   d_opac[N]     += d opacity (the [0,1] input).
//
// Forward being differentiated, per gaussian with camera-frame axes
// a_k = R Rq e_k:
//   lf_k = s_k^2 + f3,  lv_k = lf_k + fp,  fp = eps2d (|m| / ppr)^2,
//   A = sum_k a_k a_k^T / lv_k,
//   opacity' = opacity * c3 * c2,  c3 = sqrt(prod s_k^2 / lf_k),
//   c2 = sqrt(D / (D + fp T + fp^2)) (flag bit 0), with n = m / |m|,
//        D = prod lf * sum_k (a_k.n)^2 / lf_k,  T = sum lf - sum (a_k.n)^2 lf_k,
//   normal = +-a_kmin.
// The local sampling rate ppr depends on where the lens images the mean; its
// gradient is a central difference of `lens_pixels_per_radian` itself, which
// is smooth wherever the lens is - deriving it analytically would take the
// lens's second derivatives for one scalar per gaussian.
//
// pgrad is consumed here, and is left holding what the camera gradient needs
// of this gaussian beyond the per-pixel ray gradients of `splat_ray_bwd_slots`
// - the view-dependent filters hang off the camera CENTRE and the lens's
// local sampling rate, not off any one ray:
//   [0..3)  dL/dm through the filters (a camera translation moves m by -tau),
//   [3..6)  (dL/dm through the sampling rate) x m (a camera rotation turns m
//           across the lens; everything else about the gaussian turns with
//           it and is unchanged),
//   [6]     dL/d(pixels per radian),
//   [7..10) m, where `splat_ray_camera_grad` differentiates the sampling rate
//           in the calibration.

@group(0) @binding(0) var<uniform> p: View;
@group(0) @binding(1) var<storage, read>       means:   array<f32>; // N*3
@group(0) @binding(2) var<storage, read>       quats:   array<f32>; // N*4 raw
@group(0) @binding(3) var<storage, read>       scales:  array<f32>; // N*3 linear
@group(0) @binding(4) var<storage, read>       filt:    array<f32>; // N
@group(0) @binding(5) var<storage, read>       ray:     array<f32>; // N*16 fwd out
@group(0) @binding(6) var<storage, read_write> pgrad:   array<f32>; // N*13
@group(0) @binding(7) var<storage, read_write> d_gauss: array<f32>; // N*10 (+=)
@group(0) @binding(8) var<storage, read_write> d_opac:  array<f32>; // N (+=)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let pg = i * 13u;
    var gm = vec3<f32>(pgrad[pg], pgrad[pg + 1u], pgrad[pg + 2u]);
    let ga_u = array<f32, 6>(pgrad[pg + 3u], pgrad[pg + 4u], pgrad[pg + 5u], pgrad[pg + 6u], pgrad[pg + 7u], pgrad[pg + 8u]);
    let gop = pgrad[pg + 9u];
    let gnrm = vec3<f32>(pgrad[pg + 10u], pgrad[pg + 11u], pgrad[pg + 12u]);
    var mag = abs(gop) + length(gm) + length(gnrm);
    for (var k = 0u; k < 6u; k = k + 1u) { mag = mag + abs(ga_u[k]); }
    if (mag == 0.0) {
        // culled, or touched nothing: no gradient, and the forward record
        // may be stale
        for (var k = 0u; k < 13u; k = k + 1u) { pgrad[pg + k] = 0.0; }
        return;
    }
    // full symmetric dL/dA
    let G = mat3x3<f32>(
        vec3<f32>(ga_u[0], 0.5 * ga_u[3], 0.5 * ga_u[4]),
        vec3<f32>(0.5 * ga_u[3], ga_u[1], 0.5 * ga_u[5]),
        vec3<f32>(0.5 * ga_u[4], 0.5 * ga_u[5], ga_u[2]));

    // ---- recompute the forward ----
    let mw = vec3<f32>(means[i * 3u], means[i * 3u + 1u], means[i * 3u + 2u]);
    let m = view_to_camera(p, mw);
    let range = length(m);
    let rq0 = quats[i * 4u];
    let rq1 = quats[i * 4u + 1u];
    let rq2 = quats[i * 4u + 2u];
    let rq3 = quats[i * 4u + 3u];
    let qn = sqrt(rq0 * rq0 + rq1 * rq1 + rq2 * rq2 + rq3 * rq3) + 1e-8;
    let qw = rq0 / qn;
    let qx = rq1 / qn;
    let qy = rq2 / qn;
    let qz = rq3 / qn;
    var qc: array<vec3<f32>, 3>;
    qc[0] = vec3<f32>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy));
    qc[1] = vec3<f32>(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx));
    qc[2] = vec3<f32>(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
    let rot = mat3x3<f32>(p.r0.xyz, p.r1.xyz, p.r2.xyz); // transpose(R)
    var ak: array<vec3<f32>, 3>;
    for (var k = 0u; k < 3u; k = k + 1u) { ak[k] = qc[k] * rot; }
    var sv: array<f32, 3>;
    sv[0] = scales[i * 3u];
    sv[1] = scales[i * 3u + 1u];
    sv[2] = scales[i * 3u + 2u];
    let f3 = max(filt[i], 0.0);
    var lf: array<f32, 3>;
    var lv: array<f32, 3>;
    let ppr = lens_pixels_per_radian(p.lens, m);
    let fp = p.eps2d * (range / ppr) * (range / ppr);
    for (var k = 0u; k < 3u; k = k + 1u) {
        lf[k] = sv[k] * sv[k] + f3;
        lv[k] = lf[k] + fp;
    }
    let opp = ray[i * 16u + 12u]; // opacity'
    var c3 = 1.0;
    if (f3 > 0.0) { c3 = sqrt((sv[0] * sv[0] / lf[0]) * (sv[1] * sv[1] / lf[1]) * (sv[2] * sv[2] / lf[2])); }

    // gradient accumulators
    var g_lf: array<f32, 3>;
    var g_a: array<vec3<f32>, 3>;
    var g_s: array<f32, 3>;
    for (var k = 0u; k < 3u; k = k + 1u) {
        g_lf[k] = 0.0;
        g_a[k] = vec3<f32>(0.0, 0.0, 0.0);
        g_s[k] = 0.0;
    }
    var g_fp = 0.0;
    var g_mfilt = vec3<f32>(0.0, 0.0, 0.0);

    // ---- A = sum a a^T / lv ----
    for (var k = 0u; k < 3u; k = k + 1u) {
        let w = 1.0 / lv[k];
        let gak = G * ak[k];
        let gw = dot(ak[k], gak);
        g_a[k] = g_a[k] + 2.0 * w * gak;
        let glv = -w * w * gw;
        g_lf[k] = g_lf[k] + glv;
        g_fp = g_fp + glv;
    }

    // ---- normal = +-a_kmin ----
    var kmin = 2u;
    if (sv[0] <= sv[1] && sv[0] <= sv[2]) {
        kmin = 0u;
    } else if (sv[1] <= sv[2]) {
        kmin = 1u;
    }
    var sgn = 1.0;
    if (dot(ak[kmin], m) > 0.0) { sgn = -1.0; }
    g_a[kmin] = g_a[kmin] + sgn * gnrm;

    // ---- opacity' = opacity c3 c2 ----
    var c2 = 1.0;
    let lop = gop * opp; // dL/d log(opacity')
    if (f3 > 0.0) {
        for (var k = 0u; k < 3u; k = k + 1u) {
            if (sv[k] > 0.0) { g_s[k] = g_s[k] + lop * f3 / (sv[k] * lf[k]); }
        }
    }
    if ((p.flags & 1u) != 0u) {
        let nh = m / range;
        var ck: array<f32, 3>;
        var sum_c2l = 0.0;
        var sum_c2lf = 0.0;
        for (var k = 0u; k < 3u; k = k + 1u) {
            ck[k] = dot(ak[k], nh);
            sum_c2l = sum_c2l + ck[k] * ck[k] / lf[k];
            sum_c2lf = sum_c2lf + ck[k] * ck[k] * lf[k];
        }
        let prodl = lf[0] * lf[1] * lf[2];
        let dd = prodl * sum_c2l;
        let tt = lf[0] + lf[1] + lf[2] - sum_c2lf;
        let z = dd + fp * tt + fp * fp;
        c2 = sqrt(dd / z);
        let l_dd = lop * 0.5 * (1.0 / dd - 1.0 / z);
        let l_tt = -lop * 0.5 * fp / z;
        g_fp = g_fp - lop * 0.5 * (tt + 2.0 * fp) / z;
        var g_nh = vec3<f32>(0.0, 0.0, 0.0);
        for (var k = 0u; k < 3u; k = k + 1u) {
            let d_dd_dl = dd / lf[k] - prodl * ck[k] * ck[k] / (lf[k] * lf[k]);
            let d_tt_dl = 1.0 - ck[k] * ck[k];
            g_lf[k] = g_lf[k] + l_dd * d_dd_dl + l_tt * d_tt_dl;
            let g_ck = l_dd * prodl * 2.0 * ck[k] / lf[k] + l_tt * (-2.0 * ck[k] * lf[k]);
            g_a[k] = g_a[k] + g_ck * nh;
            g_nh = g_nh + g_ck * ak[k];
        }
        // n = m / |m|
        let gm_n = (g_nh - nh * dot(nh, g_nh)) / range;
        g_mfilt = g_mfilt + gm_n;
    }
    d_opac[i] = d_opac[i] + gop * c3 * c2;

    // ---- fp = eps2d |m|^2 / ppr^2 ----
    let hh = 1e-3 * range;
    var dppr = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 3u; k = k + 1u) {
        var e = vec3<f32>(0.0, 0.0, 0.0);
        e[k] = hh;
        dppr[k] = (lens_pixels_per_radian(p.lens, m + e) - lens_pixels_per_radian(p.lens, m - e)) / (2.0 * hh);
    }
    let g_ppr = -2.0 * g_fp * fp / ppr;
    g_mfilt = g_mfilt + g_fp * 2.0 * fp / (range * range) * m + g_ppr * dppr;
    let g_rot = cross(g_ppr * dppr, m);
    gm = gm + g_mfilt;

    // ---- lf = s^2 + f3 ----
    for (var k = 0u; k < 3u; k = k + 1u) {
        g_s[k] = g_s[k] + 2.0 * sv[k] * g_lf[k];
    }

    // ---- a_k = R q_k ----
    // g_q_k = R^T g_a_k; rot = transpose(R), so R^T v = rot * v
    var vrq: array<f32, 9>; // d/d Rq[r][c], row-major; column c is q_c
    for (var c = 0u; c < 3u; c = c + 1u) {
        let gq = rot * g_a[c];
        vrq[c] = gq.x;
        vrq[3u + c] = gq.y;
        vrq[6u + c] = gq.z;
    }
    let vq_w = 2.0 * (qx * (vrq[7] - vrq[5]) + qy * (vrq[2] - vrq[6]) + qz * (vrq[3] - vrq[1]));
    let vq_x = 2.0 * (-2.0 * qx * (vrq[4] + vrq[8]) + qy * (vrq[1] + vrq[3])
        + qz * (vrq[2] + vrq[6]) + qw * (vrq[7] - vrq[5]));
    let vq_y = 2.0 * (qx * (vrq[1] + vrq[3]) - 2.0 * qy * (vrq[0] + vrq[8])
        + qz * (vrq[5] + vrq[7]) + qw * (vrq[2] - vrq[6]));
    let vq_z = 2.0 * (qx * (vrq[2] + vrq[6]) + qy * (vrq[5] + vrq[7])
        - 2.0 * qz * (vrq[0] + vrq[4]) + qw * (vrq[3] - vrq[1]));
    let dotq = vq_w * qw + vq_x * qx + vq_y * qy + vq_z * qz;

    // ---- m = R mu + t ----
    let gmu = rot * gm;
    let o = i * 10u;
    d_gauss[o] = d_gauss[o] + gmu.x;
    d_gauss[o + 1u] = d_gauss[o + 1u] + gmu.y;
    d_gauss[o + 2u] = d_gauss[o + 2u] + gmu.z;
    d_gauss[o + 3u] = d_gauss[o + 3u] + g_s[0];
    d_gauss[o + 4u] = d_gauss[o + 4u] + g_s[1];
    d_gauss[o + 5u] = d_gauss[o + 5u] + g_s[2];
    d_gauss[o + 6u] = d_gauss[o + 6u] + (vq_w - dotq * qw) / qn;
    d_gauss[o + 7u] = d_gauss[o + 7u] + (vq_x - dotq * qx) / qn;
    d_gauss[o + 8u] = d_gauss[o + 8u] + (vq_y - dotq * qy) / qn;
    d_gauss[o + 9u] = d_gauss[o + 9u] + (vq_z - dotq * qz) / qn;
    pgrad[pg] = g_mfilt.x;
    pgrad[pg + 1u] = g_mfilt.y;
    pgrad[pg + 2u] = g_mfilt.z;
    pgrad[pg + 3u] = g_rot.x;
    pgrad[pg + 4u] = g_rot.y;
    pgrad[pg + 5u] = g_rot.z;
    pgrad[pg + 6u] = g_ppr;
    pgrad[pg + 7u] = m.x;
    pgrad[pg + 8u] = m.y;
    pgrad[pg + 9u] = m.z;
}
