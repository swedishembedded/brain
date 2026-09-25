// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS projection for ray evaluation through any lens (3DGUT-style)
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
// Ray-evaluated splatting, stage 0: one invocation per gaussian. Nothing is
// linearized. The rasterizer evaluates each gaussian EXACTLY along every
// pixel's own ray - its maximum response along the ray, which is the gaussian
// marginalized onto the plane across that ray - so a lens only has to say
// where each pixel looks (`lens_unproject`) and the image is right for a
// fisheye as for a pinhole. Projection here only has to decide WHICH tiles a
// gaussian can touch, and does it with the Unscented Transform (Wu et al.,
// 3DGUT, CVPR 2025): the six sigma points of the gaussian pushed through the
// real lens give its image-space spread without a Jacobian.
//
// Filters (Mip-Splatting, Yu et al., CVPR 2024), both in 3D:
// * the 3D smoothing filter: `filt[i]` is a variance added to every axis -
//   the finest detail the training views could resolve for this gaussian -
//   with opacity scaled by sqrt(|S| / |S + filt I|);
// * the per-view footprint filter: a pixel's own footprint at the gaussian's
//   range, eps2d * (range / pixels-per-radian)^2, added isotropically. Along
//   the ray it changes nothing; across it, it is the 2D Mip filter evaluated
//   on the ray marginal. With flag bit 0 the opacity is compensated by
//   sqrt(|S_perp| / |S_perp + f I|) for the marginal S_perp across the ray to
//   the mean (the exact analogue of the 2D Mip compensation), whose
//   determinant is |S| (n^T S^-1 n) without building a basis.
//
// Outputs:
//   proj[N*9]  {x2d, y2d, conic a, b, c, opacity', range, rx, ry} - the tile
//              stages' record, shared with the EWA renderer; the conic is the
//              inverse UT covariance and rx == 0 marks a culled gaussian;
//   ray[N*16]  {m (3), A (6: a00 a11 a22 a01 a02 a12), n (3), opacity',
//              colour (3)} - the mean in the mid-frame camera, the precision of
//              its filtered covariance, its surface normal (shortest axis,
//              facing the camera), and what the compositing stages read of it,
//              so they bind one record per gaussian.

@group(0) @binding(0) var<uniform> p: View;
@group(0) @binding(1) var<storage, read>       means:  array<f32>; // N*3
@group(0) @binding(2) var<storage, read>       quats:  array<f32>; // N*4 wxyz
@group(0) @binding(3) var<storage, read>       scales: array<f32>; // N*3 linear
@group(0) @binding(4) var<storage, read>       opac:   array<f32>; // N in [0,1]
@group(0) @binding(5) var<storage, read>       filt:   array<f32>; // N 3D filter variance
@group(0) @binding(6) var<storage, read>       colors: array<f32>; // N*3
@group(0) @binding(7) var<storage, read_write> proj:   array<f32>; // N*9
@group(0) @binding(8) var<storage, read_write> ray:    array<f32>; // N*16

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let o = i * 9u;
    proj[o + 7u] = 0.0; // culled until proven visible
    proj[o + 8u] = 0.0;

    let mw = vec3<f32>(means[i * 3u], means[i * 3u + 1u], means[i * 3u + 2u]);
    let m = view_to_camera(p, mw);
    let range = length(m);
    if (p.lens.code <= 1u) {
        if (m.z < p.near || m.z > p.far) { return; }
    } else {
        if (range < p.near || range > p.far) { return; }
    }

    // camera-frame axes rc_k = R Rq e_k and variances
    let qn = sqrt(quats[i * 4u] * quats[i * 4u] + quats[i * 4u + 1u] * quats[i * 4u + 1u]
        + quats[i * 4u + 2u] * quats[i * 4u + 2u] + quats[i * 4u + 3u] * quats[i * 4u + 3u]) + 1e-8;
    let qw = quats[i * 4u] / qn;
    let qx = quats[i * 4u + 1u] / qn;
    let qy = quats[i * 4u + 2u] / qn;
    let qz = quats[i * 4u + 3u] / qn;
    let q0 = vec3<f32>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy));
    let q1 = vec3<f32>(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx));
    let q2 = vec3<f32>(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
    let rot = mat3x3<f32>(p.r0.xyz, p.r1.xyz, p.r2.xyz); // columns = rows of R: transpose(R)
    let a0 = q0 * rot; // (R^T)^T q0 = R q0
    let a1 = q1 * rot;
    let a2 = q2 * rot;
    let s0 = scales[i * 3u];
    let s1 = scales[i * 3u + 1u];
    let s2 = scales[i * 3u + 2u];
    let f3 = max(filt[i], 0.0);
    let lf = vec3<f32>(s0 * s0 + f3, s1 * s1 + f3, s2 * s2 + f3);
    if (lf.x <= 0.0 || lf.y <= 0.0 || lf.z <= 0.0) { return; }
    // 3D filter compensation, sqrt(prod s^2 / prod (s^2 + f))
    var c3 = 1.0;
    if (f3 > 0.0) {
        c3 = sqrt((s0 * s0 / lf.x) * (s1 * s1 / lf.y) * (s2 * s2 / lf.z));
    }

    let ppr = lens_pixels_per_radian(p.lens, m);
    if (ppr <= 0.0) { return; }
    let fp = p.eps2d * (range / ppr) * (range / ppr);
    let lv = lf + vec3<f32>(fp, fp, fp);
    var op = opac[i] * c3;
    if ((p.flags & 1u) != 0u) {
        let nh = m / range;
        let c = vec3<f32>(dot(a0, nh), dot(a1, nh), dot(a2, nh));
        let det = lf.x * lf.y * lf.z;
        let dd = det * (c.x * c.x / lf.x + c.y * c.y / lf.y + c.z * c.z / lf.z);
        let tt = (lf.x + lf.y + lf.z) - (c.x * c.x * lf.x + c.y * c.y * lf.y + c.z * c.z * lf.z);
        op = op * sqrt(dd / (dd + fp * tt + fp * fp));
    }
    if (op < 1.0 / 255.0) { return; }

    // precision of the filtered covariance, A = sum_k a_k a_k^T / lv_k
    let w = vec3<f32>(1.0 / lv.x, 1.0 / lv.y, 1.0 / lv.z);
    let a00 = w.x * a0.x * a0.x + w.y * a1.x * a1.x + w.z * a2.x * a2.x;
    let a11 = w.x * a0.y * a0.y + w.y * a1.y * a1.y + w.z * a2.y * a2.y;
    let a22 = w.x * a0.z * a0.z + w.y * a1.z * a1.z + w.z * a2.z * a2.z;
    let a01 = w.x * a0.x * a0.y + w.y * a1.x * a1.y + w.z * a2.x * a2.y;
    let a02 = w.x * a0.x * a0.z + w.y * a1.x * a1.z + w.z * a2.x * a2.z;
    let a12 = w.x * a0.y * a0.z + w.y * a1.y * a1.z + w.z * a2.y * a2.z;

    // surface normal: the shortest axis, turned to face the camera
    var nrm = a2;
    if (s0 <= s1 && s0 <= s2) {
        nrm = a0;
    } else if (s1 <= s2) {
        nrm = a1;
    }
    if (dot(nrm, m) > 0.0) { nrm = -nrm; }

    // ---- image-space bounds by the Unscented Transform ----
    let c0 = lens_project(p.lens, m);
    if (c0.ok == 0.0) { return; }
    var uv0 = c0.uv;
    var shift = vec2<f32>(0.0, 0.0);
    if (view_rs(p)) {
        // the row a point lands on decides when it was exposed; two fixed-
        // point steps settle it to far below a pixel for any real readout
        var uv = uv0;
        for (var it = 0u; it < 2u; it = it + 1u) {
            let tau = uv.y / f32(p.height) - 0.5;
            let mt = rot_exp_apply(-tau * p.rs_w.xyz, m - tau * p.rs_v.xyz);
            let pr = lens_project(p.lens, mt);
            if (pr.ok == 0.0) { return; }
            uv = pr.uv;
        }
        shift = uv - uv0;
    }
    var cxx = 0.0;
    var cxy = 0.0;
    var cyy = 0.0;
    var whole = false;
    let sp = sqrt(3.0 * lv);
    for (var k = 0u; k < 6u; k = k + 1u) {
        var ax = a0 * sp.x;
        if (k / 2u == 1u) { ax = a1 * sp.y; }
        if (k / 2u == 2u) { ax = a2 * sp.z; }
        var x = m + ax;
        if (k % 2u == 1u) { x = m - ax; }
        let pk = lens_project(p.lens, x);
        if (pk.ok == 0.0) {
            whole = true;
        } else {
            var du = pk.uv - uv0;
            if ((p.flags & 4u) != 0u) {
                // longitude wraps: take the short way round
                let wd = f32(p.width);
                if (du.x > 0.5 * wd) { du.x = du.x - wd; }
                if (du.x < -0.5 * wd) { du.x = du.x + wd; }
            }
            cxx = cxx + du.x * du.x / 6.0;
            cxy = cxy + du.x * du.y / 6.0;
            cyy = cyy + du.y * du.y / 6.0;
        }
    }
    uv0 = uv0 + shift;
    let extend = min(3.33, sqrt(max(0.0, 2.0 * log(op * 255.0))));
    // a pixel of margin for what the transform's second-order accuracy misses
    var rx = ceil(extend * sqrt(cxx) + 1.0);
    var ry = ceil(extend * sqrt(cyy) + 1.0);
    if (whole) {
        // part of the gaussian lies where the lens cannot image it (behind a
        // perspective camera, or past its fold): it is close enough to the
        // camera to span the frame, so it may touch any tile
        rx = f32(p.width + p.height);
        ry = rx;
    }
    if (rx <= 0.0 || ry <= 0.0) { return; }
    if ((p.flags & 4u) == 0u) {
        if (uv0.x + rx <= 0.0 || uv0.x - rx >= f32(p.width) ||
            uv0.y + ry <= 0.0 || uv0.y - ry >= f32(p.height)) { return; }
    }
    let det2 = max(cxx * cyy - cxy * cxy, 1e-12);

    proj[o] = uv0.x;
    proj[o + 1u] = uv0.y;
    proj[o + 2u] = cyy / det2;
    proj[o + 3u] = -cxy / det2;
    proj[o + 4u] = cxx / det2;
    proj[o + 5u] = op;
    proj[o + 6u] = range;
    proj[o + 7u] = rx;
    proj[o + 8u] = ry;
    let r = i * 16u;
    ray[r] = m.x;
    ray[r + 1u] = m.y;
    ray[r + 2u] = m.z;
    ray[r + 3u] = a00;
    ray[r + 4u] = a11;
    ray[r + 5u] = a22;
    ray[r + 6u] = a01;
    ray[r + 7u] = a02;
    ray[r + 8u] = a12;
    ray[r + 9u] = nrm.x;
    ray[r + 10u] = nrm.y;
    ray[r + 11u] = nrm.z;
    ray[r + 12u] = op;
    ray[r + 13u] = colors[i * 3u];
    ray[r + 14u] = colors[i * 3u + 1u];
    ray[r + 15u] = colors[i * 3u + 2u];
}
