// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// 3DGS projection (gsplat-parity EWA): one invocation per gaussian.
// World mean -> camera -> 2D mean + 2D covariance (perspective Jacobian with
// FOV clamping), +eps2d anti-alias blur with optional opacity compensation,
// conic (inverse 2D covariance), opacity-aware truncation radius, and the
// full cull set (near/far, degenerate covariance, alpha < 1/255, offscreen).
//
// Output record stride 9 per gaussian:
//   {x2d, y2d, conic_a, conic_b, conic_c, opacity', depth, radius_x, radius_y}
// radius_x == 0 marks a culled gaussian for every downstream stage.

struct Params {
    n: u32,
    width: u32,
    height: u32,
    aa: u32,      // 1 = antialiased (multiply compensation into opacity)
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    near: f32,
    far: f32,
    eps2d: f32,
    pad0: u32,
    // world-to-camera rows [R | t]
    r00: f32, r01: f32, r02: f32, tx: f32,
    r10: f32, r11: f32, r12: f32, ty: f32,
    r20: f32, r21: f32, r22: f32, tz: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       means:  array<f32>; // N*3
@group(0) @binding(2) var<storage, read>       quats:  array<f32>; // N*4 wxyz
@group(0) @binding(3) var<storage, read>       scales: array<f32>; // N*3 linear
@group(0) @binding(4) var<storage, read>       opac:   array<f32>; // N in [0,1]
@group(0) @binding(5) var<storage, read_write> proj:   array<f32>; // N*9

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let o = i * 9u;
    proj[o + 7u] = 0.0; // culled until proven visible
    proj[o + 8u] = 0.0;

    // ---- world -> camera ----
    let mx = means[i * 3u];
    let my = means[i * 3u + 1u];
    let mz = means[i * 3u + 2u];
    let x = p.r00 * mx + p.r01 * my + p.r02 * mz + p.tx;
    let y = p.r10 * mx + p.r11 * my + p.r12 * mz + p.ty;
    let z = p.r20 * mx + p.r21 * my + p.r22 * mz + p.tz;
    if (z < p.near || z > p.far) { return; }

    // ---- quat (wxyz) + scale -> 3D covariance Sigma3 = R_q diag(s^2) R_q^T ----
    var qw = quats[i * 4u];
    var qx = quats[i * 4u + 1u];
    var qy = quats[i * 4u + 2u];
    var qz = quats[i * 4u + 3u];
    let qn = sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-8;
    qw = qw / qn; qx = qx / qn; qy = qy / qn; qz = qz / qn;
    let q00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    let q01 = 2.0 * (qx * qy - qw * qz);
    let q02 = 2.0 * (qx * qz + qw * qy);
    let q10 = 2.0 * (qx * qy + qw * qz);
    let q11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    let q12 = 2.0 * (qy * qz - qw * qx);
    let q20 = 2.0 * (qx * qz - qw * qy);
    let q21 = 2.0 * (qy * qz + qw * qx);
    let q22 = 1.0 - 2.0 * (qx * qx + qy * qy);
    let s0 = scales[i * 3u] * scales[i * 3u];
    let s1 = scales[i * 3u + 1u] * scales[i * 3u + 1u];
    let s2 = scales[i * 3u + 2u] * scales[i * 3u + 2u];
    // Sigma3_ij = sum_k s_k^2 Rq_ik Rq_jk (symmetric)
    let s00 = s0 * q00 * q00 + s1 * q01 * q01 + s2 * q02 * q02;
    let s01 = s0 * q00 * q10 + s1 * q01 * q11 + s2 * q02 * q12;
    let s02 = s0 * q00 * q20 + s1 * q01 * q21 + s2 * q02 * q22;
    let s11 = s0 * q10 * q10 + s1 * q11 * q11 + s2 * q12 * q12;
    let s12 = s0 * q10 * q20 + s1 * q11 * q21 + s2 * q12 * q22;
    let s22 = s0 * q20 * q20 + s1 * q21 * q21 + s2 * q22 * q22;

    // ---- Sigma_c = R Sigma3 R^T (world-to-camera rotation) ----
    // A = R * Sigma3
    let a00 = p.r00 * s00 + p.r01 * s01 + p.r02 * s02;
    let a01 = p.r00 * s01 + p.r01 * s11 + p.r02 * s12;
    let a02 = p.r00 * s02 + p.r01 * s12 + p.r02 * s22;
    let a10 = p.r10 * s00 + p.r11 * s01 + p.r12 * s02;
    let a11 = p.r10 * s01 + p.r11 * s11 + p.r12 * s12;
    let a12 = p.r10 * s02 + p.r11 * s12 + p.r12 * s22;
    let a20 = p.r20 * s00 + p.r21 * s01 + p.r22 * s02;
    let a21 = p.r20 * s01 + p.r21 * s11 + p.r22 * s12;
    let a22 = p.r20 * s02 + p.r21 * s12 + p.r22 * s22;
    let c00 = a00 * p.r00 + a01 * p.r01 + a02 * p.r02;
    let c01 = a00 * p.r10 + a01 * p.r11 + a02 * p.r12;
    let c02 = a00 * p.r20 + a01 * p.r21 + a02 * p.r22;
    let c11 = a10 * p.r10 + a11 * p.r11 + a12 * p.r12;
    let c12 = a10 * p.r20 + a11 * p.r21 + a12 * p.r22;
    let c22 = a20 * p.r20 + a21 * p.r21 + a22 * p.r22;

    // ---- EWA perspective projection with FOV-clamped Jacobian ----
    let rz = 1.0 / z;
    let tan_fovx = 0.5 * f32(p.width) / p.fx;
    let tan_fovy = 0.5 * f32(p.height) / p.fy;
    let lim_x_pos = (f32(p.width) - p.cx) / p.fx + 0.3 * tan_fovx;
    let lim_x_neg = p.cx / p.fx + 0.3 * tan_fovx;
    let lim_y_pos = (f32(p.height) - p.cy) / p.fy + 0.3 * tan_fovy;
    let lim_y_neg = p.cy / p.fy + 0.3 * tan_fovy;
    let txc = z * clamp(x * rz, -lim_x_neg, lim_x_pos);
    let tyc = z * clamp(y * rz, -lim_y_neg, lim_y_pos);
    let j00 = p.fx * rz;
    let j02 = -p.fx * txc * rz * rz;
    let j11 = p.fy * rz;
    let j12 = -p.fy * tyc * rz * rz;
    var sa = j00 * j00 * c00 + 2.0 * j00 * j02 * c02 + j02 * j02 * c22;
    let sb = j00 * (c01 * j11 + c02 * j12) + j02 * (c12 * j11 + c22 * j12);
    var sc = j11 * j11 * c11 + 2.0 * j11 * j12 * c12 + j12 * j12 * c22;

    // ---- anti-alias blur + compensation ----
    let det_orig = sa * sc - sb * sb;
    sa = sa + p.eps2d;
    sc = sc + p.eps2d;
    let det_blur = sa * sc - sb * sb;
    if (det_blur <= 0.0) { return; }
    let comp = sqrt(max(0.005 * 0.005, det_orig / det_blur));

    // ---- conic + opacity + radius ----
    var op = opac[i];
    if (p.aa != 0u) { op = op * comp; }
    if (op < 1.0 / 255.0) { return; }
    let extend = min(3.33, sqrt(max(0.0, 2.0 * log(op * 255.0))));
    let rx = ceil(extend * sqrt(sa));
    let ry = ceil(extend * sqrt(sc));
    if (rx <= 0.0 || ry <= 0.0) { return; }
    let px = p.fx * x * rz + p.cx;
    let py = p.fy * y * rz + p.cy;
    if (px + rx <= 0.0 || px - rx >= f32(p.width) ||
        py + ry <= 0.0 || py - ry >= f32(p.height)) { return; }

    proj[o] = px;
    proj[o + 1u] = py;
    proj[o + 2u] = sc / det_blur;
    proj[o + 3u] = -sb / det_blur;
    proj[o + 4u] = sa / det_blur;
    proj[o + 5u] = op;
    proj[o + 6u] = z;
    proj[o + 7u] = rx;
    proj[o + 8u] = ry;
}
