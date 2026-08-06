// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS backward, stage 5: EWA projection VJP
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// 3DGS backward, stage 5: EWA projection VJP. Per gaussian, take the reduced
// 2D gradients pgrad = {v_xy, v_conic(a,b,c), v_opacity, v_rgb(handled by
// grad_reduce)} and produce gradients w.r.t. the 3D parameters:
//   d_gauss[N*10] = {d_mean_w(3), d_scale_linear(3), d_quat_raw(4)},
//   d_opac[N] (w.r.t. the [0,1] opacity input; the antialiased compensation
//   chain is NOT modeled — run fit with aa off).
// Recomputes the forward quantities (cheap O(N)); culled gaussians (radius 0
// in proj) get zero grads. Flat local arrays only (CPU-JIT safe). One
// invocation per gaussian.

struct Params {
    n: u32,
    width: u32,
    height: u32,
    pad0: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    near: f32,
    far: f32,
    eps2d: f32,
    pad1: u32,
    r00: f32, r01: f32, r02: f32, tx: f32,
    r10: f32, r11: f32, r12: f32, ty: f32,
    r20: f32, r21: f32, r22: f32, tz: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       means:   array<f32>; // N*3
@group(0) @binding(2) var<storage, read>       quats:   array<f32>; // N*4 (raw)
@group(0) @binding(3) var<storage, read>       scales:  array<f32>; // N*3 linear
@group(0) @binding(4) var<storage, read>       proj:    array<f32>; // N*9 fwd out
@group(0) @binding(5) var<storage, read>       pgrad:   array<f32>; // N*9
@group(0) @binding(6) var<storage, read_write> d_gauss: array<f32>; // N*10 (+=)
@group(0) @binding(7) var<storage, read_write> d_opac:  array<f32>; // N (+=)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    if (proj[i * 9u + 7u] <= 0.0) { return; } // culled: no grads

    // ---- recompute forward camera-space quantities ----
    let mx = means[i * 3u];
    let my = means[i * 3u + 1u];
    let mz = means[i * 3u + 2u];
    let x = p.r00 * mx + p.r01 * my + p.r02 * mz + p.tx;
    let y = p.r10 * mx + p.r11 * my + p.r12 * mz + p.ty;
    let z = p.r20 * mx + p.r21 * my + p.r22 * mz + p.tz;
    let rz = 1.0 / z;

    var qw = quats[i * 4u];
    var qx = quats[i * 4u + 1u];
    var qy = quats[i * 4u + 2u];
    var qz = quats[i * 4u + 3u];
    let qn = sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-8;
    qw = qw / qn; qx = qx / qn; qy = qy / qn; qz = qz / qn;
    // rq: row-major 3x3
    var rq: array<f32, 9>;
    rq[0] = 1.0 - 2.0 * (qy * qy + qz * qz);
    rq[1] = 2.0 * (qx * qy - qw * qz);
    rq[2] = 2.0 * (qx * qz + qw * qy);
    rq[3] = 2.0 * (qx * qy + qw * qz);
    rq[4] = 1.0 - 2.0 * (qx * qx + qz * qz);
    rq[5] = 2.0 * (qy * qz - qw * qx);
    rq[6] = 2.0 * (qx * qz - qw * qy);
    rq[7] = 2.0 * (qy * qz + qw * qx);
    rq[8] = 1.0 - 2.0 * (qx * qx + qy * qy);
    var sv: array<f32, 3>;
    sv[0] = scales[i * 3u];
    sv[1] = scales[i * 3u + 1u];
    sv[2] = scales[i * 3u + 2u];
    // M = Rq diag(s), Sigma3 = M M^T, Sigma_c = R Sigma3 R^T
    var mm: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            mm[r * 3u + c] = rq[r * 3u + c] * sv[c];
        }
    }
    var s3: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 3u; k = k + 1u) {
                acc = acc + mm[r * 3u + k] * mm[c * 3u + k];
            }
            s3[r * 3u + c] = acc;
        }
    }
    var rr: array<f32, 9>;
    rr[0] = p.r00; rr[1] = p.r01; rr[2] = p.r02;
    rr[3] = p.r10; rr[4] = p.r11; rr[5] = p.r12;
    rr[6] = p.r20; rr[7] = p.r21; rr[8] = p.r22;
    var sc: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 3u; k = k + 1u) {
                for (var l = 0u; l < 3u; l = l + 1u) {
                    acc = acc + rr[r * 3u + k] * s3[k * 3u + l] * rr[c * 3u + l];
                }
            }
            sc[r * 3u + c] = acc;
        }
    }

    let tan_fovx = 0.5 * f32(p.width) / p.fx;
    let tan_fovy = 0.5 * f32(p.height) / p.fy;
    let lim_x_pos = (f32(p.width) - p.cx) / p.fx + 0.3 * tan_fovx;
    let lim_x_neg = p.cx / p.fx + 0.3 * tan_fovx;
    let lim_y_pos = (f32(p.height) - p.cy) / p.fy + 0.3 * tan_fovy;
    let lim_y_neg = p.cy / p.fy + 0.3 * tan_fovy;
    let xrz = x * rz;
    let yrz = y * rz;
    let txc = z * clamp(xrz, -lim_x_neg, lim_x_pos);
    let tyc = z * clamp(yrz, -lim_y_neg, lim_y_pos);
    let x_clamped = xrz < -lim_x_neg || xrz > lim_x_pos;
    let y_clamped = yrz < -lim_y_neg || yrz > lim_y_pos;
    // J (2x3, flat): rows [fx/z, 0, -fx*txc/z^2], [0, fy/z, -fy*tyc/z^2]
    var jm: array<f32, 6>;
    jm[0] = p.fx * rz; jm[1] = 0.0; jm[2] = -p.fx * txc * rz * rz;
    jm[3] = 0.0; jm[4] = p.fy * rz; jm[5] = -p.fy * tyc * rz * rz;
    // blurred 2D covariance + conic
    let ba = jm[0] * jm[0] * sc[0] + 2.0 * jm[0] * jm[2] * sc[2] + jm[2] * jm[2] * sc[8] + p.eps2d;
    let bb = jm[0] * (sc[1] * jm[4] + sc[2] * jm[5]) + jm[2] * (sc[5] * jm[4] + sc[8] * jm[5]);
    let bc = jm[4] * jm[4] * sc[4] + 2.0 * jm[4] * jm[5] * sc[5] + jm[5] * jm[5] * sc[8] + p.eps2d;
    let det = ba * bc - bb * bb;
    let ka = bc / det;
    let kb = -bb / det;
    let kc = ba / det;

    // ---- upstream 2D grads ----
    let v_mx2 = pgrad[i * 9u];
    let v_my2 = pgrad[i * 9u + 1u];
    let va = pgrad[i * 9u + 2u];
    let vb = pgrad[i * 9u + 3u];
    let vc = pgrad[i * 9u + 4u];
    d_opac[i] = d_opac[i] + pgrad[i * 9u + 5u];

    // conic -> blurred Sigma2: VS2 = -C G C, C=[[ka,kb],[kb,kc]],
    // G=[[va, vb/2],[vb/2, vc]] (b appears twice in the quadratic form)
    let g00 = va;
    let g01 = 0.5 * vb;
    let g11 = vc;
    let t00 = ka * g00 + kb * g01;
    let t01 = ka * g01 + kb * g11;
    let t10 = kb * g00 + kc * g01;
    let t11 = kb * g01 + kc * g11;
    let w00 = -(t00 * ka + t01 * kb);
    let w01 = -(t00 * kb + t01 * kc);
    let w10 = -(t10 * ka + t11 * kb);
    let w11 = -(t10 * kb + t11 * kc);
    // symmetric VS2 (flat 2x2)
    var vs: array<f32, 4>;
    vs[0] = w00;
    vs[1] = 0.5 * (w01 + w10);
    vs[2] = vs[1];
    vs[3] = w11;

    // VSc = J^T VS2 J (3x3 sym); VJ = 2 VS2 J Sc (2x3)
    var vsc: array<f32, 9>;
    for (var m = 0u; m < 3u; m = m + 1u) {
        for (var nn = 0u; nn < 3u; nn = nn + 1u) {
            var acc = 0.0;
            for (var r = 0u; r < 2u; r = r + 1u) {
                for (var q = 0u; q < 2u; q = q + 1u) {
                    acc = acc + jm[r * 3u + m] * vs[r * 2u + q] * jm[q * 3u + nn];
                }
            }
            vsc[m * 3u + nn] = acc;
        }
    }
    var vj: array<f32, 6>;
    for (var r = 0u; r < 2u; r = r + 1u) {
        for (var nn = 0u; nn < 3u; nn = nn + 1u) {
            var acc = 0.0;
            for (var q = 0u; q < 2u; q = q + 1u) {
                for (var k = 0u; k < 3u; k = k + 1u) {
                    acc = acc + 2.0 * vs[r * 2u + q] * jm[q * 3u + k] * sc[k * 3u + nn];
                }
            }
            vj[r * 3u + nn] = acc;
        }
    }

    // ---- mean grads (projection + J dependence) ----
    var v_xc = p.fx * rz * v_mx2;
    var v_yc = p.fy * rz * v_my2;
    var v_zc = -p.fx * x * rz * rz * v_mx2 - p.fy * y * rz * rz * v_my2;
    v_zc = v_zc + vj[0] * (-p.fx * rz * rz) + vj[4] * (-p.fy * rz * rz);
    v_zc = v_zc + vj[2] * (2.0 * p.fx * txc * rz * rz * rz)
        + vj[5] * (2.0 * p.fy * tyc * rz * rz * rz);
    if (x_clamped) {
        v_zc = v_zc + vj[2] * (-p.fx * rz * rz) * (txc * rz);
    } else {
        v_xc = v_xc + vj[2] * (-p.fx * rz * rz);
    }
    if (y_clamped) {
        v_zc = v_zc + vj[5] * (-p.fy * rz * rz) * (tyc * rz);
    } else {
        v_yc = v_yc + vj[5] * (-p.fy * rz * rz);
    }
    let dmx = p.r00 * v_xc + p.r10 * v_yc + p.r20 * v_zc;
    let dmy = p.r01 * v_xc + p.r11 * v_yc + p.r21 * v_zc;
    let dmz = p.r02 * v_xc + p.r12 * v_yc + p.r22 * v_zc;

    // ---- covariance chain: VS3 = R^T VSc R; VM = 2 sym(VSc3) M ----
    var vs3: array<f32, 9>;
    for (var m = 0u; m < 3u; m = m + 1u) {
        for (var nn = 0u; nn < 3u; nn = nn + 1u) {
            var acc = 0.0;
            for (var r = 0u; r < 3u; r = r + 1u) {
                for (var q = 0u; q < 3u; q = q + 1u) {
                    acc = acc + rr[r * 3u + m] * vsc[r * 3u + q] * rr[q * 3u + nn];
                }
            }
            vs3[m * 3u + nn] = acc;
        }
    }
    var vmm: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            var acc = 0.0;
            for (var q = 0u; q < 3u; q = q + 1u) {
                acc = acc + (vs3[r * 3u + q] + vs3[q * 3u + r]) * mm[q * 3u + c];
            }
            vmm[r * 3u + c] = acc;
        }
    }
    // v_s[k] = sum_r VM[r][k] Rq[r][k]; v_Rq[r][k] = VM[r][k] s[k]
    var d_s: array<f32, 3>;
    var vrq: array<f32, 9>;
    for (var c = 0u; c < 3u; c = c + 1u) {
        var acc = 0.0;
        for (var r = 0u; r < 3u; r = r + 1u) {
            acc = acc + vmm[r * 3u + c] * rq[r * 3u + c];
            vrq[r * 3u + c] = vmm[r * 3u + c] * sv[c];
        }
        d_s[c] = acc;
    }
    // v_Rq -> v_quat (normalized wxyz), then the normalization chain
    let vq_w = 2.0 * (qx * (vrq[7] - vrq[5]) + qy * (vrq[2] - vrq[6]) + qz * (vrq[3] - vrq[1]));
    let vq_x = 2.0 * (-2.0 * qx * (vrq[4] + vrq[8]) + qy * (vrq[1] + vrq[3])
        + qz * (vrq[2] + vrq[6]) + qw * (vrq[7] - vrq[5]));
    let vq_y = 2.0 * (qx * (vrq[1] + vrq[3]) - 2.0 * qy * (vrq[0] + vrq[8])
        + qz * (vrq[5] + vrq[7]) + qw * (vrq[2] - vrq[6]));
    let vq_z = 2.0 * (qx * (vrq[2] + vrq[6]) + qy * (vrq[5] + vrq[7])
        - 2.0 * qz * (vrq[0] + vrq[4]) + qw * (vrq[3] - vrq[1]));
    let dot = vq_w * qw + vq_x * qx + vq_y * qy + vq_z * qz;
    let dq_w = (vq_w - dot * qw) / qn;
    let dq_x = (vq_x - dot * qx) / qn;
    let dq_y = (vq_y - dot * qy) / qn;
    let dq_z = (vq_z - dot * qz) / qn;

    let o = i * 10u;
    d_gauss[o] = d_gauss[o] + dmx;
    d_gauss[o + 1u] = d_gauss[o + 1u] + dmy;
    d_gauss[o + 2u] = d_gauss[o + 2u] + dmz;
    d_gauss[o + 3u] = d_gauss[o + 3u] + d_s[0];
    d_gauss[o + 4u] = d_gauss[o + 4u] + d_s[1];
    d_gauss[o + 5u] = d_gauss[o + 5u] + d_s[2];
    d_gauss[o + 6u] = d_gauss[o + 6u] + dq_w;
    d_gauss[o + 7u] = d_gauss[o + 7u] + dq_x;
    d_gauss[o + 8u] = d_gauss[o + 8u] + dq_y;
    d_gauss[o + 9u] = d_gauss[o + 9u] + dq_z;
}
