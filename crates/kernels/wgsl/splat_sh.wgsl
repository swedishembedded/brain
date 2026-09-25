// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Spherical-harmonic view-dependent colour, forward and VJP in one kernel
// @how   one thread per gaussian
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Colour that depends on where you look from. Without it a fit has only one
// way to explain a surface that is brighter from one side - move geometry - so
// every glossy object in a capture is reconstructed as a smear of duplicated
// surfaces at slightly different depths.
//
// The colour a gaussian shows along direction d is
//   c = base + sum_k Y_k(d) * sh[k]
// which is LINEAR in the coefficients, so the same basis evaluation serves the
// forward and the backward: d_sh[k] = Y_k(d) * d_c, d_base = d_c. mode picks
// which one runs; both are one invocation per gaussian, no reductions.
//
// sh layout is Inria's, channel-major: all K coefficients of R, then G, then B.
//
// `limit[i]` caps gaussian i's own coefficients per channel (0, 3, 8 or 15):
// the view dependence the views that see IT can support. Coefficients past
// the cap neither colour the gaussian nor receive a gradient.

struct Params {
    n: u32,
    k: u32,       // coefficients per channel: 0, 3, 8 or 15
    mode: u32,    // 0 = forward (write colors), 1 = VJP (accumulate d_base/d_sh)
    skip: u32,    // highest coefficients held out (progressive bands), 0 = none
    eye_x: f32,
    eye_y: f32,
    eye_z: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       means:  array<f32>; // N*3
@group(0) @binding(2) var<storage, read>       base:   array<f32>; // N*3
@group(0) @binding(3) var<storage, read>       sh:     array<f32>; // N*3K
@group(0) @binding(4) var<storage, read_write> colors: array<f32>; // N*3 (fwd out)
@group(0) @binding(5) var<storage, read_write> d_base: array<f32>; // N*3   (+=)
@group(0) @binding(6) var<storage, read_write> d_sh:   array<f32>; // N*3K  (+=)
@group(0) @binding(7) var<storage, read>       limit:  array<u32>; // N

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }

    var y: array<f32, 15>;
    for (var t = 0u; t < 15u; t = t + 1u) { y[t] = 0.0; }

    if (p.k > 0u) {
        let dx0 = means[i * 3u] - p.eye_x;
        let dy0 = means[i * 3u + 1u] - p.eye_y;
        let dz0 = means[i * 3u + 2u] - p.eye_z;
        let l = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
        // A gaussian sitting exactly at the camera has no view direction; it
        // is also invisible, so degree 0 is the right answer rather than a NaN.
        let inv = select(0.0, 1.0 / l, l > 1e-12);
        let x = dx0 * inv;
        let yy = dy0 * inv;
        let z = dz0 * inv;

        y[0] = -0.4886025119029199 * yy;
        y[1] = 0.4886025119029199 * z;
        y[2] = -0.4886025119029199 * x;
        if (p.k > 3u) {
            let xx = x * x;
            let y2 = yy * yy;
            let zz = z * z;
            y[3] = 1.0925484305920792 * x * yy;
            y[4] = -1.0925484305920792 * yy * z;
            y[5] = 0.31539156525252005 * (2.0 * zz - xx - y2);
            y[6] = -1.0925484305920792 * x * z;
            y[7] = 0.5462742152960396 * (xx - y2);
            if (p.k > 8u) {
                y[8] = -0.5900435899266435 * yy * (3.0 * xx - y2);
                y[9] = 2.890611442640554 * x * yy * z;
                y[10] = -0.4570457994644658 * yy * (4.0 * zz - xx - y2);
                y[11] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * y2);
                y[12] = -0.4570457994644658 * x * (4.0 * zz - xx - y2);
                y[13] = 1.445305721320277 * z * (xx - y2);
                y[14] = -0.5900435899266435 * x * (xx - 3.0 * y2);
            }
        }
    }

    // Progressive bands: a fit enables SH one degree at a time, so the
    // coefficients above the current degree neither colour the splat nor
    // receive a gradient until their band switches on.
    for (var t = min(p.k - min(p.skip, p.k), limit[i]); t < 15u; t = t + 1u) { y[t] = 0.0; }

    for (var c = 0u; c < 3u; c = c + 1u) {
        let o = i * 3u + c;
        let sbase = i * 3u * p.k + c * p.k;
        if (p.mode == 0u) {
            var acc = base[o];
            for (var t = 0u; t < p.k; t = t + 1u) {
                acc = acc + y[t] * sh[sbase + t];
            }
            // Radiance cannot be negative, and letting it go there lets the
            // optimizer park error below zero where no view can see it.
            colors[o] = max(acc, 0.0);
        } else {
            // The clamp above is part of the forward: where it bit, nothing
            // downstream depends on these coefficients.
            var acc = base[o];
            for (var t = 0u; t < p.k; t = t + 1u) {
                acc = acc + y[t] * sh[sbase + t];
            }
            if (acc <= 0.0) { continue; }
            let vc = colors[o];
            d_base[o] = d_base[o] + vc;
            for (var t = 0u; t < p.k; t = t + 1u) {
                d_sh[sbase + t] = d_sh[sbase + t] + y[t] * vc;
            }
        }
    }
}
