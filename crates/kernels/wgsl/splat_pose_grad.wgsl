// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Camera pose gradient, as a strided partial reduction over gaussian gradients
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Moving a camera is exactly the same thing as moving the scene the other way,
// so the gradient with respect to a pose is a reduction over gradients the
// backward pass has already produced - positions AND orientations, because a
// rigid motion turns every gaussian as well as shifting it. Nothing here
// approximates the chain rule; it reuses it.
//
// Accumulates, for a scene motion expressed as a rotation about the camera
// centre plus a translation:
//   a = sum  dL/dmean_i                                   (translation)
//   b = sum (mean_i - centre) x dL/dmean_i  +  dq term    (rotation)
// Rotating about the CAMERA CENTRE rather than the world origin is what keeps
// the two blocks from fighting each other: about the origin, a far-away camera
// turns rotation into mostly translation and the problem conditions badly.
//
// Strided so every thread walks its own share and writes its own 6 floats -
// no workgroup memory and no barriers, so the CPU backend runs the same code.
// The host sums `total` partials, which is a few thousand values.

struct Params {
    n: u32,
    total: u32,   // number of partial slots == number of threads dispatched
    pad0: u32,
    pad1: u32,
    ex: f32,
    ey: f32,
    ez: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       means: array<f32>; // N*3
@group(0) @binding(2) var<storage, read>       quats: array<f32>; // N*4 raw
@group(0) @binding(3) var<storage, read>       dg:    array<f32>; // N*10
@group(0) @binding(4) var<storage, read_write> out:   array<f32>; // total*6

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= p.total) { return; }

    var ax = 0.0;
    var ay = 0.0;
    var az = 0.0;
    var bx = 0.0;
    var by = 0.0;
    var bz = 0.0;
    var i = t;
    loop {
        if (i >= p.n) { break; }
        let o = i * 10u;
        let gx = dg[o];
        let gy = dg[o + 1u];
        let gz = dg[o + 2u];
        ax = ax + gx;
        ay = ay + gy;
        az = az + gz;
        let wx = means[i * 3u] - p.ex;
        let wy = means[i * 3u + 1u] - p.ey;
        let wz = means[i * 3u + 2u] - p.ez;
        bx = bx + (wy * gz - wz * gy);
        by = by + (wz * gx - wx * gz);
        bz = bz + (wx * gy - wy * gx);

        // A rigid motion composes with each gaussian's orientation:
        // q' = dq * q with dq = (1, u/2), so the orientation gradients carry
        // their own share of the rotation. Leaving this out silently biases
        // the pose whenever the scene is anisotropic, which is always.
        let qw = quats[i * 4u];
        let qx = quats[i * 4u + 1u];
        let qy = quats[i * 4u + 2u];
        let qz = quats[i * 4u + 3u];
        let dw = dg[o + 6u];
        let dx = dg[o + 7u];
        let dy = dg[o + 8u];
        let dz = dg[o + 9u];
        bx = bx + 0.5 * (-qx * dw + qw * dx - qz * dy + qy * dz);
        by = by + 0.5 * (-qy * dw + qz * dx + qw * dy - qx * dz);
        bz = bz + 0.5 * (-qz * dw - qy * dx + qx * dy + qw * dz);

        i = i + p.total;
    }

    let w = t * 6u;
    out[w] = ax;
    out[w + 1u] = ay;
    out[w + 2u] = az;
    out[w + 3u] = bx;
    out[w + 4u] = by;
    out[w + 5u] = bz;
}
