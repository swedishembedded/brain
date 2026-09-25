// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Multi-view stereo on the device: what the mvs_* kernels share. Imported
// after `camera`.
//
// A hypothesis is a plane in the REFERENCE camera's frame, carried per pixel
// as the range along that pixel's unit ray (`lens_unproject`) and the plane's
// unit normal, facing the camera (dot(n, ray) < 0). Every other view is
// described relative to the reference by `MvsCam`: the rigid transform that
// takes a reference-frame point into that view's frame, and its lens.
//
// Planar per-view images: a view's `k`-th channel starts at `k * plane`.

struct MvsCam {
    r0: vec4<f32>,   // rows of the rotation reference -> this view, translation in w
    r1: vec4<f32>,
    r2: vec4<f32>,
    dims: vec4<u32>, // width, height, (unused), (unused)
    lens: Lens,
};

// Reference-frame point `x` in view `c`'s frame.
fn mvs_to_cam(c: MvsCam, x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(c.r0.xyz, x) + c.r0.w, dot(c.r1.xyz, x) + c.r1.w, dot(c.r2.xyz, x) + c.r2.w);
}

// View `c`'s frame point `y` back in the reference frame: R^T (y - t).
fn mvs_from_cam(c: MvsCam, y: vec3<f32>) -> vec3<f32> {
    let d = y - vec3<f32>(c.r0.w, c.r1.w, c.r2.w);
    return c.r0.xyz * d.x + c.r1.xyz * d.y + c.r2.xyz * d.z;
}

// View `c`'s centre in the reference frame.
fn mvs_centre(c: MvsCam) -> vec3<f32> {
    return mvs_from_cam(c, vec3<f32>(0.0, 0.0, 0.0));
}

// Range along unit ray `d` (from the reference centre) to the plane through
// `x0` with normal `n`; negative where the ray does not meet the plane in
// front of the camera or meets it too obliquely to trust.
fn mvs_plane_range(n: vec3<f32>, x0: vec3<f32>, d: vec3<f32>) -> f32 {
    let den = dot(n, d);
    if (den > -1e-4) { return -1.0; }
    return dot(n, x0) / den;
}

// Integer hash (Jarzynski & Olano, "Hash Functions for GPU Rendering", JCGT
// 2020: the PCG-derived one-round hash) - every random draw is a pure function
// of (pixel, pass, seed, draw), so a run is reproducible on any backend.
fn mvs_hash(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Uniform in [0, 1), advancing `state`.
fn mvs_rand(state: ptr<function, u32>) -> f32 {
    *state = mvs_hash(*state);
    return f32(*state >> 8u) * (1.0 / 16777216.0);
}

// A uniformly random unit vector.
fn mvs_rand_dir(state: ptr<function, u32>) -> vec3<f32> {
    let z = 2.0 * mvs_rand(state) - 1.0;
    let phi = 6.2831853 * mvs_rand(state);
    let s = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(s * cos(phi), s * sin(phi), z);
}

// `n` turned to face a camera looking along `d`.
fn mvs_face(n: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if (dot(n, d) > 0.0) { return -n; }
    return n;
}
