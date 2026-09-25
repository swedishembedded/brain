// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  observational support of a rendered view: which training views saw each pixel's surface, from how far away in angle, and which saw through it
// @how   one thread per output element
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
// @import mvs
//
// For every pixel of a view (the reference: a novel camera, or a training
// camera judged against the others), the surface the scene renders there -
// the median range of `splat_ray_diagnose.wgsl` along the pixel's ray - is
// tested against every training view's own rendered range by multi-view
// stereo's consistency test (`mvs_filter.wgsl`, through `lib/mvs.wgsl`'s
// `mvs_seen` / `mvs_measured`): training view k images the point, reads its
// range there, and the surface it measures, carried back into the
// reference, must land within `max_reproj` pixels and `max_rel` in range.
//
//   [0] supporting views: k that measure the same surface
//   [1] the smallest angle, in degrees, between the reference's direction to
//       the surface and a supporting view's (-1 = no support): how far the
//       reference is from anything that observed the point
//   [2] free-space violations: k whose measured surface lies BEYOND the
//       point - k saw empty space where the reference renders a surface
//   [3] the angular radius, in degrees, of the supporting views'
//       directions about their mean: how widely the point was triangulated
//
// A pixel with alpha under a half has no surface to test and is all zero
// except [1] = -1.

struct Support {
    n_src: u32,
    plane: u32,
    skip: u32,     // a source left out (the reference itself), 0xffffffff = none
    pad1: u32,
    max_reproj: f32,
    max_rel: f32,
    pad2: f32,
    pad3: f32,
};

struct Params {
    v: View,
    s: Support,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       diag:   array<f32>;    // W*H*12, the reference's diagnostics
@group(0) @binding(2) var<storage, read>       cams:   array<MvsCam>; // n_src, relative to the reference
@group(0) @binding(3) var<storage, read>       ranges: array<f32>;    // n_src planes of rendered range (0 = none)
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;    // W*H*4

const MAX_SRC: u32 = 256u;

// Training view k against the reference's surface point `x0` (pixel `pc`,
// range `r` along a ray from `org`): 1 = k measures the same surface,
// -1 = k measures a surface beyond it (it saw through), 0 = neither (outside
// k's frame, no measurement, or occluded); with k's direction from `x0`.
struct Verdict {
    status: i32,
    to_k: vec3<f32>,
};

// The nearest surface view k measured at the four pixels around `uv`; 0 if
// any of them measured none (k cannot say the space there is empty).
fn nearest_around(k: u32, cam: MvsCam, uv: vec2<f32>) -> f32 {
    let base = vec2<i32>(floor(uv - vec2<f32>(0.5, 0.5)));
    var nearest = 3.0e38;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let q = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0, 0), vec2<i32>(i32(cam.dims.x) - 1, i32(cam.dims.y) - 1));
            let rq = ranges[k * p.s.plane + u32(q.y) * cam.dims.x + u32(q.x)];
            if (rq <= 0.0) { return 0.0; }
            nearest = min(nearest, rq);
        }
    }
    return nearest;
}

fn verdict(k: u32, x0: vec3<f32>, org: vec3<f32>, pc: vec2<f32>, r: f32) -> Verdict {
    var v = Verdict(0, vec3<f32>(0.0, 0.0, 0.0));
    let cam = cams[k];
    let seen = mvs_seen(cam, x0);
    if (!seen.ok) { return v; }
    let rk = ranges[k * p.s.plane + u32(seen.uv.y) * cam.dims.x + u32(seen.uv.x)];
    if (rk <= 0.0) { return v; }
    // k saw through the point only if every surface it measured around the
    // point's projection lies beyond it: at a depth edge the nearest sample
    // can be the background behind a surface k does see
    if (nearest_around(k, cam, seen.uv) > length(seen.y) * (1.0 + p.s.max_rel)) {
        v.status = -1;
        return v;
    }
    let xb = mvs_measured(cam, seen.y, rk);
    let back = lens_project(p.v.lens, xb);
    if (back.ok == 0.0 || length(back.uv - pc) > p.s.max_reproj || abs(length(xb - org) - r) > p.s.max_rel * r) { return v; }
    v.status = 1;
    v.to_k = normalize(mvs_centre(cam) - x0);
    return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.v.width * p.v.height) { return; }
    let o = i * 4u;
    out[o] = 0.0;
    out[o + 1u] = -1.0;
    out[o + 2u] = 0.0;
    out[o + 3u] = 0.0;
    let alpha = diag[i * 12u];
    var r = diag[i * 12u + 2u];
    if (r <= 0.0) { r = diag[i * 12u + 1u]; }
    if (alpha < 0.5 || r <= 0.0) { return; }
    let pc = vec2<f32>(f32(i % p.v.width) + 0.5, f32(i / p.v.width) + 0.5);
    let pr = pixel_ray(p.v, pc);
    if (pr.ok == 0.0) { return; }
    let x0 = pr.o + r * pr.d;
    let to_ref = normalize(pr.o - x0);
    let n = min(p.s.n_src, MAX_SRC);
    var support = 0u;
    var violations = 0u;
    var closest = -1.0; // largest cosine to a supporting view
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < n; k = k + 1u) {
        if (k == p.s.skip) { continue; }
        let v = verdict(k, x0, pr.o, pc, r);
        if (v.status < 0) { violations = violations + 1u; }
        if (v.status > 0) {
            support = support + 1u;
            closest = max(closest, dot(to_ref, v.to_k));
            sum = sum + v.to_k;
        }
    }
    out[o] = f32(support);
    out[o + 2u] = f32(violations);
    if (support == 0u) { return; }
    out[o + 1u] = degrees(acos(clamp(closest, -1.0, 1.0)));
    if (support < 2u) { return; }
    // the angular radius of the supporting directions about their mean
    let mean = normalize(sum);
    var widest = 1.0;
    for (var k = 0u; k < n; k = k + 1u) {
        if (k == p.s.skip) { continue; }
        let v = verdict(k, x0, pr.o, pc, r);
        if (v.status > 0) { widest = min(widest, dot(mean, v.to_k)); }
    }
    out[o + 3u] = degrees(acos(clamp(widest, -1.0, 1.0)));
}
