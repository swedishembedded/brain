// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  free-space carving: per gaussian, the views that saw empty space where it sits and the views that saw it on their surface
// @how   one thread per output element
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
//
// A camera that measured a surface at range r along a ray observed the ray
// before r to be empty. A gaussian whose whole extent lies on such a ray
// segment, in front of what that camera measured, is where the camera saw
// nothing: a floater, however well it serves some other view's colour. Per
// gaussian i and view k (the view's own record, `View`, as the renderer
// takes it): project the mean through k's lens; if k measured a surface
// there - `ranges` plane k, at every pixel of the 2x2 neighbourhood (a depth
// edge's nearest sample can be the background behind a surface k does see)
// - compare its range along k's ray with the gaussian's:
//
//   violation: range + 3 sigma < (1 - margin) * every measured range around
//              (the gaussian is wholly in front of the surface k saw)
//   support:   |range - measured| <= margin * measured + 3 sigma, for the
//              nearest pixel (the gaussian is on the surface k saw)
//
// with sigma the gaussian's largest standard deviation. out[i*2] counts
// violations, out[i*2 + 1] supports.

struct Carve {
    n: u32,
    n_views: u32,
    plane: u32,
    pad0: u32,
    margin: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
};

@group(0) @binding(0) var<uniform> p: Carve;
@group(0) @binding(1) var<storage, read>       means:  array<f32>;  // N*3
@group(0) @binding(2) var<storage, read>       scales: array<f32>;  // N*3
@group(0) @binding(3) var<storage, read>       views:  array<View>; // n_views
@group(0) @binding(4) var<storage, read>       ranges: array<f32>;  // n_views planes, 0 = no surface measured
@group(0) @binding(5) var<storage, read_write> out:    array<f32>;  // N*2

// The nearest range view k measured at the four pixels around `uv`; 0 if any
// of them measured none.
fn nearest_around(k: u32, w: u32, h: u32, uv: vec2<f32>) -> f32 {
    let base = vec2<i32>(floor(uv - vec2<f32>(0.5, 0.5)));
    var nearest = 3.0e38;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let q = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0, 0), vec2<i32>(i32(w) - 1, i32(h) - 1));
            let rq = ranges[k * p.plane + u32(q.y) * w + u32(q.x)];
            if (rq <= 0.0) { return 0.0; }
            nearest = min(nearest, rq);
        }
    }
    return nearest;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let m = vec3<f32>(means[i * 3u], means[i * 3u + 1u], means[i * 3u + 2u]);
    let sigma = max(scales[i * 3u], max(scales[i * 3u + 1u], scales[i * 3u + 2u]));
    var violations = 0u;
    var supports = 0u;
    for (var k = 0u; k < p.n_views; k = k + 1u) {
        let v = views[k];
        let y = view_to_camera(v, m);
        let pr = lens_project(v.lens, y);
        if (pr.ok == 0.0 || pr.uv.x < 0.0 || pr.uv.y < 0.0 || pr.uv.x >= f32(v.width) || pr.uv.y >= f32(v.height)) { continue; }
        let dist = length(y);
        let at = ranges[k * p.plane + u32(pr.uv.y) * v.width + u32(pr.uv.x)];
        if (at <= 0.0) { continue; }
        if (abs(dist - at) <= p.margin * at + 3.0 * sigma) {
            supports = supports + 1u;
            continue;
        }
        let around = nearest_around(k, v.width, v.height, pr.uv);
        if (around > 0.0 && dist + 3.0 * sigma < (1.0 - p.margin) * around) {
            violations = violations + 1u;
        }
    }
    out[i * 2u] = f32(violations);
    out[i * 2u + 1u] = f32(supports);
}
