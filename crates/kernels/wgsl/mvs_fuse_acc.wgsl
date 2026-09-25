// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MVS fusion, one view pair: gather another view's agreeing measurement per reference pixel
// @how   one thread per reference pixel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import mvs
//
// Depth-map fusion by consistency (Schönberger et al., ECCV 2016, §5), as a
// gather so it needs no atomics: dispatched once per (reference, other view)
// pair, every reference pixel looks up the other view's filtered range at the
// pixel its point images to and, when the two agree - the other view's point
// carried back lands within `max_reproj` pixels of the reference pixel's
// centre, within `max_rel` of its range, and its normal within `max_normal`
// (cosine) of the reference normal - adds that measurement to its
// accumulator: the range it implies along the reference pixel's own ray (so
// averaging never moves a point sideways), the normal (reference frame),
// the colour and the confidence, and one to the support count.
//
// Ownership. Every agreeing view sees the same surface point, and it must
// come out of fusion once. It is emitted by the view that samples it most
// finely - the smallest pixel footprint, range / pixels-per-radian, ties to
// the lower view index (`other_first`) - so a reference pixel that finds an
// agreeing view with a finer footprint marks itself as owned elsewhere.
//
// acc planes: range sum, normal sum (3), colour sum (3), count, confidence
// sum, owned-elsewhere flag.

struct Params {
    w: u32,
    h: u32,
    plane: u32,
    oplane: u32,
    other_first: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    max_reproj: f32,
    max_rel: f32,
    max_normal: f32,
    pad3: f32,
    lens: Lens,
    other: MvsCam,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       refm:  array<f32>; // range, normal (3), confidence
@group(0) @binding(2) var<storage, read>       rays:  array<f32>;
@group(0) @binding(3) var<storage, read>       otherm: array<f32>; // the other view's, same layout
@group(0) @binding(4) var<storage, read>       orgb:  array<f32>;  // the other view's colour planes
@group(0) @binding(5) var<storage, read_write> acc:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.w * p.h) { return; }
    let pl = p.plane;
    let r = refm[i];
    if (r <= 0.0) { return; }
    let d = vec3<f32>(rays[i], rays[pl + i], rays[2u * pl + i]);
    let n = vec3<f32>(refm[pl + i], refm[2u * pl + i], refm[3u * pl + i]);
    let cam = p.other;
    let y = mvs_to_cam(cam, r * d);
    let pr = lens_project(cam.lens, y);
    if (pr.ok == 0.0 || pr.uv.x < 0.0 || pr.uv.y < 0.0 || pr.uv.x >= f32(cam.dims.x) || pr.uv.y >= f32(cam.dims.y)) { return; }
    let op = p.oplane;
    let j = u32(pr.uv.y) * cam.dims.x + u32(pr.uv.x);
    let rj = otherm[j];
    if (rj <= 0.0) { return; }
    let yj = normalize(y) * rj;
    let xb = mvs_from_cam(cam, yj);
    let back = lens_project(p.lens, xb);
    if (back.ok == 0.0) { return; }
    let pc = vec2<f32>(f32(i % p.w) + 0.5, f32(i / p.w) + 0.5);
    if (length(back.uv - pc) > p.max_reproj || abs(length(xb) - r) > p.max_rel * r) { return; }
    let nj = mvs_from_cam(cam, vec3<f32>(otherm[op + j], otherm[2u * op + j], otherm[3u * op + j]))
        - mvs_from_cam(cam, vec3<f32>(0.0, 0.0, 0.0));
    if (dot(nj, n) < p.max_normal) { return; }
    acc[i] = acc[i] + dot(xb, d);
    acc[pl + i] = acc[pl + i] + nj.x;
    acc[2u * pl + i] = acc[2u * pl + i] + nj.y;
    acc[3u * pl + i] = acc[3u * pl + i] + nj.z;
    acc[4u * pl + i] = acc[4u * pl + i] + orgb[j];
    acc[5u * pl + i] = acc[5u * pl + i] + orgb[op + j];
    acc[6u * pl + i] = acc[6u * pl + i] + orgb[2u * op + j];
    acc[7u * pl + i] = acc[7u * pl + i] + 1.0;
    acc[8u * pl + i] = acc[8u * pl + i] + otherm[4u * op + j];
    // who samples this point more finely
    let fi = r / max(lens_pixels_per_radian(p.lens, d), 1e-12);
    let fj = rj / max(lens_pixels_per_radian(cam.lens, y), 1e-12);
    if (fj < fi * (1.0 - 1e-3) || (fj <= fi * (1.0 + 1e-3) && p.other_first == 1u)) {
        acc[9u * pl + i] = 1.0;
    }
}
