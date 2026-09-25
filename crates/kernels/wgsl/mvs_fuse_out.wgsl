// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MVS fusion, per view: the fused world-space point each owned, supported pixel emits
// @how   one thread per pixel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import mvs
//
// Closes `mvs_fuse_acc`'s accumulation for one view: a pixel with a filtered
// range, owned by this view (no agreeing view samples it more finely) and
// supported by at least `min_support` views counting itself, emits the mean
// of its own and every agreeing measurement - range along its own ray, normal,
// colour, confidence - as a world-space point, with the radius of its pixel
// footprint at that range (range / pixels-per-radian: the world size one
// pixel covers there).
//
// `c2w` carries this view's camera-to-world rotation rows with the centre in
// w, and its lens. out planes: x, y, z, normal x, y, z, r, g, b, confidence, radius,
// support (0 = no point).

struct Params {
    w: u32,
    h: u32,
    plane: u32,
    min_support: u32,
    c2w: MvsCam,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       refm: array<f32>; // range, normal (3), confidence
@group(0) @binding(2) var<storage, read>       rays: array<f32>;
@group(0) @binding(3) var<storage, read>       rgb:  array<f32>;
@group(0) @binding(4) var<storage, read>       acc:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.w * p.h) { return; }
    let pl = p.plane;
    out[11u * pl + i] = 0.0;
    let r = refm[i];
    let support = acc[7u * pl + i] + 1.0;
    if (r <= 0.0 || acc[9u * pl + i] != 0.0 || support < f32(p.min_support)) { return; }
    let d = vec3<f32>(rays[i], rays[pl + i], rays[2u * pl + i]);
    let inv = 1.0 / support;
    let range = (r + acc[i]) * inv;
    let n = normalize(vec3<f32>(refm[pl + i] + acc[pl + i], refm[2u * pl + i] + acc[2u * pl + i], refm[3u * pl + i] + acc[3u * pl + i]));
    let x = mvs_to_cam(p.c2w, range * d);
    let nw = mvs_to_cam(p.c2w, n) - mvs_to_cam(p.c2w, vec3<f32>(0.0, 0.0, 0.0));
    out[i] = x.x;
    out[pl + i] = x.y;
    out[2u * pl + i] = x.z;
    out[3u * pl + i] = nw.x;
    out[4u * pl + i] = nw.y;
    out[5u * pl + i] = nw.z;
    for (var c = 0u; c < 3u; c = c + 1u) {
        out[(6u + c) * pl + i] = (rgb[c * pl + i] + acc[(4u + c) * pl + i]) * inv;
    }
    out[9u * pl + i] = (refm[4u * pl + i] + acc[8u * pl + i]) * inv;
    out[10u * pl + i] = range / max(lens_pixels_per_radian(p.c2w.lens, d), 1e-12);
    out[11u * pl + i] = support;
}
