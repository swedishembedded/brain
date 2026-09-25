// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MVS filter: keep hypotheses multi-view consistent, facing the camera; per-pixel confidence
// @how   one thread per pixel, serial loop over source views
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import mvs
//
// The last word on a reference view's range map (Schönberger et al., ECCV
// 2016, §4.4 / §5 - the consistency test also used for fusion). Source view
// k agrees with pixel p's hypothesis X = r d_p when k's own range map, read at
// the pixel X images to, puts the surface where p does: that range is taken
// along k's ray to X and the point carried back into the reference, and it
// must land within `max_reproj` pixels of p's centre AND within `max_rel` of
// r in range. It only counts when the two rays meet at an angle of at least
// `min_angle` (radians) - a view on the same line of sight agrees with
// anything. p is kept when at least `min_views` views agree, its normal
// faces the camera (n . -d_p >= `min_facing`), and its PatchMatch cost is at
// most `max_cost`; its confidence is
//   min(1, agree / conf_views) * (1 - cost / max_cost),
// in [0, 1]: more agreeing views and a better photometric match are both
// evidence.
//
// A patch with no texture at all costs the maximum (`mvs_pm.wgsl`) and so
// never passes, however consistent its plane: a plane propagated out over a
// featureless region is consistent in every view by construction, whether
// the region is a blank surface or empty background (measured: 13.5 % of
// one view's measurements off the surface of a synthetic scene when such
// pixels were kept on consistency alone).
//
// state: the PatchMatch state (range, normal, cost); depth: source k's range
// plane in slot k; out: range (0 = rejected), normal x, y, z, confidence.

struct Params {
    w: u32,
    h: u32,
    plane: u32,
    nsrc: u32,
    min_views: u32,
    conf_views: u32,
    pad0: u32,
    pad1: u32,
    max_reproj: f32,
    max_rel: f32,
    min_angle: f32,
    min_facing: f32,
    max_cost: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
    lens: Lens,
    src: array<MvsCam, 16>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       state: array<f32>;
@group(0) @binding(2) var<storage, read>       rays:  array<f32>;
@group(0) @binding(3) var<storage, read>       depth: array<f32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.w * p.h) { return; }
    let pl = p.plane;
    for (var c = 0u; c < 5u; c = c + 1u) { out[c * pl + i] = 0.0; }
    let d = vec3<f32>(rays[i], rays[pl + i], rays[2u * pl + i]);
    let r = state[i];
    let n = vec3<f32>(state[pl + i], state[2u * pl + i], state[3u * pl + i]);
    let cost = state[4u * pl + i];
    if (r <= 0.0 || dot(d, d) == 0.0 || cost > p.max_cost || dot(n, -d) < p.min_facing) { return; }
    let x0 = r * d;
    let pc = vec2<f32>(f32(i % p.w) + 0.5, f32(i / p.w) + 0.5);
    let cos_min = cos(p.min_angle);
    var agree = 0u;
    for (var k = 0u; k < p.nsrc; k = k + 1u) {
        let cam = p.src[k];
        let seen = mvs_seen(cam, x0);
        if (!seen.ok) { continue; }
        let rk = depth[k * pl + u32(seen.uv.y) * cam.dims.x + u32(seen.uv.x)];
        if (rk <= 0.0) { continue; }
        let xb = mvs_measured(cam, seen.y, rk);
        let back = lens_project(p.lens, xb);
        if (back.ok == 0.0) { continue; }
        if (length(back.uv - pc) > p.max_reproj) { continue; }
        if (abs(length(xb) - r) > p.max_rel * r) { continue; }
        // triangulation angle at X between the two centres
        let to_k = normalize(mvs_centre(cam) - x0);
        if (dot(-d, to_k) > cos_min) { continue; }
        agree = agree + 1u;
    }
    if (agree < p.min_views) { return; }
    out[i] = r;
    out[pl + i] = n.x;
    out[2u * pl + i] = n.y;
    out[3u * pl + i] = n.z;
    let views = min(1.0, f32(agree) / f32(max(p.conf_views, 1u)));
    out[4u * pl + i] = views * clamp(1.0 - cost / p.max_cost, 0.0, 1.0);
}
