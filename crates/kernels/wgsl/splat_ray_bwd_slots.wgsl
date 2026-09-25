// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated splat backward: per-pixel partials into tile slots
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
// @import splat_ray_pair
// @import splat_ray_window
//
// The ray renderer's backward, per pixel, over a BAND of tiles
// [tile0, tile0 + n/256) and instances [k0, k1) - the slot layout and banding
// of `splat_bwd_slots.wgsl`, so `splat_bwd_tile_reduce` and everything after
// it are shared. One thread per pixel replays the forward walk twice, in the
// forward's own per-pixel order (`splat_ray_window`): once for the totals,
// once subtracting each contribution to get the strict suffixes the alpha
// gradient needs. A pair's partials are written when it leaves the pixel's
// window, into its instance's slot, so the order is free to differ from the
// list's.
//
// This is the reference form of the backward and the one the CPU JIT runs:
// every thread is independent. On a GPU `splat_ray_bwd_tile.wgsl` computes
// the same records without the slot grid.
//
// Upstream (all per pixel): dimg = dL/d(rgb, alpha out); daux = dL/d(the
// ACCUMULATED range sum w t, the accumulated normal sum w n (3), the
// distortion). A caller supervising EXPECTED range splits its gradient
// between daux[0] and dimg's alpha (`splat::renderer::add_expected_depth_vjp`).
//
// Channels (17 per instance, per pixel, in slots[((k - k0) * 17 + c) * 256 +
// local]), as `splat_ray_pair` lists them.
//
// With `want_ray` set, every pixel also writes dL/d(its ray origin) and
// dL/d(its ray direction) to dray[pix*6] - what `splat_ray_camera_grad`
// turns into pose, rolling-shutter and lens gradients. A tile split across
// bands writes them from the band that reaches its list's end, the only one
// that walks all of it.

struct Band {
    tile0: u32,
    k0: u32,
    k1: u32,
    n: u32,        // threads in the band: tiles * 256
    want_ray: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    v: View,
    b: Band,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ray:    array<vec4<f32>>; // N*4
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted instance -> gaussian id
@group(0) @binding(3) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(4) var<storage, read>       dimg:   array<f32>; // W*H*4
@group(0) @binding(5) var<storage, read>       daux:   array<f32>; // W*H*5
@group(0) @binding(6) var<storage, read_write> slots:  array<f32>; // band_instances*17*256
@group(0) @binding(7) var<storage, read_write> dray:   array<f32>; // W*H*6

const CH: u32 = 17u;

fn ray_record(g: u32) -> RayRec {
    return ray_rec4(ray[g * 4u], ray[g * 4u + 1u], ray[g * 4u + 2u], ray[g * 4u + 3u]);
}

fn zeros() -> array<f32, 17> {
    var z: array<f32, 17>;
    for (var c = 0u; c < CH; c = c + 1u) { z[c] = 0.0; }
    return z;
}

// Instance k's partials from this pixel, if k is in the band.
fn put(k: u32, local: u32, part: array<f32, 17>) {
    if (k < p.b.k0 || k >= p.b.k1) { return; }
    let base = (k - p.b.k0) * CH * 256u + local;
    for (var c = 0u; c < CH; c = c + 1u) {
        slots[base + c * 256u] = part[c];
    }
}

// A pair that left the window: its partials, and its share of the ray's
// gradient.
fn composited(s: RayBack, pair: RayBackPair, local: u32) -> RayBack {
    var b = s;
    let pg = ray_pair_grad(ray_record(pair.g), b.o, b.d, b.gc, b.gn, pair.g_alpha, pair.g_t, pair.w);
    put(pair.tag, local, pg.part);
    b.g_o = b.g_o + pg.g_o;
    b.g_d = b.g_d + pg.g_d;
    return b;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.b.n) { return; }
    let tiles_x = (p.v.width + 15u) / 16u;
    let tile = p.b.tile0 + idx / 256u;
    let local = idx % 256u;
    let px = (tile % tiles_x) * 16u + local % 16u;
    let py = (tile / tiles_x) * 16u + local / 16u;
    let inside = px < p.v.width && py < p.v.height;
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    let pr = pixel_ray(p.v, vec2<f32>(f32(px) + 0.5, f32(py) + 0.5));
    let live0 = inside && pr.ok != 0.0;

    var dimg4 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var daux4 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var gl = 0.0;
    if (inside) {
        let pix = py * p.v.width + px;
        dimg4 = vec4<f32>(dimg[pix * 4u], dimg[pix * 4u + 1u], dimg[pix * 4u + 2u], dimg[pix * 4u + 3u]);
        daux4 = vec4<f32>(daux[pix * 5u], daux[pix * 5u + 1u], daux[pix * 5u + 2u], daux[pix * 5u + 3u]);
        gl = daux[pix * 5u + 4u];
    }

    var w = ray_walk(pr.o, pr.d, live0);
    for (var j = start; j < end && w.live; j = j + 1u) {
        let g = vals[j];
        w = ray_walk_join(w, g, ray_record(g));
    }
    w = ray_walk_drain(w);
    var b = ray_back(w, live0, dimg4, daux4, gl, p.v.bg.xyz);
    // Every instance of the band is written once: a miss as the walk passes
    // it, a pair when it leaves the window (zero if the walk had stopped).
    // The walk goes on past the band, whose pairs can leave after later ones.
    for (var j = start; j < end; j = j + 1u) {
        if (!b.live) {
            if (j >= p.b.k1) { break; }
            put(j, local, zeros());
            continue;
        }
        let joined = b.win.joined;
        let g = vals[j];
        let st = ray_back_join(b, g, j, ray_record(g));
        b = st.b;
        if (b.win.joined == joined) { put(j, local, zeros()); }
        if (st.leaves) { b = composited(b, st.out, local); }
    }
    loop {
        let st = ray_back_drain(b);
        if (!st.leaves) { break; }
        b = composited(st.b, st.out, local);
    }
    if (p.b.want_ray != 0u && p.b.k1 >= end && inside) {
        let pix = (py * p.v.width + px) * 6u;
        dray[pix] = b.g_o.x;
        dray[pix + 1u] = b.g_o.y;
        dray[pix + 2u] = b.g_o.z;
        dray[pix + 3u] = b.g_d.x;
        dray[pix + 4u] = b.g_d.y;
        dray[pix + 5u] = b.g_d.z;
    }
}
