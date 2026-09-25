// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pop-free differentiable splat renderers for
// its clients. If your team needs expertise in gaussian splatting then you
// can procure our services by sending an email to info@swedishembedded.com.

// @what  ray-evaluated splat backward, stage 1: each pixel walks its list in its own order again, logging every pair's gradient
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
// The ray renderer's backward as the GPU runs it, first stage. Every pixel
// composites in its own order (`splat_ray_window`), so at any point of a
// tile's list different pixels composite different instances, and the
// per-instance reduction (`splat_ray_bwd_tile`) cannot run in step with the
// walks. This stage runs them: one thread per pixel (the 256 of a tile in
// consecutive workgroups, walking one list) starts from the totals the
// forward left (`splat_ray_rasterize`'s `totals`), walks the list again
// subtracting each pair's contribution for the strict suffixes the alpha
// gradient needs, and logs each pair as it leaves the window - at its RANK,
// the order it joined in, which is list order.
//
// The log (u32 words, floats as their bits), for a W x H frame:
//   [pix * 6]            dL/d(its ray origin) (3), dL/d(its ray direction)
//                        (3) of pixel pix, for `splat_ray_camera_grad`,
//                        written with `want_ray`;
//   [6 W H + pix * 6]    dL/d(rgb) (3), dL/d(sum w n) (3) of pixel pix;
//   [12 W H + 4 (offs[pix] + rank)]
//                        {gaussian, dL/dalpha, dL/dt*, w} of the pixel's
//                        rank-th pair, all zero but the gaussian for a pair
//                        the walk stopped at or never composited.
// `offs` is the exclusive scan of the forward's per-pixel count of pairs
// (`splat_ray_rasterize`'s `joined`); a pixel logs at most that many.

struct Tail {
    want_ray: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    v: View,
    t: Tail,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ray:    array<vec4<f32>>; // N*4
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted instance -> gaussian id
@group(0) @binding(3) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(4) var<storage, read>       dimg:   array<f32>; // W*H*4
@group(0) @binding(5) var<storage, read>       daux:   array<f32>; // W*H*5
@group(0) @binding(6) var<storage, read>       offs:   array<u32>; // W*H + 1
@group(0) @binding(7) var<storage, read>       totals: array<f32>; // W*H*10
@group(0) @binding(8) var<storage, read_write> pairs:  array<u32>; // 12 W H + 4 offs[W H]

fn ray_record(g: u32) -> RayRec {
    return ray_rec4(ray[g * 4u], ray[g * 4u + 1u], ray[g * 4u + 2u], ray[g * 4u + 3u]);
}

// Log a pair that left the window at its rank (dropped past the pixel's
// count, which only a forward that disagreed with this walk could reach), and
// add its share of the ray's gradient.
fn logged(s: RayBack, pair: RayBackPair, entries: u32, count: u32) -> RayBack {
    var b = s;
    if (pair.tag < count) {
        let e = entries + 4u * pair.tag;
        pairs[e] = pair.g;
        pairs[e + 1u] = bitcast<u32>(pair.g_alpha);
        pairs[e + 2u] = bitcast<u32>(pair.g_t);
        pairs[e + 3u] = bitcast<u32>(pair.w);
    }
    if (p.t.want_ray != 0u && pair.w != 0.0) {
        let pg = ray_pair_grad(ray_record(pair.g), b.o, b.d, b.gc, b.gn, pair.g_alpha, pair.g_t, pair.w);
        b.g_o = b.g_o + pg.g_o;
        b.g_d = b.g_d + pg.g_d;
    }
    return b;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let tiles_x = (p.v.width + 15u) / 16u;
    let tiles_y = (p.v.height + 15u) / 16u;
    let tile = idx / 256u;
    if (tile >= tiles_x * tiles_y) { return; }
    let local = idx % 256u;
    let px = (tile % tiles_x) * 16u + local % 16u;
    let py = (tile / tiles_x) * 16u + local / 16u;
    if (px >= p.v.width || py >= p.v.height) { return; }
    let n_px = p.v.width * p.v.height;
    let pix = py * p.v.width + px;
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    let pr = pixel_ray(p.v, vec2<f32>(f32(px) + 0.5, f32(py) + 0.5));
    let live0 = pr.ok != 0.0;

    let dimg4 = vec4<f32>(dimg[pix * 4u], dimg[pix * 4u + 1u], dimg[pix * 4u + 2u], dimg[pix * 4u + 3u]);
    let daux4 = vec4<f32>(daux[pix * 5u], daux[pix * 5u + 1u], daux[pix * 5u + 2u], daux[pix * 5u + 3u]);
    let gl = daux[pix * 5u + 4u];
    let h = 6u * (n_px + pix);
    pairs[h] = bitcast<u32>(dimg4.x);
    pairs[h + 1u] = bitcast<u32>(dimg4.y);
    pairs[h + 2u] = bitcast<u32>(dimg4.z);
    pairs[h + 3u] = bitcast<u32>(daux4.y);
    pairs[h + 4u] = bitcast<u32>(daux4.z);
    pairs[h + 5u] = bitcast<u32>(daux4.w);

    // the forward's walk, as it ended
    var w = ray_walk(pr.o, pr.d, live0);
    let f = pix * 10u;
    w.col = vec3<f32>(totals[f], totals[f + 1u], totals[f + 2u]);
    w.nsum = vec3<f32>(totals[f + 3u], totals[f + 4u], totals[f + 5u]);
    w.dsum = totals[f + 6u];
    w.wsum = totals[f + 7u];
    w.dist = totals[f + 8u];
    w.tr = totals[f + 9u];
    var b = ray_back(w, live0, dimg4, daux4, gl, p.v.bg.xyz);
    let entries = 12u * n_px + 4u * offs[pix];
    let count = offs[pix + 1u] - offs[pix];
    for (var j = start; j < end && b.live; j = j + 1u) {
        let g = vals[j];
        let st = ray_back_join(b, g, b.win.joined, ray_record(g));
        b = st.b;
        if (st.leaves) { b = logged(b, st.out, entries, count); }
    }
    loop {
        let st = ray_back_drain(b);
        if (!st.leaves) { break; }
        b = logged(st.b, st.out, entries, count);
    }
    if (p.t.want_ray != 0u) {
        let i = pix * 6u;
        pairs[i] = bitcast<u32>(b.g_o.x);
        pairs[i + 1u] = bitcast<u32>(b.g_o.y);
        pairs[i + 2u] = bitcast<u32>(b.g_o.z);
        pairs[i + 3u] = bitcast<u32>(b.g_d.x);
        pairs[i + 4u] = bitcast<u32>(b.g_d.y);
        pairs[i + 5u] = bitcast<u32>(b.g_d.z);
    }
}
