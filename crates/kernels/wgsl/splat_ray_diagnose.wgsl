// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated splat diagnostics per pixel: alpha, expected and median range, range spread, contribution entropy, dominant gaussian, normal
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
// What a frame of the ray renderer is made of, pixel by pixel: the same
// walk as `splat_ray_rasterize.wgsl` (the window library's join, drain and
// composite, in each pixel's own range order), with the statistics that tell
// a surface from a stack of translucent gaussians that only looks like one
// from the training cameras:
//
//   [0]  alpha out, 1 - T
//   [1]  expected range, sum w t / sum w
//   [2]  median range: where the accumulated weight first reaches 1/2 (0 =
//        never; the pixel is less than half covered)
//   [3]  range spread, sqrt(sum w t^2 / sum w - expected^2)
//   [4]  contribution entropy of p_i = w_i / sum w: 0 for one gaussian,
//        ln(k) for k equal ones
//   [5]  gaussians composited
//   [6]  the largest share w_i / sum w
//   [7]  the gaussian with that share (u32 bits; 0xffffffff = none)
//   [8..11) the composited normal, unit length (zero where there is none)
//   [11] sum w
//
// An opaque surface has spread near zero, low entropy and a dominant share
// near one; a stack cooperating to fake a colour has a large spread or a
// high entropy even where its expected range is right.

@group(0) @binding(0) var<uniform> p: View;
@group(0) @binding(1) var<storage, read>       ray:    array<vec4<f32>>; // N*4
@group(0) @binding(2) var<storage, read>       vals:   array<u32>;       // sorted gaussian ids
@group(0) @binding(3) var<storage, read>       ranges: array<u32>;       // n_tiles*2
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;       // W*H*12

const DIAG: u32 = 12u;

fn ray_record(g: u32) -> RayRec {
    return ray_rec4(ray[g * 4u], ray[g * 4u + 1u], ray[g * 4u + 2u], ray[g * 4u + 3u]);
}

struct Diag {
    w: RayWalk,
    t2: f32,     // sum w t^2
    wlogw: f32,  // sum w ln w
    median: f32,
    count: f32,
    wmax: f32,
    gmax: u32,
};

// Composite `h` into the walk and fold it into the statistics.
fn diag_composite(s: Diag, h: RayHeld, sh: RayShade) -> Diag {
    var d = s;
    let wt = h.alpha * d.w.tr;
    let before = d.w.wsum;
    d.w = ray_walk_composite(d.w, h, sh);
    if (d.w.wsum == before) { return d; } // the walk stopped short of it
    d.t2 = d.t2 + wt * h.t * h.t;
    d.wlogw = d.wlogw + wt * log(max(wt, 1e-30));
    if (d.median == 0.0 && d.w.wsum >= 0.5) { d.median = h.t; }
    d.count = d.count + 1.0;
    if (wt > d.wmax) {
        d.wmax = wt;
        d.gmax = h.g;
    }
    return d;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let tiles_x = (p.width + 15u) / 16u;
    let tiles_y = (p.height + 15u) / 16u;
    let tile = idx / 256u;
    if (tile >= tiles_x * tiles_y) { return; }
    let local = idx % 256u;
    let px = (tile % tiles_x) * 16u + local % 16u;
    let py = (tile / tiles_x) * 16u + local / 16u;
    if (px >= p.width || py >= p.height) { return; }
    let pr = pixel_ray(p, vec2<f32>(f32(px) + 0.5, f32(py) + 0.5));
    var d = Diag(ray_walk(pr.o, pr.d, pr.ok != 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, RAY_NO_GAUSSIAN);
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    for (var j = start; j < end && d.w.live; j = j + 1u) {
        let g = vals[j];
        let r = ray_record(g);
        let st = ray_window_join(d.w.win, ray_pair(r, d.w.o, d.w.d), r, g, 0u);
        d.w.win = st.win;
        if (st.leaves) { d = diag_composite(d, st.out, st.shade); }
    }
    loop {
        if (!d.w.live) { break; }
        let st = ray_window_take(d.w.win);
        if (!st.leaves) { break; }
        d.w.win = st.win;
        d = diag_composite(d, st.out, st.shade);
    }
    let o = (py * p.width + px) * DIAG;
    let sw = d.w.wsum;
    out[o] = 1.0 - d.w.tr;
    var mean = 0.0;
    var spread = 0.0;
    var entropy = 0.0;
    var share = 0.0;
    if (sw > 1e-8) {
        mean = d.w.dsum / sw;
        spread = sqrt(max(d.t2 / sw - mean * mean, 0.0));
        entropy = max(log(sw) - d.wlogw / sw, 0.0);
        share = d.wmax / sw;
    }
    out[o + 1u] = mean;
    out[o + 2u] = d.median;
    out[o + 3u] = spread;
    out[o + 4u] = entropy;
    out[o + 5u] = d.count;
    out[o + 6u] = share;
    out[o + 7u] = bitcast<f32>(d.gmax);
    let nl = length(d.w.nsum);
    var n = vec3<f32>(0.0, 0.0, 0.0);
    if (nl > 1e-8) { n = d.w.nsum / nl; }
    out[o + 8u] = n.x;
    out[o + 9u] = n.y;
    out[o + 10u] = n.z;
    out[o + 11u] = sw;
}
