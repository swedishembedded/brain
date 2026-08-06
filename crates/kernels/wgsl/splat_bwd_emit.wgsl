// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS backward, stage 2: replay each pixel's compositing walk and emit one gradient record per contributing gaussian (gsplat blend-backward math)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// 3DGS backward, stage 2: replay each pixel's compositing walk and emit one
// gradient record per contributing gaussian (gsplat blend-backward math).
// Forward walk recomputes each gaussian's T_i; the color-suffix S needed by
// v_alpha is obtained by first accumulating the pixel's full color, then
// subtracting contributions as the walk advances.
//
// Record stride 10 in `recs`: {v_xy(2), v_conic(3), v_opacity, v_rgb(3), pad};
// record keys (gaussian id) go to `keys` for the radix sort; offsets come
// from the scanned per-pixel counts. dimg is the upstream RGBA gradient
// (dL/d rgb, dL/d alpha_out). One invocation per pixel.

struct Params {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:    array<f32>; // N*9
@group(0) @binding(2) var<storage, read>       colors:  array<f32>; // N*3
@group(0) @binding(3) var<storage, read>       vals:    array<u32>; // sorted ids
@group(0) @binding(4) var<storage, read>       ranges:  array<u32>; // n_tiles*2
@group(0) @binding(5) var<storage, read>       dimg:    array<f32>; // W*H*4
@group(0) @binding(6) var<storage, read>       offsets: array<u32>; // W*H (scanned counts)
@group(0) @binding(7) var<storage, read_write> recs:    array<f32>; // cap*10 (+ keys via bitcast slot 9)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.width * p.height) { return; }
    let px = idx % p.width;
    let py = idx / p.width;
    let fx = f32(px) + 0.5;
    let fy = f32(py) + 0.5;
    let tile = (py / 16u) * p.tiles_x + (px / 16u);
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    let vr = dimg[idx * 4u];
    let vg = dimg[idx * 4u + 1u];
    let vb = dimg[idx * 4u + 2u];
    let va_out = dimg[idx * 4u + 3u];

    // pass 1: total composited color (to derive suffixes) + final T
    var t = 1.0;
    var sr = 0.0;
    var sg = 0.0;
    var sb = 0.0;
    for (var j = start; j < end; j = j + 1u) {
        let g = vals[j];
        let o = g * 9u;
        let dx = proj[o] - fx;
        let dy = proj[o + 1u] - fy;
        let sigma = 0.5 * (proj[o + 2u] * dx * dx + proj[o + 4u] * dy * dy)
            + proj[o + 3u] * dx * dy;
        if (sigma < 0.0) { continue; }
        let alpha = min(0.99, proj[o + 5u] * exp(-sigma));
        if (alpha < 1.0 / 255.0) { continue; }
        let next_t = t * (1.0 - alpha);
        if (next_t <= 1e-4) { break; }
        let w = alpha * t;
        sr = sr + colors[g * 3u] * w;
        sg = sg + colors[g * 3u + 1u] * w;
        sb = sb + colors[g * 3u + 2u] * w;
        t = next_t;
    }
    let t_final = t;

    // pass 2: same walk, emitting records; S becomes the strict suffix by
    // subtracting each contribution as we pass it.
    var pos = offsets[idx];
    t = 1.0;
    for (var j = start; j < end; j = j + 1u) {
        let g = vals[j];
        let o = g * 9u;
        let dx = proj[o] - fx;
        let dy = proj[o + 1u] - fy;
        let ca = proj[o + 2u];
        let cb = proj[o + 3u];
        let cc = proj[o + 4u];
        let op = proj[o + 5u];
        let sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy;
        if (sigma < 0.0) { continue; }
        let vis = exp(-sigma);
        let alpha = min(0.99, op * vis);
        if (alpha < 1.0 / 255.0) { continue; }
        let next_t = t * (1.0 - alpha);
        if (next_t <= 1e-4) { break; }
        let w = alpha * t;
        let cr = colors[g * 3u];
        let cg2 = colors[g * 3u + 1u];
        let cb2 = colors[g * 3u + 2u];
        sr = sr - cr * w;
        sg = sg - cg2 * w;
        sb = sb - cb2 * w;
        let om = 1.0 - alpha;
        // v_alpha: own color at T_i minus what this alpha attenuates behind it
        // (suffix colors + background + the alpha output).
        var v_alpha = (cr * t - sr / om) * vr + (cg2 * t - sg / om) * vg
            + (cb2 * t - sb / om) * vb;
        v_alpha = v_alpha + (t_final / om) * va_out;
        v_alpha = v_alpha - (t_final / om) * (p.bg_r * vr + p.bg_g * vg + p.bg_b * vb);
        // through alpha = op*vis (zero grad when clamped at 0.99)
        var v_sigma = 0.0;
        var v_op = 0.0;
        if (op * vis < 0.99) {
            v_sigma = -op * vis * v_alpha;
            v_op = vis * v_alpha;
        }
        let r = pos * 10u;
        recs[r] = v_sigma * (ca * dx + cb * dy);      // d sigma/d mean_x
        recs[r + 1u] = v_sigma * (cb * dx + cc * dy); // d sigma/d mean_y
        recs[r + 2u] = v_sigma * 0.5 * dx * dx;
        recs[r + 3u] = v_sigma * dx * dy;
        recs[r + 4u] = v_sigma * 0.5 * dy * dy;
        recs[r + 5u] = v_op;
        recs[r + 6u] = w * vr;
        recs[r + 7u] = w * vg;
        recs[r + 8u] = w * vb;
        recs[r + 9u] = bitcast<f32>(g); // gaussian id (sort key source)
        pos = pos + 1u;
        t = next_t;
    }
}
