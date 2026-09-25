// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS backward, stage 1: replay each pixel's compositing walk and write its gradient partials into a fixed (instance, channel, pixel) slot grid (gsplat blend-backward math)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The backward's first stage, over a BAND of tiles [tile0, tile0 + n/256).
// One thread per pixel of the band: thread i is local pixel i % 256 of tile
// tile0 + i / 256. Each pixel walks its tile's depth-sorted instance list
// exactly as the forward composited it and, for EVERY instance k in that list,
// writes 11 partials to
//
//   slots[((k - k0) * 11 + c) * 256 + local]
//
// - zeros where the instance does not touch the pixel or the walk has already
// saturated - so no clear is needed and splat_bwd_tile_reduce can sum each
// instance's 256 pixels as one fixed-size contiguous block, with no sort.
// Within a warp the 32 lanes are consecutive pixels of one tile walking the
// same list, so the writes are coalesced.
//
// Channels: {d sigma/d mean_x, d sigma/d mean_y, 3 conic partials, opacity,
// weighted r, g, b, weighted depth, |d/d mean_xy|}. The last is the per-pixel
// MAGNITUDE of the position partial (AbsGS), which a per-instance sum could not
// recover afterwards.
//
// The colour suffix S needed by v_alpha comes from first accumulating the
// pixel's full colour, then subtracting contributions as the walk advances.
// dimg is the upstream RGBA gradient (dL/d rgb, dL/d alpha_out) and ddepth
// dL/d(sum_i z_i alpha_i T_i), the UNNORMALIZED accumulated depth; a caller
// supervising expected depth splits its gradient between the two
// (`splat::renderer::add_expected_depth_vjp`).

struct Params {
    width: u32,
    height: u32,
    tiles_x: u32,
    tile0: u32,   // first tile of the band
    k0: u32,      // first instance of the band (the band's slot origin)
    n: u32,       // threads in the band: tiles * 256
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:    array<f32>; // N*9
@group(0) @binding(2) var<storage, read>       colors:  array<f32>; // N*3
@group(0) @binding(3) var<storage, read>       vals:    array<u32>; // sorted instance -> gaussian id
@group(0) @binding(4) var<storage, read>       ranges:  array<u32>; // n_tiles*2
@group(0) @binding(5) var<storage, read>       dimg:    array<f32>; // W*H*4
@group(0) @binding(6) var<storage, read>       ddepth:  array<f32>; // W*H
@group(0) @binding(7) var<storage, read_write> slots:   array<f32>; // band_instances*11*256

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    let tile = p.tile0 + idx / 256u;
    let local = idx % 256u;
    let px = (tile % p.tiles_x) * 16u + local % 16u;
    let py = (tile / p.tiles_x) * 16u + local / 16u;
    let inside = px < p.width && py < p.height;
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    let fx = f32(px) + 0.5;
    let fy = f32(py) + 0.5;

    // Upstream gradients; a pixel past the frame edge has none and only
    // writes zeros.
    var vr = 0.0;
    var vg = 0.0;
    var vb = 0.0;
    var va_out = 0.0;
    var vd = 0.0;
    if (inside) {
        let abs = py * p.width + px;
        vr = dimg[abs * 4u];
        vg = dimg[abs * 4u + 1u];
        vb = dimg[abs * 4u + 2u];
        va_out = dimg[abs * 4u + 3u];
        vd = ddepth[abs];
    }

    // pass 1: total composited colour (to derive suffixes) + final T
    var t = 1.0;
    var sr = 0.0;
    var sg = 0.0;
    var sb = 0.0;
    var sd = 0.0;
    if (inside) {
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
            sd = sd + proj[o + 6u] * w;
            t = next_t;
        }
    }
    let t_final = t;

    // pass 2: the same walk, every instance of the list gets its 11 slots;
    // S becomes the strict suffix by subtracting each contribution as the walk
    // passes it.
    t = 1.0;
    var live = inside;
    for (var j = start; j < end; j = j + 1u) {
        var part: array<f32, 11>;
        for (var c = 0u; c < 11u; c = c + 1u) { part[c] = 0.0; }
        if (live) {
            let g = vals[j];
            let o = g * 9u;
            let dx = proj[o] - fx;
            let dy = proj[o + 1u] - fy;
            let ca = proj[o + 2u];
            let cb = proj[o + 3u];
            let cc = proj[o + 4u];
            let op = proj[o + 5u];
            let sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy;
            let vis = exp(-sigma);
            let alpha = min(0.99, op * vis);
            if (sigma >= 0.0 && alpha >= 1.0 / 255.0) {
                let next_t = t * (1.0 - alpha);
                if (next_t <= 1e-4) {
                    live = false;
                } else {
                    let w = alpha * t;
                    let cr = colors[g * 3u];
                    let cg2 = colors[g * 3u + 1u];
                    let cb2 = colors[g * 3u + 2u];
                    let zg = proj[o + 6u];
                    sr = sr - cr * w;
                    sg = sg - cg2 * w;
                    sb = sb - cb2 * w;
                    sd = sd - zg * w;
                    let om = 1.0 - alpha;
                    // v_alpha: own colour at T_i minus what this alpha
                    // attenuates behind it (suffix colours + background + the
                    // alpha output). Depth composites by the same weights.
                    var v_alpha = (cr * t - sr / om) * vr + (cg2 * t - sg / om) * vg
                        + (cb2 * t - sb / om) * vb;
                    v_alpha = v_alpha + (zg * t - sd / om) * vd;
                    v_alpha = v_alpha + (t_final / om) * va_out;
                    v_alpha = v_alpha - (t_final / om) * (p.bg_r * vr + p.bg_g * vg + p.bg_b * vb);
                    // through alpha = op*vis (zero grad when clamped at 0.99)
                    var v_sigma = 0.0;
                    var v_op = 0.0;
                    if (op * vis < 0.99) {
                        v_sigma = -op * vis * v_alpha;
                        v_op = vis * v_alpha;
                    }
                    part[0] = v_sigma * (ca * dx + cb * dy);
                    part[1] = v_sigma * (cb * dx + cc * dy);
                    part[2] = v_sigma * 0.5 * dx * dx;
                    part[3] = v_sigma * dx * dy;
                    part[4] = v_sigma * 0.5 * dy * dy;
                    part[5] = v_op;
                    part[6] = w * vr;
                    part[7] = w * vg;
                    part[8] = w * vb;
                    part[9] = w * vd;
                    part[10] = sqrt(part[0] * part[0] + part[1] * part[1]);
                    t = next_t;
                }
            }
        }
        let base = (j - p.k0) * 11u * 256u + local;
        for (var c = 0u; c < 11u; c = c + 1u) {
            slots[base + c * 256u] = part[c];
        }
    }
}
