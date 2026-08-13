// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled 3DGS, stage 5: front-to-back compositing
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Tiled 3DGS, stage 5: front-to-back compositing. One 64-invocation workgroup
// owns one 16×16 tile (dispatch threads = n_tiles*64; tile = idx/64), each
// invocation owns 4 pixels (lane, lane+64, lane+128, lane+192). All 64
// invocations walk the SAME sorted range, so loads are coherent — no
// workgroup memory, no barriers (CPU-JIT safe by construction).
// mode 0 = color, 1 = expected depth. img is RGBA f32.

struct Params {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    mode: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:   array<f32>; // N*9
@group(0) @binding(2) var<storage, read>       colors: array<f32>; // N*3
@group(0) @binding(3) var<storage, read>       vals:   array<u32>; // sorted gaussian ids
@group(0) @binding(4) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(5) var<storage, read_write> img:    array<f32>; // W*H*4

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let tile = idx / 64u;
    let lane = idx % 64u;
    if (tile >= p.tiles_x * p.tiles_y) { return; }
    let tile_x = tile % p.tiles_x;
    let tile_y = tile / p.tiles_x;

    // Per-slot pixel coords + accumulators (k-th pixel = lane + k*64).
    var fx: array<f32, 4>;
    var fy: array<f32, 4>;
    var live: array<u32, 4>; // 0 = off-image or terminated
    var tr: array<f32, 4>;
    var cr: array<f32, 4>;
    var cg: array<f32, 4>;
    var cb: array<f32, 4>;
    var dep: array<f32, 4>;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let pix = lane + k * 64u;
        let px = tile_x * 16u + (pix % 16u);
        let py = tile_y * 16u + (pix / 16u);
        fx[k] = f32(px) + 0.5;
        fy[k] = f32(py) + 0.5;
        live[k] = u32(px < p.width && py < p.height);
        tr[k] = 1.0;
        cr[k] = 0.0;
        cg[k] = 0.0;
        cb[k] = 0.0;
        dep[k] = 0.0;
    }

    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let g = vals[j];
        let o = g * 9u;
        let mx = proj[o];
        let my = proj[o + 1u];
        let ca = proj[o + 2u];
        let cbn = proj[o + 3u];
        let cc = proj[o + 4u];
        let op = proj[o + 5u];
        let z = proj[o + 6u];
        let gr = colors[g * 3u];
        let gg = colors[g * 3u + 1u];
        let gb = colors[g * 3u + 2u];
        var any_live = 0u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            if (live[k] == 0u) { continue; }
            any_live = 1u;
            let dx = mx - fx[k];
            let dy = my - fy[k];
            let sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cbn * dx * dy;
            if (sigma < 0.0) { continue; }
            let alpha = min(0.99, op * exp(-sigma));
            if (alpha < 1.0 / 255.0) { continue; }
            let next_t = tr[k] * (1.0 - alpha);
            if (next_t <= 1e-4) {
                // Terminating gaussian excluded (gsplat semantics): T keeps
                // its pre-termination value, the pixel just stops.
                live[k] = 0u;
                continue;
            }
            let w = alpha * tr[k];
            cr[k] = cr[k] + gr * w;
            cg[k] = cg[k] + gg * w;
            cb[k] = cb[k] + gb * w;
            dep[k] = dep[k] + z * w;
            tr[k] = next_t;
        }
        if (any_live == 0u) { break; }
    }

    for (var k = 0u; k < 4u; k = k + 1u) {
        let pix = lane + k * 64u;
        let px = tile_x * 16u + (pix % 16u);
        let py = tile_y * 16u + (pix / 16u);
        if (px >= p.width || py >= p.height) { continue; }
        let t = tr[k];
        let a = 1.0 - t;
        let o = (py * p.width + px) * 4u;
        if (p.mode == 1u) {
            var d = 0.0;
            if (a > 1e-6) { d = dep[k] / a; }
            img[o] = d;
            img[o + 1u] = d;
            img[o + 2u] = d;
        } else {
            img[o] = cr[k] + t * p.bg_r;
            img[o + 1u] = cg[k] + t * p.bg_g;
            img[o + 2u] = cb[k] + t * p.bg_b;
        }
        img[o + 3u] = a;
    }
}
