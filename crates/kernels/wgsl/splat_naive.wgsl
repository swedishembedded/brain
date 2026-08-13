// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Naive 3DGS compositing - the correctness oracle and tiny-scene path
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Naive 3DGS compositing — the correctness oracle and tiny-scene path. One
// invocation per pixel, front-to-back over ALL projected gaussians in buffer
// order (the host pre-sorts by camera depth). No tiles, no sort kernels.
// mode 0 = color, 1 = expected depth ((sum w*z)/(1-T)) replicated to rgb.
// img is RGBA f32: rgb = color + T*bg, a = 1 - T.

struct Params {
    n: u32,
    width: u32,
    height: u32,
    mode: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:   array<f32>; // N*9
@group(0) @binding(2) var<storage, read>       colors: array<f32>; // N*3
@group(0) @binding(3) var<storage, read_write> img:    array<f32>; // W*H*4

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.width * p.height) { return; }
    let px = f32(idx % p.width) + 0.5;
    let py = f32(idx / p.width) + 0.5;

    var t = 1.0;
    var cr = 0.0;
    var cg = 0.0;
    var cb = 0.0;
    var dep = 0.0;
    for (var g = 0u; g < p.n; g = g + 1u) {
        let o = g * 9u;
        let rx = proj[o + 7u];
        if (rx <= 0.0) { continue; }
        let dx = proj[o] - px;
        let dy = proj[o + 1u] - py;
        // conic quadratic form; outside the truncation bbox contributes ~0 and
        // the alpha threshold below rejects it, matching the tiled path.
        if (abs(dx) > rx || abs(dy) > proj[o + 8u]) { continue; }
        let sigma = 0.5 * (proj[o + 2u] * dx * dx + proj[o + 4u] * dy * dy)
            + proj[o + 3u] * dx * dy;
        if (sigma < 0.0) { continue; }
        let alpha = min(0.99, proj[o + 5u] * exp(-sigma));
        if (alpha < 1.0 / 255.0) { continue; }
        let next_t = t * (1.0 - alpha);
        if (next_t <= 1e-4) {
            // gsplat semantics: the terminating gaussian is EXCLUDED.
            break;
        }
        let w = alpha * t;
        cr = cr + colors[g * 3u] * w;
        cg = cg + colors[g * 3u + 1u] * w;
        cb = cb + colors[g * 3u + 2u] * w;
        dep = dep + proj[o + 6u] * w;
        t = next_t;
    }
    let a = 1.0 - t;
    if (p.mode == 1u) {
        var d = 0.0;
        if (a > 1e-6) { d = dep / a; }
        img[idx * 4u] = d;
        img[idx * 4u + 1u] = d;
        img[idx * 4u + 2u] = d;
    } else {
        img[idx * 4u] = cr + t * p.bg_r;
        img[idx * 4u + 1u] = cg + t * p.bg_g;
        img[idx * 4u + 2u] = cb + t * p.bg_b;
    }
    img[idx * 4u + 3u] = a;
}
