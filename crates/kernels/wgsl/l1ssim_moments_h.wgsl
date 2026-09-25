// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  L1 + D-SSIM loss, pass 1: horizontal 11-tap gaussian sums of x, y, x^2, y^2, xy per channel
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The first of four separable passes of the SSIM term (Wang et al., IEEE TIP
// 2004; the window 3D Gaussian Splatting trains with: 11 taps, sigma 1.5,
// zero padding). One thread per pixel reads the prediction (RGBA, stride 4)
// and the truth (RGB, stride 3) along its row and writes the five window
// moments of each channel - the squares and the product are formed here, so
// they are never materialized as planes.
//
// Output `h` is 15 planes of W*H: plane (moment * 3 + channel), moments in
// the order x, y, x^2, y^2, xy.

// The window tap at offset i is exp(-i^2 / 4.5) * 0.266011725, the latter
// being 1 / (sum of the 11 unnormalized taps), so the window sums to 1.

struct Params {
    w: u32,
    h: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>; // W*H*4
@group(0) @binding(2) var<storage, read>       truth: array<f32>; // W*H*3
@group(0) @binding(3) var<storage, read_write> hsum:   array<f32>; // 15*W*H

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let px = p.w * p.h;
    if (idx >= px) { return; }
    let x = i32(idx % p.w);
    let y = idx / p.w;
    for (var c = 0u; c < 3u; c = c + 1u) {
        var sx = 0.0;
        var sy = 0.0;
        var sxx = 0.0;
        var syy = 0.0;
        var sxy = 0.0;
        for (var i = -5; i <= 5; i = i + 1) {
            let xx = x + i;
            if (xx < 0 || xx >= i32(p.w)) { continue; }
            let q = y * p.w + u32(xx);
            let k = exp(-f32(i * i) / 4.5) * 0.266011725;
            let a = pred[q * 4u + c];
            let b = truth[q * 3u + c];
            sx = sx + k * a;
            sy = sy + k * b;
            sxx = sxx + k * a * a;
            syy = syy + k * b * b;
            sxy = sxy + k * a * b;
        }
        hsum[c * px + idx] = sx;
        hsum[(3u + c) * px + idx] = sy;
        hsum[(6u + c) * px + idx] = sxx;
        hsum[(9u + c) * px + idx] = syy;
        hsum[(12u + c) * px + idx] = sxy;
    }
}
