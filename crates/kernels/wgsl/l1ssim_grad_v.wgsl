// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  L1 + D-SSIM (or MSE) loss, pass 4: vertical adjoint pass and the per-pixel gradient with respect to the prediction, as RGBA
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// mode 1 (L1 + D-SSIM): finishes carrying the SSIM partials back through the
// window (vertical half), and per channel
//
//   dL/dx = (1 - lam) m sign(x - y) / norm + b_mx + 2 x b_mxx + y b_mxy
//
// where b_* are the window-carried partials of the mean, second moment and
// cross moment. The per-pixel loss was written by l1ssim_map_v.
//
// mode 0 (MSE): dL/dx = 2 m (x - y) / norm, and this pass also writes the
// per-pixel loss; the three SSIM passes do not run and `hpart` is unread.
//
// The gradient is written as RGBA (alpha 0), the upstream layout the
// rasterizer's backward consumes, so a fit with no camera model hands it on
// without a host round trip.

// The window tap at offset i is exp(-i^2 / 4.5) * 0.266011725, the latter
// being 1 / (sum of the 11 unnormalized taps), so the window sums to 1.

struct Params {
    w: u32,
    h: u32,
    lam: f32,
    inv: f32,
    mode: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       hpart:  array<f32>; // 9*W*H
@group(0) @binding(2) var<storage, read>       pred:   array<f32>; // W*H*4
@group(0) @binding(3) var<storage, read>       truth: array<f32>; // W*H*3
@group(0) @binding(4) var<storage, read>       weight: array<f32>; // W*H
@group(0) @binding(5) var<storage, read_write> dimg:   array<f32>; // W*H*4
@group(0) @binding(6) var<storage, read_write> loss:   array<f32>; // W*H

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let px = p.w * p.h;
    if (idx >= px) { return; }
    let m = weight[idx];
    if (p.mode == 0u) {
        var sq = 0.0;
        for (var c = 0u; c < 3u; c = c + 1u) {
            let d = pred[idx * 4u + c] - truth[idx * 3u + c];
            sq = sq + d * d;
            dimg[idx * 4u + c] = 2.0 * p.inv * m * d;
        }
        dimg[idx * 4u + 3u] = 0.0;
        loss[idx] = p.inv * m * sq;
        return;
    }
    let x = idx % p.w;
    let y = i32(idx / p.w);
    for (var c = 0u; c < 3u; c = c + 1u) {
        var bmx = 0.0;
        var bmxx = 0.0;
        var bmxy = 0.0;
        for (var i = -5; i <= 5; i = i + 1) {
            let yy = y + i;
            if (yy < 0 || yy >= i32(p.h)) { continue; }
            let src = u32(yy) * p.w + x;
            let kk = exp(-f32(i * i) / 4.5) * 0.266011725;
            bmx = bmx + kk * hpart[c * px + src];
            bmxx = bmxx + kk * hpart[(3u + c) * px + src];
            bmxy = bmxy + kk * hpart[(6u + c) * px + src];
        }
        let a = pred[idx * 4u + c];
        let b = truth[idx * 3u + c];
        let d = a - b;
        var sgn = 0.0;
        if (d > 0.0) { sgn = 1.0; } else if (d < 0.0) { sgn = -1.0; }
        dimg[idx * 4u + c] = (1.0 - p.lam) * p.inv * m * sgn + bmx + 2.0 * a * bmxx + b * bmxy;
    }
    dimg[idx * 4u + 3u] = 0.0;
}
