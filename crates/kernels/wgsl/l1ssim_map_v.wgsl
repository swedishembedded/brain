// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  L1 + D-SSIM loss, pass 2: vertical window sums, the SSIM map, the per-pixel loss and the SSIM partials
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Finishes the window moments with the vertical 11-tap pass over
// l1ssim_moments_h's planes, then per channel
//
//   S = A1 A2 / (B1 B2),  A1 = 2 mx my + C1, A2 = 2 cov + C2,
//                         B1 = mx^2 + my^2 + C1, B2 = var_x + var_y + C2
//
// with C1 = 0.01^2, C2 = 0.03^2, and writes:
//
// * `loss[p]`: the pixel's share of (1 - lam) L1 + lam (1 - SSIM), each a
//   weighted mean over supervised pixels and channels - summed on the host;
// * `part`: 9 planes, the partials of S with respect to the three window
//   moments of the PREDICTION that it depends on (mean, second moment, cross
//   moment; the truth's own moments carry no gradient), already scaled by
//   the pixel's weight and by -lam/norm. l1ssim_partials_h and l1ssim_grad_v
//   carry them back through the window, which is its own adjoint.

// The window tap at offset i is exp(-i^2 / 4.5) * 0.266011725, the latter
// being 1 / (sum of the 11 unnormalized taps), so the window sums to 1.

struct Params {
    w: u32,
    h: u32,
    lam: f32,   // SSIM weight
    inv: f32,   // 1 / (3 * sum of weights)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       hsum:   array<f32>; // 15*W*H
@group(0) @binding(2) var<storage, read>       pred:   array<f32>; // W*H*4
@group(0) @binding(3) var<storage, read>       truth: array<f32>; // W*H*3
@group(0) @binding(4) var<storage, read>       weight: array<f32>; // W*H
@group(0) @binding(5) var<storage, read_write> part:   array<f32>; // 9*W*H
@group(0) @binding(6) var<storage, read_write> loss:   array<f32>; // W*H

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let px = p.w * p.h;
    if (idx >= px) { return; }
    let x = idx % p.w;
    let y = i32(idx / p.w);
    let m = weight[idx];
    let c1 = 0.0001;
    let c2 = 0.0009;
    let k = -p.lam * p.inv;
    var ssim = 0.0;
    var l1 = 0.0;
    for (var c = 0u; c < 3u; c = c + 1u) {
        var mom: array<f32, 5>;
        for (var q = 0u; q < 5u; q = q + 1u) { mom[q] = 0.0; }
        for (var i = -5; i <= 5; i = i + 1) {
            let yy = y + i;
            if (yy < 0 || yy >= i32(p.h)) { continue; }
            let src = u32(yy) * p.w + x;
            let kk = exp(-f32(i * i) / 4.5) * 0.266011725;
            for (var q = 0u; q < 5u; q = q + 1u) {
                mom[q] = mom[q] + kk * hsum[(q * 3u + c) * px + src];
            }
        }
        let ux = mom[0];
        let uy = mom[1];
        let a1 = 2.0 * ux * uy + c1;
        let a2 = 2.0 * (mom[4] - ux * uy) + c2;
        let b1 = ux * ux + uy * uy + c1;
        let b2 = (mom[2] - ux * ux) + (mom[3] - uy * uy) + c2;
        let s = a1 * a2 / (b1 * b2);
        ssim = ssim + s;
        let d_ux = (2.0 * uy * a2 - 2.0 * uy * a1) / (b1 * b2) - s * (2.0 * ux / b1 - 2.0 * ux / b2);
        part[c * px + idx] = k * m * d_ux;
        part[(3u + c) * px + idx] = k * m * (-s / b2);
        part[(6u + c) * px + idx] = k * m * (2.0 * a1 / (b1 * b2));
        l1 = l1 + abs(pred[idx * 4u + c] - truth[idx * 3u + c]);
    }
    loss[idx] = p.inv * m * ((1.0 - p.lam) * l1 + p.lam * (3.0 - ssim));
}
