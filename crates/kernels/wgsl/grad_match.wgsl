// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Multi-scale gradient-matching loss terms and their gradient wrt the residual
// @how   one thread per pixel, gathering its neighbours on each of four scales
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The MiDaS gradient-matching term over the residual r of ssi_resid.wgsl:
//
//   L = sum_n c_n sum_{k=0}^{3} sum_i M^k_i ( |r^k_{i+dx} - r^k_i| + |r^k_{i+dy} - r^k_i| )
//
// with r^k the residual subsampled by 2^k (every 2^k-th pixel in both axes, as MiDaS does), the differences taken
// between adjacent subsampled pixels, i.e. 2^k apart in the image, and a difference counted only where BOTH its
// pixels are valid. c_n = 1 / (N |M_n|) folds the per-image valid count and the batch mean in.
//
// One thread owns one pixel and writes three things, so no atomics are needed:
//   term[i] : c_n * the differences this pixel OWNS (its right and down neighbours on each grid it lies on)
//   g[i]    : dL/dr_i = c_n * sum of the signs of every difference the pixel takes part in (a pixel is the
//             minuend of the pair to its left/top and the subtrahend of the pair to its right/bottom)
//   gp[i]   : g[i] * pred[i], whose per-image sum is the sum_j g_j p_j the host needs for the through-(s,b) term
// sign(0) = 0: |d| has no derivative at 0 and 0 is the subgradient the reference selects.
//
//   resid, pred, mask : [N * hw] (mask exactly 0.0 or 1.0)    c : [N]
//   term, g, gp : [N * hw]  read_write
//
// Cost (gpu_core::cost) counts every difference as valid; a masked difference skips its arithmetic.

struct Params {
    total: u32,   // N * hw
    hw: u32,      // pixels per image
    h: u32,       // image height
    w: u32,       // image width, hw = h * w
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       resid: array<f32>;
@group(0) @binding(2) var<storage, read>       pred:  array<f32>;
@group(0) @binding(3) var<storage, read>       mask:  array<f32>;
@group(0) @binding(4) var<storage, read>       c:     array<f32>;
@group(0) @binding(5) var<storage, read_write> term:  array<f32>;
@group(0) @binding(6) var<storage, read_write> g:     array<f32>;
@group(0) @binding(7) var<storage, read_write> gp:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let n = idx / p.hw;
    let i = idx % p.hw;
    let y = i / p.w;
    let x = i % p.w;
    let img = n * p.hw;
    let r0 = resid[idx];
    let m0 = mask[idx];
    var acc_term = 0.0;
    var acc_g = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let step = 1u << k;
        // Only pixels on the 2^k grid take part at scale k.
        if (y % step == 0u && x % step == 0u) {
            if (x + step < p.w) {
                let j = img + i + step;
                let v = m0 * mask[j];
                let d = resid[j] - r0;
                acc_term = acc_term + v * abs(d);
                acc_g = acc_g - v * sign(d);
            }
            if (y + step < p.h) {
                let j = img + i + step * p.w;
                let v = m0 * mask[j];
                let d = resid[j] - r0;
                acc_term = acc_term + v * abs(d);
                acc_g = acc_g - v * sign(d);
            }
            if (x >= step) {
                let j = img + i - step;
                let v = m0 * mask[j];
                acc_g = acc_g + v * sign(r0 - resid[j]);
            }
            if (y >= step) {
                let j = img + i - step * p.w;
                let v = m0 * mask[j];
                acc_g = acc_g + v * sign(r0 - resid[j]);
            }
        }
    }
    let cn = c[n];
    term[idx] = cn * acc_term;
    let gi = cn * acc_g;
    g[idx] = gi;
    gp[idx] = gi * pred[idx];
}
