// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-pixel terms of the five masked moments of a scale-and-shift fit
// @how   one thread per pixel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The scale-and-shift-invariant depth loss fits, per image, (s, b) = argmin sum_M (s p + b - t)^2 over the valid
// pixels M, whose normal equations need five sums: a00 = sum p^2, a01 = sum p, a11 = |M|, b0 = sum p t and
// b1 = sum t. This kernel emits each pixel's contribution to all five; the HOST sums each plane per image in f64,
// the same split masked_l1.wgsl uses (a binary32 serial sum over a whole image is the larger error, and the
// readback is small beside the net's forward).
//
//   pred, tgt : [N * hw]      mask : [N * hw]  (exactly 0.0 or 1.0)
//   out : [N, 5, hw]  read_write; image n, term j at out[(n*5 + j)*hw + i], j = 0..4 in the order
//         m p^2, m p, m, m p t, m t
//
// Pair: ssi_resid.wgsl (the residual under the solved fit), grad_match.wgsl, ssi_grad.wgsl. The whole objective is
// derived in `zipdepth::loss`.

struct Params {
    total: u32,   // N * hw
    hw: u32,      // pixels per image
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:  array<f32>;
@group(0) @binding(3) var<storage, read>       mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let n = idx / p.hw;
    let i = idx % p.hw;
    let base = n * 5u * p.hw + i;
    let m = mask[idx];
    let mp = m * pred[idx];
    out[base] = mp * pred[idx];
    out[base + p.hw] = mp;
    out[base + 2u * p.hw] = m;
    out[base + 3u * p.hw] = mp * tgt[idx];
    out[base + 4u * p.hw] = m * tgt[idx];
}
