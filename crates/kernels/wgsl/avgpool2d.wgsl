// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Adaptive/box average-pool forward, NCHW, arbitrary output size
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Adaptive/box average-pool forward, NCHW, arbitrary output size.
//   x : [N, C, H,  W ]
//   y : [N, C, Ho, Wo]   one invocation per OUTPUT element
//
// Covers everything ZipDepth's default path pools with, via one index rule —
// torch's adaptive_avg_pool2d semantics, of which the fixed k=stride box pool is
// the exact divisor case:
//   h0 = floor(ho*H/Ho), h1 = ceil((ho+1)*H/Ho)   (likewise w)
//   y  = mean of x over [h0,h1) x [w0,w1)
//
// The three call sites this serves:
//   * ChannelAttention / SE:  adaptive_avg_pool2d(x, 1)      -> Ho=Wo=1
//   * MinimalCrossScale:      adaptive_avg_pool2d(x, x_low)  -> exact 2x2 at 24->12
//   * the loss's multi-scale gradient term: avg_pool2d(k) with stride==k
// Using the adaptive rule for all three means one kernel and no divisor
// special-case; at Ho|H it reduces to a plain box pool bit-for-bit.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    Ho: u32,
    Wo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.Ho * p.Wo;
    if (idx >= total) { return; }

    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // Integer floor/ceil: h1 = ceil((ho+1)*H/Ho) = ((ho+1)*H + Ho - 1)/Ho.
    let h0 = (ho * p.H) / p.Ho;
    let h1 = ((ho + 1u) * p.H + p.Ho - 1u) / p.Ho;
    let w0 = (wo * p.W) / p.Wo;
    let w1 = ((wo + 1u) * p.W + p.Wo - 1u) / p.Wo;

    let base = (n * p.C + c) * p.H;
    var acc = 0.0;
    for (var hi: u32 = h0; hi < h1; hi = hi + 1u) {
        for (var wi: u32 = w0; wi < w1; wi = wi + 1u) {
            acc = acc + x[(base + hi) * p.W + wi];
        }
    }
    y[idx] = acc / f32((h1 - h0) * (w1 - w0));
}
