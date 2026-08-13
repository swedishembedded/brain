// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nearest-neighbour resize, NCHW, ARBITRARY output size
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Nearest-neighbour resize, NCHW, ARBITRARY output size.
//   x : [N, C, H,  W ]
//   y : [N, C, Ho, Wo]   one invocation per OUTPUT element
//
// The generic form of upsample2.wgsl, whose 2x factor is hardcoded. ZipDepth's
// MinimalCrossScale interpolates to an arbitrary target size (x_high's spatial
// dims), which 2x cannot express.
//
// Uses torch's `nearest` rule (== ONNX "asymmetric" + nearest_mode "floor"):
//   src = floor(o * in / out)
// integer arithmetic, so no floor() call and no half-pixel subtlety — unlike
// bilinear, the two nearest conventions coincide here for the ratios in use.

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
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.Ho * p.Wo;
    if (idx >= total) { return; }
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let nc = t1 / p.Ho;
    let hi = min((ho * p.H) / p.Ho, p.H - 1u);
    let wi = min((wo * p.W) / p.Wo, p.W - 1u);
    y[idx] = x[(nc * p.H + hi) * p.W + wi];
}
