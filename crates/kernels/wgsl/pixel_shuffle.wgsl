// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Pixel shuffle (depth-to-space) forward, NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Pixel shuffle (depth-to-space) forward, NCHW.
//   x : [N, C*S*S, H,   W  ]
//   y : [N, C,      H*S, W*S]   one invocation per OUTPUT element
//
// torch's F.pixel_shuffle / ONNX DepthToSpace with mode="CRD":
//   y[n, c, h*S + sh, w*S + sw] = x[n, (c*S + sh)*S + sw, h, w]
// i.e. the sub-pixel offsets are the FASTEST-varying part of the input channel
// index. The other convention (mode="DCR", offsets slowest) is a different
// permutation that produces an equally plausible-looking image — CRD is what
// torch does and therefore what ZipDepth's checkpoint expects.
//
// Used by FastConvexUpsample to fold the S*S sub-pixel predictions of each
// half-resolution pixel into the full-resolution map.

struct Params {
    N: u32,
    C: u32,    // OUTPUT channels
    H: u32,    // INPUT height
    W: u32,    // INPUT width
    S: u32,    // upscale factor
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
    let Ho = p.H * p.S;
    let Wo = p.W * p.S;
    let total = p.N * p.C * Ho * Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (n, c, ho, wo).
    let wo = idx % Wo;
    let t1 = idx / Wo;
    let ho = t1 % Ho;
    let t2 = t1 / Ho;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let h  = ho / p.S;
    let sh = ho % p.S;
    let w  = wo / p.S;
    let sw = wo % p.S;

    let cin = (c * p.S + sh) * p.S + sw;
    let x_idx = ((n * (p.C * p.S * p.S) + cin) * p.H + h) * p.W + w;
    y[idx] = x[x_idx];
}
