// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nearest-neighbour x2 upsample
// @how   one thread per output element
// @opt   3
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
//
// Nearest-neighbour x2 upsample. One invocation per OUTPUT element. Output is
// 2H x 2W; y[n,c,ho,wo] = x[n,c,ho/2,wo/2]. NCHW layout. Params carry the INPUT
// dims (N,C,H,W); the output grid is N*C*(2H)*(2W).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
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
    let OH = p.H * 2u;
    let OW = p.W * 2u;
    let total = p.N * p.C * OH * OW;
    if (idx >= total) { return; }

    // Decompose output flat index into (n,c,ho,wo).
    let wo = idx % OW;
    let t1 = idx / OW;
    let ho = t1 % OH;
    let t2 = t1 / OH;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let hi = ho / 2u;
    let wi = wo / 2u;
    let ii = ((n * p.C + c) * p.H + hi) * p.W + wi;
    y[idx] = x[ii];
}
