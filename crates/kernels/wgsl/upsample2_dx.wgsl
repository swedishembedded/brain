// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nearest-neighbour x2 upsample backward, GATHER form
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Nearest-neighbour x2 upsample backward, GATHER form. One invocation per INPUT
// element (n,c,hi,wi). Each input pixel feeds the 2x2 output block
// {2hi, 2hi+1} x {2wi, 2wi+1}, so its grad is the sum of dy over that block.
// NCHW layout. Params carry the INPUT dims; output dims are 2H x 2W.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let ii = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (ii >= total) { return; }

    // Decompose input flat index into (n,c,hi,wi).
    let wi = ii % p.W;
    let t1 = ii / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let OH = p.H * 2u;
    let OW = p.W * 2u;
    var acc: f32 = 0.0;
    for (var dh: u32 = 0u; dh < 2u; dh = dh + 1u) {
        let ho = hi * 2u + dh;
        for (var dw: u32 = 0u; dw < 2u; dw = dw + 1u) {
            let wo = wi * 2u + dw;
            let oi = ((n * p.C + c) * OH + ho) * OW + wo;
            acc = acc + dy[oi];
        }
    }
    dx[ii] = acc;
}
