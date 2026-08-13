// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Concat backward / channel-slice copy
// @how   one thread per output element
// @opt   3
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Concat backward / channel-slice copy. Copies a contiguous channel range out of
// a source-grad dy[N,Ctot,H,W] into da[N,Csrc,H,W], reading dy at channel
// c + c_off. One invocation per OUTPUT element of `da` (n,c,h,w).
//
// To split a concat2 gradient back to its two inputs, run this twice:
//   - for `a`: Csrc=Ca, c_off=0
//   - for `b`: Csrc=Cb, c_off=Ca

struct Params {
    N:     u32,
    Ctot:  u32,
    Csrc:  u32,
    c_off: u32,
    H:     u32,
    W:     u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> da: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Csrc * p.H * p.W;
    if (idx >= total) { return; }

    // Decompose da flat index into (n,c,h,w).
    let w  = idx % p.W;
    let t1 = idx / p.W;
    let h  = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.Csrc;
    let n  = t2 / p.Csrc;

    let cy = c + p.c_off;
    let yi = ((n * p.Ctot + cy) * p.H + h) * p.W + w;
    da[idx] = dy[yi];
}
