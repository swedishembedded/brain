// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Concatenate two NCHW tensors along the channel axis
// @how   one thread per output element
// @opt   3
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
//
// Concatenate two NCHW tensors along the channel axis:
//   y[N, Ca+Cb, H, W]  from  a[N,Ca,H,W] and b[N,Cb,H,W].
// One invocation per OUTPUT element (n,c,h,w). If c < Ca read a at channel c,
// else read b at channel c-Ca. Output channel count is Ca+Cb.

struct Params {
    N:  u32,
    Ca: u32,
    Cb: u32,
    H:  u32,
    W:  u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let Ctot = p.Ca + p.Cb;
    let total = p.N * Ctot * p.H * p.W;
    if (idx >= total) { return; }

    // Decompose output flat index into (n,c,h,w).
    let w  = idx % p.W;
    let t1 = idx / p.W;
    let h  = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % Ctot;
    let n  = t2 / Ctot;

    if (c < p.Ca) {
        let ai = ((n * p.Ca + c) * p.H + h) * p.W + w;
        y[idx] = a[ai];
    } else {
        let cb = c - p.Ca;
        let bi = ((n * p.Cb + cb) * p.H + h) * p.W + w;
        y[idx] = b[bi];
    }
}
