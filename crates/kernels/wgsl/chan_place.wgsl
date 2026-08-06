// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Place a tensor into a contiguous channel range of a larger NCHW tensor
// @how   one thread per output element
// @opt   3
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
//
// Place a tensor into a contiguous channel range of a larger NCHW tensor:
//   dst[N, Ctot, H, W][c_off + c] = src[N, Csrc, H, W][c]
// The inverse of concat_split (which copies a channel range OUT). Used to build
// a multi-way channel concat in a single pass: each source chunk is written once
// into its slice of the destination (vs a left-fold that re-copies the growing
// prefix). One invocation per SOURCE element (n,c,h,w); writes are disjoint
// across chunks, so several chan_place dispatches can share one submit.

struct Params {
    N:     u32,
    Ctot:  u32,
    Csrc:  u32,
    c_off: u32,
    H:     u32,
    W:     u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Csrc * p.H * p.W;
    if (idx >= total) { return; }

    // Decompose src flat index into (n,c,h,w).
    let w  = idx % p.W;
    let t1 = idx / p.W;
    let h  = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.Csrc;
    let n  = t2 / p.Csrc;

    let cd = c + p.c_off;
    let di = ((n * p.Ctot + cd) * p.H + h) * p.W + w;
    dst[di] = src[idx];
}
