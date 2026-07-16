// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Strip pooling INPUT gradient: broadcast dy back over the reduced axis, NCHW.
//   axis = 0: dy : [N, C, H, 1]  ->  dx[n,c,h,w] = dy[n,c,h] / W
//   axis = 1: dy : [N, C, 1, W]  ->  dx[n,c,h,w] = dy[n,c,w] / H
//
// The adjoint of strip_pool.wgsl. A mean over an axis has a trivial adjoint —
// every input on that axis receives an equal 1/len share — so this is a pure
// gather with one invocation per INPUT element and no atomics.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    axis: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }

    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let keep = select(p.W, p.H, p.axis == 0u);
    let k    = select(wi, hi, p.axis == 0u);
    let len  = select(p.H, p.W, p.axis == 0u);   // length of the REDUCED axis
    let dy_idx = (n * p.C + c) * keep + k;
    dx[idx] = dy[dy_idx] / f32(len);
}
