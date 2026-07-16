// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// weighted_gap gradient wrt the FEATURE MAP.
//   dy : [N, C]
//   m  : [N, 1, H*W]
//   dx : [N, C, H*W]   read_write (one invocation per INPUT element)
//
//   dx[n,c,hw] = dy[n,c] * m[n,hw]
//
// The product is bilinear, so each argument's adjoint is a scaled broadcast of
// the other. Pure gather, no accumulation.

struct Params {
    N: u32,
    C: u32,
    HW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       m:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.HW;
    if (idx >= total) { return; }
    let i  = idx % p.HW;
    let t1 = idx / p.HW;
    let c  = t1 % p.C;
    let n  = t1 / p.C;
    dx[idx] = dy[n * p.C + c] * m[n * p.HW + i];
}
