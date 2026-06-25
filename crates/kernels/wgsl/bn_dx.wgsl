// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// BatchNorm backward w.r.t. x. One invocation per element (N*C*H*W threads).
// With M = N*H*W, xhat = (x-mean)/sqrt(var+eps):
//   dx = (gamma/sqrt(var+eps)) * (dy - dsum/M - xhat*dxhat_sum/M)
//
// `bp` is the [5C] packed buffer from bn_dstats (stride 5 per channel):
//   bp[5c+0]=mean  bp[5c+1]=var  bp[5c+2]=gamma  bp[5c+3]=dsum  bp[5c+4]=dxhat_sum
// Activation index: ((n*C+c)*H+h)*W+w; channel = (idx / (H*W)) % C.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read>       bp: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }
    let hw = p.H * p.W;
    let c = (idx / hw) % p.C;
    let M = f32(p.N * p.H * p.W);

    let mean = bp[5u * c + 0u];
    let va = bp[5u * c + 1u];
    let gamma = bp[5u * c + 2u];
    let dsum = bp[5u * c + 3u];
    let dxhat_sum = bp[5u * c + 4u];

    let inv = inverseSqrt(va + 1e-5);
    let xhat = (x[idx] - mean) * inv;
    dx[idx] = (gamma * inv) * (dy[idx] - dsum / M - xhat * dxhat_sum / M);
}
