// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  BatchNorm forward using BATCH statistics, NCHW tensor x[N,C,H,W]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// BatchNorm forward using BATCH statistics, NCHW tensor x[N,C,H,W].
//   y = (x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c],  eps = 1e-5
// One invocation per output element (N*C*H*W threads).
//
// To stay <=4 storage buffers the per-channel stats and affine params are
// passed as INTERLEAVED [2C] arrays:
//   mv[2c] = mean[c],  mv[2c+1] = var[c]
//   gb[2c] = gamma[c], gb[2c+1] = beta[c]
// Activation index: ((n*C+c)*H+h)*W+w; channel = (idx / (H*W)) % C.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       mv:  array<f32>;
@group(0) @binding(3) var<storage, read>       gb:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

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

    let mean = mv[2u * c];
    let va = mv[2u * c + 1u];
    let gamma = gb[2u * c];
    let beta = gb[2u * c + 1u];
    let inv = inverseSqrt(va + 1e-5);
    out[idx] = (x[idx] - mean) * inv * gamma + beta;
}
