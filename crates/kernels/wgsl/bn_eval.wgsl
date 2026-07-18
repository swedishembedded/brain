// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// BatchNorm forward for INFERENCE using RUNNING statistics, NCHW x[N,C,H,W].
// Identical math/packing to bn_train (separate name for clarity); the caller
// passes running mean/var in `mv` instead of batch stats.
//   y = act((x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]),  eps = 1e-5
// One invocation per output element (N*C*H*W threads).
//   mv[2c] = mean[c],  mv[2c+1] = var[c]
//   gb[2c] = gamma[c], gb[2c+1] = beta[c]
//
// `act` selects the fused activation like the conv_act* kernels (0 identity,
// 1 relu, 2 silu, 3 sigmoid) — a BN followed by an activation is otherwise a
// second full-tensor pass AND a dependent dispatch hop, which is what actually
// costs on the measured Intel Arc. A caller that predates the field passes a
// 4-word uniform whose 16-byte pad reads as 0 = identity, i.e. the old
// behavior — on every backend (the CPU pads uniforms the same way).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    act: u32,
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
    var z = (x[idx] - mean) * inv * gamma + beta;
    if (p.act == 1u) { z = max(z, 0.0); }
    else if (p.act == 2u) { z = z / (1.0 + exp(-z)); }
    else if (p.act == 3u) { z = 1.0 / (1.0 + exp(-z)); }
    out[idx] = z;
}
