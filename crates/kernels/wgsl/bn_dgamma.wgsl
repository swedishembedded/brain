// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// BatchNorm backward w.r.t. gamma. One invocation per channel (C threads).
//   dgamma[c] += sum_{n,h,w} dy * xhat,   xhat = (x-mean)/sqrt(var+eps), eps=1e-5
// Accumulates into the (pre-zeroed) grad buffer.
//
// `mv` is [2C] interleaved (mean|var):  mv[2c]=mean[c], mv[2c+1]=var[c].
// Activation index: ((n*C+c)*H+h)*W+w.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       dy:     array<f32>;
@group(0) @binding(3) var<storage, read>       mv:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dgamma: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let c = gidx;
    if (c >= p.C) { return; }
    let N = p.N;
    let C = p.C;
    let H = p.H;
    let W = p.W;

    let mean = mv[2u * c];
    let va = mv[2u * c + 1u];
    let inv = inverseSqrt(va + 1e-5);

    var acc = 0.0;
    for (var n: u32 = 0u; n < N; n = n + 1u) {
        for (var h: u32 = 0u; h < H; h = h + 1u) {
            for (var w: u32 = 0u; w < W; w = w + 1u) {
                let i = ((n * C + c) * H + h) * W + w;
                let xhat = (x[i] - mean) * inv;
                acc = acc + dy[i] * xhat;
            }
        }
    }
    dgamma[c] = dgamma[c] + acc;
}
