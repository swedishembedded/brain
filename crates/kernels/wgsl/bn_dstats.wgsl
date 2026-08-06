// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  BatchNorm backward: per-channel reduced sums for the input-grad formula
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// BatchNorm backward: per-channel reduced sums for the input-grad formula.
// One invocation per channel (C threads). NCHW tensor x[N,C,H,W].
//   xhat       = (x - mean) / sqrt(var + eps),  eps = 1e-5
//   dsum[c]      = sum_{n,h,w} dy
//   dxhat_sum[c] = sum_{n,h,w} dy * xhat
//
// INPUT packing — `mvg` is [3C] interleaved (mean|var|gamma):
//   mvg[3c] = mean[c], mvg[3c+1] = var[c], mvg[3c+2] = gamma[c]
// OUTPUT packing — `bp` is [5C], stride 5 per channel:
//   bp[5c+0] = mean[c]   (copied through)
//   bp[5c+1] = var[c]    (copied through)
//   bp[5c+2] = gamma[c]  (copied through)
//   bp[5c+3] = dsum[c]
//   bp[5c+4] = dxhat_sum[c]
// This `bp` layout is consumed by bn_dx so the chain stays <=4 buffers.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       dy:  array<f32>;
@group(0) @binding(3) var<storage, read>       mvg: array<f32>;
@group(0) @binding(4) var<storage, read_write> bp:  array<f32>;

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

    let mean = mvg[3u * c];
    let va = mvg[3u * c + 1u];
    let gamma = mvg[3u * c + 2u];
    let inv = inverseSqrt(va + 1e-5);

    var dsum = 0.0;
    var dxhat_sum = 0.0;
    for (var n: u32 = 0u; n < N; n = n + 1u) {
        for (var h: u32 = 0u; h < H; h = h + 1u) {
            for (var w: u32 = 0u; w < W; w = w + 1u) {
                let i = ((n * C + c) * H + h) * W + w;
                let d = dy[i];
                let xhat = (x[i] - mean) * inv;
                dsum = dsum + d;
                dxhat_sum = dxhat_sum + d * xhat;
            }
        }
    }

    bp[5u * c + 0u] = mean;
    bp[5u * c + 1u] = va;
    bp[5u * c + 2u] = gamma;
    bp[5u * c + 3u] = dsum;
    bp[5u * c + 4u] = dxhat_sum;
}
