// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  BatchNorm training batch statistics for an NCHW tensor x[N,C,H,W]
// @how   one thread per output element, 6 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// BatchNorm training batch statistics for an NCHW tensor x[N,C,H,W].
// Normalizes per-channel over (N,H,W): one invocation per channel c (C threads).
//   mean[c] = mean_{n,h,w}(x);  var[c] = mean_{n,h,w}((x-mean)^2)  (population/biased)
// Activation index: ((n*C+c)*H+h)*W+w.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> vari: array<f32>;

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
    let M = f32(N * H * W);

    var m = 0.0;
    for (var n: u32 = 0u; n < N; n = n + 1u) {
        for (var h: u32 = 0u; h < H; h = h + 1u) {
            for (var w: u32 = 0u; w < W; w = w + 1u) {
                m = m + x[((n * C + c) * H + h) * W + w];
            }
        }
    }
    m = m / M;

    var va = 0.0;
    for (var n: u32 = 0u; n < N; n = n + 1u) {
        for (var h: u32 = 0u; h < H; h = h + 1u) {
            for (var w: u32 = 0u; w < W; w = w + 1u) {
                let d = x[((n * C + c) * H + h) * W + w] - m;
                va = va + d * d;
            }
        }
    }
    mean[c] = m;
    vari[c] = va / M;
}
