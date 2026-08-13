// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Softmax over a STRIDED axis of length K, NCHW-flattened
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Softmax over a STRIDED axis of length K, NCHW-flattened.
//   x, y : [N, K, M]   normalized over the K axis, independently per (n, m)
//
// ZipDepth's FastConvexUpsample softmaxes over the 9 NEIGHBOUR axis of a
// [N, 9, S*S, H, W] mask — i.e. over a strided axis with M = S*S*H*W trailing
// elements. That is a different reduction from softmax_hw (which normalizes a
// whole contiguous feature map), hence a separate kernel.
//
// One invocation per (n, m) — i.e. per softmax GROUP, not per element — reducing
// serially over K. K is 9 here, so the loop is trivial and the strided reads stay
// coalesced across neighbouring invocations (consecutive m are adjacent).
//
// Max-subtracted for stability.

struct Params {
    N: u32,
    K: u32,
    M: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.M;
    if (idx >= total) { return; }
    let m = idx % p.M;
    let n = idx / p.M;
    let base = n * p.K * p.M + m;   // stride between K entries is M

    var mx = x[base];
    for (var k: u32 = 1u; k < p.K; k = k + 1u) {
        mx = max(mx, x[base + k * p.M]);
    }
    var s = 0.0;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        let e = exp(x[base + k * p.M] - mx);
        y[base + k * p.M] = e;
        s = s + e;
    }
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        y[base + k * p.M] = y[base + k * p.M] / s;
    }
}
