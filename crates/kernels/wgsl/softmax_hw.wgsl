// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Softmax over the flattened SPATIAL positions of a 1-channel map, NCHW.
//   x : [N, 1, H, W]  ->  y : [N, 1, H, W], each image summing to 1 over H*W.
// One invocation per IMAGE (n), reducing serially over H*W — there are only N of
// them (N=1 at inference), so this is a tiny dispatch, and a serial reduction
// keeps it barrier-free and atomic-free.
//
// brain's existing attn_softmax* kernels are seq/head-shaped (they normalize a
// [T,T] score row); this normalizes a whole feature map, which is a different
// reduction axis, hence a separate kernel.
//
// ZipDepth's GlobalContextBlock uses it to build a LEARNED weighted global
// average pool: softmax over a 1-ch attention map, then bmm against the C-channel
// features. Note that upstream's ONNX export monkey-patches this whole block into
// a uniform avg_pool2d, DROPPING the softmax and changing the function; brain
// emits Softmax natively, so brain's export stays faithful to the trained model.
//
// Max-subtracted for numerical stability (two passes over H*W).

struct Params {
    N: u32,
    HW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.N) { return; }
    let base = n * p.HW;

    var m = x[base];
    for (var i: u32 = 1u; i < p.HW; i = i + 1u) {
        m = max(m, x[base + i]);
    }
    var s = 0.0;
    for (var i: u32 = 0u; i < p.HW; i = i + 1u) {
        let e = exp(x[base + i] - m);
        y[base + i] = e;
        s = s + e;
    }
    for (var i: u32 = 0u; i < p.HW; i = i + 1u) {
        y[base + i] = y[base + i] / s;
    }
}
