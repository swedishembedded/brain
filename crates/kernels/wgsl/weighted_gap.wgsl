// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Weighted global average pool: contract a feature map against a 1-channel
// spatial weight map, per image.
//   x : [N, C, H*W]
//   m : [N, 1, H*W]   (softmax'd attention weights, summing to 1 per image)
//   y : [N, C]        y[n,c] = sum_hw x[n,c,hw] * m[n,hw]
//
// ZipDepth's GlobalContextBlock does this as a bmm:
//   bmm(x.view(B,C,HW), mask.view(B,HW,1)) -> [B,C,1,1]
// i.e. a LEARNED weighted global average pool — the weights come from a conv +
// softmax rather than being uniform 1/HW.
//
// Expressed as its own kernel rather than through `matmul`: matmul is a 2D
// [M,K]x[K,N] GEMM with no batch axis, so a per-image contraction would need N
// separate dispatches with sliced buffers. One invocation per (n,c) reducing
// serially over HW is both simpler and the same amount of arithmetic.
//
// NOTE this is the block whose semantics upstream's ONNX export silently
// replaces with a uniform avg_pool2d (dropping the learned softmax). brain emits
// the real thing.

struct Params {
    N: u32,
    C: u32,
    HW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       m: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C;
    if (idx >= total) { return; }
    let c = idx % p.C;
    let n = idx / p.C;
    let xb = (n * p.C + c) * p.HW;
    let mb = n * p.HW;
    var acc = 0.0;
    for (var i: u32 = 0u; i < p.HW; i = i + 1u) {
        acc = acc + x[xb + i] * m[mb + i];
    }
    y[idx] = acc;
}
