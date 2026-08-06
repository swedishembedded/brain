// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Conv-as-GEMM epilogue: per-channel affine (BN-eval collapsed) + activation
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Conv-as-GEMM epilogue: per-channel affine (BN-eval collapsed) + activation.
//   dst[c, hw] = act( src[c, hw] * sb[2c] + sb[2c+1] )
// `src` is the raw matmul_reg2 conv output [Cout, Ho*Wo]; `sb` is the same
// [scale|bias] pair-per-channel `conv_act_reg` consumes, so im2col +
// matmul_reg2 + this epilogue reproduces the fused conv exactly. act: 0 = id,
// 1 = ReLU, 2 = SiLU, 3 = sigmoid. One invocation per output element.

struct Params { cout: u32, hw: u32, act: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       sb:  array<f32>;
@group(0) @binding(2) var<storage, read>       src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.cout * p.hw;
    if (idx >= total) { return; }
    let c = idx / p.hw;
    var z = src[idx] * sb[2u * c] + sb[2u * c + 1u];
    if (p.act == 1u) { z = max(z, 0.0); }
    else if (p.act == 2u) { z = z / (1.0 + exp(-z)); }
    else if (p.act == 3u) { z = 1.0 / (1.0 + exp(-z)); }
    dst[idx] = z;
}
