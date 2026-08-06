// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bidirectional (non-causal) attention scores with an additive per-head bias and a CONFIGURABLE scalar scale — the spatial-attention primitive for GenieRedux's ST transformer
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Bidirectional (non-causal) attention scores with an additive per-head bias
// and a CONFIGURABLE scalar scale — the spatial-attention primitive for
// GenieRedux's ST transformer:
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,h,:]) * scale + bias[h,i,j]
// `scale` is a caller-supplied constant (GenieRedux uses 8 with L2-normalized
// q,k, NOT the usual 1/sqrt(head_dim)). `bias` is the ContinuousPositionBias
// added to every (i,j) pair, shared across the batch: bias[(h*T + i)*T + j].
// q,k read from the fused qkv buffer. One invocation per (b,h,i,j).
// scores layout: ((b*H + h)*T + i)*T + j.

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,        // T
    head_dim: u32,
    qkv_stride: u32,   // 3*d_model (or q/k projection stride)
    q_off: u32,
    k_off: u32,
    scale: u32,        // f32 bits — the score multiplier
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       qkv:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias:   array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.bsz * p.n_heads * T * T;
    let idx = gidx;
    if (idx >= total) { return; }

    let j = idx % T;
    let r1 = idx / T;
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let hd = p.head_dim;
    let q_base = (b * T + i) * p.qkv_stride + p.q_off + h * hd;
    let k_base = (b * T + j) * p.qkv_stride + p.k_off + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + qkv[q_base + d] * qkv[k_base + d];
    }
    let bidx = (h * T + i) * T + j;
    scores[idx] = s * bitcast<f32>(p.scale) + bias[bidx];
}
