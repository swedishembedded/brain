// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Attention scores with causal mask AND key-padding mask (no RoPE)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Attention scores with causal mask AND key-padding mask (no RoPE):
//   scores[b,h,i,j] = -inf            if j > i              (causal)
//                   = -inf            if token[b,j] == PAD  (key padding)
//                   = (q.k)/sqrt(hd)  otherwise
// q,k read from the fused qkv buffer (q at q_off, k at k_off; no rotation).
// One invocation per (b,h,i,j). scores layout: ((b*H + h)*T + i)*T + j.

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,        // T
    head_dim: u32,
    qkv_stride: u32,   // 3*d_model
    q_off: u32,
    k_off: u32,
    pad: u32,          // PAD token id
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       qkv:    array<f32>;
@group(0) @binding(2) var<storage, read>       tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
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

    if (j > i) { scores[idx] = -3.4e38; return; }
    if (tokens[b * T + j] == p.pad) { scores[idx] = -3.4e38; return; }

    let hd = p.head_dim;
    let q_base = (b * T + i) * p.qkv_stride + p.q_off + h * hd;
    let k_base = (b * T + j) * p.qkv_stride + p.k_off + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + qkv[q_base + d] * qkv[k_base + d];
    }
    scores[idx] = s * inverseSqrt(f32(hd));
}
