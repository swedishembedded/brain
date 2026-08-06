// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Full (bidirectional, NON-causal) attention scores with an additive key mask and NO 1/sqrt(head_dim) scaling — the Chronos-2 encoder contract
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Full (bidirectional, NON-causal) attention scores with an additive key mask
// and NO 1/sqrt(head_dim) scaling — the Chronos-2 encoder contract:
//   scores[b,h,i,j] = ( q[b,i,h,:] . k[b,j,h,:] ) + key_mask[b,j]
// key_mask is additive (0 for observed, ~-inf for padded/masked keys), broadcast
// over query index i and head h. Separate q,k buffers (not fused), each laid out
// [b, S, qk_stride] with head h at channel offset h*head_dim. One invocation per
// (b,h,i,j). scores layout: ((b*H + h)*S + i)*S + j.
//
// Contrast `attn_scores.wgsl` (causal + 1/sqrt(hd) scaled) — Chronos-2 needs
// neither; the softmax then runs over the full row.

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,      // S
    head_dim: u32,
    qk_stride: u32,  // n_heads * head_dim (per-token channel stride of q/k)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read>       kmask:  array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let S = p.tcols;
    let total = p.bsz * p.n_heads * S * S;
    if (gidx >= total) { return; }

    let j = gidx % S;
    let r1 = gidx / S;
    let i = r1 % S;
    let r2 = r1 / S;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let hd = p.head_dim;
    let q_base = (b * S + i) * p.qk_stride + h * hd;
    let k_base = (b * S + j) * p.qk_stride + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * k[k_base + d];
    }
    scores[gidx] = s + kmask[b * S + j];   // additive mask, UNSCALED
}
