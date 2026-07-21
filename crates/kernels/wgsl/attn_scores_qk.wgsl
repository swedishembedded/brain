// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Attention scores from SEPARATE q,k buffers, with a configurable scale and an
// optional causal mask — covers Kronos's two attention modes:
//   - decoder/tokenizer self-attention: causal=1, scale=1/sqrt(head_dim)
//   - dependency-layer cross-attention (inference): causal=0, scale=1/sqrt(head_dim)
//     (q from the sibling embedding, k from the hidden states)
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,h,:]) * scale,  and -inf if causal && j>i.
// Single-sequence inference => no external key-padding mask. q,k laid out
// [b, S, qk_stride], head h at channel offset h*head_dim. One invocation per
// (b,h,i,j). scores layout ((b*H + h)*S + i)*S + j.

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,       // S
    head_dim: u32,
    qk_stride: u32,   // channel stride of q/k per token
    causal: u32,      // 1 = mask j>i
    scale: f32,       // usually 1/sqrt(head_dim)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

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

    if (p.causal != 0u && j > i) { scores[gidx] = -3.4e38; return; }

    let hd = p.head_dim;
    let q_base = (b * S + i) * p.qk_stride + h * hd;
    let k_base = (b * S + j) * p.qk_stride + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * k[k_base + d];
    }
    scores[gidx] = s * p.scale;
}
