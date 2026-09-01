// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Attention scores from SEPARATE q,k buffers, with a configurable scale, an optional causal mask, AND an additive per-key mask - the TimesFM-3 contract (one kernel serves both its sequence and variate attention)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `attn_scores_qk` (causal, separate q/k, configurable scale, no external
// mask) plus `attn_scores_full`'s additive per-key mask, combined: TimesFM-3
// needs both at once (its causal SEQUENCE attention and non-causal VARIATE
// attention both need to hide masked/padded patches, and its own attention
// scale is folded entirely into the query projection ahead of this kernel -
// see `timesfm3::model`'s doc for why `scale` is passed as 1.0 here - so
// neither existing kernel's fixed 1/sqrt(head_dim) is usable).
//   scores[b,h,i,j] = -inf                          if causal != 0 && j > i
//                   = (q[b,i,h,:] . k[b,j,h,:])*scale + kmask[b,j]   otherwise
// kmask is additive (0 for a valid key, a large negative value for a masked
// one), broadcast over query index i and head h - same convention and shape
// as `attn_scores_full`'s `kmask`. q,k laid out [b, S, qk_stride], head h at
// channel offset h*head_dim. One invocation per (b,h,i,j). scores layout:
// ((b*H + h)*S + i)*S + j.

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,       // S
    head_dim: u32,
    qk_stride: u32,   // channel stride of q/k per token
    causal: u32,      // 1 = also mask j>i
    scale: f32,
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

    if (p.causal != 0u && j > i) { scores[gidx] = -3.4e38; return; }

    let hd = p.head_dim;
    let q_base = (b * S + i) * p.qk_stride + h * hd;
    let k_base = (b * S + j) * p.qk_stride + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * k[k_base + d];
    }
    scores[gidx] = s * p.scale + kmask[b * S + j];
}
