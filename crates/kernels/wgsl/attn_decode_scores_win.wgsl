// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Windowed decode-step attention scores
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Windowed decode-step attention scores: a SINGLE query (the new token) against
// the cached keys, but only over the sliding window [w0, t); positions j < w0
// are masked to -inf so the downstream `decode_softmax` (max-subtract, exp) gives
// them ~0 weight and `attn_decode_apply` contributes nothing for them. This is
// the decode twin of Kronos's sliding-window self-attention (window = max_context).
//   q     : [n_heads*head_dim]          the new token's (QK-normed, RoPE'd) query
//   kcache: [max_T, n_kv*head_dim]      persistent key cache (rows 0..t valid)
//   scores: [n_heads, cap]              cap = max_T stride (only 0..t written)
// GQA: kvhead(h) = h / (n_heads/n_kv_heads). One invocation per (h, j<t).
// Barrier-free → JITs to CPU.

struct Params {
    n_heads: u32,
    group: u32,      // n_heads / n_kv_heads
    head_dim: u32,
    t: u32,          // cached length (positions 0..t)
    cap: u32,        // scores row stride (max_T)
    kv_stride: u32,  // n_kv_heads * head_dim (cache row width)
    w0: u32,         // window start: mask j < w0
    scale: f32,      // 1/sqrt(head_dim)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       kcache: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n_heads * p.t;
    if (idx >= total) { return; }
    let h = idx / p.t;
    let j = idx % p.t;
    if (j < p.w0) { scores[h * p.cap + j] = -3.4e38; return; }
    let hd = p.head_dim;
    let kvh = h / p.group;
    let qb = h * hd;
    let kb = j * p.kv_stride + kvh * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[qb + d] * kcache[kb + d];
    }
    scores[h * p.cap + j] = s * p.scale;
}
