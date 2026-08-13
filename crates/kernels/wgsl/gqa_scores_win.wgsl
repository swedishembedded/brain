// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GQA attention scores with a sliding-window causal mask - `gqa_scores` plus a lower key bound `i - j < window`
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// GQA attention scores with a SLIDING-WINDOW causal mask (batched, full
// sequence — the prefill/one-shot twin of `attn_decode_scores_win`'s single-
// query decode-step form):
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,hkv,:]) / sqrt(head_dim)   for i-window < j <= i
//                   = -inf                                            otherwise
// `window` bounds how far back key `j` may sit from query `i`; `window >= T`
// degenerates to `gqa_scores`'s plain causal mask exactly (every `i-j <
// window` holds since `i-j <= T-1`), so this kernel is a strict
// generalization, not a second copy — a caller with no sliding-window
// requirement passes `window = tcols` (or anything `>= tcols`) rather than
// dispatching `gqa_scores` separately. Layouts identical to `gqa_scores`;
// `attn_softmax` is reused unchanged. One invocation per (b,h,i,j).

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,        // T
    head_dim: u32,
    group: u32,        // n_heads / n_kv_heads
    window: u32,       // sliding-window size: key j lives iff i-window < j <= i
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
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

    if (j > i || (i - j) >= p.window) { scores[idx] = -3.4e38; return; }

    let hd = p.head_dim;
    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let k_row = p.n_kv_heads * hd;
    let q_base = (b * T + i) * q_row + h * hd;
    let k_base = (b * T + j) * k_row + hkv * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * k[k_base + d];
    }
    scores[idx] = s * inverseSqrt(f32(hd));
}
