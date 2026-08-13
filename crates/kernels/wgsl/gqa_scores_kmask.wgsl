// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GQA attention scores with an additive per-key mask - `gqa_scores` plus `kmask[j]` added to every finite score
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// GQA attention scores with an additive per-key mask — `gqa_scores` plus
// `kmask[j]` added to every finite score:
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,hkv,:]) / sqrt(head_dim) + kmask[j]   for j <= i
//                   = -inf                                                       for j >  i
// `kmask` is 0 for live keys and a large negative value (-3.4e38) for excluded
// ones — the right-padding key mask of a padded encoder batch (Qwen3 as the
// FLUX.2 text encoder: pad tokens are queries with real outputs but must never
// be attended as keys). Layouts identical to `gqa_scores`; `attn_softmax` is
// reused unchanged. One invocation per (b,h,i,j).

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,        // T
    head_dim: u32,
    group: u32,        // n_heads / n_kv_heads
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read>       kmask:  array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

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
    scores[idx] = s * inverseSqrt(f32(hd)) + kmask[j];
}
