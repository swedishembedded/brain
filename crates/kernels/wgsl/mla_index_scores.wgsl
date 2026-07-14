// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// DSA indexer scores (forward, detached). For each (query s, key t<=s):
//   index_scores[b,s,t] = sum_h (weights[b,s,h] * H^-0.5)
//                             * relu( (q[b,s,h,:] . k[b,t,:]) * D^-0.5 )
// q is [B*T, H*D] (post sub-slice RoPE), k the shared single-head indexer key
// [B*T, D] (post LayerNorm + RoPE), weights [B*T, H] (= weights_proj(hidden)).
// Future keys (t>s) get -inf so the top-k selection stays causal. One invocation
// per (b,s,t). scores layout: (b*T + s)*T + t.

struct Params {
    bsz: u32,
    n_heads: u32,  // index_n_heads
    tcols: u32,    // T
    head_dim: u32, // index_head_dim
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:       array<f32>;
@group(0) @binding(2) var<storage, read>       k:       array<f32>;
@group(0) @binding(3) var<storage, read>       weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.bsz * T * T;
    if (gidx >= total) { return; }

    let t = gidx % T;
    let r = gidx / T;
    let s = r % T;
    let b = r / T;

    if (t > s) { scores[gidx] = -3.4e38; return; }

    let H = p.n_heads;
    let D = p.head_dim;
    let qscale = inverseSqrt(f32(D));
    let wscale = inverseSqrt(f32(H));
    let k_base = (b * T + t) * D;
    var acc = 0.0;
    for (var h: u32 = 0u; h < H; h = h + 1u) {
        let q_base = (b * T + s) * H * D + h * D;
        var dot = 0.0;
        for (var d: u32 = 0u; d < D; d = d + 1u) {
            dot = dot + q[q_base + d] * k[k_base + d];
        }
        let rel = max(dot * qscale, 0.0);
        acc = acc + weights[(b * T + s) * H + h] * wscale * rel;
    }
    scores[gidx] = acc;
}
