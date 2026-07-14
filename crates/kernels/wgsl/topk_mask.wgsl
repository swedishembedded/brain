// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// DSA top-k selection mask (forward). Turn per-query indexer scores into an
// additive attention mask: keep the top-`index_topk` causal keys per query
// (0), block the rest (-inf). One invocation per (b, query s).
//   mask[b,s,t] = 0     if t<=s and score[b,s,t] is among the top-k causal
//               = -inf  otherwise
// When index_topk >= s+1 every causal key is kept (the all-pass regime that
// makes this exactly dense attention — the invariant tiny models rely on).
// Ties at the boundary keep all tied keys (torch.topk breaks them arbitrarily;
// keeping ties is a superset, harmless for a mask). O(T^2) per row.

struct Params {
    bsz: u32,
    tcols: u32,     // T
    index_topk: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    if (gidx >= p.bsz * T) { return; }

    let s = gidx % T;
    let b = gidx / T;
    let base = (b * T + s) * T;
    let causal_len = s + 1u;
    let count = min(p.index_topk, causal_len);

    if (count >= causal_len) {
        for (var t: u32 = 0u; t < T; t = t + 1u) {
            mask[base + t] = select(-3.4e38, 0.0, t <= s);
        }
        return;
    }
    for (var t: u32 = 0u; t < T; t = t + 1u) {
        if (t > s) { mask[base + t] = -3.4e38; continue; }
        let v = scores[base + t];
        var greater = 0u;
        for (var t2: u32 = 0u; t2 <= s; t2 = t2 + 1u) {
            if (scores[base + t2] > v) { greater = greater + 1u; }
        }
        mask[base + t] = select(-3.4e38, 0.0, greater < count);
    }
}
