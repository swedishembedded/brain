// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  DSA top-k selection mask (forward)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// DSA top-k selection mask (forward). Turn per-query indexer scores into an
// additive attention mask: keep the top-`index_topk` causal keys per query
// (0), block the rest (-inf). One invocation per (b, query s, key t).
//   mask[b,s,t] = 0     if t<=s and score[b,s,t] is among the top-k causal
//               = -inf  otherwise
// When index_topk >= s+1 every causal key is kept (the all-pass regime that
// makes this exactly dense attention - the invariant tiny models rely on).
// Ties at the boundary keep all tied keys (torch.topk breaks them arbitrarily;
// keeping ties is a superset, harmless for a mask). Still O(T) per `t` (the
// causal rank count), so still O(T^2) per row in total - a genuinely
// sub-quadratic top-k (a partial sort/selection) would change that; this
// kernel only fixes how that O(T^2) is spread across threads (see below), not
// the total FLOP count.
//
// The previous revision gave one THREAD the entire row: an outer serial loop
// over every `t` in `0..T`, each iteration paying its own O(s) causal rank
// count - a nested serial chain with no parallelism ACROSS `t` at all, so the
// single slowest row (`s=T-1`) alone set the whole dispatch's wall time while
// every other invocation sat idle after finishing its own, much shorter,
// row. Every `t` in a row is independent of every other `t` (no shared
// state, no ordering requirement between them), so that outer loop was
// serialising work that was already embarrassingly parallel. This revision
// gives one thread per `(b,s,t)` CELL instead of per `(b,s)` row - same
// per-cell formula, same float-op order, so bit-identical output - and lets
// every `t` in a row run concurrently instead of one after another.
// Dispatch: `threads = bsz*T*T`, one per cell (mirrors `mla_scores.wgsl`'s
// own `(b,h,i,j)` decomposition), not `bsz*T` per row.

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
    let total = p.bsz * T * T;
    if (gidx >= total) { return; }

    let t = gidx % T;
    let r1 = gidx / T;
    let s = r1 % T;
    let b = r1 / T;
    let base = (b * T + s) * T;

    if (t > s) {
        mask[base + t] = -3.4e38;
        return;
    }
    let causal_len = s + 1u;
    let count = min(p.index_topk, causal_len);
    if (count >= causal_len) {
        mask[base + t] = 0.0;
        return;
    }
    let v = scores[base + t];
    var greater = 0u;
    for (var t2: u32 = 0u; t2 <= s; t2 = t2 + 1u) {
        if (scores[base + t2] > v) { greater = greater + 1u; }
    }
    mask[base + t] = select(-3.4e38, 0.0, greater < count);
}
