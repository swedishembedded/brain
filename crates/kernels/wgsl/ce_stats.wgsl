// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Per-row softmax statistics for the cross-entropy backward: for each row,
//   stats[2n] = max_v logits[n,v]
//   stats[2n+1] = sum_v exp(logits[n,v] - max)      (0 for ignored rows)
// One invocation per ROW (not per element). This is what lets `ce_grad_stats`
// run in O(rows*vocab) instead of the naive per-element softmax recompute's
// O(rows*vocab^2) — the difference between ~10 ms and ~56 s at a 151936 vocab.
struct Params { n_rows: u32, u_bins: u32, ignore: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats:   array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = gid.y * (nwg.x * 64u) + gid.x;
    if (n >= p.n_rows) { return; }
    if (targets[n] == p.ignore) { stats[2u * n] = 0.0; stats[2u * n + 1u] = 1.0; return; }
    let base = n * p.u_bins;
    var mx = -3.4e38;
    for (var c: u32 = 0u; c < p.u_bins; c = c + 1u) { mx = max(mx, logits[base + c]); }
    var sum = 0.0;
    for (var c: u32 = 0u; c < p.u_bins; c = c + 1u) { sum = sum + exp(logits[base + c] - mx); }
    stats[2u * n] = mx;
    stats[2u * n + 1u] = sum;
}
