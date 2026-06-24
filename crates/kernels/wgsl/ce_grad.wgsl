// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Cross-entropy gradient w.r.t. logits, averaged over all positions:
//   d_logits[n, v] = (softmax(logits[n])_v - [v == target[n]]) / n_rows
// One invocation per (row, vocab); recomputes the row softmax (vocab is small).

struct Params {
    n_rows: u32,
    vocab: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> dlogits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.n_rows * p.vocab;
    if (idx >= total) { return; }
    let n = idx / p.vocab;
    let v = idx % p.vocab;
    let base = n * p.vocab;

    var mx = -3.4e38;
    for (var c: u32 = 0u; c < p.vocab; c = c + 1u) { mx = max(mx, logits[base + c]); }
    var sum = 0.0;
    for (var c: u32 = 0u; c < p.vocab; c = c + 1u) { sum = sum + exp(logits[base + c] - mx); }
    let prob = exp(logits[base + v] - mx) / sum;

    let onehot = select(0.0, 1.0, v == targets[n]);
    dlogits[idx] = (prob - onehot) / f32(p.n_rows);
}
