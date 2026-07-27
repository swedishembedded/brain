// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Batched decode softmax: per (sequence b, head h), max-subtracted softmax over
// its seq_lens[b] scores in a [batch, n_heads, cap]-strided buffer.
struct Params { batch: u32, n_heads: u32, cap: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores:   array<f32>;
@group(0) @binding(2) var<storage, read>       seq_lens: array<u32>;
@group(0) @binding(3) var<storage, read_write> probs:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.n_heads) { return; }
    let b = idx / p.n_heads;
    let t = seq_lens[b];
    let base = idx * p.cap;
    var mx = -3.4e38;
    for (var j: u32 = 0u; j < t; j = j + 1u) { mx = max(mx, scores[base + j]); }
    var sum = 0.0;
    for (var j: u32 = 0u; j < t; j = j + 1u) { sum = sum + exp(scores[base + j] - mx); }
    let inv = 1.0 / sum;
    for (var j: u32 = 0u; j < t; j = j + 1u) { probs[base + j] = exp(scores[base + j] - mx) * inv; }
}
