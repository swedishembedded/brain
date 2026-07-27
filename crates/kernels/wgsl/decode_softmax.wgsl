// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Decode-step softmax: max-subtracted softmax over the `t` cached scores of each
// query head, in place per row of a [n_heads, cap]-strided buffer. Matches the
// CPU reference (subtract row max, exp, normalise). One invocation per head.
// `scores` (read) and `probs` (write) are distinct buffers (no output alias).

struct Params {
    n_heads: u32,
    t: u32,     // cached length
    cap: u32,   // row stride (max_T)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let h = gid.y * (nwg.x * 64u) + gid.x;
    if (h >= p.n_heads) { return; }
    let base = h * p.cap;
    var mx = -3.4e38;
    for (var j: u32 = 0u; j < p.t; j = j + 1u) { mx = max(mx, scores[base + j]); }
    var sum = 0.0;
    for (var j: u32 = 0u; j < p.t; j = j + 1u) { sum = sum + exp(scores[base + j] - mx); }
    let inv = 1.0 / sum;
    for (var j: u32 = 0u; j < p.t; j = j + 1u) {
        probs[base + j] = exp(scores[base + j] - mx) * inv;
    }
}
