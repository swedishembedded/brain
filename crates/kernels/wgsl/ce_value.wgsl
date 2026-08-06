// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-position cross-entropy loss (for logging)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Per-position cross-entropy loss (for logging). out[n] = logsumexp - logit[target].
// The host sums these and divides by n_rows to get the mean CE.

struct Params {
    n_rows: u32,
    vocab: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> out:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.n_rows) { return; }
    let base = n * p.vocab;
    var mx = -3.4e38;
    for (var c: u32 = 0u; c < p.vocab; c = c + 1u) { mx = max(mx, logits[base + c]); }
    var sum = 0.0;
    for (var c: u32 = 0u; c < p.vocab; c = c + 1u) { sum = sum + exp(logits[base + c] - mx); }
    out[n] = (mx + log(sum)) - logits[base + targets[n]];
}
