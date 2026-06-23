// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Per-position cross-entropy over U_BINS, with ignore_index (sentinel IGNORE).
//   out[n] = 0                                  if target[n] == IGNORE
//          = logsumexp(logits[n]) - logit[tgt]  otherwise
// Host sums out[] and divides by the count of non-ignored rows (mean CE,
// matching F.cross_entropy(..., ignore_index=-100)).

struct Params {
    n_rows: u32,
    u_bins: u32,
    ignore: u32,
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
    let tgt = targets[n];
    if (tgt == p.ignore) { out[n] = 0.0; return; }
    let base = n * p.u_bins;
    var mx = -3.4e38;
    for (var c: u32 = 0u; c < p.u_bins; c = c + 1u) { mx = max(mx, logits[base + c]); }
    var sum = 0.0;
    for (var c: u32 = 0u; c < p.u_bins; c = c + 1u) { sum = sum + exp(logits[base + c] - mx); }
    out[n] = (mx + log(sum)) - logits[base + tgt];
}
