// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Cross-entropy gradient over U_BINS with ignore_index, normalised by the count of non-ignored positions (passed in as a float)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Cross-entropy gradient over U_BINS with ignore_index, normalised by the count
// of non-ignored positions (passed in as a float):
//   d_logits[n,v] = 0                                if target[n] == IGNORE
//                 = (softmax(logits[n])_v - [v==tgt]) / count   otherwise
// One invocation per (row, bin); recomputes the row softmax (U_BINS small).

struct Params {
    n_rows: u32,
    u_bins: u32,
    ignore: u32,
    count: f32,
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
    if (idx >= p.n_rows * p.u_bins) { return; }
    let n = idx / p.u_bins;
    let v = idx % p.u_bins;
    let tgt = targets[n];
    if (tgt == p.ignore) { dlogits[idx] = 0.0; return; }

    let base = n * p.u_bins;
    var mx = -3.4e38;
    for (var c: u32 = 0u; c < p.u_bins; c = c + 1u) { mx = max(mx, logits[base + c]); }
    var sum = 0.0;
    for (var c: u32 = 0u; c < p.u_bins; c = c + 1u) { sum = sum + exp(logits[base + c] - mx); }
    let prob = exp(logits[base + v] - mx) / sum;
    let onehot = select(0.0, 1.0, v == tgt);
    dlogits[idx] = (prob - onehot) / p.count;
}
