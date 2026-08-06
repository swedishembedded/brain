// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Cross-entropy gradient using precomputed per-row softmax stats (see ce_stats)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Cross-entropy gradient using precomputed per-row softmax stats (see ce_stats):
//   d_logits[n,v] = 0                                    if target[n]==IGNORE
//                 = (exp(logits[n,v]-max[n])/sum[n] - [v==tgt]) / count  else
// One invocation per (row, bin), O(1) each — no per-element softmax recompute.
struct Params { n_rows: u32, u_bins: u32, ignore: u32, count: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       targets: array<u32>;
@group(0) @binding(3) var<storage, read>       stats:   array<f32>;
@group(0) @binding(4) var<storage, read_write> dlogits: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n_rows * p.u_bins) { return; }
    let n = idx / p.u_bins;
    let v = idx % p.u_bins;
    let tgt = targets[n];
    if (tgt == p.ignore) { dlogits[idx] = 0.0; return; }
    let mx = stats[2u * n];
    let sum = stats[2u * n + 1u];
    let prob = exp(logits[n * p.u_bins + v] - mx) / sum;
    let onehot = select(0.0, 1.0, v == tgt);
    dlogits[idx] = (prob - onehot) / p.count;
}
