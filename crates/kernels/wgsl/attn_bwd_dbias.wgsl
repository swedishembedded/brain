// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Backward w.r.t. the additive score bias for attn_scores_{bidir,causal}_bias.
// The bias is added directly to every score and is shared across the batch, so
// its gradient is the batch sum of the incoming score gradient:
//   d_bias[h,i,j] = sum_b d_score[b,h,i,j]
// For the causal variant the j>i entries never affect any output (softmax made
// their probability 0, so d_score is 0 there); with `causal != 0` we write 0
// explicitly. One invocation per (h,i,j). d_bias layout: (h*T + i)*T + j.

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,        // T
    causal: u32,       // 0 = bidir, else zero the j>i entries
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_bias:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.n_heads * T * T;
    let idx = gidx;
    if (idx >= total) { return; }

    let j = idx % T;
    let r1 = idx / T;
    let i = r1 % T;
    let h = r1 / T;

    if (p.causal != 0u && j > i) {
        d_bias[idx] = 0.0;
        return;
    }

    var acc = 0.0;
    for (var b: u32 = 0u; b < p.bsz; b = b + 1u) {
        acc = acc + d_scores[((b * p.n_heads + h) * T + i) * T + j];
    }
    d_bias[idx] = acc;
}
