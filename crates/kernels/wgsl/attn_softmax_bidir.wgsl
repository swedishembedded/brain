// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row-wise bidirectional softmax over the full key axis
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Row-wise bidirectional softmax over the full key axis. One invocation per
// (b,h,i): normalises scores[b,h,i, 0..T] into probs (non-causal; no j>i zeroing,
// cf. attn_softmax.wgsl which only sums j<=i).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,   // T
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.bsz * p.n_heads * T;
    let idx = gidx;
    if (idx >= total) { return; }

    let i = idx % T;
    let r = idx / T;          // b*H + h
    let base = (r * T + i) * T;

    var mx = -3.4e38;
    for (var j: u32 = 0u; j < T; j = j + 1u) { mx = max(mx, scores[base + j]); }
    var sum = 0.0;
    for (var j: u32 = 0u; j < T; j = j + 1u) { sum = sum + exp(scores[base + j] - mx); }
    let inv = 1.0 / sum;
    for (var j: u32 = 0u; j < T; j = j + 1u) {
        probs[base + j] = exp(scores[base + j] - mx) * inv;
    }
}
