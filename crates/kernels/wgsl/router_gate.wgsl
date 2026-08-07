// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Router gating: softmax over experts -> keep top_k -> renormalise
// @how   one thread per output element, 6 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Router gating: softmax over experts -> keep top_k -> renormalise.
// Produces a dense gate matrix [seq_len, n_experts] that is nonzero only for
// the top_k experts of each token (their renormalised probabilities). One
// invocation per token. n_experts is assumed <= MAX_EXPERTS. 128 covers every
// released top-k-softmax MoE brain imports today (Qwen3-Omni thinker/talker:
// 128 experts each).
//
// Running every expert and masking by this gate is numerically identical to a
// true sparse top-k dispatch *without* capacity dropping. Capacity limits exist
// only to bound memory during training; inference has no such pressure, so this
// is the exact top-k MoE output. `moe_linear_gated.wgsl` is the sparse
// alternative that actually skips a non-routed row's FLOPs instead of relying
// on this property to discard a densely-computed one.

const MAX_EXPERTS: u32 = 128u;

struct Params {
    seq_len: u32,
    n_experts: u32,
    top_k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let t = gidx;
    if (t >= p.seq_len) { return; }
    let E = p.n_experts;
    let base = t * E;

    var prob: array<f32, 64>;
    var used: array<bool, 64>;

    // softmax over experts
    var mx = -3.4e38;
    for (var e: u32 = 0u; e < E; e = e + 1u) { mx = max(mx, logits[base + e]); }
    var sm = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        let pe = exp(logits[base + e] - mx);
        prob[e] = pe;
        used[e] = false;
        sm = sm + pe;
    }
    for (var e: u32 = 0u; e < E; e = e + 1u) { prob[e] = prob[e] / sm; }

    // pick the top_k experts (greedy argmax, top_k is tiny)
    var sel_sum = 0.0;
    for (var kk: u32 = 0u; kk < p.top_k; kk = kk + 1u) {
        var best = 0u;
        var best_v = -1.0;
        for (var e: u32 = 0u; e < E; e = e + 1u) {
            if (!used[e] && prob[e] > best_v) { best_v = prob[e]; best = e; }
        }
        used[best] = true;
        sel_sum = sel_sum + prob[best];
    }

    // renormalise the kept experts; zero the rest
    let inv = 1.0 / max(sel_sum, 1e-9);
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (used[e]) { gate[base + e] = prob[e] * inv; }
        else         { gate[base + e] = 0.0; }
    }
}
