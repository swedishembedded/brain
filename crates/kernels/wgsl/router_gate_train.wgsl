// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Router gating (training variant)
// @how   one thread per output element, 6 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Router gating (training variant): like router_gate.wgsl but also writes the
// full softmax probabilities (needed by the backward pass). One invocation per
// token row. n_experts <= MAX_EXPERTS.

const MAX_EXPERTS: u32 = 64u;

struct Params {
    n_rows: u32,
    n_experts: u32,
    top_k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate:   array<f32>;
@group(0) @binding(3) var<storage, read_write> probs:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let t = gidx;
    if (t >= p.n_rows) { return; }
    let E = p.n_experts;
    let base = t * E;

    var pr: array<f32, 64>;
    var used: array<bool, 64>;

    var mx = -3.4e38;
    for (var e: u32 = 0u; e < E; e = e + 1u) { mx = max(mx, logits[base + e]); }
    var sm = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        let pe = exp(logits[base + e] - mx);
        pr[e] = pe;
        used[e] = false;
        sm = sm + pe;
    }
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        pr[e] = pr[e] / sm;
        probs[base + e] = pr[e];
    }

    var sel_sum = 0.0;
    for (var kk: u32 = 0u; kk < p.top_k; kk = kk + 1u) {
        var best = 0u;
        var best_v = -1.0;
        for (var e: u32 = 0u; e < E; e = e + 1u) {
            if (!used[e] && pr[e] > best_v) { best_v = pr[e]; best = e; }
        }
        used[best] = true;
        sel_sum = sel_sum + pr[best];
    }

    let inv = 1.0 / max(sel_sum, 1e-9);
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (used[e]) { gate[base + e] = pr[e] * inv; }
        else         { gate[base + e] = 0.0; }
    }
}
