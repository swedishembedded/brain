// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Router gating (training variant): softmax + full probs + top_k gate (optional renorm + scale)
// @how   one thread per token, array-free (no expert-count cap)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Router gating (training variant): like `router_gate.wgsl` but also writes
// the full softmax probabilities (needed by the backward pass). One
// invocation per token row.
//
// Same fix as `router_gate.wgsl` (see its header for the full rationale) -
// removed the `n_experts`-capped `array<f32,128>`/`array<bool,128>` scratch.
// Here the softmax itself is computed straight into the `probs` OUTPUT buffer
// (unnormalised numerator, then normalised in place in a second pass) instead
// of a local cache, since `probs` is exactly `[n_rows, n_experts]` already.
// The only remaining `var<function>` array is `sel_idx`, bounded by
// `MAX_TOP_K`, never by `n_experts`.

struct Params {
    n_rows: u32,
    n_experts: u32,
    top_k: u32,
    norm: u32,      // 1 = renormalise the selected probabilities to sum to 1
    scale: f32,     // routed_scaling_factor (1.0 = none)
};

const MAX_TOP_K: u32 = 32u; // bounds ONLY the top-k bookkeeping, never n_experts

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
    let k = min(p.top_k, MAX_TOP_K);

    // Pass 1: row max.
    var mx = -3.4e38;
    for (var e: u32 = 0u; e < E; e = e + 1u) { mx = max(mx, logits[base + e]); }

    // Pass 2: unnormalised numerator into `probs` (scratch reuse) + its sum.
    var sm = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        let pe = exp(logits[base + e] - mx);
        probs[base + e] = pe;
        sm = sm + pe;
    }
    // Pass 2b: normalise `probs` in place - it is now the real softmax.
    for (var e: u32 = 0u; e < E; e = e + 1u) { probs[base + e] = probs[base + e] / sm; }

    // Pass 3: greedy top-k over the now-normalised `probs`.
    var sel_idx: array<u32, 32>;
    var sel_sum = 0.0;
    for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
        var best = 0u;
        var best_v = -1.0;
        for (var e: u32 = 0u; e < E; e = e + 1u) {
            var already = false;
            for (var s: u32 = 0u; s < kk; s = s + 1u) {
                if (sel_idx[s] == e) { already = true; break; }
            }
            if (!already && probs[base + e] > best_v) { best_v = probs[base + e]; best = e; }
        }
        sel_idx[kk] = best;
        sel_sum = sel_sum + best_v;
    }

    // Pass 4: finalise `gate` - keep the selected experts (renormalised iff
    // `norm`, then scaled), zero the rest. Same two knobs as
    // `router_gate.wgsl`; see its header for what they mean.
    let inv = select(1.0, 1.0 / max(sel_sum, 1e-9), p.norm != 0u);
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        var selected = false;
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            if (sel_idx[s] == e) { selected = true; break; }
        }
        if (selected) {
            gate[base + e] = probs[base + e] * inv * p.scale;
        } else {
            gate[base + e] = 0.0;
        }
    }
}
