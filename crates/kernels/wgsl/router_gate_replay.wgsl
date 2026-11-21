// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for reinforcement learning on mixture-of-experts models whose trainer must
// route exactly as the serving engine did for its clients. If your team needs expertise in MoE routing replay then you
// can procure our services by sending an email to info@swedishembedded.com.

// @what  MoE router gate under a recorded top-k selection (routing replay): the router's own weight formula at given experts
// @how   one thread per row, array-free in n_experts, no barrier
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Routing replay. A rollout engine records which `top_k` experts each token row selected at each MoE call; the trainer
// then forces that SELECTION while computing the WEIGHTS from its own current logits with the router's own formula, so
// a near-tie that floating-point rounding flipped between the engine and the trainer no longer changes which experts
// the trained forward runs, and the gradient still flows into the logits of the selected experts. Run right after the
// router's own gate kernel on the same `gate` buffer: a row whose recorded ids are not all `< n_experts` (the host
// writes `0xFFFFFFFF` for a position whose routing was not recorded) is left as that kernel wrote it.
//
//   kind = 0, softmax (`router_gate.wgsl`):   p = softmax(logits), w_e = p_e * (norm ? 1/max(sum_S p, 1e-9) : 1) * scale
//   kind = 1, sigmoid (`router_gate_sigmoid.wgsl`): s = sigmoid(logits), w_e = s_e * (norm ? 1/(sum_S s + 1e-20) : 1) * scale
//
// with S the recorded experts and every other column 0, so `router_bwd.wgsl` / `router_bwd_sigmoid.wgsl`, which read
// the selected set as the gate's nonzero pattern, differentiate the replayed gate unchanged. The selected scores are
// summed in the order the router's own greedy top-k would have picked them (descending selection score - the
// probability for softmax, `s + bias` for sigmoid - ties to the lower expert index), so replaying the router's own
// selection reproduces its gate to the bit. Group-limited selection (`n_group > 1`) is not re-applied: the recorded
// set is the selection. `bias` is read by the sigmoid kind only (the softmax kind binds any read-only buffer there).
//
// The host guarantees distinct ids per recorded row (`model::routing::RoutingRecord::validate`); `top_k <= 32`.
//
//   params  : u32 [rows, n_experts, top_k, kind, norm, scale (f32 bits)]
//   logits  : [rows, n_experts]   experts : [rows, top_k] u32   bias : [n_experts]   gate : [rows, n_experts] read_write

struct Params {
    rows: u32,
    n_experts: u32,
    top_k: u32,
    kind: u32,      // 0 = softmax router, 1 = sigmoid router
    norm: u32,      // 1 = renormalise the selected weights
    scale: f32,     // routed_scaling_factor (1.0 = none)
};

const MAX_TOP_K: u32 = 32u; // bounds ONLY the top-k bookkeeping, never n_experts

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       experts: array<u32>;
@group(0) @binding(3) var<storage, read>       bias:    array<f32>;
@group(0) @binding(4) var<storage, read_write> gate:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= p.rows) { return; }
    let E = p.n_experts;
    let base = t * E;
    let k = min(p.top_k, MAX_TOP_K);
    let ids = t * p.top_k;

    // An unrecorded row (or any id out of range) keeps the router's own gate.
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        if (experts[ids + j] >= E) { return; }
    }

    // Softmax constants, in `router_gate.wgsl`'s order.
    var mx = -3.4e38;
    var sm = 1.0;
    if (p.kind == 0u) {
        for (var e: u32 = 0u; e < E; e = e + 1u) { mx = max(mx, logits[base + e]); }
        sm = 0.0;
        for (var e: u32 = 0u; e < E; e = e + 1u) { sm = sm + exp(logits[base + e] - mx); }
    }

    // Sum the selected values in the greedy order the router's own top-k would visit them.
    var used: array<bool, 32>;
    for (var j: u32 = 0u; j < k; j = j + 1u) { used[j] = false; }
    var sel_sum = 0.0;
    for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
        var best = 0u;
        var best_e = 0xFFFFFFFFu;
        var best_v = -3.4e38;
        for (var j: u32 = 0u; j < k; j = j + 1u) {
            if (used[j]) { continue; }
            let e = experts[ids + j];
            var v = 0.0;
            if (p.kind == 0u) {
                v = exp(logits[base + e] - mx) / sm;
            } else {
                v = 1.0 / (1.0 + exp(-logits[base + e])) + bias[e];
            }
            if (v > best_v || (v == best_v && e < best_e)) { best_v = v; best_e = e; best = j; }
        }
        used[best] = true;
        if (p.kind == 0u) {
            sel_sum = sel_sum + best_v;
        } else {
            sel_sum = sel_sum + 1.0 / (1.0 + exp(-logits[base + best_e]));
        }
    }

    var inv = 1.0;
    if (p.norm != 0u) {
        if (p.kind == 0u) { inv = 1.0 / max(sel_sum, 1e-9); } else { inv = 1.0 / (sel_sum + 1e-20); }
    }
    for (var e: u32 = 0u; e < E; e = e + 1u) { gate[base + e] = 0.0; }
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        let e = experts[ids + j];
        if (p.kind == 0u) {
            gate[base + e] = (exp(logits[base + e] - mx) / sm) * inv * p.scale;
        } else {
            gate[base + e] = (1.0 / (1.0 + exp(-logits[base + e]))) * inv * p.scale;
        }
    }
}
