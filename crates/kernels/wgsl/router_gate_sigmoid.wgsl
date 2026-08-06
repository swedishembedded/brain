// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GLM/DeepSeek-V3 "noaux_tc" MoE router (forward)
// @how   one thread per output element, 6 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// GLM/DeepSeek-V3 "noaux_tc" MoE router (forward). Per token row:
//   s[e]      = sigmoid(logit[e])                     (combine score; -> probs)
//   choice[e] = s[e] + bias[e]                         (selection score only)
//   group-limited top-k: split E into `n_group` groups, score each group by the
//     sum of its top-2 `choice`, keep the best `topk_group` groups, then take the
//     global top-`top_k` `choice` among the kept groups.
//   gate[e]   = selected ? s[e] : 0 ; if norm: gate /= sum_selected s ; *= scale
// The bias steers *selection* only; the combine weight is the raw sigmoid score
// (aux-loss-free load balancing). E <= 64, n_group <= 64. One invocation/row.

const MAX_E: u32 = 64u;

struct Params {
    n_rows: u32,
    n_experts: u32,
    top_k: u32,
    n_group: u32,
    topk_group: u32,
    norm: u32,      // 1 = renormalise selected weights
    scale: f32,     // routed_scaling_factor
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       bias:   array<f32>;
@group(0) @binding(3) var<storage, read_write> gate:   array<f32>;
@group(0) @binding(4) var<storage, read_write> probs:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let t = gidx;
    if (t >= p.n_rows) { return; }
    let E = p.n_experts;
    let base = t * E;

    var s: array<f32, 64>;
    var choice: array<f32, 64>;
    var used: array<bool, 64>;
    // `bias` is a per-expert vector [E] (indexed by `e`, not per-row).
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        let se = 1.0 / (1.0 + exp(-logits[base + e]));
        s[e] = se;
        probs[base + e] = se;
        choice[e] = se + bias[e];
        used[e] = false;
    }

    // ---- group-limited masking: keep only experts in the top `topk_group` groups ----
    let ng = max(p.n_group, 1u);
    let per = E / ng;
    var group_keep: array<bool, 64>;
    for (var g: u32 = 0u; g < ng; g = g + 1u) { group_keep[g] = (ng == 1u); }
    if (ng > 1u) {
        // group score = sum of top-2 choice within the group
        var gscore: array<f32, 64>;
        for (var g: u32 = 0u; g < ng; g = g + 1u) {
            var b1 = -3.4e38;
            var b2 = -3.4e38;
            for (var m: u32 = 0u; m < per; m = m + 1u) {
                let cv = choice[g * per + m];
                if (cv > b1) { b2 = b1; b1 = cv; }
                else if (cv > b2) { b2 = cv; }
            }
            gscore[g] = b1 + b2;
        }
        // keep the top `topk_group` groups
        var gused: array<bool, 64>;
        for (var g: u32 = 0u; g < ng; g = g + 1u) { gused[g] = false; }
        for (var kk: u32 = 0u; kk < p.topk_group; kk = kk + 1u) {
            var best = 0u;
            var bestv = -3.4e38;
            for (var g: u32 = 0u; g < ng; g = g + 1u) {
                if (!gused[g] && gscore[g] > bestv) { bestv = gscore[g]; best = g; }
            }
            gused[best] = true;
            group_keep[best] = true;
        }
    }
    // mask out experts whose group is not kept
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (!group_keep[e / per]) { choice[e] = -3.4e38; }
    }

    // ---- global top-k over the (masked) choice scores ----
    var sel_sum = 0.0;
    for (var kk: u32 = 0u; kk < p.top_k; kk = kk + 1u) {
        var best = 0u;
        var bestv = -3.4e38;
        for (var e: u32 = 0u; e < E; e = e + 1u) {
            if (!used[e] && choice[e] > bestv) { bestv = choice[e]; best = e; }
        }
        used[best] = true;
        sel_sum = sel_sum + s[best];
    }

    let denom = select(1.0, 1.0 / max(sel_sum, 1e-20), p.norm != 0u);
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (used[e]) { gate[base + e] = s[e] * denom * p.scale; }
        else         { gate[base + e] = 0.0; }
    }
}
