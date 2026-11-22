// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mixture-of-experts serving kernels for its
// clients. If your team needs expertise in routing tokens to experts at the
// speed of the device then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  GLM/DeepSeek-V3 "noaux_tc" MoE router (forward), one WORKGROUP per row: router_gate_sigmoid's drop-in twin
// @how   64-thread workgroup; each top-k round is a strided scan plus an (value, index) argmax tree in workgroup memory
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Same Params, bindings and outputs as router_gate_sigmoid (read its header for the routing rule), and the same BITS:
// every value is the reference's own expression on the same operands (the sigmoid per expert, the selected scores
// summed by one thread in selection order, the same renormalisation), and the selection is exact - an argmax with
// ties to the lower expert index is what the reference's ascending strict-greater scan picks, a NaN choice is never
// taken, and a round with no candidate above the floor picks expert 0 as the reference does. The reference gives one
// THREAD a row and scans all experts top_k times from global memory; at a prompt's tens of rows that is tens of
// threads. Here 64 threads share a row. A transparent `gpu_core::upgrade` row (`Op::RouterGate`), so every caller of
// router_gate_sigmoid inherits it; the gate is `gpu-core` `tests/it/router_gate_upgrade.rs` (bits, not a tolerance).
//
// Dispatch: rows * 64 invocations (one workgroup per row), 2D-grid safe.

const WG: u32 = 64u;
const MAX_TOP_K: u32 = 32u;
const MAX_GROUP: u32 = 64u;
const FLOOR: f32 = -3.4e38;

struct Params {
    n_rows: u32,
    n_experts: u32,
    top_k: u32,
    n_group: u32,
    topk_group: u32,
    norm: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       bias:   array<f32>;
@group(0) @binding(3) var<storage, read_write> gate:   array<f32>;
@group(0) @binding(4) var<storage, read_write> probs:  array<f32>;

var<workgroup> red_v: array<f32, WG>;
var<workgroup> red_i: array<u32, WG>;
var<workgroup> sel_idx: array<u32, MAX_TOP_K>;
var<workgroup> group_keep: array<u32, MAX_GROUP>;
var<workgroup> gscore: array<f32, MAX_GROUP>;
var<workgroup> denom_wg: f32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Uniform across the workgroup, so every barrier below stays uniform.
    let t = wg.y * nwg.x + wg.x;
    if (t >= p.n_rows) { return; }
    let lane = li.x;
    let E = p.n_experts;
    let base = t * E;
    let k = min(p.top_k, MAX_TOP_K);
    let ng = max(p.n_group, 1u);
    let per = E / ng;

    for (var e = lane; e < E; e = e + WG) {
        probs[base + e] = 1.0 / (1.0 + exp(-logits[base + e]));
    }
    storageBarrier();
    workgroupBarrier();

    // Group-limited masking, the reference's serial rule: one lane per group scores it, lane 0 keeps the best groups.
    if (lane < ng) {
        group_keep[lane] = select(0u, 1u, ng == 1u);
        if (ng > 1u) {
            var b1 = FLOOR;
            var b2 = FLOOR;
            for (var m: u32 = 0u; m < per; m = m + 1u) {
                let cv = probs[base + lane * per + m] + bias[lane * per + m];
                if (cv > b1) { b2 = b1; b1 = cv; }
                else if (cv > b2) { b2 = cv; }
            }
            gscore[lane] = b1 + b2;
        }
    }
    workgroupBarrier();
    if (lane == 0u && ng > 1u) {
        var gused: array<bool, MAX_GROUP>;
        for (var g: u32 = 0u; g < ng; g = g + 1u) { gused[g] = false; }
        for (var kk: u32 = 0u; kk < p.topk_group; kk = kk + 1u) {
            var best = 0u;
            var bestv = FLOOR;
            for (var g: u32 = 0u; g < ng; g = g + 1u) {
                if (!gused[g] && gscore[g] > bestv) { bestv = gscore[g]; best = g; }
            }
            gused[best] = true;
            group_keep[best] = 1u;
        }
    }
    workgroupBarrier();

    // Global top-k: per round, each lane's first strict maximum over its experts (ascending), then the argmax tree
    // with ties to the lower index - together the reference's ascending strict-greater scan.
    for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
        var best = 0u;
        var bestv = FLOOR;
        for (var e = lane; e < E; e = e + WG) {
            if (group_keep[e / per] == 0u) { continue; }
            var already = false;
            for (var s: u32 = 0u; s < kk; s = s + 1u) {
                if (sel_idx[s] == e) { already = true; break; }
            }
            if (!already) {
                let cv = probs[base + e] + bias[e];
                if (cv > bestv) { bestv = cv; best = e; }
            }
        }
        red_v[lane] = bestv;
        red_i[lane] = best;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s = s >> 1u) {
            if (lane < s) {
                let ov = red_v[lane + s];
                let oi = red_i[lane + s];
                // A lane that found nothing reports (FLOOR, 0), which loses to any candidate and ties to index 0.
                if (ov > red_v[lane] || (ov == red_v[lane] && oi < red_i[lane])) {
                    red_v[lane] = ov;
                    red_i[lane] = oi;
                }
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            sel_idx[kk] = red_i[0];
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        var sel_sum = 0.0;
        for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
            sel_sum = sel_sum + probs[base + sel_idx[kk]];
        }
        denom_wg = select(1.0, 1.0 / (sel_sum + 1e-20), p.norm != 0u);
    }
    workgroupBarrier();
    let denom = denom_wg;
    for (var e = lane; e < E; e = e + WG) {
        var selected = false;
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            if (sel_idx[s] == e) { selected = true; break; }
        }
        if (selected) { gate[base + e] = probs[base + e] * denom * p.scale; }
        else          { gate[base + e] = 0.0; }
    }
}
