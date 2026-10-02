// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MoE router head: softmax over experts, top-k ids and renormalised weights, plus the shared expert's sigmoid weight, one workgroup per row
// @how   256-thread workgroup, tree reductions for max/sum/argmax, top_k rounds of argmax
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The decode-regime replacement for `router_gate.wgsl` + `router_topk_compact
// .wgsl`. Those two run one THREAD per row, a serial O(top_k^2 * n_experts)
// selection that at decode (one row) is a single lane walking ~17k
// comparisons per layer while the rest of the device idles; this kernel gives
// each row a whole workgroup (thread `t` owns experts `t, t+256, ...`) and
// finds the top_k with `top_k` tree-argmax rounds, so a row costs
// `top_k * log2(256)` barrier steps instead of `top_k^2 * n_experts` serial
// ones.
//
// It also writes what the expert GEMVs actually consume - the COMPACT id and
// weight of each slot - instead of a dense `[rows, n_experts]` gate that a
// second kernel has to scan: `ids[row, s]` is the s-th selected expert and
// `weight[row, s]` its renormalised softmax probability. The semantics are
// `router_gate.wgsl`'s with `norm = 1, scale = 1`: probabilities are the
// softmax over ALL experts, the `top_k` largest are kept, and the kept ones
// are divided by their own sum (`max(sum, 1e-9)`). Ties go to the LOWER
// expert index, as in that kernel's strict `>` scan. The softmax denominator
// is summed in a tree rather than left to right, so a weight can differ from
// `router_gate`'s by an ulp; the selection can only differ on a tie within
// that ulp.
//
// SHARED EXPERT SLOT. Qwen3.5-style MoE adds one always-on shared expert,
// scaled by `sigmoid(shared_gate . x)`. With `p.has_shared == 1` the logits row
// carries ONE extra column (`n_experts + 1` wide: the router weight with the
// shared gate's row appended), the softmax and top-k ignore it, and slot
// `top_k` is written as expert `n_experts` (the index the shared expert has in
// the expert bank, one past the routed ones) with weight `sigmoid(logit)`.
// The shared expert then rides through the routed experts' GEMVs as a ninth
// slot instead of costing its own four dispatches.
//
//   logits : [rows, n_experts + shared]   router matmul output
//   ids    : [rows, top_k + shared] u32
//   weight : [rows, top_k + shared] f32
//
// REQUIRES n_experts <= 1024 and top_k <= 16. Barriers are at the top level
// of `main`, so every thread of the workgroup reaches each of them.

struct Params {
    rows: u32,
    n_experts: u32,
    top_k: u32,
    has_shared: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids:    array<u32>;
@group(0) @binding(3) var<storage, read_write> weight: array<f32>;

const WG: u32 = 256u;
const MAX_PER_THREAD: u32 = 4u;      // n_experts <= WG * MAX_PER_THREAD

var<workgroup> red_v: array<f32, 256>;
var<workgroup> red_i: array<u32, 256>;
var<workgroup> sel_i: array<u32, 16>;
var<workgroup> sel_p: array<f32, 16>;
var<workgroup> row_max: array<f32, 1>;
var<workgroup> row_sum: array<f32, 1>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = row < p.rows;
    let e_n = p.n_experts;
    let width = e_n + p.has_shared;
    let base = row * width;

    // ---- row max ---------------------------------------------------------
    var mx = -3.4e38;
    if (live) {
        for (var j: u32 = 0u; j < MAX_PER_THREAD; j = j + 1u) {
            let e = t + j * WG;
            if (e < e_n) { mx = max(mx, logits[base + e]); }
        }
    }
    red_v[t] = mx;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red_v[t] = max(red_v[t], red_v[t + s]); }
        workgroupBarrier();
    }
    if (t == 0u) { row_max[0] = red_v[0]; }
    workgroupBarrier();

    // ---- softmax numerators and their sum --------------------------------
    var pe: array<f32, 4>;
    var part = 0.0;
    for (var j: u32 = 0u; j < MAX_PER_THREAD; j = j + 1u) {
        let e = t + j * WG;
        var v = 0.0;
        if (live && e < e_n) { v = exp(logits[base + e] - row_max[0]); }
        pe[j] = v;
        part = part + v;
    }
    red_v[t] = part;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red_v[t] = red_v[t] + red_v[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) { row_sum[0] = red_v[0]; }
    workgroupBarrier();
    let sm = row_sum[0];

    // ---- top_k rounds of argmax (value desc, index asc on ties) -----------
    let k = min(p.top_k, 16u);
    for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
        var best_v = -1.0;
        var best_i = 0xffffffffu;
        for (var j: u32 = 0u; j < MAX_PER_THREAD; j = j + 1u) {
            let e = t + j * WG;
            if (e < e_n) {
                var taken = false;
                for (var s: u32 = 0u; s < kk; s = s + 1u) {
                    if (sel_i[s] == e) { taken = true; }
                }
                let v = pe[j] / sm;
                // Ascending `j` is ascending `e`, so a strict `>` keeps the lower index.
                if (!taken && v > best_v) { best_v = v; best_i = e; }
            }
        }
        red_v[t] = best_v;
        red_i[t] = best_i;
        workgroupBarrier();
        for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
            if (t < s) {
                let ov = red_v[t + s];
                let oi = red_i[t + s];
                if (ov > red_v[t] || (ov == red_v[t] && oi < red_i[t])) {
                    red_v[t] = ov;
                    red_i[t] = oi;
                }
            }
            workgroupBarrier();
        }
        if (t == 0u) {
            sel_i[kk] = red_i[0];
            sel_p[kk] = red_v[0];
        }
        workgroupBarrier();
    }

    // ---- renormalise and write ------------------------------------------
    if (t == 0u && live) {
        var sel_sum = 0.0;
        for (var kk: u32 = 0u; kk < k; kk = kk + 1u) { sel_sum = sel_sum + sel_p[kk]; }
        let inv = 1.0 / max(sel_sum, 1e-9);
        let out_w = k + p.has_shared;
        for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
            ids[row * out_w + kk] = sel_i[kk];
            weight[row * out_w + kk] = sel_p[kk] * inv;
        }
        if (p.has_shared != 0u) {
            ids[row * out_w + k] = e_n;
            weight[row * out_w + k] = 1.0 / (1.0 + exp(-logits[base + e_n]));
        }
    }
}
