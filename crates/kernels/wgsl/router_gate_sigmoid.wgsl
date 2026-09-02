// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GLM/DeepSeek-V3 "noaux_tc" MoE router (forward)
// @how   one thread per output element, array-free (no expert-count cap)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// GLM/DeepSeek-V3 "noaux_tc" MoE router (forward). Per token row:
//   s[e]      = sigmoid(logit[e])                     (combine score; -> probs)
//   choice[e] = s[e] + bias[e]                         (selection score only)
//   group-limited top-k: split E into `n_group` groups, score each group by the
//     sum of its top-2 `choice`, keep the best `topk_group` groups, then take the
//     global top-`top_k` `choice` among the kept groups.
//   gate[e]   = selected ? s[e] : 0 ; if norm: gate /= sum_selected s ; *= scale
// The bias steers *selection* only; the combine weight is the raw sigmoid score
// (aux-loss-free load balancing). One invocation per row.
//
// Used to hard-cap at `n_experts <= 64` via `array<f32/bool,64> s/choice/used`
// scratch - silent out-of-bounds writes above that (a familiar failure shape:
// `router_gate.wgsl`/`router_bwd.wgsl` already carry this exact story in their
// own headers). `crates/glmdsa`'s own `GlmConfig::glm5_2()` (256 routed
// experts) hit this wall directly and had to assert the model unbuildable
// rather than build it wrong - see that crate's `new_impl_on` for the guard
// this fix removes.
//
// Fixed the same way `router_gate.wgsl` was: nothing here is cached in an
// array sized by `n_experts`.
//   - `s[e]` is never cached - `probs[base+e]` (an output buffer this kernel
//     already owns) holds it after pass 1, and every later use re-reads it.
//   - `choice[e]` is never cached either - it is `probs[base+e] + bias[e]`,
//     recomputed inline at every use, with the group mask applied as a guard
//     (`group_keep[e/per]`) rather than a pre-baked `-inf` sentinel value.
//   - `used[e]` becomes `sel_idx: array<u32, MAX_TOP_K>`, bounded by `top_k`
//     (single digits at every real config), never by `n_experts` - the same
//     "already picked" O(kk) scan `router_gate.wgsl` uses instead of an O(1)
//     `used[e]` lookup.
// `group_keep`/`gscore`/`gused` stay `n_group`-sized arrays (`MAX_GROUP`, a
// GENUINELY different and much smaller bound - real configs use single-digit
// group counts) - renamed from the old kernel's overloaded `MAX_E` so the
// bound each array actually carries is named, not implied.
//
// Cost: the top-k selection becomes O(top_k^2 * n_experts) instead of the old
// O(top_k * n_experts) - at the real scale (top_k=8, E=256) that is ~16k vs
// ~2k scalar ops per token, still trivial next to a single GEMM. A
// workgroup-cooperative rewrite is a valid follow-on once profiling justifies
// it; this fix is correctness-first (kernel-performance.md M5.7).

const MAX_TOP_K: u32 = 32u; // bounds ONLY the top-k bookkeeping, never n_experts
const MAX_GROUP: u32 = 64u; // bounds ONLY n_group, never n_experts

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
    let k = min(p.top_k, MAX_TOP_K);

    // Pass 1: sigmoid -> `probs` (also serves as the `s[e]` scratch for every
    // later read - no separate E-sized array). `bias` is a per-expert vector
    // [E] (indexed by `e`, not per-row).
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        probs[base + e] = 1.0 / (1.0 + exp(-logits[base + e]));
    }

    // ---- group-limited masking: which groups may be selected from ----
    let ng = max(p.n_group, 1u);
    let per = E / ng;
    var group_keep: array<bool, MAX_GROUP>;
    for (var g: u32 = 0u; g < ng; g = g + 1u) { group_keep[g] = (ng == 1u); }
    if (ng > 1u) {
        // group score = sum of top-2 choice within the group
        var gscore: array<f32, MAX_GROUP>;
        for (var g: u32 = 0u; g < ng; g = g + 1u) {
            var b1 = -3.4e38;
            var b2 = -3.4e38;
            for (var m: u32 = 0u; m < per; m = m + 1u) {
                let cv = probs[base + g * per + m] + bias[g * per + m];
                if (cv > b1) { b2 = b1; b1 = cv; }
                else if (cv > b2) { b2 = cv; }
            }
            gscore[g] = b1 + b2;
        }
        // keep the top `topk_group` groups
        var gused: array<bool, MAX_GROUP>;
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

    // ---- global top-k over the (group-masked) choice scores ----
    // `sel_idx` is bounded by MAX_TOP_K, not E.
    var sel_idx: array<u32, 32>;
    var sel_sum = 0.0;
    for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
        var best = 0u;
        var bestv = -3.4e38;
        for (var e: u32 = 0u; e < E; e = e + 1u) {
            if (!group_keep[e / per]) { continue; }
            var already = false;
            for (var s: u32 = 0u; s < kk; s = s + 1u) {
                if (sel_idx[s] == e) { already = true; break; }
            }
            if (!already) {
                let cv = probs[base + e] + bias[e];
                if (cv > bestv) { bestv = cv; best = e; }
            }
        }
        sel_idx[kk] = best;
        sel_sum = sel_sum + probs[base + best];
    }

    // ---- finalise: keep the selected experts (renormalised iff `norm`, then
    // scaled), zero the rest ----
    let denom = select(1.0, 1.0 / max(sel_sum, 1e-20), p.norm != 0u);
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        var selected = false;
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            if (sel_idx[s] == e) { selected = true; break; }
        }
        if (selected) { gate[base + e] = probs[base + e] * denom * p.scale; }
        else          { gate[base + e] = 0.0; }
    }
}
