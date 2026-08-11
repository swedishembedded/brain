// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Router gating: softmax over experts -> keep top_k -> renormalise
// @how   one thread per token, array-free (no expert-count cap)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Router gating: softmax over experts -> keep top_k -> renormalise. Produces a
// dense gate matrix [seq_len, n_experts] that is nonzero only for the top_k
// experts of each token (their renormalised probabilities). One invocation
// per token.
//
// Running every expert and masking by this gate is numerically identical to a
// true sparse top-k dispatch *without* capacity dropping. Capacity limits exist
// only to bound memory during training; inference has no such pressure, so this
// is the exact top-k MoE output. `moe_linear_gated.wgsl` is the sparse
// alternative that actually skips a non-routed row's FLOPs instead of relying
// on this property to discard a densely-computed one.
//
// Used to hard-cap at `n_experts <= 128` via `array<f32,128> prob`/
// `array<bool,128> used` scratch — silent out-of-bounds writes above that
// (.agents/rules/lessons.md #35b's failure shape: a `const` bump without its array
// literal is a silent out-of-bounds write; Qwen3.5-35B-A3B's 256 experts hit
// this wall directly). Fixed the same way `router_bwd.wgsl` already was:
// nothing here is cached in an array sized by `n_experts`. The softmax
// numerator is stashed in the `gate` OUTPUT buffer itself (we already own
// read_write access to `[seq_len, n_experts]`, so it doubles as scratch — no
// second E-sized buffer needed) and re-divided by `sm` on every later read
// instead of being cached. The only `var<function>` array left is
// `sel_idx: array<u32, MAX_TOP_K>`, bounded by `top_k` (8 at the real 256-
// expert scale), never by `n_experts` — this is the actual fix `docs/
// lessons.md` #35b calls for ("an array-free top-k rewrite... no
// n_experts-sized function array"), not a bigger constant.
//
// Cost: the top-k selection becomes O(top_k^2 * n_experts) instead of the
// old O(top_k * n_experts) (excluding an already-picked expert now scans the
// small `sel_idx` set instead of an O(1) `used[e]` lookup) — at the real
// scale (top_k=8, E=256) that is ~17k vs ~3.5k scalar ops per token, still
// trivial next to a single GEMM. A workgroup-cooperative (`_rows`-style)
// rewrite is a valid follow-on per `.agents/rules/kernels.md`'s
// measure-before-optimizing rule; this fix is correctness-first.

struct Params {
    seq_len: u32,
    n_experts: u32,
    top_k: u32,
};

const MAX_TOP_K: u32 = 32u; // bounds ONLY the top-k bookkeeping, never n_experts

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
    let k = min(p.top_k, MAX_TOP_K);

    // Pass 1: row max (no array — a running scalar).
    var mx = -3.4e38;
    for (var e: u32 = 0u; e < E; e = e + 1u) { mx = max(mx, logits[base + e]); }

    // Pass 2: unnormalised softmax numerator, stashed into `gate` (scratch
    // reuse of the output buffer) + its sum.
    var sm = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        let pe = exp(logits[base + e] - mx);
        gate[base + e] = pe;
        sm = sm + pe;
    }

    // Pass 3: greedy top-k. `sel_idx` is bounded by MAX_TOP_K, not E.
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
            if (!already) {
                let pr_e = gate[base + e] / sm;
                if (pr_e > best_v) { best_v = pr_e; best = e; }
            }
        }
        sel_idx[kk] = best;
        sel_sum = sel_sum + best_v;
    }

    // Pass 4: finalise — renormalise the kept experts, zero the rest.
    let inv = 1.0 / max(sel_sum, 1e-9);
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        var selected = false;
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            if (sel_idx[s] == e) { selected = true; break; }
        }
        if (selected) {
            gate[base + e] = (gate[base + e] / sm) * inv;
        } else {
            gate[base + e] = 0.0;
        }
    }
}
