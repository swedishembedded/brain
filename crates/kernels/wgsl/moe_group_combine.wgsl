// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped MoE, stage 4: gate-scaled row-major combine of a compacted expert-major GEMM output
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Grouped MoE, stage 4 (the scatter half): unlike `moe_scatter_scaled_add
// .wgsl` (safe because it is dispatched once PER EXPERT, so its `idx` names
// rows that are distinct WITHIN that one call), a single dispatch spanning
// EVERY expert cannot scatter-add compacted rows straight back into a
// `[rows, d]` accumulator - a row selected by `top_k > 1` experts would need
// several different threads to `+=` the SAME output address with no atomics
// to make that safe. This kernel sidesteps the race by going the other
// direction: one thread per `(row, column)` OUTPUT element GATHERS its own
// `top_k` contributions (via `moe_group_perm_emit.wgsl`'s `pos_for_slot`
// inverse permutation) and sums them itself - the standard one-thread-per-
// output-element reduction shape, no cross-thread write ever shared.
//
// Summing in ascending `k` (equivalently ascending expert id, since
// `top_ids` is written by `router_topk_compact.wgsl` in ascending-column
// scan order) reproduces the dense oracle's own accumulation order bit for
// bit: the dense path's `scale_add` walks experts `0..n_experts` unconditionally
// and adds `gate[r,e] * expert_out_e[r,:]`, which is an exact no-op at every
// `gate[r,e] == 0` (`0.0 * x + acc == acc` in IEEE-754), so its only
// arithmetically-live steps are the SAME `top_k` terms in the SAME order this
// kernel visits via `top_ids`.
//
// WRITES (not accumulates) `out` - this one dispatch already sums every
// selected expert's contribution for the row, so unlike
// [`crate::moe::expert_fwd_compact_layer`]'s per-expert loop there is no
// earlier partial value for the caller to have pre-zeroed.

struct Params {
    rows: u32,
    d: u32,
    n_experts: u32,
    top_k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate:         array<f32>;
@group(0) @binding(2) var<storage, read>       top_ids:      array<u32>;
@group(0) @binding(3) var<storage, read>       pos_for_slot: array<u32>;
@group(0) @binding(4) var<storage, read>       expert_out:   array<f32>; // [compacted_rows, d]
@group(0) @binding(5) var<storage, read_write> out:          array<f32>; // [rows, d]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.d;
    if (gidx >= total) { return; }
    let r = gidx / p.d;
    let c = gidx % p.d;
    var acc = 0.0;
    let base = r * p.top_k;
    for (var k: u32 = 0u; k < p.top_k; k = k + 1u) {
        let e = top_ids[base + k];
        if (e >= p.n_experts) { continue; } // router_topk_compact's defensive sentinel
        let g = gate[r * p.n_experts + e];
        let pos = pos_for_slot[base + k];
        acc = acc + g * expert_out[pos * p.d + c];
    }
    out[r * p.d + c] = acc;
}
