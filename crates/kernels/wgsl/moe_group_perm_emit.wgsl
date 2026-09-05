// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped MoE, stage 2: emit the row-permutation and its inverse from the scanned per-expert offsets
// @how   one thread per expert, serial scan over rows
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Grouped MoE, stage 2: turn `router_topk_compact.wgsl`'s per-row top-k
// expert ids plus `moe_group_counts.wgsl` -> `record_scan`'s exclusive
// per-expert row offsets into two arrays a single grouped GEMM dispatch (see
// `matmul_reg3_grouped.wgsl`) and its combine step need:
//
//  - `perm[pos] = r`          -- expert-major: compacted row `pos` came from
//                                 original row `r` (feeds `embed.wgsl`'s
//                                 existing row-gather unchanged).
//  - `pos_for_slot[r*top_k+k] = pos` -- the INVERSE, row-major: original row
//                                 `r`'s `k`-th selected expert's contribution
//                                 lives at compacted row `pos` (feeds
//                                 `moe_group_combine.wgsl`'s per-row sum).
//
// One invocation per expert `e` walks every row in order, so its own writes
// land in the DISJOINT range `[row_start[e], row_start[e]+count_e)` that
// `record_scan`'s exclusive scan already reserved for it - no atomics, no
// cross-thread ordering requirement, and (unlike a chunked histogram
// scatter) no bound on `n_experts` at all. Finding which of `top_ids`'
// `top_k` slots this row's expert-`e` selection sits in is an O(top_k)
// inner scan - cheap (`top_k` is a handful, not `n_experts`), and it is the
// step that lets the combine kernel below sum only `top_k` compacted rows
// per output row instead of scanning all `n_experts` of them.

struct Params {
    rows: u32,
    n_experts: u32,
    top_k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate:         array<f32>;
@group(0) @binding(2) var<storage, read>       top_ids:      array<u32>;
@group(0) @binding(3) var<storage, read>       row_start:    array<u32>;
@group(0) @binding(4) var<storage, read_write> perm:         array<u32>;
@group(0) @binding(5) var<storage, read_write> pos_for_slot: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let e = gidx;
    if (e >= p.n_experts) { return; }
    var pos = row_start[e];
    for (var r: u32 = 0u; r < p.rows; r = r + 1u) {
        if (gate[r * p.n_experts + e] <= 0.0) { continue; }
        perm[pos] = r;
        let base = r * p.top_k;
        for (var k: u32 = 0u; k < p.top_k; k = k + 1u) {
            if (top_ids[base + k] == e) {
                pos_for_slot[base + k] = pos;
                break;
            }
        }
        pos = pos + 1u;
    }
}
