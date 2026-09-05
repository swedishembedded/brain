// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped MoE, stage 1: per-expert routed-row count and tile count, both host-readback-free
// @how   one thread per expert, serial reduction over rows
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Grouped MoE (device-side permutation + grouped GEMM), stage 1: for every
// expert `e`, count how many rows the router sent it (`row_start[e]`, u32 -
// `expert_counts.wgsl`'s own float/aux-loss sibling, minus the
// `/ (rows*top_k)` division that kernel exists for) AND, in the SAME pass,
// how many `bm`-row GEMM tiles that count needs (`tile_start[e] =
// ceil(count/bm)`). One dispatch instead of two: both values fall out of the
// same per-expert row scan for free.
//
// `row_count[e]` is the raw count, kept UNTOUCHED by the exclusive scan the
// caller runs next (`record_group_scan` over `scan_block.wgsl`/`scan_add
// .wgsl`, which scans a buffer IN PLACE) - `matmul_reg3_grouped.wgsl` needs
// both the scanned start offset AND the original count (to bound `m_end`),
// so `row_start`/`tile_start` start as a COPY of the same raw values the
// scan then overwrites in place, while `row_count` survives as the
// unscanned reference.

struct Params {
    rows: u32,
    n_experts: u32,
    bm: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate:       array<f32>;
@group(0) @binding(2) var<storage, read_write> row_count:  array<u32>;
@group(0) @binding(3) var<storage, read_write> row_start:  array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_start: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let e = gidx;
    if (e >= p.n_experts) { return; }
    var count: u32 = 0u;
    for (var r: u32 = 0u; r < p.rows; r = r + 1u) {
        if (gate[r * p.n_experts + e] > 0.0) { count = count + 1u; }
    }
    row_count[e] = count;
    row_start[e] = count;
    tile_start[e] = (count + p.bm - 1u) / p.bm;
}
