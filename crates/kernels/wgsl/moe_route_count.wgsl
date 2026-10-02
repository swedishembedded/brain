// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE routing tables, stage 1: how many slots each expert was given
// @how   one workgroup per expert, strided scan of the slot ids, tree reduction
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The grouped expert GEMM (`moe_i8_grouped.wgsl`) wants each expert's slots
// together, so one launch can stream an expert's weights once for every token
// that chose it. Three kernels turn the router's `ids[slots]` (expert of each
// `row * slots_per_row + s` slot, as `moe_router_topk.wgsl` writes it, the shared
// expert being expert `n_experts`) into the tables that launch reads:
//
//   moe_route_count  counts[e]  = number of slots routed to expert e
//   moe_route_scan   tab        = exclusive offsets of those counts, and of the
//                                 MR-slot tiles each expert's slots fill
//   moe_route_emit   perm[pos]  = the slots in expert-major order, ascending
//                                 slot index within an expert (stable, so a
//                                 launch is deterministic)
//
// No atomics: an expert's workgroup scans every slot itself, so each count and
// each position is written by exactly one workgroup, and the order is a function
// of the ids alone.
//
//   params : u32 [slots, ne]        ne = experts counted (n_experts + shared)
//   ids    : [slots] u32
//   counts : [ne] u32

struct Params {
    slots: u32,
    ne: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids:    array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;

var<workgroup> red: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let e = wg.y * nwg.x + wg.x;
    let t = li.x;
    var n = 0u;
    if (e < p.ne) {
        for (var i = t; i < p.slots; i = i + 256u) {
            if (ids[i] == e) { n = n + 1u; }
        }
    }
    red[t] = n;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u && e < p.ne) { counts[e] = red[0]; }
}
