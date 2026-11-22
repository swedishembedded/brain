// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE routing tables, stage 1 without workgroup memory: how many slots each expert was given
// @how   one thread per expert scans every slot id serially (no barrier, so it runs on the CPU JIT)
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The same `counts` as `moe_route_count.wgsl` (whose tree reduction needs a workgroup barrier in a loop). A prefill
// chunk routes a few thousand slots to a few hundred experts, so a serial scan per expert is cheap anywhere; with
// `moe_route_scan.wgsl` and `moe_route_emit_serial.wgsl` it builds the table a grouped expert product reads on every
// backend.
//
//   params : u32 [slots, ne]        ne = experts counted
//   ids    : [slots] u32
//   counts : [ne] u32

struct Params {
    slots: u32,
    ne: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids:    array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let e = gid.y * (nwg.x * 64u) + gid.x;
    if (e >= p.ne) { return; }
    var n = 0u;
    for (var i = 0u; i < p.slots; i = i + 1u) {
        if (ids[i] == e) { n = n + 1u; }
    }
    counts[e] = n;
}
