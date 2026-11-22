// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE routing tables, stage 3 without workgroup memory: the slots in expert-major order
// @how   one thread per expert writes its slots in ascending order from its table offset (no barrier, CPU JIT safe)
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The same `perm` as `moe_route_emit.wgsl`: `perm[tab[e] + j]` is the j-th slot (ascending) routed to expert e, with
// `tab` from `moe_route_scan.wgsl` over `moe_route_count_serial.wgsl`'s counts.
//
//   params : u32 [slots, ne]
//   ids    : [slots] u32
//   tab    : [2 * (ne + 1)] u32
//   perm   : [slots] u32

struct Params {
    slots: u32,
    ne: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids:  array<u32>;
@group(0) @binding(2) var<storage, read>       tab:  array<u32>;
@group(0) @binding(3) var<storage, read_write> perm: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let e = gid.y * (nwg.x * 64u) + gid.x;
    if (e >= p.ne) { return; }
    var pos = tab[e];
    for (var i = 0u; i < p.slots; i = i + 1u) {
        if (ids[i] == e) {
            perm[pos] = i;
            pos = pos + 1u;
        }
    }
}
