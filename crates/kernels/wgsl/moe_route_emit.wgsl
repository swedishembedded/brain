// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE routing tables, stage 3: the slots in expert-major order
// @how   one workgroup per expert, each thread owns a contiguous run of slots, a Hillis-Steele scan of the run counts places them
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// See `moe_route_count.wgsl`. Expert `e`'s workgroup splits the slot range into
// 256 contiguous runs, counts the matches in its run, scans the 256 counts to
// find where each run starts inside the expert's block, then writes the slot
// indices of its run there. Runs are contiguous and scanned in order, so inside
// an expert the slots are ascending - the same order a serial scan produces.
//
//   params : u32 [slots, ne]
//   ids    : [slots] u32
//   tab    : [2 * (ne + 1)] u32   from `moe_route_scan.wgsl`; `tab[e]` is the block
//   perm   : [slots] u32          perm[tab[e] + j] = j-th slot routed to expert e

struct Params {
    slots: u32,
    ne: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids:  array<u32>;
@group(0) @binding(2) var<storage, read>       tab:  array<u32>;
@group(0) @binding(3) var<storage, read_write> perm: array<u32>;

var<workgroup> scan: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let e = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = e < p.ne;
    let run = (p.slots + 255u) / 256u;
    let lo = min(t * run, p.slots);
    let hi = min(lo + run, p.slots);

    var mine = 0u;
    if (live) {
        for (var i = lo; i < hi; i = i + 1u) {
            if (ids[i] == e) { mine = mine + 1u; }
        }
    }
    scan[t] = mine;
    workgroupBarrier();
    // Inclusive Hillis-Steele scan of the run counts.
    for (var s: u32 = 1u; s < 256u; s = s << 1u) {
        var add = 0u;
        if (t >= s) { add = scan[t - s]; }
        workgroupBarrier();
        scan[t] = scan[t] + add;
        workgroupBarrier();
    }
    if (live) {
        var pos = tab[e] + scan[t] - mine;
        for (var i = lo; i < hi; i = i + 1u) {
            if (ids[i] == e) {
                perm[pos] = i;
                pos = pos + 1u;
            }
        }
    }
}
