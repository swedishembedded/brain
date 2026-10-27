// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory inverse bank map: slot_of[seq, block, group, position] = slot + 1 of the bank entry holding that position (0 = none)
// @how   one thread per entry, two phases (clear, then scatter) dispatched in order
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The backward of the indexed read must add each record's key and value
// gradient from every row that read it. Scattering from the rows would need
// atomics; this map lets the record gather instead (`anchor_read_bwd_kv`).
// A record occupies at most one slot of a block's bank, so the scatter has no
// write conflicts.
//
//   phase 0: slot_of[i] = 0 for every entry         (threads: n_seq * n_blocks * groups * t)
//   phase 1: slot_of[(b, s, g, ref - 1)] = c + 1    (threads: n_seq * n_blocks * groups * capacity)
//
//   refs    : [n_blocks, n_seq, groups, capacity]   (`anchor_commit`'s bank, block-major)
//   slot_of : [n_blocks, n_seq, groups, t]

struct Params {
    n_seq: u32,
    t: u32,
    groups: u32,
    capacity: u32,
    chunk: u32,
    phase: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       refs:    array<u32>;
@group(0) @binding(2) var<storage, read_write> slot_of: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n_blocks = (p.t + p.chunk - 1u) / p.chunk;
    if (p.phase == 0u) {
        if (gidx >= p.n_seq * n_blocks * p.groups * p.t) { return; }
        slot_of[gidx] = 0u;
        return;
    }
    if (gidx >= p.n_seq * n_blocks * p.groups * p.capacity) { return; }
    let c = gidx % p.capacity;
    let bg = gidx / p.capacity;           // (b * n_seq + s) * groups + g
    let slot = refs[gidx];
    if (slot != 0u) {
        slot_of[bg * p.t + slot - 1u] = c + 1u;
    }
}
