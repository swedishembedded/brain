// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Advance the batched-decode metadata one sub-step, on the device (A4)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype n/a
//
// Advance the batched-decode metadata one sub-step, on the device (A4):
// position and attended length grow by one, and the K/V write slot for the
// next token comes from a host-precomputed schedule — the host allocated the
// window's blocks up front, so no readback is needed between steps.
//
//   sched : [window-1, bsz, 3] u32 — (block, offset, bt_index) per row per
//           sub-step; bt_index = NO_BT means the append stayed inside the
//           row's current block and the block table is unchanged.
//
// One invocation per row.

struct Params {
    bsz: u32,
    /// Sub-step being prepared (indexes `sched` row `s`).
    s: u32,
    /// Stride of one row of the per-sequence block table.
    mbt: u32,
    /// Sentinel bt_index meaning "no new block this step".
    no_bt: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       sched:  array<u32>; // [w-1, bsz, 3]
@group(0) @binding(2) var<storage, read_write> pos:    array<u32>; // [bsz]
@group(0) @binding(3) var<storage, read_write> seqlen: array<u32>; // [bsz]
@group(0) @binding(4) var<storage, read_write> blk:    array<u32>; // [bsz]
@group(0) @binding(5) var<storage, read_write> off:    array<u32>; // [bsz]
@group(0) @binding(6) var<storage, read_write> bt:     array<u32>; // [bsz, mbt]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.bsz) { return; }
    pos[i] = pos[i] + 1u;
    seqlen[i] = seqlen[i] + 1u;
    let base = (p.s * p.bsz + i) * 3u;
    let b = sched[base];
    blk[i] = b;
    off[i] = sched[base + 1u];
    let bti = sched[base + 2u];
    if (bti != p.no_bt) {
        bt[i * p.mbt + bti] = b;
    }
}
