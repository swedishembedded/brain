// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Extract one row of a per-token adaLN table and add a per-block table row
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// PixArt/adaLN-single modulation, without a host round trip:
//   out[r, d] = tbl[row*D + d] + tab[r*(NR*D) + row*D + d]        (plus_one = 0)
//   out[r, d] = 1.0 + (tbl[row*D + d] + tab[r*(NR*D) + row*D + d]) (plus_one = 1)
//
//   tab : [R, NR, D]  the MODEL-level per-token adaLN table, one per forward
//   tbl : [NR, D]     this BLOCK's own scale_shift_table
//   out : [R, D]      one of the NR modulation vectors
//
// The alternative this replaces is `dit::adaln::add_table` + a slice per row on
// the HOST, once per block: at the real 22B/720p shape that is a [3520, 36864]
// f32 combine (519 MB written) plus nine [3520, 4096] slices, then nine 57.7 MB
// uploads - per block, times 48 blocks, times every denoise step, for a table
// whose only per-block input is the 147 KB `tbl`. Measured at 36.0 s of a
// 103.3 s real forward before this kernel existed.
//
// BIT-IDENTICAL to that host form on purpose, and the operand order is why:
// `add_table` computes `table[i] + v[r*width+i]` (block table first, model
// value second) and `slice_mod`'s `one_plus` computes `1.0 + x`. Both are
// reproduced here exactly - one f32 add, then one more for `plus_one` - so a
// device-side modulation changes no number a downstream kernel reads.
//
// Coalesced by construction: consecutive threads walk consecutive `d`, which is
// contiguous in `out` AND contiguous in `tab` (the row axis is the slowest), so
// each 32-lane group reads and writes one full sector run. No reduction, no
// shared memory, no barrier - it is a pure streaming op at the bandwidth roof,
// which is the right shape for something that exists to replace PCIe traffic.

struct Params {
    R: u32,   // token rows
    D: u32,   // per-row width (the model's inner dim)
    NR: u32,  // rows in the adaLN table (9 for the video stream)
    row: u32, // which of the NR rows to extract
    plus_one: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tab: array<f32>;
@group(0) @binding(2) var<storage, read>       tbl: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.R * p.D) { return; }
    let r = idx / p.D;
    let d = idx % p.D;
    let off = p.row * p.D + d;
    let v = tbl[off] + tab[r * p.NR * p.D + off];
    if (p.plus_one == 1u) {
        out[idx] = 1.0 + v;
    } else {
        out[idx] = v;
    }
}
