// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory commit, key/value half: move the records the selection kept into the new bank, in rank order
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `anchor_commit` decides which candidate lands in each slot (`out_src`); this
// kernel moves the records. Slot `c` of group `g` takes its key and value from
// the in bank (`src < K`) or from the pending block (`src >= K`, pending row
// `new_row0 + src - K`), or becomes zero for an empty slot.
//
//   in_key / in_value / out_key / out_value : [rows, K, groups * head_dim]
//   pend_key / pend_value                   : [rows, C, groups * head_dim]
//   plan, out_src                           : as `anchor_commit` (`in_row`, `out_row`, `new_row0`)

struct Params {
    n_seq: u32,
    groups: u32,
    capacity: u32,
    head_dim: u32,
};

const EMPTY: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       plan:       array<u32>;
@group(0) @binding(2) var<storage, read>       out_src:    array<u32>;
@group(0) @binding(3) var<storage, read>       in_key:     array<f32>;
@group(0) @binding(4) var<storage, read>       in_value:   array<f32>;
@group(0) @binding(5) var<storage, read>       pend_key:   array<f32>;
@group(0) @binding(6) var<storage, read>       pend_value: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_key:    array<f32>;
@group(0) @binding(8) var<storage, read_write> out_value:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let hd = p.head_dim;
    if (gidx >= p.n_seq * p.groups * p.capacity * hd) { return; }

    let d = gidx % hd;
    let r = gidx / hd;
    let c = r % p.capacity;
    let r2 = r / p.capacity;
    let g = r2 % p.groups;
    let s = r2 / p.groups;
    let gd = p.groups * hd;
    let src = out_src[(plan[s * 5u + 1u] * p.groups + g) * p.capacity + c];
    let dst = (plan[s * 5u + 1u] * p.capacity + c) * gd + g * hd + d;
    if (src == EMPTY) {
        out_key[dst] = 0.0;
        out_value[dst] = 0.0;
    } else if (src < p.capacity) {
        let from_i = (plan[s * 5u] * p.capacity + src) * gd + g * hd + d;
        out_key[dst] = in_key[from_i];
        out_value[dst] = in_value[from_i];
    } else {
        let from_i = (plan[s * 5u + 2u] + (src - p.capacity)) * gd + g * hd + d;
        out_key[dst] = pend_key[from_i];
        out_value[dst] = pend_value[from_i];
    }
}
