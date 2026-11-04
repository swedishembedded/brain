// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Native sparse attention: emit one (key row, query) pair per selected key, the input of the selection's inversion
// @how   one 64-thread workgroup per query row, padding past the row's count with an all-ones key that sorts last
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype n/a
//
// Swedish Embedded AB implements long-context models trained with sparse
// attention in time and memory linear in the context, for clients whose
// sequences outgrow quadratic attention. If your team needs expertise in
// sparse attention training kernels, you can procure our services by sending
// an email to info@swedishembedded.com.
//
// The first step of inverting a native selection on the device: query row
// `r` (`b * t + i`) of the index (`[count, row, live, positions...]` rows of
// `stride` words) emits, at `r * emit_width + e`, the key row `b * t + j` of
// its `e`-th selected position `j` and the query position `i`. Slots past
// the row's count get key 0xFFFFFFFF. A stable sort by key then lists every
// key's queries in ascending order, and `lsa_native_offsets` turns that into
// the index's inverse.

const WG: u32 = 64u;
const HEADER: u32 = 3u;

struct Params {
    rows: u32,
    t: u32,
    stride: u32,
    emit_width: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       index: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys:  array<u32>;
@group(0) @binding(3) var<storage, read_write> vals:  array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wgid.y * nwg.x + wgid.x;
    if (r >= p.rows) { return; }
    let srow = r * p.stride;
    let count = min(index[srow], p.emit_width);
    let seq0 = (r / p.t) * p.t;
    let i = r % p.t;
    for (var e = lid.x; e < p.emit_width; e = e + WG) {
        let at = r * p.emit_width + e;
        keys[at] = select(0xFFFFFFFFu, seq0 + index[srow + HEADER + e], e < count);
        vals[at] = i;
    }
}
