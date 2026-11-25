// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row sums over three tables: out[r] = sum over the taps of row r of table_{tag}[row], a tap naming its table in its top two bits
// @how   one thread per output element, serial sum over the row's taps in order
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The input rows of a model whose every row is a sum of embedding rows drawn from
// several tables (a speech decoder's frame rows: one row per codebook, a text row,
// a speaker row). For r < rows and c < width:
//
//   out[r * width + c] = sum_{k < taps, idx[r * taps + k] != 0xFFFFFFFF} table_{tag}[row * width + c]
//
// where `tag = idx >> 30` picks `t0`, `t1` or `t2` and `row = idx & 0x3FFFFFFF`.
// A tap of 0xFFFFFFFF is padding and adds nothing, so a row with fewer terms than
// `taps`, or a row of no terms at all (written as zeros), shares one fixed layout.
// The terms of a row are added in tap order, starting from 0.0, in f32: a caller
// that lists them in the order its reference sums them gets the reference's bits.
// The adjoint with respect to each table is `rows_gather_weighted_bwd.wgsl` over
// that table's taps transposed (a host-built CSR, unit weights).
//
//   params: rows, taps, width

struct Params {
    rows: u32,
    taps: u32,
    width: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       idx: array<u32>;
@group(0) @binding(2) var<storage, read>       t0:  array<f32>;
@group(0) @binding(3) var<storage, read>       t1:  array<f32>;
@group(0) @binding(4) var<storage, read>       t2:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= p.rows * p.width) { return; }
    let r = t / p.width;
    let c = t % p.width;
    var acc = 0.0;
    for (var k = 0u; k < p.taps; k = k + 1u) {
        let tap = idx[r * p.taps + k];
        if (tap == 0xFFFFFFFFu) { continue; }
        let at = (tap & 0x3FFFFFFFu) * p.width + c;
        let tag = tap >> 30u;
        if (tag == 0u) {
            acc = acc + t0[at];
        } else if (tag == 1u) {
            acc = acc + t1[at];
        } else {
            acc = acc + t2[at];
        }
    }
    out[t] = acc;
}
