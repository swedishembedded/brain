// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Engram n-gram table rows of the token at pos: hashed with 32-bit wrapping arithmetic, 0xFFFFFFFF where the n-gram is incomplete
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// For table t of order o (orders list, `hashes` tables per order): acc = seed * (order_index * hashes + h + 1);
// for j < o: acc = (acc ^ tok[pos - j]) * prime (tokens before position 0 count as 0); acc ^= acc >> 15;
// row = t * slots + acc % slots. The host decoder computes the same values in u64 masked to 32 bits.
// hist: [batch, maxt]; rows: [batch, tables]. Params: batch, maxt, pos, tables, hashes, slots, o0, o1, o2, o3.
// Dispatch: batch*tables threads.

struct Params { batch: u32, maxt: u32, pos: u32, tables: u32, hashes: u32, slots: u32, o0: u32, o1: u32, o2: u32, o3: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       hist: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.tables) { return; }
    let t = idx % p.tables;
    let b = idx / p.tables;
    let oi = t / p.hashes;
    let hh = t % p.hashes;
    var order = p.o0;
    if (oi == 1u) { order = p.o1; }
    if (oi == 2u) { order = p.o2; }
    if (oi == 3u) { order = p.o3; }
    var acc = 0x9E3779B9u * (oi * p.hashes + hh + 1u);
    for (var j = 0u; j < order; j = j + 1u) {
        var tok = 0u;
        if (p.pos >= j) { tok = hist[b * p.maxt + p.pos - j]; }
        acc = (acc ^ tok) * 0x01000193u;
    }
    acc = acc ^ (acc >> 15u);
    var r = 0xFFFFFFFFu;
    if (p.pos + 1u >= order) { r = t * p.slots + acc % p.slots; }
    rows[idx] = r;
}
