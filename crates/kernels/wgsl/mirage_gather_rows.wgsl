// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row gather from a table of sub-width rows with an absent marker: out[b, t*sub + i] = row[b,t] == 0xFFFFFFFF ? 0 : table[row[b,t]*sub + i] (engram lookup)
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: batch, tables, sub. Dispatch: batch*tables*sub threads.

struct Params { batch: u32, tables: u32, sub: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       rows: array<u32>;
@group(0) @binding(2) var<storage, read>       table: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.tables * p.sub) { return; }
    let i = idx % p.sub;
    let t = (idx / p.sub) % p.tables;
    let b = idx / (p.sub * p.tables);
    let row = rows[b * p.tables + t];
    var v = 0.0;
    if (row != 0xFFFFFFFFu) { v = table[row * p.sub + i]; }
    out[idx] = v;
}
