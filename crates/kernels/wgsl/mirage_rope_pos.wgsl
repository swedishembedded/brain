// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Rotary embedding of one position shared by every row: rotate-half pairs (i, i + hd/2), frequency base^(-2i/hd)
// @how   one thread per (row, rotation pair)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// t: [rows, hd] (a row is one head of one sequence). The position is read from `pos[0]`. Params: rows, hd, base (f32 bits). Dispatch: rows*hd/2 threads.

struct Params { rows: u32, hd: u32, base: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> t: array<f32>;
@group(0) @binding(2) var<storage, read>       pos: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.hd / 2u;
    if (idx >= p.rows * half) { return; }
    let i = idx % half;
    let r = idx / half;
    let angle = f32(pos[0]) * exp(-log(p.base) * f32(2u * i) / f32(p.hd));
    let s = sin(angle);
    let co = cos(angle);
    let at = r * p.hd + i;
    let a = t[at];
    let b = t[at + half];
    t[at] = a * co - b * s;
    t[at + half] = b * co + a * s;
}
