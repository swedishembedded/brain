// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  2D rotary embedding on image-cell tokens, registers left unrotated (Mirage encoder attention)
// @how   one thread per (token, head, rotation pair)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// t: [batch * tokens, heads, hd], rotate-half pairs (j, j + hd/2). The first half of the pairs encodes the
// cell row, the second half the cell column, with frequency base^(-(j mod per_axis)/per_axis).
// Tokens at index >= cells (the registers) are not rotated.
// Params: tokens, cells, grid, heads, hd, base (f32 bits), rows. Dispatch: rows*heads*hd/2 threads.

struct Params { tokens: u32, cells: u32, grid: u32, heads: u32, hd: u32, base: f32, rows: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> t: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let pairs = p.hd / 2u;
    if (idx >= p.rows * p.heads * pairs) { return; }
    let j = idx % pairs;
    let head = (idx / pairs) % p.heads;
    let row = idx / (pairs * p.heads);
    let tok = row % p.tokens;
    if (tok >= p.cells) { return; }
    let per_axis = pairs / 2u;
    var pos = f32(tok % p.grid);
    if (j < per_axis) { pos = f32(tok / p.grid); }
    let freq = exp(-log(p.base) * f32(j % per_axis) / f32(per_axis));
    let angle = pos * freq;
    let s = sin(angle);
    let co = cos(angle);
    let at = (row * p.heads + head) * p.hd + j;
    let a = t[at];
    let b = t[at + pairs];
    t[at] = a * co - b * s;
    t[at + pairs] = b * co + a * s;
}
