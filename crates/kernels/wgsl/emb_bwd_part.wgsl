// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Stage one of the embedding backward: per block of rows, each vocabulary row's partial gradient
// @how   one thread per (row block, vocabulary row, channel), serial scan of its block
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// part[blk, v, c] = sum over rows r of block blk with tokens[r] == v of dx[r, c]
// Blocks are contiguous runs of ceil(n_rows / P) rows. `dw_splitk_reduce`
// folds the P blocks into the table's gradient (accumulating). `emb_bwd`
// gives each (vocabulary row, channel) one thread that scans every row: with
// a small vocabulary and tens of thousands of rows that is too few threads,
// each with a long serial walk. Splitting the rows across P blocks keeps the
// same comparisons but runs them P times wider. No atomics: one thread owns
// each partial.

struct Params {
    n_rows: u32,
    d: u32,
    vocab: u32,
    P: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens: array<u32>;
@group(0) @binding(2) var<storage, read>       dx:     array<f32>;
@group(0) @binding(3) var<storage, read_write> part:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let per = p.vocab * p.d;
    if (gidx >= per * p.P) { return; }
    let blk = gidx / per;
    let rem = gidx % per;
    let v = rem / p.d;
    let c = rem % p.d;
    let len = (p.n_rows + p.P - 1u) / p.P;
    let r0 = blk * len;
    let r1 = min(p.n_rows, r0 + len);
    var acc = 0.0;
    for (var r = r0; r < r1; r = r + 1u) {
        if (tokens[r] == v) {
            acc = acc + dx[r * p.d + c];
        }
    }
    part[gidx] = acc;
}
