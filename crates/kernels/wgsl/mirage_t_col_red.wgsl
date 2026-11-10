// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training column reduction: dst[c] += sum over rows of g[r, c] (times o[r, c] when used)
// @how   64-thread workgroup per 8 columns (8 row lanes), 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Tiled column reduction: a 64-thread workgroup owns 8 adjacent columns; thread (rl, cl) sums rows rl, rl+8, ... of
// column cl, then the 8 row lanes are folded in a fixed order (one barrier), so the sum is deterministic and the
// loads of a row lane touch 8 adjacent columns.
// The gradient of a per-channel bias (use_o 0) or scale (use_o 1, o is the scaled input) over `rows` rows of width
// `period`; a scalar gradient is period 1. cw (1, 2, 4 or 8) is the number of columns a workgroup owns, so a narrow
// period still gets all 64 threads as row lanes (64 / cw of them).
// Params: rows, period, use_o, cw. Dispatch: ceil(period / cw) workgroups.

struct Params { rows: u32, period: u32, use_o: u32, cw: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  g: array<f32>;
@group(0) @binding(2) var<storage, read>  o: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let col0 = row * p.cw;
    let cl = t % p.cw;
    let rl = t / p.cw;
    let c = col0 + cl;
    var s = 0.0;
    if (c < p.period) {
        for (var r = rl; r < p.rows; r = r + 64u / p.cw) {
            let i = r * p.period + c;
            if (p.use_o != 0u) {
                s = s + g[i] * o[i];
            } else {
                s = s + g[i];
            }
        }
    }
    partial[t] = s;
    workgroupBarrier();
    if (t < p.cw) {
        let c2 = col0 + t;
        if (c2 < p.period) {
            var tot = 0.0;
            for (var k = 0u; k < 64u / p.cw; k = k + 1u) { tot = tot + partial[k * p.cw + t]; }
            dst[c2] = dst[c2] + tot;
        }
    }
}
