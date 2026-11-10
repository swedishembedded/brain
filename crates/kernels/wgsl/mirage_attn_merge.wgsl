// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fold the key-split partials of single-query attention into the output row
// @how   one 64-thread workgroup per row, a lane per value column, no barrier
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// part: [rows, splits, vd + 2] holding (acc[0..vd), max, sum) per split as mirage_attn_decode writes them.
// out[r, i] = sum_s exp(max_s - M) acc_s[i] / sum_s exp(max_s - M) sum_s, M = max_s max_s; zero when no split saw a
// live key (M at the -1e30 floor). Params: rows, vd, splits. Dispatch: one workgroup per row.

struct Params { rows: u32, vd: u32, splits: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows || t >= p.vd) { return; }
    let stride = p.vd + 2u;
    var gm = -1.0e30;
    for (var s = 0u; s < p.splits; s = s + 1u) { gm = max(gm, part[(row * p.splits + s) * stride + p.vd]); }
    var num = 0.0;
    var den = 0.0;
    if (gm > -1.0e29) {
        for (var s = 0u; s < p.splits; s = s + 1u) {
            let base = (row * p.splits + s) * stride;
            let e = exp(part[base + p.vd] - gm);
            num = num + e * part[base + t];
            den = den + e * part[base + p.vd + 1u];
        }
    }
    var r = 0.0;
    if (den > 0.0) { r = num / den; }
    out[row * p.vd + t] = r;
}
