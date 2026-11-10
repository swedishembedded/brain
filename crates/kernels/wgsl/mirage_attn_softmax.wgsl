// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Key-parallel masked GQA attention, pass 2: row softmax in place (rows with no live key become zeros)
// @how   64-thread workgroup per row, online max/sum combine, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// scores: [rows, cols] in place. Each thread folds a strided slice into a (max, sum of exp) pair, one barrier,
// every thread combines the 64 pairs and normalises its slice. Params: rows, cols. Dispatch: one workgroup per row.

struct Params { rows: u32, cols: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;

var<workgroup> pm: array<f32, 64>;
var<workgroup> pl: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var m = -1.0e30;
    var l = 0.0;
    for (var c = t; c < p.cols; c = c + 64u) {
        let v = scores[base + c];
        let nm = max(m, v);
        l = l * exp(m - nm) + exp(v - nm);
        m = nm;
    }
    pm[t] = m;
    pl[t] = l;
    workgroupBarrier();
    var gm = -1.0e30;
    for (var i = 0u; i < 64u; i = i + 1u) { gm = max(gm, pm[i]); }
    var gl = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) { gl = gl + pl[i] * exp(pm[i] - gm); }
    for (var c = t; c < p.cols; c = c + 64u) {
        var r = 0.0;
        if (gm > -1.0e29) { r = exp(scores[base + c] - gm) / gl; }
        scores[base + c] = r;
    }
}
