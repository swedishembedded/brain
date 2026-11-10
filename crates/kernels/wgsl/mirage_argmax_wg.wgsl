// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row-wise argmax with a workgroup per row: out[m] = f32(argmax_n x[m, n]) (ties go to the lowest n)
// @how   64-thread workgroup per row, strided scan then one barrier and a 64-way fold
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Same contract as argmax_row, which gives each row to a single thread. Params: m, n. Dispatch: one workgroup per row.

struct Params { m: u32, n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> pv: array<f32, 64>;
var<workgroup> pi: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.m) { return; }
    var best = -3.4e38;
    var at = 0xFFFFFFFFu;
    for (var c = t; c < p.n; c = c + 64u) {
        let v = x[row * p.n + c];
        if (v > best) { best = v; at = c; }
    }
    pv[t] = best;
    pi[t] = at;
    workgroupBarrier();
    var gb = -3.4e38;
    var gi = 0xFFFFFFFFu;
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (pv[i] > gb || (pv[i] == gb && pi[i] < gi)) { gb = pv[i]; gi = pi[i]; }
    }
    if (t == 0u) { out[row] = f32(gi); }
}
