// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training mask loss forward: bw * mean BCE(logits, t) + dw * (1 - (2 sum(p t) + 1) / (sum p + sum t + 1))
// @how   one 64-thread workgroup, four strided partial sums, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// stats receives (sum p, sum p t, sum t) for the backward. Params: n, bw, dw (f32 bits). Dispatch: 1 workgroup.

struct Params { n: u32, bw: f32, dw: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;

var<workgroup> p_bce: array<f32, 64>;
var<workgroup> p_sp: array<f32, 64>;
var<workgroup> p_spt: array<f32, 64>;
var<workgroup> p_st: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    var bce = 0.0;
    var sp = 0.0;
    var spt = 0.0;
    var st = 0.0;
    for (var i = t; i < p.n; i = i + 64u) {
        let v = x[i];
        let tg = tgt[i];
        bce = bce + max(v, 0.0) + log(1.0 + exp(-abs(v))) - tg * v;
        let pr = 1.0 / (1.0 + exp(-v));
        sp = sp + pr;
        spt = spt + pr * tg;
        st = st + tg;
    }
    p_bce[t] = bce;
    p_sp[t] = sp;
    p_spt[t] = spt;
    p_st[t] = st;
    workgroupBarrier();
    if (t == 0u) {
        var a = 0.0;
        var b = 0.0;
        var c = 0.0;
        var d = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            a = a + p_bce[i];
            b = b + p_sp[i];
            c = c + p_spt[i];
            d = d + p_st[i];
        }
        let dice = 1.0 - (2.0 * c + 1.0) / (b + d + 1.0);
        out[0] = p.bw * a / f32(p.n) + p.dw * dice;
        stats[0] = b;
        stats[1] = c;
        stats[2] = d;
    }
}
