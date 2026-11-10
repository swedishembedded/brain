// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Engram gate: y += sigmoid(dot(rms_unit(y), rms_unit(key)) / sqrt(d)) * value, per row
// @how   64-thread workgroup per row, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// y, key, value: [rows, d]. Params: d, rows, eps (f32 bits), inv_sqrt_d (f32 bits). Dispatch: one workgroup per row.

struct Params { d: u32, rows: u32, eps: f32, inv_sqrt_d: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read>       key: array<f32>;
@group(0) @binding(3) var<storage, read>       value: array<f32>;

var<workgroup> pyy: array<f32, 64>;
var<workgroup> pkk: array<f32, 64>;
var<workgroup> pyk: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.d;
    var yy = 0.0;
    var kk = 0.0;
    var yk = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        let a = y[base + c];
        let b = key[base + c];
        yy = yy + a * a;
        kk = kk + b * b;
        yk = yk + a * b;
    }
    pyy[t] = yy;
    pkk[t] = kk;
    pyk[t] = yk;
    workgroupBarrier();
    var syy = 0.0;
    var skk = 0.0;
    var syk = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        syy = syy + pyy[i];
        skk = skk + pkk[i];
        syk = syk + pyk[i];
    }
    let iy = 1.0 / sqrt(syy / f32(p.d) + p.eps);
    let ik = 1.0 / sqrt(skk / f32(p.d) + p.eps);
    let alpha = 1.0 / (1.0 + exp(-(syk * iy * ik * p.inv_sqrt_d)));
    for (var c = t; c < p.d; c = c + 64u) {
        y[base + c] = y[base + c] + alpha * value[base + c];
    }
}
