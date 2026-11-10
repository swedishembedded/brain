// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training global-response-norm backward statistics: de[c] and dgamma[c] from the scale gradient
// @how   one 64-thread workgroup, strided partials, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// With E the per-channel energy, den = mean(E) + eps and cross = sum_c gs[c] gamma[c] E[c]:
//   dgamma[c] += gs[c] E[c] / den,   de[c] = gs[c] gamma[c] / den - cross / (den^2 d).
// Params: d, eps (f32 bits). Dispatch: 1 workgroup.

struct Params { d: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  energy: array<f32>;
@group(0) @binding(2) var<storage, read>  gamma: array<f32>;
@group(0) @binding(3) var<storage, read>  gs: array<f32>;
@group(0) @binding(4) var<storage, read_write> de: array<f32>;
@group(0) @binding(5) var<storage, read_write> dgamma: array<f32>;

var<workgroup> p_sum: array<f32, 64>;
var<workgroup> p_cross: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    var s = 0.0;
    var x = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        s = s + energy[c];
        x = x + gs[c] * gamma[c] * energy[c];
    }
    p_sum[t] = s;
    p_cross[t] = x;
    workgroupBarrier();
    var tot = 0.0;
    var cross = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        tot = tot + p_sum[i];
        cross = cross + p_cross[i];
    }
    let den = tot / f32(p.d) + p.eps;
    for (var c = t; c < p.d; c = c + 64u) {
        dgamma[c] = dgamma[c] + gs[c] * energy[c] / den;
        de[c] = gs[c] * gamma[c] / den - cross / (den * den * f32(p.d));
    }
}
