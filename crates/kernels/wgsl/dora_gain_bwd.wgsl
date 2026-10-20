// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Weight-decomposed adapter gain backward:  gmu += dg*ratio,  c = -dg*g/v2
// @how   one thread per output channel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Given dg_o = sum_m dy[m,o] * y_pre[m,o] and the gains [g | ratio | 1/v2]
// written by dora_gain: the magnitude gradient is gmu_o += dg_o * ratio_o, and
// the gradient that flows through the row norm into the direction is
// dV_o = c_o * V_o with c_o = -dg_o * g_o / v2_o. The gradients of the factors
// A and B follow from c without forming V densely.

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dg:   array<f32>;
@group(0) @binding(2) var<storage, read>       gain: array<f32>;
@group(0) @binding(3) var<storage, read_write> gmu:  array<f32>;
@group(0) @binding(4) var<storage, read_write> c:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let o = gid.y * (nwg.x * 64u) + gid.x;
    if (o >= p.n) { return; }
    gmu[o] = gmu[o] + dg[o] * gain[p.n + o];
    c[o] = -dg[o] * gain[o] * gain[2u * p.n + o];
}
