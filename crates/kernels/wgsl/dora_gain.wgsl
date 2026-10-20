// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Weight-decomposed adapter gain: g = (1+mu) * sqrt(w2 / v2) from the adapted row norms
// @how   one thread per output channel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// DoRA keeps a linear's direction W + s*B*A and rescales each output row to a
// learned magnitude. With w2 = |W_o|^2, s1 = <W_o, (BA)_o>-terms and
// s2 = |(BA)_o|^2 precomputed, the new row norm is v2 = w2 + 2*s*s1 + s^2*s2
// and the per-channel gain is g_o = (1 + mu_o) * |W_o| / |V_o|.
// Writes g into out[0..n), the ratio |W_o|/|V_o| into out[n..2n) and 1/v2 into
// out[2n..3n) (the backward reads all three).

struct Params {
    n: u32,
    s: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w2:  array<f32>;
@group(0) @binding(2) var<storage, read>       s1:  array<f32>;
@group(0) @binding(3) var<storage, read>       s2:  array<f32>;
@group(0) @binding(4) var<storage, read>       mu:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let o = gid.y * (nwg.x * 64u) + gid.x;
    if (o >= p.n) { return; }
    let v2 = max(w2[o] + 2.0 * p.s * s1[o] + p.s * p.s * s2[o], 1e-30);
    let ratio = sqrt(w2[o] / v2);
    out[o] = (1.0 + mu[o]) * ratio;
    out[p.n + o] = ratio;
    out[2u * p.n + o] = 1.0 / v2;
}
