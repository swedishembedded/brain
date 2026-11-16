// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nucleus cut among the tokens at the threshold logit
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Nucleus cut among the tokens at the threshold logit: one thread per row of
// draw flag 2 sums the weight above the threshold tau (nucleus_step) and walks
// the row's tokens at tau in column order until the nucleus holds top_p of the
// row's mass, as the host sampler's best-first prefix takes equal logits
// lowest column first; the last one taken is the cut column draw_part's pass
// 2 keeps tokens at tau up to. Serial over the row: the reference the native
// draw_rows_f32 is checked against, not a fast path.
//
//   logits: [M, N]
//   spec  : [M, 8] u32 (see draw_part)
//   out   : [M, 4] (see draw_final)
//   state : [M, 4] (see draw_part): writes the cut at 4r+3
//
// params: m, n

struct Params {
    m: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       spec:   array<u32>;
@group(0) @binding(3) var<storage, read>       out:    array<f32>;
@group(0) @binding(4) var<storage, read_write> state:  array<f32>;

fn unkey(k: u32) -> f32 {
    if ((k & 0x80000000u) != 0u) { return bitcast<f32>(k & 0x7fffffffu); }
    return bitcast<f32>(~k);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= pr.m) { return; }
    if (spec[8u * row + 3u] != 2u) { return; }
    let temp = max(bitcast<f32>(spec[8u * row]), 1e-6);
    let target_mass = bitcast<f32>(spec[8u * row + 4u]) * out[4u * row + 3u];
    let top = state[4u * row];
    let tau = unkey(bitcast<u32>(state[4u * row + 1u]));
    var above = 0.0;
    for (var i = 0u; i < pr.n; i = i + 1u) {
        let l = logits[row * pr.n + i];
        if (l > tau) { above = above + exp(l / temp - top); }
    }
    let w = exp(tau / temp - top);
    var acc = above;
    var cut = 0u;
    for (var i = 0u; i < pr.n; i = i + 1u) {
        if (logits[row * pr.n + i] == tau) {
            cut = i;
            acc = acc + w;
            if (acc >= target_mass) { break; }
        }
    }
    state[4u * row + 3u] = bitcast<f32>(cut);
}
