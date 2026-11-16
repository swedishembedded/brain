// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nucleus threshold search over logits rows, STEP stage
// @how   one thread per output element
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Nucleus threshold search over logits rows, STEP stage: one thread per row
// folds nucleus_part's slices into the mass at or above the midpoint of the
// row's search interval and keeps the half that holds the threshold: the
// largest logit tau whose tokens at or above it carry top_p of the row's mass.
// The interval is over ordered keys - a float's bits mapped so that integer
// order is float order (a negative float's bits inverted, a positive one's
// sign bit set) - so 32 steps from the whole key range end on adjacent keys:
// low is tau, whose tokens at or above it carry the nucleus, and nothing above
// it does.
//
//   spec  : [M, 8] u32 (see draw_part)
//   out   : [M, 4] (see draw_final): the row's mass at 4r+3
//   part  : [M, P]
//   state : [M, 4] (see draw_part)
//
// params: m, p (slices per row)

struct Params {
    m: u32,
    p: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       spec:  array<u32>;
@group(0) @binding(2) var<storage, read>       out:   array<f32>;
@group(0) @binding(3) var<storage, read>       part:  array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= pr.m) { return; }
    if (spec[8u * row + 3u] != 2u) { return; }
    let lo = bitcast<u32>(state[4u * row + 1u]);
    let hi = bitcast<u32>(state[4u * row + 2u]);
    if (hi - lo <= 1u) { return; }
    var mass = 0.0;
    for (var c = 0u; c < pr.p; c = c + 1u) {
        mass = mass + part[row * pr.p + c];
    }
    let mid = lo + (hi - lo) / 2u;
    if (mass >= bitcast<f32>(spec[8u * row + 4u]) * out[4u * row + 3u]) {
        state[4u * row + 1u] = bitcast<f32>(mid);
    } else {
        state[4u * row + 2u] = bitcast<f32>(mid);
    }
}
