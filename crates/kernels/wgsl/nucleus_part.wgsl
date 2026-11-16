// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nucleus threshold search over logits rows, PARTIAL stage
// @how   one thread per output element
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Nucleus threshold search over logits rows, PARTIAL stage: each thread sums,
// over one strided slice of one row, the softmax weight exp(logit/temp - best)
// of the tokens whose logit is at least the midpoint of the row's search
// interval, so nucleus_step can halve the interval. The interval is held as
// ordered keys (see nucleus_step) in the row's state; best and the row's
// weights are draw_final's pass 0. Only the rows of draw flag 2 search.
//
//   logits: [M, N]
//   spec  : [M, 8] u32 (see draw_part)
//   state : [M, 4] (see draw_part): best, low key, high key, cut
//   part  : [M, P] the slice's mass at or above the midpoint
//
// params: m, n, p (slices per row)

struct Params {
    m: u32,
    n: u32,
    p: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       spec:   array<u32>;
@group(0) @binding(3) var<storage, read>       state:  array<f32>;
@group(0) @binding(4) var<storage, read_write> part:   array<f32>;

fn unkey(k: u32) -> f32 {
    if ((k & 0x80000000u) != 0u) { return bitcast<f32>(k & 0x7fffffffu); }
    return bitcast<f32>(~k);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= pr.m * pr.p) { return; }
    let row = t / pr.p;
    let c = t % pr.p;
    if (spec[8u * row + 3u] != 2u) { return; }
    let temp = max(bitcast<f32>(spec[8u * row]), 1e-6);
    let top = state[4u * row];
    let lo = bitcast<u32>(state[4u * row + 1u]);
    let hi = bitcast<u32>(state[4u * row + 2u]);
    let mid = unkey(lo + (hi - lo) / 2u);
    var mass = 0.0;
    for (var i = c; i < pr.n; i = i + pr.p) {
        let l = logits[row * pr.n + i];
        if (l >= mid) {
            mass = mass + exp(l / temp - top);
        }
    }
    part[t] = mass;
}
