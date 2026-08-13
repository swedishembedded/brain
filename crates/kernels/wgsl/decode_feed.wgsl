// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Feed the greedy head's output back as the next decode step's input, on the device (A4)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Feed the greedy head's output back as the next decode step's input, on the
// device (A4): tok[i] = u32(argmax[i]), and record the token into the window
// history at row `s` so the host reads the whole window's tokens ONCE at the
// end instead of once per step. One invocation per row.

struct Params {
    bsz: u32,
    /// Window row to record into (`hist[s * bsz + i]`).
    s: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       amax: array<f32>; // [bsz] greedy indices
@group(0) @binding(2) var<storage, read_write> tok:  array<u32>; // [bsz] next inputs
@group(0) @binding(3) var<storage, read_write> hist: array<f32>; // [window, bsz]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.bsz) { return; }
    let t = amax[i];
    tok[i] = u32(t);
    hist[p.s * p.bsz + i] = t;
}
