// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Feed a keyed draw's tokens back as the next decode step's input, on the device
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Feed a keyed draw's tokens back as the next decode step's input, on the
// device: tok[i] = the token draw_final (or draw_rows_f32) drew for row i, and
// the token and its log-probability recorded into the window history at step
// `s`, so a sampled decode window reads its tokens back once at the end, as
// the greedy window does (decode_feed). One invocation per row.
//
//   out    : [bsz, 4] the draw's (token as f32, log-probability, best, mass)
//   tok    : [bsz] u32 next inputs
//   hist   : [window, bsz] tokens (as f32)
//   lphist : [window, bsz] log-probabilities
//
// params: bsz, s

struct Params {
    bsz: u32,
    s: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       drawn:  array<f32>;
@group(0) @binding(2) var<storage, read_write> tok:    array<u32>;
@group(0) @binding(3) var<storage, read_write> hist:   array<f32>;
@group(0) @binding(4) var<storage, read_write> lphist: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.bsz) { return; }
    let t = drawn[4u * i];
    tok[i] = u32(t);
    hist[p.s * p.bsz + i] = t;
    lphist[p.s * p.bsz + i] = drawn[4u * i + 1u];
}
