// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Moondream prefix-LM attention mask, added into the scores before softmax
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Moondream prefix-LM attention mask, added into the scores before softmax:
//   allow(i,j) = (i < P && j < P) || (j <= i)
// i.e. the first `prefix` positions (bos + image tokens) attend BIDIRECTIONALLY
// to each other, everything else is causal. Disallowed pairs get a large negative
// so softmax zeroes them. scores: [B*H*T*T] = ((b*H+h)*T+i)*T+j. One invocation per
// (b,h,i,j). No backward: it's a constant mask, and the softmax backward yields
// ~0 gradient on the masked (≈0-probability) entries.

struct Params {
    bsz: u32,
    heads: u32,
    tcols: u32,  // T
    prefix: u32, // P
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let t = p.tcols;
    if (idx >= p.bsz * p.heads * t * t) { return; }
    let j = idx % t;
    let i = (idx / t) % t;
    let prefix_pair = (i < p.prefix) && (j < p.prefix);
    let causal = j <= i;
    if (!(prefix_pair || causal)) {
        scores[idx] = scores[idx] - 1.0e30;
    }
}
