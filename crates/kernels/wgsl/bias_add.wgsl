// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Add a per-output-feature bias in place
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Add a per-output-feature bias in place:  out[m,n] += bias[n].
// Used after every biased linear (qkv, attn-out, ffn value/gate/down, u_head).
// One invocation per element (M*N).

struct Params {
    m: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out:  array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.m * p.n) { return; }
    out[idx] = out[idx] + bias[idx % p.n];
}
