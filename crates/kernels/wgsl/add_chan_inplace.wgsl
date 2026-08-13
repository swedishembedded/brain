// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  In-place per-channel bias over NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// In-place per-channel bias over NCHW: out[n,c,hw] += v[c]. Single
// read_write binding (no input aliasing — wgpu usage-scope safe), the
// conv-bias companion of add_chan_bcast.

struct Params {
    total: u32, // N*C*HW
    c: u32,
    hw: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read>       v:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    out[idx] = out[idx] + v[(idx / p.hw) % p.c];
}
