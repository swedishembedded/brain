// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  SwiGLU backward, part 2 - gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// SwiGLU backward, part 2 — gradient w.r.t. the up projection `b`.
//   h = SiLU(a) * b   =>   d_b = dH * SiLU(a)
// Elementwise.

struct Params { total: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:  array<f32>;   // gate_pre
@group(0) @binding(2) var<storage, read>       dh: array<f32>;
@group(0) @binding(3) var<storage, read_write> db: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let av = a[idx];
    let silu = av / (1.0 + exp(-av));
    db[idx] = dh[idx] * silu;
}
