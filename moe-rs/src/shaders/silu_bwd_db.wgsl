// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// SwiGLU backward, part 2 — gradient w.r.t. the up projection `b`.
//   h = SiLU(a) * b   =>   d_b = dH * SiLU(a)
// Elementwise.

struct Params { total: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:  array<f32>;   // gate_pre
@group(0) @binding(2) var<storage, read>       dh: array<f32>;
@group(0) @binding(3) var<storage, read_write> db: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    let av = a[idx];
    let silu = av / (1.0 + exp(-av));
    db[idx] = dh[idx] * silu;
}
