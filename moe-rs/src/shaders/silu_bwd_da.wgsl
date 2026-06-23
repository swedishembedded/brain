// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// SwiGLU backward, part 1 — gradient w.r.t. the gate pre-activation `a`.
//   h = SiLU(a) * b,  SiLU(a) = a*sigmoid(a)
//   d_a = dH * b * d/da[SiLU(a)],   d/da SiLU = sig + a*sig*(1-sig)
// Elementwise.

struct Params { total: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;   // gate_pre
@group(0) @binding(2) var<storage, read>       b:   array<f32>;   // up
@group(0) @binding(3) var<storage, read>       dh:  array<f32>;
@group(0) @binding(4) var<storage, read_write> da:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    let av = a[idx];
    let sig = 1.0 / (1.0 + exp(-av));
    let dsilu = sig + av * sig * (1.0 - sig);
    da[idx] = dh[idx] * b[idx] * dsilu;
}
