// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Elementwise Hadamard product — spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Elementwise Hadamard product — spec: docs/world-models/specs/P1.glue.md §3.1/§4.1.
//   y[i] = a[i] * b[i]         one invocation per element, i in [0, n)
// Backward COMPOSES from this same kernel (no new kernel):
//   da = mul(dy, b);  db = mul(dy, a)
// GEGLU (documented decision): GEGLU(u,v) = gelu(u) ⊙ v = existing `gelu`
// into a fresh SSA buffer, then `mul` — bwd reuses gelu_bwd + mul.
//

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    y[i] = a[i] * b[i];
}
