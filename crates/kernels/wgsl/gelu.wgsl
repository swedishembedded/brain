// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GELU activation (tanh approximation, as used by GPT-2-style MLPs)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// GELU activation (tanh approximation, as used by GPT-2-style MLPs):
//   out[i] = 0.5 * x * (1 + tanh( k * (x + 0.044715 * x^3) )),  k = sqrt(2/pi)
// Elementwise over seq_len * d_ff. The matching derivative is in gelu_bwd.wgsl.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let v = x[idx];
    let k = 0.7978845608028654;          // sqrt(2/pi)
    let inner = k * (v + 0.044715 * v * v * v);
    out[idx] = 0.5 * v * (1.0 + tanh(inner));
}
