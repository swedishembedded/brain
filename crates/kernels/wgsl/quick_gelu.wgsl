// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  QuickGELU activation (OpenAI CLIP's sigmoid approximation)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// QuickGELU activation (OpenAI CLIP's sigmoid approximation):
//   out[i] = x * sigmoid(1.702 * x)
// The third member of the GELU family already here: `gelu.wgsl` is the tanh
// approximation (GPT-2), `gelu_erf.wgsl` is the exact erf form (torch's default
// F.gelu), and this is what `transformers`' `hidden_act = "quick_gelu"` selects
// — CLIP-L / OpenAI CLIP text and image towers. The three differ by up to ~1e-2
// in absolute value, so they are NOT interchangeable for parity.
//
// NOT SiLU: `silu.wgsl` is x*sigmoid(x); the 1.702 factor is the whole point.
// Elementwise over `total`. A matching derivative (`quick_gelu_bwd`) does not
// exist yet — add it with the CLIP backward, gated by `gradcheck`.

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
    out[idx] = v / (1.0 + exp(-1.702 * v));
}
