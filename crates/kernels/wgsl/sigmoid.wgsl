// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Sigmoid activation:  y = 1 / (1 + exp(-x)).
// Elementwise. Matching derivative in sigmoid_bwd.wgsl.
//
// brain had sigmoid only FUSED inside other kernels (silu, bce_logits_grad,
// router_gate_sigmoid); ZipDepth needs it standalone as a gate in three places
// (ChannelAttention/SE, StripPoolingAttention, and the NPU upsampler's blend).

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
    out[idx] = 1.0 / (1.0 + exp(-x[idx]));
}
