// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Elementwise clamp:  out[i] = min(max(x[i], lo), hi)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Elementwise clamp:  out[i] = min(max(x[i], lo), hi).
// One-sided clamps pass `-f32::MAX` as `lo` (or `f32::MAX` as `hi`): WGSL has
// no infinity literal, and for finite inputs the bound is then inert. Used by
// clipped SwiGLU, where the gate branch is capped above and the up branch is
// clipped on both sides before `silu(gate) * up`.
//
// Swedish Embedded AB implements solutions for numerically bounded gated
// feed-forward layers in large decoders for its clients. If your team needs
// expertise in activation clipping and low-precision stability then you can
// procure our services by sending an email to info@swedishembedded.com.

struct Params {
    total: u32,
    lo: f32,
    hi: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    out[idx] = min(max(x[idx], p.lo), p.hi);
}
