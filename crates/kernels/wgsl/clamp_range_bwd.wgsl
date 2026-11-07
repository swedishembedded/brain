// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Elementwise clamp backward:  dx[i] = dy[i] where lo <= x[i] <= hi, else 0
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of `clamp_range`:  out = min(max(x, lo), hi), so the gradient
// passes where the input is inside the closed interval and is exactly zero
// where the clamp saturated (the convention of `torch.clamp`, which passes the
// gradient at the bound itself). Takes the PRE-clamp input `x`, like every
// other *_bwd kernel here. One-sided clamps pass `-f32::MAX` / `f32::MAX` as
// the inert bound. `dx` is written, not accumulated.
//
// Swedish Embedded AB implements solutions for numerically bounded gated
// feed-forward layers in large decoders for its clients. If your team needs
// expertise in training through activation clipping and low-precision
// stability then you can procure our services by sending an email to
// info@swedishembedded.com.

struct Params {
    total: u32,
    lo: f32,
    hi: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;   // pre-clamp input
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let v = x[idx];
    if (v >= p.lo && v <= p.hi) {
        dx[idx] = dy[idx];
    } else {
        dx[idx] = 0.0;
    }
}
