// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Asymmetric crop on NCHW (gather) — the exact adjoint of pad2d. Spec:
// docs/world-models/specs/P1.glue.md §3.7/§4.7. Params layout is IDENTICAL
// to pad2d: h, w are the CROPPED (output) dims, l/r/t/b the offsets; the
// INPUT is the padded tensor [NC, h+t+b, w+l+r]. One thread per OUTPUT
// element idx < total (total = NC*h*w):
//   p = idx/(h*w); r0 = idx % (h*w); ho = r0/w; wo = r0 % w      (u32 ops)
//   y[idx] = x[p*hp*wp + (ho+t)*wp + (wo+l)]     hp = h+t+b, wp = w+l+r
// crop2d = pad2d^T (discards exactly what pad2d zero-fills). Backward:
// dx = pad2d(dy) with the same offsets. Pure copy: output bits are exact
// images of input bits.
//

struct Params {
    total: u32,
    h: u32,
    w: u32,
    l: u32,
    r: u32,
    t: u32,
    b: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let hp = p.h + p.t + p.b;
    let wp = p.w + p.l + p.r;
    let img = idx / (p.h * p.w);
    let r0 = idx % (p.h * p.w);
    let ho = r0 / p.w;
    let wo = r0 % p.w;
    y[idx] = x[img * hp * wp + (ho + p.t) * wp + (wo + p.l)];
}
