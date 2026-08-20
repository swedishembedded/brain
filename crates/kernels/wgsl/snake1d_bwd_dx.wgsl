// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Snake activation backward - input gradient
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Input gradient of `snake1d.wgsl`'s forward
// (`y = x + (a+eps)^-1 * sin(a*x)^2`, `a = alpha[c]`):
//   dy/dx = 1 + (a/(a+eps)) * sin(2*a*x)      (via 2*sin(t)*cos(t) = sin(2t))
//   dx = dy * dy/dx
// Layout matches the forward: [rows, C, inner], c = (idx / inner) % C.

struct Params {
    total: u32,
    c: u32,
    inner: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:    array<f32>;
@group(0) @binding(2) var<storage, read>       x:     array<f32>;
@group(0) @binding(3) var<storage, read>       alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let c = (idx / p.inner) % p.c;
    let a = alpha[c];
    let dydx = 1.0 + (a / (a + p.eps)) * sin(2.0 * a * x[idx]);
    dx[idx] = dy[idx] * dydx;
}
