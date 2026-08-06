// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of glu.wgsl (F.glu over the middle dim)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of glu.wgsl (F.glu over the middle dim). With a = x[o,c,i],
// b = x[o,d+c,i], s = sigmoid(b), out = a*s:
//   dx[o,c,i]   = dy * s
//   dx[o,d+c,i] = dy * a * s*(1-s)
// One thread per output element (o,c,i) writes both halves of dx (disjoint,
// no races). Inputs dy [outer,d,inner], x [outer,2d,inner]; output dx [outer,2d,inner].

struct Params {
    outer: u32,
    d: u32,
    inner: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.outer * p.d * p.inner;
    if (idx >= total) { return; }
    let di = p.d * p.inner;
    let o = idx / di;
    let rem = idx % di;
    let c = rem / p.inner;
    let i = rem % p.inner;
    let base = o * (2u * p.d) * p.inner + i;
    let ia = base + c * p.inner;
    let ib = base + (p.d + c) * p.inner;
    let a = x[ia];
    let s = 1.0 / (1.0 + exp(-x[ib]));
    let g = dy[idx];
    dx[ia] = g * s;
    dx[ib] = g * a * s * (1.0 - s);
}
