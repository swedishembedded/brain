// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated Linear Unit over the middle dim, matching torch `F.glu(x, dim=1)`
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Gated Linear Unit over the middle dim, matching torch `F.glu(x, dim=1)`:
//   x   : [outer, 2*d, inner]  row-major
//   out : [outer,   d, inner]  out[o,c,i] = x[o,c,i] * sigmoid(x[o,d+c,i])
// The Conformer convolution module applies this after pointwise_conv1 (channels
// 2C → C). Elementwise over `outer*d*inner`; each thread reads the two halves.

struct Params {
    outer: u32,
    d: u32,
    inner: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

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
    let a = x[base + c * p.inner];
    let b = x[base + (p.d + c) * p.inner];
    out[idx] = a * (1.0 / (1.0 + exp(-b)));
}
