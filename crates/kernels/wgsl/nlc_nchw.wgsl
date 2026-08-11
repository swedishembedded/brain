// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Layout permutation NLC [N, L=H*W, C] -> NCHW (gather) — exact inverse AND adjoint of nchw_nlc
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Layout permutation NLC [N, L=H*W, C] -> NCHW (gather) — exact inverse AND
// adjoint of nchw_nlc. total = N*c*hw. One thread per OUTPUT (NCHW) element:
//   n = idx/(c*hw); r0 = idx % (c*hw); ch = r0/hw; l = r0 % hw   (u32 ops)
//   y[idx] = x[(n*hw + l)*c + ch]
// Backward: dx = nchw_nlc(dy). Pure copy: output bits are exact images of
// input bits.
//

struct Params {
    total: u32,
    c: u32,
    hw: u32,
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
    let n = idx / (p.c * p.hw);
    let r0 = idx % (p.c * p.hw);
    let ch = r0 / p.hw;
    let l = r0 % p.hw;
    y[idx] = x[(n * p.hw + l) * p.c + ch];
}
