// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Layout permutation NCHW -> NLC [N, L=H*W, C] (gather) — spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Layout permutation NCHW -> NLC [N, L=H*W, C] (gather) — spec:
// docs/world-models/specs/P1.glue.md §3.8/§4.8. total = N*c*hw. One thread
// per OUTPUT (NLC) element:
//   n = idx/(hw*c); r0 = idx % (hw*c); l = r0/c; ch = r0 % c    (u32 ops)
//   y[idx] = x[(n*c + ch)*hw + l]
// Permutation matrix: inverse == transpose == adjoint == nlc_nchw (same
// params), which is also its backward: dx = nlc_nchw(dy). Required:
// nlc_nchw(nchw_nlc(x)) == x BITWISE and <nchw_nlc(x), y> == <x, nlc_nchw(y)>.
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
    let n = idx / (p.hw * p.c);
    let r0 = idx % (p.hw * p.c);
    let l = r0 / p.c;
    let ch = r0 % p.c;
    y[idx] = x[(n * p.c + ch) * p.hw + l];
}
