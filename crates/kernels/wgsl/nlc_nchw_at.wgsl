// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  A window of rows of a larger NLC tensor back to an NCHW map: y[n, c, l] = x[n, base + l, c]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The inverse of `nchw_nlc_at`, reading a WINDOW of a larger row tensor:
//   x : [N, A, C], rows [base, base+HW) of each image read
//   y : [N, C, HW]                     (an NCHW map, H*W flattened)
//   y[(n*C + ch)*HW + l] = x[(n*A + base + l)*C + ch]
//
// A detector's loss produces its gradient as ONE [N, A, C] tensor of anchor
// rows; each pyramid scale's head takes its own window back as an NCHW map.
// Pure data movement. One invocation per OUTPUT element, so consecutive
// threads write consecutive floats.

struct Params {
    n: u32,
    c: u32,
    hw: u32,
    a: u32,
    base: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n * p.c * p.hw) { return; }
    let n = idx / (p.c * p.hw);
    let r = idx % (p.c * p.hw);
    let ch = r / p.hw;
    let l = r % p.hw;
    y[idx] = x[(n * p.a + p.base + l) * p.c + ch];
}
