// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  NCHW map into a window of rows of a larger NLC tensor: y[n, base + l, c] = x[n, c, l]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `nchw_nlc` writing into a WINDOW of a larger row tensor instead of a tensor
// of its own:
//   x : [N, C, HW]                     (an NCHW map, H*W flattened)
//   y : [N, A, C], rows [base, base+HW) of each image written, the rest untouched
//   y[(n*A + base + l)*C + ch] = x[(n*C + ch)*HW + l]
//
// A detector's head emits one NCHW map per pyramid scale, and its loss reads
// all of them as ONE [N, A, C] tensor of anchor rows, scale after scale. One
// dispatch per scale places that scale's map in its window, on the device, so
// the concatenated tensor never has to be assembled on the host. Pure data
// movement: every value is copied, none is computed. One invocation per OUTPUT
// element, so consecutive threads write consecutive floats.

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
    if (idx >= p.n * p.hw * p.c) { return; }
    let n = idx / (p.hw * p.c);
    let r = idx % (p.hw * p.c);
    let l = r / p.c;
    let ch = r % p.c;
    y[(n * p.a + p.base + l) * p.c + ch] = x[(n * p.c + ch) * p.hw + l];
}
