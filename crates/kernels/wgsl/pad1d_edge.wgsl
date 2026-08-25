// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Asymmetric REPLICATE (edge-clamp) pad on NCL (gather) - spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Asymmetric replicate pad on NCL. The sibling of pad2d, which zero-fills:
// this one repeats the EDGE sample instead, which is what an antialiasing
// resampler's boundary needs (a zero-filled edge is a step discontinuity the
// lowpass filter then rings on). Params carry the UNPADDED length l and the
// pad amounts left/right (u32, each may be 0); the padded length is derived,
// lp = l + left + right. Batch and channel are combined (row index), so an
// [N, C, L] tensor is dispatched as N*C rows exactly like pad2d combines its
// own batch and channel. One thread per OUTPUT element idx < total
// (total = rows*lp):
//   row = idx / lp; lo = idx % lp                                (u32 ops)
//   si  = clamp(lo, left, left + l - 1) - left
//   y[idx] = x[row * l + si]
// The clamp is what makes this replicate rather than zero pad: an output
// position left of the interior reads source 0, one right of it reads source
// l-1, and an interior one reads lo - left. Pure movement, no arithmetic:
// every output bit is an exact image of an input bit, so the interior is
// bitwise identical to crop2d's copy contract and the two pad regions hold
// exact copies of the first and last source sample.
//
// NOT a linear-injection adjoint pair with any crop the way pad2d/crop2d are:
// replicate padding sums into the edge samples on the way back, so its
// transpose is a crop PLUS an edge fold, not a bare crop. No backward is
// declared here because every dispatch site is inference-only.
//

struct Params {
    total: u32,
    l: u32,
    left: u32,
    right: u32,
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
    let lp = p.l + p.left + p.right;
    let row = idx / lp;
    let lo = idx % lp;
    // max() then min() rather than a single clamp() call so the u32 subtraction
    // below can never underflow: lo is forced to at least `left` first.
    let hi = p.left + p.l - 1u;
    var s = lo;
    if (s < p.left) { s = p.left; }
    if (s > hi) { s = hi; }
    y[idx] = x[row * p.l + (s - p.left)];
}
