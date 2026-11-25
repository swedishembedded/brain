// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Replicate (edge) pad of a padded NCL batch at each item's own length (gather)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `pad1d_edge.wgsl` for a batch whose items have different valid lengths:
// item `b` of the `[n, c, l]` source is valid on `[0, lens[b])`, and its row of
// the `[n, c, lp]` output is that row with `left` copies of its first sample
// in front and its LAST VALID sample repeated to the end - what a replicate-
// padded causal conv over the unpadded item reads, whatever the batch padding
// holds:
//
//   row = idx / lp; j = idx % lp; b = row / c
//   y[idx] = x[row*l + clamp(j - left, 0, lens[b] - 1)]
//
// Params: total = n*c*lp, c, l, left, lp. `lens[b] >= 1`. Dispatch: total
// invocations. Pure movement: every output bit is a copy of an input bit.

struct Params {
    total: u32,
    c: u32,
    l: u32,
    left: u32,
    lp: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> y:    array<f32>;
@group(0) @binding(3) var<storage, read>       lens: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let row = idx / p.lp;
    let j = idx % p.lp;
    let last = lens[row / p.c] - 1u;
    var s = 0u;
    if (j > p.left) { s = j - p.left; }
    if (s > last) { s = last; }
    y[idx] = x[row * p.l + s];
}
