// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Residual splice (backward)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Residual splice (backward): route the residual-stream gradient at the spliced
// region into the compact image-embedding gradient, and ZERO it in the residual
// grad so the downstream embedding backward (emb_bwd) does not scatter it into
// the placeholder token's `tok.weight` row (the placeholder embedding was
// overwritten in the forward and never influenced the loss, so its true grad is
// zero — leaving it non-zero would fail the gradient check).
//   d_src[i] = d_dst[base + i];  d_dst[base + i] = 0   for i in 0..n.
// Mirrors splice.wgsl. One invocation per element.

struct Params {
    n:    u32,
    base: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> d_dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_src: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    d_src[idx] = d_dst[p.base + idx];
    d_dst[p.base + idx] = 0.0;
}
