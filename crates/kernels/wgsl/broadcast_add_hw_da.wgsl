// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of broadcast_add_hw wrt ONE strip
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Gradient of broadcast_add_hw wrt ONE strip: sum the full-map gradient over the
// broadcast axis.
//   axis = 0: da[n,c,h] = sum_w dy[n,c,h,w]   (the [N,C,H,1] strip)
//   axis = 1: db[n,c,w] = sum_h dy[n,c,h,w]   (the [N,C,1,W] strip)
//
// The adjoint of a broadcast is a sum over the broadcast axis — which is exactly
// what strip_pool does WITHOUT its 1/len mean factor. One invocation per output
// strip element, reducing serially: no atomics, no barrier.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    axis: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> ds: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let keep = select(p.W, p.H, p.axis == 0u);
    let total = p.N * p.C * keep;
    if (idx >= total) { return; }
    let k  = idx % keep;
    let nc = idx / keep;
    let base = nc * p.H;
    var acc = 0.0;
    if (p.axis == 0u) {
        for (var wi: u32 = 0u; wi < p.W; wi = wi + 1u) { acc = acc + dy[(base + k) * p.W + wi]; }
    } else {
        for (var hi: u32 = 0u; hi < p.H; hi = hi + 1u) { acc = acc + dy[(base + hi) * p.W + k]; }
    }
    ds[idx] = acc;
}
