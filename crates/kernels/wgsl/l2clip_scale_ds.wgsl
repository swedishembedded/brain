// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Scale gradient of l2clip_scale: the column sum of the per-element partials over the vectors of each head
// @how   one thread per (head, dim), serial sum over the vectors of that head
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// ds[h, c] += sum over vectors v with v % heads == h of gs[v, c], where gs is what l2clip_scale_dx wrote
// (dL/da * x). Accumulates into ds (pre-zeroed by the caller, like rmsnorm_dw), so gradients of several
// micro-batches add.
//
// gs: [n_vec, d]; ds: [heads, d]; params: n_vec, heads, d.
// Dispatch: heads * d invocations.

struct Params {
    n_vec: u32,
    heads: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gs: array<f32>;
@group(0) @binding(2) var<storage, read_write> ds: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.heads * p.d) { return; }
    let h = idx / p.d;
    let c = idx % p.d;
    var acc = 0.0;
    for (var v = h; v < p.n_vec; v = v + p.heads) {
        acc = acc + gs[v * p.d + c];
    }
    ds[idx] = ds[idx] + acc;
}
