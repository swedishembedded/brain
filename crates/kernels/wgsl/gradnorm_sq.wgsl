// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-parameter sum of squares of its gradient, written to norms[slot]
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Per-parameter sum of squares of its gradient, written to norms[slot].
// One single-thread dispatch per parameter buffer (numel small enough); the
// host reads the tiny `norms` array, sums it, and computes the global clip
// coefficient on the CPU (mirrors how forward() reduces ce_buf on the host).

struct Params {
    numel: u32,
    slot: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       grad:  array<f32>;
@group(0) @binding(2) var<storage, read_write> norms: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx != 0u) { return; }
    var acc = 0.0;
    for (var i: u32 = 0u; i < p.numel; i = i + 1u) {
        let g = grad[i];
        acc = acc + g * g;
    }
    norms[p.slot] = acc;
}
