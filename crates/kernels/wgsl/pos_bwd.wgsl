// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Positional-embedding backward (scatter)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Positional-embedding backward (scatter):
//   dpos[i, c] += sum_b d_x[(b*T + i)*D + c]      for i in 0..T
// One invocation per (i, c); loops batch b to avoid atomics. The dpos buffer is
// block_size*D — only the first T rows are written, so the whole buffer must be
// zeroed in the backward clears.

struct Params {
    b: u32,
    t: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_x:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dpos: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.t * p.d_model) { return; }
    let i = idx / p.d_model;
    let c = idx % p.d_model;
    var acc = 0.0;
    for (var bb: u32 = 0u; bb < p.b; bb = bb + 1u) {
        acc = acc + d_x[(bb * p.t + i) * p.d_model + c];
    }
    dpos[i * p.d_model + c] = dpos[i * p.d_model + c] + acc;
}
