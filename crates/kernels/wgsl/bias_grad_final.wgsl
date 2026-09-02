// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bias gradient, STAGE 2 of 2 - fold the partials and ACCUMULATE
// @how   one thread per output element, serial fold over P partials
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Bias gradient, STAGE 2 of 2 - fold the partials `bias_grad_part` wrote and
// ACCUMULATE into the caller's gradient buffer, mirroring `gn_dsum2`/
// `gn_dgb2`'s own fold stage:
//
//   dbias[col] += sum_chunk part[chunk * n + col]
//
// `+=`, not `=` - matches `bias_grad.wgsl`'s own accumulate contract (a
// parameter gradient accumulates over the whole step and is zeroed exactly
// once by the model's `zero_grads`). One invocation per column; `P` is small
// by construction (`bias_grad_part`'s own chunk count), so this fold is cheap
// relative to the reduction it replaces, and its access is coalesced the same
// way `bias_grad_part`'s was - adjacent threads read adjacent columns at
// every step.

struct Params {
    m: u32, // rows (unused here, kept so both stages share one uniform layout)
    n: u32, // columns (the bias width)
    P: u32, // chunks per column
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dbias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let col = gidx;
    if (col >= p.n) { return; }

    var acc = 0.0;
    for (var c: u32 = 0u; c < p.P; c = c + 1u) {
        acc = acc + part[c * p.n + col];
    }
    dbias[col] = dbias[col] + acc;
}
