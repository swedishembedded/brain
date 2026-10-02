// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE combine: out[row] = sum over a row's slots of weight * slot output, slot order ascending
// @how   one thread per output element, serial sum over the row's slots
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The last step of the gather-layout expert pipeline: `moe_i8_gemv_gather`
// leaves one projected row per SLOT (`slot = row * slots_per_row + s`), and a
// token's MoE output is the weighted sum of its slots - the routed experts'
// router weights and, as the last slot, the shared expert's sigmoid weight
// (`moe_router_topk.wgsl`'s `weight`). A one-thread-per-output-element
// reduction rather than a scatter-add, so no two threads ever write one
// element and no atomics are needed.
//
//   y      : [rows * slots_per_row, d] f32   per-slot expert outputs
//   weight : [rows * slots_per_row]    f32   per-slot combine weight
//   out    : [rows, d] f32
//   params : u32 [rows, d, slots_per_row]

struct Params {
    rows: u32,
    d: u32,
    slots_per_row: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       y:      array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.d) { return; }
    let row = idx / p.d;
    let c = idx % p.d;
    var acc = 0.0;
    for (var s: u32 = 0u; s < p.slots_per_row; s = s + 1u) {
        let slot = row * p.slots_per_row + s;
        acc = acc + weight[slot] * y[slot * p.d + c];
    }
    out[idx] = acc;
}
