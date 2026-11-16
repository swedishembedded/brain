// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm gain-weight gradient, stage 2: sum the partials into dW
// @how   one thread per channel, serial over the chunks
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The second stage of the two-stage gain gradient (see rmsnorm_dw_part):
//   dW[c] += sum_j part[j, c]
// Accumulates into the (pre-zeroed) weight-grad buffer, as rmsnorm_dw does.

struct Params {
    d_model: u32,
    chunks: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> dw:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.d_model) { return; }
    var acc = 0.0;
    for (var j: u32 = 0u; j < p.chunks; j = j + 1u) {
        acc = acc + part[j * p.d_model + c];
    }
    dw[c] = dw[c] + acc;
}
