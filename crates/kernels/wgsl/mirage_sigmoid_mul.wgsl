// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sigmoid gate on an attention output: out = att * sigmoid(gate)
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The gate may sit inside a wider fused projection: element (r, c) is gate[r * gate_stride + gate_off + c].
// Params: total, w, gate_stride, gate_off. Dispatch: total threads.

struct Params { total: u32, w: u32, gate_stride: u32, gate_off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       att: array<f32>;
@group(0) @binding(2) var<storage, read>       gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    out[idx] = att[idx] / (1.0 + exp(-gate[(idx / p.w) * p.gate_stride + p.gate_off + idx % p.w]));
}
