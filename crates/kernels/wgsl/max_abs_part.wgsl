// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Partial per-tensor max/x/ for dynamic int8 quantization
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
//
// Partial per-tensor max|x| for dynamic int8 quantization. P threads each scan a
// strided stripe of the [total] input and write their local max into part[P];
// max_abs_final then reduces part[P] to the scale. Two-pass (no atomics).

struct Params { total: u32, p: u32 };

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= pr.p) { return; }
    var m = 0.0;
    var i = t;
    loop {
        if (i >= pr.total) { break; }
        m = max(m, abs(x[i]));
        i = i + pr.p;
    }
    part[t] = m;
}
