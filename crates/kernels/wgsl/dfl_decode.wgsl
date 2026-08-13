// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  DFL decode: for each (anchor, side) softmax over `reg_max` logits then take the expectation E = sum_i i * p_i
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// DFL decode: for each (anchor, side) softmax over `reg_max` logits then take
// the expectation E = sum_i i * p_i. Logits laid out [A, 4, reg_max]; output
// dist[A, 4]. One thread per (anchor, side) = A*4 threads. reg_max <= 16.

struct Params {
    A: u32,
    reg_max: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> dist:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;                       // (anchor*4 + side)
    let total = p.A * 4u;
    if (idx >= total) { return; }

    let base = idx * p.reg_max;           // contiguous reg_max logits

    var mx = -3.4e38;
    for (var i: u32 = 0u; i < p.reg_max; i = i + 1u) {
        mx = max(mx, logits[base + i]);
    }
    var sum = 0.0;
    for (var i: u32 = 0u; i < p.reg_max; i = i + 1u) {
        sum = sum + exp(logits[base + i] - mx);
    }
    var e = 0.0;
    for (var i: u32 = 0u; i < p.reg_max; i = i + 1u) {
        let pi = exp(logits[base + i] - mx) / sum;
        e = e + f32(i) * pi;
    }
    dist[idx] = e;
}
