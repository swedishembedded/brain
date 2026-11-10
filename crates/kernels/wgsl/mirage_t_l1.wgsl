// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training weighted L1 per element: forward term w |x - t| or subgradient w sign(x - t) g / norm
// @how   one thread per element; accumulates in the backward mode
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Mode 0 writes the term, mode 1 accumulates the gradient (zero at equality). Params: n, norm (f32 bits), mode.
// Dispatch: n threads.

struct Params { n: u32, norm: f32, mode: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<f32>;
@group(0) @binding(3) var<storage, read>       weight: array<f32>;
@group(0) @binding(4) var<storage, read>       g: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    let diff = x[idx] - tgt[idx];
    if (p.mode == 0u) {
        out[idx] = weight[idx] * abs(diff);
    } else {
        var sg = 0.0;
        if (diff > 0.0) { sg = 1.0; }
        if (diff < 0.0) { sg = -1.0; }
        out[idx] = out[idx] + weight[idx] * sg * g[0] / p.norm;
    }
}
