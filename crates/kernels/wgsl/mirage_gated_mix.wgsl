// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated depthwise mixer product g = silu(g + b_gate) * (v + b_value) in place (Mirage body mixer)
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: rows, d. Dispatch: rows*d threads.

struct Params { rows: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> g: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       bg: array<f32>;
@group(0) @binding(4) var<storage, read>       bv: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.d) { return; }
    let c = idx % p.d;
    let a = g[idx] + bg[c];
    g[idx] = a / (1.0 + exp(-a)) * (v[idx] + bv[c]);
}
