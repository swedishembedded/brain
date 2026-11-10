// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated linear unit with a SiLU gate from one fused projection: z = uv[:, :d] * silu(uv[:, d:])
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// uv: [rows, 2d], z: [rows, d]. Params: rows, d. Dispatch: rows*d threads.

struct Params { rows: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       uv: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.d) { return; }
    let r = idx / p.d;
    let c = idx % p.d;
    let g = uv[r * 2u * p.d + p.d + c];
    z[idx] = uv[r * 2u * p.d + c] * (g / (1.0 + exp(-g)));
}
