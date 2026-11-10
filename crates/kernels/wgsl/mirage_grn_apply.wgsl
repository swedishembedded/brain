// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Global response norm, part 3 with the block residual: x[r, c] += m[r] * (y[r, c] * scale[c] + beta[c])
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// x, y: [batch, rows, d]; m: [batch, rows]; scale: [batch, d]; beta: [d].
// Params: batch, rows, d. Dispatch: batch*rows*d threads.

struct Params { batch: u32, rows: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read>       y: array<f32>;
@group(0) @binding(3) var<storage, read>       m: array<f32>;
@group(0) @binding(4) var<storage, read>       scale: array<f32>;
@group(0) @binding(5) var<storage, read>       beta: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.rows * p.d) { return; }
    let c = idx % p.d;
    let r = idx / p.d;
    let b = r / p.rows;
    x[idx] = x[idx] + m[r] * (y[idx] * scale[b * p.d + c] + beta[c]);
}
