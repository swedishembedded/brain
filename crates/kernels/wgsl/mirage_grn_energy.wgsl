// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Global response norm, part 1: per image and channel, sqrt(sum over valid rows of y^2 + eps)
// @how   one thread per (image, channel), serial over rows
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// y: [batch, rows, d], m: [batch, rows] (1 valid, 0 padding), energy: [batch, d].
// Params: batch, rows, d, eps (f32 bits). Dispatch: batch*d threads.

struct Params { batch: u32, rows: u32, d: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       y: array<f32>;
@group(0) @binding(2) var<storage, read>       m: array<f32>;
@group(0) @binding(3) var<storage, read_write> energy: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.d) { return; }
    let c = idx % p.d;
    let b = idx / p.d;
    var acc = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        let v = y[(b * p.rows + r) * p.d + c];
        acc = acc + m[b * p.rows + r] * v * v;
    }
    energy[idx] = sqrt(acc + p.eps);
}
