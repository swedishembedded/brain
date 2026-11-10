// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Global response norm, part 2: scale[c] = gamma[c] * energy[c] / (mean(energy) + eps) + 1 per image
// @how   one thread per (image, channel), serial mean over channels
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: batch, d, eps (f32 bits). Dispatch: batch*d threads.

struct Params { batch: u32, d: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       energy: array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> scale: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.d) { return; }
    let c = idx % p.d;
    let b = idx / p.d;
    var s = 0.0;
    for (var i = 0u; i < p.d; i = i + 1u) { s = s + energy[b * p.d + i]; }
    scale[idx] = gamma[c] * energy[idx] / (s / f32(p.d) + p.eps) + 1.0;
}
