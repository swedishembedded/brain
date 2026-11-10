// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training Kronecker rotation, left factor: out[r, x, l] (+)= sum_y m[y sy + x sx] in[r, y, l]
// @how   one thread per output element, looping over the contraction
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Each row is viewed as [fa, fb]. The forward A^T tmp uses strides (fa, 1), the backward A dOut uses (1, fa).
// Params: total (rows * fa * fb), fa, fb, sy, sx, acc. Dispatch: total threads.

struct Params { total: u32, fa: u32, fb: u32, sy: u32, sx: u32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  a: array<f32>;
@group(0) @binding(2) var<storage, read>  m: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let l = idx % p.fb;
    let x = (idx / p.fb) % p.fa;
    let r = idx / (p.fb * p.fa);
    var acc = 0.0;
    for (var y = 0u; y < p.fa; y = y + 1u) {
        acc = acc + m[y * p.sy + x * p.sx] * a[(r * p.fa + y) * p.fb + l];
    }
    if (p.acc != 0u) {
        out[idx] = out[idx] + acc;
    } else {
        out[idx] = acc;
    }
}
