// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training Kronecker rotation, right factor: out[q, x] (+)= sum_y in[q, y] m[y sy + x sx]
// @how   one thread per output element, looping over the contraction
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Rows q of width ny times a small matrix addressed through the strides (sy, sx): the forward Z B uses (fb, 1), the
// backward dZ = dtmp B^T uses (1, fb) with acc 1. Params: total (q * nx), ny, nx, sy, sx, acc. Dispatch: total threads.

struct Params { total: u32, ny: u32, nx: u32, sy: u32, sx: u32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  a: array<f32>;
@group(0) @binding(2) var<storage, read>  m: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let x = idx % p.nx;
    let q = idx / p.nx;
    var acc = 0.0;
    for (var y = 0u; y < p.ny; y = y + 1u) {
        acc = acc + a[q * p.ny + y] * m[y * p.sy + x * p.sx];
    }
    if (p.acc != 0u) {
        out[idx] = out[idx] + acc;
    } else {
        out[idx] = acc;
    }
}
