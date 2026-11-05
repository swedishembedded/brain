// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  What int8 activation quantization leaves out: r = x - clamp(round(x / sx), -127, 127) * sx, per (row, 32-group) scale
// @how   one thread per element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The second half of a two-level int8 activation: `quant_pack` rounds x onto
// its group's int8 grid, and this writes the rounding error that grid drops.
// Quantized in turn (`max_abs_row` + `quant_pack`) and multiplied by the same
// int8 weight, it restores the activation to ~1/(127*254) of its group scale -
// the activation side of a W8A8 linear then loses essentially nothing next to
// its weights. The rounding here must be quant_pack's exactly, or the residual
// would describe a different first level than the one the GEMM reads.

struct Params { m: u32, k: u32 };  // k = full K (multiple of 32)

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       sx: array<f32>;
@group(0) @binding(3) var<storage, read_write> r:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.m * p.k) { return; }
    let m = idx / p.k;
    let s = sx[m * (p.k / 32u) + (idx % p.k) / 32u];
    let inv = 1.0 / s;
    let q = clamp(round(x[idx] * inv), -127.0, 127.0);
    r[idx] = x[idx] - q * s;
}
