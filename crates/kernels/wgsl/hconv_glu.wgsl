// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated product of the HConvGLU mixer: gelu_tanh(gate + bias) * value, the two branches read from one fused projection
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// HConvGLU keeps its gate and value branches in ONE projection output
// `ab: [rows, 2n]` (gate in columns 0..n, value in columns n..2n), because both
// branches read the same transformed input and a single GEMM streams that input
// once. This kernel forms the gated product that feeds the output projection:
//   z[r,c] = gelu_tanh(ab[r, c] + bias[c]) * ab[r, n + c]
// The bias belongs to the gate branch only. gelu_tanh is the tanh approximation
// (0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))), identical to gelu.wgsl.
// Params: rows, n. Dispatch: rows * n threads.

struct Params {
    rows: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ab:   array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> z:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.n) { return; }
    let r = idx / p.n;
    let c = idx % p.n;
    let a = ab[r * 2u * p.n + c] + bias[c];
    let v = ab[r * 2u * p.n + p.n + c];
    let k = 0.7978845608028654;
    let g = 0.5 * a * (1.0 + tanh(k * (a + 0.044715 * a * a * a)));
    z[idx] = g * v;
}
