// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  SwiGLU and group-wise int8 activation quantisation in one pass: out = quant(SiLU(gate) * up), one workgroup per row
// @how   128-thread workgroup, a lane per 32-element scale group, two passes over each group
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// `silu_mul.wgsl` + `max_abs_row.wgsl` + `quant_pack.wgsl` fused for the one
// place a sparse-MoE expert needs all three back to back: the activation
// between its gate/up and down projections. Besides saving two dispatches per
// expert layer, it replaces `max_abs_row`'s ONE THREAD per row (a serial walk
// of the whole row) with a workgroup per row, lane `t` owning the scale groups
// `t, t + 128, ...`.
//
// The result is BIT-IDENTICAL to the three kernels run in sequence, because it
// evaluates the same expressions: `h = (x / (1 + exp(-x))) * u`, a max over
// `|h|` of each 32-element group (order-free), `sh = max(max|h|, 1e-8) / 127`,
// and `q = clamp(round(h * (1 / sh)), -127, 127)` packed four per word
// little-endian along the row.
//
//   gate : [rows, k] f32   up : [rows, k] f32
//   hq   : [rows, k/4] u32 sh : [rows, k/32] f32
//   params : u32 [rows, k]   k a multiple of 32

struct Params {
    rows: u32,
    k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<storage, read>       up:   array<f32>;
@group(0) @binding(3) var<storage, read_write> hq:   array<u32>;
@group(0) @binding(4) var<storage, read_write> sh:   array<f32>;

const WG: u32 = 128u;

fn swiglu(g: f32, u: f32) -> f32 {
    return (g / (1.0 + exp(-g))) * u;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.rows) { return; }
    let groups = p.k / 32u;
    let base = row * p.k;
    for (var g = li.x; g < groups; g = g + WG) {
        let c0 = base + g * 32u;
        var a = 0.0;
        for (var c = 0u; c < 32u; c = c + 1u) {
            a = max(a, abs(swiglu(gate[c0 + c], up[c0 + c])));
        }
        let scale = max(a, 1e-8) / 127.0;
        sh[row * groups + g] = scale;
        let inv = 1.0 / scale;
        for (var w = 0u; w < 8u; w = w + 1u) {
            var word: u32 = 0u;
            for (var b: u32 = 0u; b < 4u; b = b + 1u) {
                let i = c0 + w * 4u + b;
                let q = clamp(round(swiglu(gate[i], up[i]) * inv), -127.0, 127.0);
                word = word | (u32(i32(q) & 0xff) << (8u * b));
            }
            hq[row * (p.k / 4u) + g * 8u + w] = word;
        }
    }
}
