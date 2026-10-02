// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  SwiGLU and per-row int8 activation quantisation in one pass: out = quant(SiLU(gate) * up), one workgroup per row
// @how   128-thread workgroup, one tree max-reduction, two passes over the row
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// `silu_mul.wgsl` + `max_abs_row.wgsl` + `quant_pack.wgsl` fused for the one
// place a sparse-MoE expert needs all three back to back: the activation
// between its gate/up and down projections. Besides saving two dispatches per
// expert layer, it replaces `max_abs_row`'s ONE THREAD per row (a serial walk
// of the whole row) with a workgroup per row.
//
// The result is BIT-IDENTICAL to the three kernels run in sequence, because it
// evaluates the same expressions: `h = (x / (1 + exp(-x))) * u`, a max over
// `|h|` (order-free), `sh = max(max|h|, 1e-8) / 127`, and `q = clamp(round(h *
// (1 / sh)), -127, 127)` packed four per word little-endian along the row.
//
//   gate : [rows, k] f32   up : [rows, k] f32
//   hq   : [rows, k/4] u32 sh : [rows] f32
//   params : u32 [rows, k]   k a multiple of 4

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

var<workgroup> red: array<f32, 128>;
var<workgroup> row_scale: array<f32, 1>;

fn swiglu(g: f32, u: f32) -> f32 {
    return (g / (1.0 + exp(-g))) * u;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = row < p.rows;
    let kg = p.k / 4u;
    let base = row * p.k;

    var a = 0.0;
    if (live) {
        for (var w = t; w < kg; w = w + WG) {
            for (var b: u32 = 0u; b < 4u; b = b + 1u) {
                let i = base + w * 4u + b;
                a = max(a, abs(swiglu(gate[i], up[i])));
            }
        }
    }
    red[t] = a;
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = max(red[t], red[t + s]); }
        workgroupBarrier();
    }
    if (t == 0u) {
        row_scale[0] = max(red[0], 1e-8) / 127.0;
        if (live) { sh[row] = row_scale[0]; }
    }
    workgroupBarrier();

    if (live) {
        let inv = 1.0 / row_scale[0];
        for (var w = t; w < kg; w = w + WG) {
            var word: u32 = 0u;
            for (var b: u32 = 0u; b < 4u; b = b + 1u) {
                let i = base + w * 4u + b;
                let q = clamp(round(swiglu(gate[i], up[i]) * inv), -127.0, 127.0);
                word = word | (u32(i32(q) & 0xff) << (8u * b));
            }
            hq[row * kg + w] = word;
        }
    }
}
