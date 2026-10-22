// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-(row, 32-group) max/x/ -> int8 activation scale, the scale of every quantized activation
// @how   one thread per row, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The int8 activation scale, one per 32-element group of each row:
//
//   x : [m, k]    sx: [m, k/32]    params: m, k (k a multiple of 32)
//   sx[row, g] = max(max|x[row, 32g .. 32g+32]|, 1e-8) / 127
//
// The group is the weight scale's own (`model::int8::GROUP`, Q8_0's block),
// so a kernel applies both scales once per group. One scale per row lets a
// single outlier channel set the step for the whole row; LLM activations
// carry such channels, and on Qwen3-0.6B a per-row scale cost a next-token
// KL of 2.8e-2 against 1.9e-3 for this layout.
//
// One thread per row; `max_abs_rows` is the cooperative form a GPU upgrades
// this to, bit-identical because `max` is exact.

struct Params { m: u32, k: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read_write> sx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let m = gid.y * (nwg.x * 64u) + gid.x;
    if (m >= p.m) { return; }
    let groups = p.k / 32u;
    for (var g: u32 = 0u; g < groups; g = g + 1u) {
        let base = m * p.k + g * 32u;
        var a = 0.0;
        for (var c: u32 = 0u; c < 32u; c = c + 1u) {
            a = max(a, abs(x[base + c]));
        }
        sx[m * groups + g] = max(a, 1e-8) / 127.0;
    }
}
