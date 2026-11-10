// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Batched product of rotated activations with a CQ2/CQ3/CQ4 weight read in place: y[m, n] = sum_g norm * (codebook[idx] . xr) - the prefill, encoder and batched-decode regime
// @how   one thread per (weight row, block of MAXM activation rows): each code is unpacked once and applied to all MAXM rows, no barriers
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// The activation rows were rotated by `cq_rotate` (xr), so the weight is never expanded: a lane's contribution is
// `codebook[index] * xr`, summed over a 128-lane group and scaled by the group's fp16 norm. Indices are an LSB-first
// bit stream per row; a group of 128 lanes is exactly `4 * BITS` words, and 32 lanes `BITS` words, so every group
// starts on a word. BITS (2, 3 or 4) and MAXM are template constants (`kernels::template::cq_variants`).
// The loop has no barriers, so the CPU JIT runs it; `matmul_cq_gemv` is the cooperative decode-regime sibling.
//
//   params : m, kpad, n, 0   activation rows, padded width, weight rows
//   xr     : [m, kpad] f32          rotated activations
//   w      : [n, kpad * BITS / 32] u32   packed indices
//   s      : fp16 norms [n, kpad / 128], two to a word, flat
//   book   : [2^BITS] f32 levels
//   y      : [m, n] f32

struct Params {
    m: u32,
    kpad: u32,
    n: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xr: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<u32>;
@group(0) @binding(3) var<storage, read>       s: array<u32>;
@group(0) @binding(4) var<storage, read>       book: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

const BITS: u32 = 2u;
const MAXM: u32 = 8u;
const GROUP: u32 = 128u;

fn f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1Fu;
    let subnormal = bitcast<f32>(0x38800000u | ((h & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
    let normal = bitcast<f32>((h & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
    let inf_or_nan = bitcast<f32>(0x7F800000u | ((h & 0x3FFu) << 13u));
    let mag = select(select(subnormal, normal, exp != 0u), inf_or_nan, exp == 31u);
    return bitcast<f32>(bitcast<u32>(mag) | sign);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let blocks = (p.m + MAXM - 1u) / MAXM;
    if (idx >= p.n * blocks) { return; }
    let col = idx % p.n;
    let r0 = (idx / p.n) * MAXM;
    let groups = p.kpad / GROUP;
    let row_words = p.kpad * BITS / 32u;
    var y_acc: array<f32, MAXM>;
    for (var r = 0u; r < MAXM; r = r + 1u) { y_acc[r] = 0.0; }
    for (var g = 0u; g < groups; g = g + 1u) {
        let flat = col * groups + g;
        let norm = f16_to_f32((s[flat >> 1u] >> ((flat & 1u) * 16u)) & 0xFFFFu);
        var acc: array<f32, MAXM>;
        for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
        let gw = col * row_words + g * 4u * BITS;
        for (var j = 0u; j < GROUP; j = j + 1u) {
            let bit = j * BITS;
            let word = bit >> 5u;
            let sh = bit & 31u;
            var v = w[gw + word] >> sh;
            if (sh + BITS > 32u) { v = v | (w[gw + word + 1u] << (32u - sh)); }
            let level = book[v & ((1u << BITS) - 1u)];
            for (var r = 0u; r < MAXM; r = r + 1u) {
                if (r0 + r < p.m) { acc[r] = acc[r] + level * xr[(r0 + r) * p.kpad + g * GROUP + j]; }
            }
        }
        for (var r = 0u; r < MAXM; r = r + 1u) { y_acc[r] = y_acc[r] + norm * acc[r]; }
    }
    for (var r = 0u; r < MAXM; r = r + 1u) {
        if (r0 + r < p.m) { y[(r0 + r) * p.n + col] = y_acc[r]; }
    }
}
