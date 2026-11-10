// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Decode-regime GEMV of a CQ2/CQ3/CQ4 weight read in place against up to MAXM rotated f32 activation rows
// @how   one 64-thread workgroup per weight row, threads stream 32-lane units of the packed row, codebook in shared memory, one barrier then an ordered fold
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// The bandwidth-bound sibling of `matmul_cq_gemm`: the packed row (`kpad * BITS / 8` bytes) is read once, coalesced
// (consecutive threads take consecutive `BITS`-word units), and applied to every activation row. A unit is 32 lanes
// = `BITS` words, four per 128-lane group, so a unit's norm is its group's. Same operands and layout as
// `matmul_cq_gemm`; BITS and MAXM (1, 2, 4, 8) are template constants.
//
//   params : m, kpad, n, 0
//   xr     : [m, kpad] f32   (rotated by cq_rotate)
//   w, s, book, y : as matmul_cq_gemm

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
const THREADS: u32 = 64u;

var<workgroup> levels: array<f32, 16>;
var<workgroup> partial: array<f32, 64u * MAXM>;

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
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = li.x;
    if (t < (1u << BITS)) { levels[t] = book[t]; }
    workgroupBarrier();
    let col = wg.y * nwg.x + wg.x;
    let live = col < p.n;
    let groups = p.kpad / 128u;
    let units = p.kpad / 32u;
    let row_words = p.kpad * BITS / 32u;
    var acc: array<f32, MAXM>;
    for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
    if (live) {
        for (var u = t; u < units; u = u + THREADS) {
            let flat = col * groups + u / 4u;
            let norm = f16_to_f32((s[flat >> 1u] >> ((flat & 1u) * 16u)) & 0xFFFFu);
            let uw = col * row_words + u * BITS;
            var part: array<f32, MAXM>;
            for (var r = 0u; r < MAXM; r = r + 1u) { part[r] = 0.0; }
            for (var j = 0u; j < 32u; j = j + 1u) {
                let bit = j * BITS;
                let word = bit >> 5u;
                let sh = bit & 31u;
                var v = w[uw + word] >> sh;
                if (sh + BITS > 32u) { v = v | (w[uw + word + 1u] << (32u - sh)); }
                let level = levels[v & ((1u << BITS) - 1u)];
                for (var r = 0u; r < MAXM; r = r + 1u) {
                    if (r < p.m) { part[r] = part[r] + level * xr[r * p.kpad + u * 32u + j]; }
                }
            }
            for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = acc[r] + norm * part[r]; }
        }
    }
    for (var r = 0u; r < MAXM; r = r + 1u) { partial[t * MAXM + r] = acc[r]; }
    workgroupBarrier();
    if (t < p.m && live) {
        var sum = 0.0;
        for (var i = 0u; i < THREADS; i = i + 1u) { sum = sum + partial[i * MAXM + t]; }
        y[t * p.n + col] = sum;
    }
}
