// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook lookup tables then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Row gather from a CQ2/CQ3/CQ4 matrix: out[o, :] = scale * dequantised row ids[o / rep] (zeros for an absent marker)
// @how   workgroup per (output row, 128-lane group): each lane decodes its index into workgroup memory, one barrier, then a 128-term signed sum (inverse Walsh) per lane
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// A stored row is groups of 128 lanes: lane value = codebook[index] * norm, and the group is stored in the Walsh
// domain, so the row element is sum_j lane[j] * H[j][e] / sqrt(128) with H the natural-order +-1 Hadamard matrix
// (sign = parity of the bits of j & e). This is the lookup of the tied embedding (token ids, one row repeated over the
// hyper-connection lanes) and of the engram tables (hashed row ids, the absent marker 0xFFFFFFFF reads as zeros)
// straight from the packed archive, with no dense table resident.
//
//   params : count k bits rep scale  count output rows, row width (multiple of 128), index width, repeat, f32 scale
//   ids    : u32 row id per output row / rep
//   w      : [rows, k * bits / 32] u32 packed indices, LSB first
//   s      : fp16 norms [rows, k / 128], two to a word, flat
//   book   : [2^bits] f32 levels
//   out    : [count, k] f32
// Dispatch: one workgroup per (output row, group), count * k / 128 workgroups.

struct Params { count: u32, k: u32, bits: u32, rep: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids:  array<u32>;
@group(0) @binding(2) var<storage, read>       w:    array<u32>;
@group(0) @binding(3) var<storage, read>       s:    array<u32>;
@group(0) @binding(4) var<storage, read>       book: array<f32>;
@group(0) @binding(5) var<storage, read_write> out:  array<f32>;

var<workgroup> lane: array<f32, 128>;

fn f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1Fu;
    let subnormal = bitcast<f32>(0x38800000u | ((h & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
    let normal = bitcast<f32>((h & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
    let inf_or_nan = bitcast<f32>(0x7F800000u | ((h & 0x3FFu) << 13u));
    let mag = select(select(subnormal, normal, exp != 0u), inf_or_nan, exp == 31u);
    return bitcast<f32>(bitcast<u32>(mag) | sign);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let at = wg.y * nwg.x + wg.x;
    let groups = p.k / 128u;
    let o = at / groups;
    let g = at % groups;
    let t = li.x;
    let row = ids[o / p.rep];
    let present = row != 0xFFFFFFFFu;
    if (present) {
        let bit = (g * 128u + t) * p.bits;
        let word = row * (p.k * p.bits / 32u) + (bit >> 5u);
        let sh = bit & 31u;
        var v = w[word] >> sh;
        if (sh + p.bits > 32u) { v = v | (w[word + 1u] << (32u - sh)); }
        let flat = row * groups + g;
        let norm = f16_to_f32((s[flat >> 1u] >> ((flat & 1u) * 16u)) & 0xFFFFu);
        lane[t] = book[v & ((1u << p.bits) - 1u)] * norm;
    } else {
        lane[t] = 0.0;
    }
    workgroupBarrier();
    var acc = 0.0;
    for (var j = 0u; j < 128u; j = j + 1u) {
        var x = j & t;
        x = x ^ (x >> 4u);
        x = x ^ (x >> 2u);
        x = x ^ (x >> 1u);
        let sign = select(1.0, -1.0, (x & 1u) == 1u);
        acc = acc + sign * lane[j];
    }
    out[o * p.k + g * 128u + t] = acc * 0.08838834764831845 * p.scale;
}
