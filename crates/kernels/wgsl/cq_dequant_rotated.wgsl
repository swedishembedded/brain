// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Look up the codebook level times the group norm of every stored index of a CQ2/CQ3/CQ4 weight: the weight in the ROTATED domain as dense f32, the operand of a batched product on the dense GEMM
// @how   one thread per element: bit-stream index extraction, a codebook load and one multiply, no Walsh transform
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// y = W x = sum_g norm_g * (codebook[idx_g] . (H x_g)), so against the rotated activations (`cq_rotate`) the weight
// is just `codebook[idx] * norm`. For many activation rows that dense matrix is cheaper to build once and multiply
// with the tensor-core GEMM than to look up per row; it costs a lookup per stored weight and no transform (the
// transform lives on the activation side). BITS (2, 3 or 4) is a template constant.
//
//   params : n, kpad   weight rows, padded width (a multiple of 128)
//   w      : [n, kpad * BITS / 32] u32   packed indices
//   s      : fp16 norms [n, kpad / 128], two to a word, flat
//   book   : [2^BITS] f32 levels
//   out    : [n, kpad] f32

struct Params {
    n: u32,
    kpad: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w: array<u32>;
@group(0) @binding(2) var<storage, read>       s: array<u32>;
@group(0) @binding(3) var<storage, read>       book: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const BITS: u32 = 2u;

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
    if (idx >= p.n * p.kpad) { return; }
    let row = idx / p.kpad;
    let c = idx % p.kpad;
    let bit = c * BITS;
    let at = row * (p.kpad * BITS / 32u) + (bit >> 5u);
    let sh = bit & 31u;
    var v = w[at] >> sh;
    if (sh + BITS > 32u) { v = v | (w[at + 1u] << (32u - sh)); }
    let flat = row * (p.kpad / 128u) + c / 128u;
    let norm = f16_to_f32((s[flat >> 1u] >> ((flat & 1u) * 16u)) & 0xFFFFu);
    out[idx] = book[v & ((1u << BITS) - 1u)] * norm;
}
