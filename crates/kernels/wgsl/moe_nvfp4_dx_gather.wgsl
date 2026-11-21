// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fine-tuning of quantized mixture-of-experts models on their published weights for its
// clients. If your team needs expertise in training through frozen microscaled expert banks then you can procure our
// services by sending an email to info@swedishembedded.com.

// @what  Input gradient of the NVFP4 expert-bank gather product: dx[s, :] = W[ids[s]]^T dy[s, :] through a frozen bank, the decode-shape backward of moe_nvfp4_gemv_gather
// @how   one thread per (slot, input column), each weight decoded in place (E2M1 x E4M3 x per-expert f32), ascending fold over the bank rows, no barrier
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant nvfp4
// @dtype f32
//
// The forward `moe_nvfp4_gemv_gather.wgsl` computes `out[s, col] = sum_i W_e[col, i] * x[s / xdiv, i]` with
// `e = ids[s]` and the bank frozen; its input gradient is the transposed product of the SAME decoded weights:
//
//   dx[s, i] (+)= scale_e * sum_col W_e[col, i] * dy[s, col]      (col ascending)
//
// written per slot (`[slots, k]`): a caller whose slots share an input row (`xdiv > 1`) or route back to rows
// (row-compacted training) sums or scatters them. The weights are never written and no weight gradient exists:
// the bank is a frozen base (a trainable delta rides beside it as an adapter).
//
//   params : u32 [slots, k, n, accumulate]   k a multiple of 64; accumulate 1 adds into dx
//   dy     : [slots, n] f32
//   ids    : [slots] u32                     expert (bank block) of each slot
//   wq     : [blocks * n, k/8] u32     eight E2M1 codes per word, low nibble first (the forward's bank)
//   sc     : [blocks * n, k/64] u32    four E4M3 block scales (one per 16 inputs) per word
//   gl     : [blocks] f32              each expert's second-level scale
//   dx     : [slots, k] f32
//
// Dispatch: one thread per output element (`slots * k`).

struct Params {
    slots: u32,
    k: u32,
    n: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:  array<f32>;
@group(0) @binding(2) var<storage, read>       ids: array<u32>;
@group(0) @binding(3) var<storage, read>       wq:  array<u32>;
@group(0) @binding(4) var<storage, read>       sc:  array<u32>;
@group(0) @binding(5) var<storage, read>       gl:  array<f32>;
@group(0) @binding(6) var<storage, read_write> dx:  array<f32>;

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    var v = f32(m) * (1.0 / 512.0);
    if (e != 0u) { v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    if ((b & 128u) != 0u) { v = -v; }
    return v;
}

// The value of an E2M1 code: sign, 2-bit exponent (bias 1), 1-bit mantissa.
fn fp4_value(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let m = c & 1u;
    var v = f32(m) * 0.5;
    if (e != 0u) { v = bitcast<f32>(((126u + e) << 23u) | (m << 22u)); }
    if ((c & 8u) != 0u) { v = -v; }
    return v;
}

// Element `i` of bank row `row_w`, without the bank's per-expert scale: code (low nibble first) times its E4M3 scale.
fn weight(row_w: u32, i: u32) -> f32 {
    let code = (wq[row_w * (p.k / 8u) + i / 8u] >> ((i % 8u) * 4u)) & 15u;
    let g = i / 16u;
    let s = (sc[row_w * (p.k / 64u) + g / 4u] >> ((g % 4u) * 8u)) & 255u;
    return fp4_value(code) * e4m3(s);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.slots * p.k) { return; }
    let s = idx / p.k;
    let i = idx % p.k;
    let e = ids[s];
    var acc = 0.0;
    for (var col = 0u; col < p.n; col = col + 1u) {
        acc = acc + weight(e * p.n + col, i) * dy[s * p.n + col];
    }
    acc = acc * gl[e];
    if (p.accumulate != 0u) { acc = acc + dx[idx]; }
    dx[idx] = acc;
}
