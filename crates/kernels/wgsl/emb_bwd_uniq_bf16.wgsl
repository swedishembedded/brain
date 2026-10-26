// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of large
// language models for its clients. If your team needs expertise in training
// vocabulary-sized tables without a full-precision gradient copy, you can
// procure our services by sending an email to info@swedishembedded.com.

// @what  Embedding backward into a packed bf16 gradient, over the DISTINCT looked-up rows only
// @how   one thread per (distinct row, channel pair), serial scan of the tokens, round to nearest even
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The bf16-gradient twin of emb_bwd_uniq: a table trained with a bf16 gradient
// (`GradDtype::Bf16`, two elements per u32 word, low half first) accumulates
//
//   grad[uniq[u], c] = bf16(f32(grad[uniq[u], c]) + sum_{n : tokens[n] == uniq[u]} d_x[n, c])
//
// One invocation owns one packed word (channels 2j and 2j + 1 of one distinct
// row), so no two invocations share a word and no atomics are needed. The sums
// are taken in f32 in ascending token order, exactly as emb_bwd_uniq takes
// them, and rounded once per call, exactly as accum_bf16 rounds a dense
// weight gradient tile. `d_model` must be even.
//
// **`uniq` must contain every distinct value in `tokens`** (build it with
// `model::block::uniq_u32`, or `model::ops::uniq_padded` for a fixed-length
// list): a missing id's gradient is silently dropped. An entry of 0xFFFFFFFF
// is padding and is skipped.

struct Params {
    n_rows: u32,
    d_model: u32,
    n_uniq: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens:   array<u32>;
@group(0) @binding(2) var<storage, read>       uniq:     array<u32>;
@group(0) @binding(3) var<storage, read>       d_x:      array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_emb: array<u32>;

fn to_bf16(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if (v != v) { return (b >> 16u) | 0x40u; }
    return (b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let pairs = p.d_model / 2u;
    if (gidx >= p.n_uniq * pairs) { return; }
    let v = uniq[gidx / pairs];
    if (v == 0xFFFFFFFFu) { return; }
    let c = 2u * (gidx % pairs);
    var lo = 0.0;
    var hi = 0.0;
    for (var n: u32 = 0u; n < p.n_rows; n = n + 1u) {
        if (tokens[n] == v) {
            lo = lo + d_x[n * p.d_model + c];
            hi = hi + d_x[n * p.d_model + c + 1u];
        }
    }
    let wi = (v * p.d_model + c) / 2u;
    let old = grad_emb[wi];
    let a = bitcast<f32>((old & 0xFFFFu) << 16u) + lo;
    let b = bitcast<f32>(old & 0xFFFF0000u) + hi;
    grad_emb[wi] = to_bf16(a) | (to_bf16(b) << 16u);
}
