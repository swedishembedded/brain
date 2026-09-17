// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Embedding backward over the DISTINCT looked-up rows only - compact twin of emb_bwd
// @how   one thread per (distinct row, channel), serial scan of the tokens
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// emb_bwd.wgsl gives one invocation to every (vocab row, channel) pair and has
// each scan the tokens for a match, so its cost is `vocab * d_model * n_rows`
// however few rows were actually looked up. A call cannot touch more than
// `n_rows` distinct vocabulary rows, and normally touches far fewer, so every
// other row's invocation exists only to prove its own accumulator is zero.
//
// Here the caller passes `uniq`, the distinct token ids, and one invocation is
// spent per (distinct row, channel): the cost becomes `n_uniq * d_model *
// n_rows`. On a 30522-row table looked up by ~100 distinct tokens that is
// ~300x less work for the same answer.
//
//   grad_emb[uniq[u], c] += sum_{n : tokens[n] == uniq[u]} d_x[n, c]
//
// Still no atomics and still no race: one invocation owns each output element,
// exactly as in emb_bwd.
//
// **`uniq` must contain every distinct value in `tokens`.** A missing id is
// not an error this kernel can see - its gradient is simply never scattered,
// and the parameter it belongs to stops learning while everything else keeps
// working. `model::block::uniq_u32` is the one supported way to build it.
//
// Results are bit-identical to emb_bwd: the same summands are added in the
// same ascending order. The only divergence is for a vocabulary row that was
// NOT looked up, where emb_bwd stores `g + 0.0` and this kernel leaves `g`
// alone - a difference only when `g` is -0.0, which `zero_grads` never leaves
// behind.

struct Params {
    n_rows: u32,
    d_model: u32,
    n_uniq: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens:   array<u32>;
@group(0) @binding(2) var<storage, read>       uniq:     array<u32>;
@group(0) @binding(3) var<storage, read>       d_x:      array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_emb: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n_uniq * p.d_model;
    if (gidx >= total) { return; }
    let v = uniq[gidx / p.d_model];
    let c = gidx % p.d_model;
    var acc = 0.0;
    for (var n: u32 = 0u; n < p.n_rows; n = n + 1u) {
        if (tokens[n] == v) {
            acc = acc + d_x[n * p.d_model + c];
        }
    }
    let idx = v * p.d_model + c;
    grad_emb[idx] = grad_emb[idx] + acc;
}
