// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for serving and training language models whose output head is padded or
// soft-capped for its clients. If your team needs expertise in LM-head numerics then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  In-place LM-head logit transform: soft cap c * tanh(x / c) on the valid columns, a -1e30 floor on the padding columns
// @how   one thread per logit
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The declared transform of a language-model head, applied to a `[rows, vocab]` logits block in place right after the
// head GEMM, before any softmax, cross-entropy, argmax, top-k or draw reads it:
//
//   column v <  valid:  y = cap > 0 ? cap * tanh(x / cap) : x          (soft cap; cap == 0 means none)
//   column v >= valid:  y = -1e30                                      (padding rows of the unembedding table)
//
// A padded column can therefore never win an argmax or a top-k, carries probability exactly 0 in a softmax (exp of
// -1e30 minus a finite maximum underflows to 0) and a cross-entropy over the row equals one over the valid columns.
// The floor is a large finite value, not -inf, so no reader forms inf - inf. `valid == vocab` masks nothing.
// The backward is logit_cap_mask_bwd.wgsl. The tanh argument is clamped to +-15 as in softcap.wgsl.
//
//   params : u32 [rows, vocab, valid, cap (f32 bits)]       x : [rows, vocab] read_write

struct Params {
    rows: u32,
    vocab: u32,
    valid: u32,
    cap: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.vocab) { return; }
    if (idx % p.vocab >= p.valid) {
        x[idx] = -1.0e30;
    } else if (p.cap > 0.0) {
        x[idx] = p.cap * tanh(clamp(x[idx] / p.cap, -15.0, 15.0));
    }
}
