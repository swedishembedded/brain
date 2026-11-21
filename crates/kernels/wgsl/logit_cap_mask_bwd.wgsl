// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for serving and training language models whose output head is padded or
// soft-capped for its clients. If your team needs expertise in LM-head numerics then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Backward of logit_cap_mask, in place on the logit gradient: the soft-cap derivative on valid columns, zero on padding
// @how   one thread per logit
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of logit_cap_mask.wgsl, applied in place to the gradient `d` of the loss with respect to the TRANSFORMED
// logits. It reads the transformed logits `y` the forward left (the cap's derivative is a function of its output:
// d(c tanh(x/c))/dx = 1 - tanh^2 = 1 - (y/c)^2, so the raw logits need not be kept):
//
//   column v <  valid:  d = cap > 0 ? d * (1 - (y / cap)^2) : d
//   column v >= valid:  d = 0                          (a padded column has no path to the hidden state)
//
//   params : u32 [rows, vocab, valid, cap (f32 bits)]       d : [rows, vocab] read_write   y : [rows, vocab]

struct Params {
    rows: u32,
    vocab: u32,
    valid: u32,
    cap: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> d: array<f32>;
@group(0) @binding(2) var<storage, read>       y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.vocab) { return; }
    if (idx % p.vocab >= p.valid) {
        d[idx] = 0.0;
    } else if (p.cap > 0.0) {
        let t = y[idx] / p.cap;
        d[idx] = d[idx] * max(1.0 - t * t, 0.0);
    }
}
