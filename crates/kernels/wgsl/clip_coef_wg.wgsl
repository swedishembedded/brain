// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Global grad-norm clip coefficient from `gradnorm_part`'s partial sums —
// the cooperative counterpart of `clip_coef.wgsl`:
//   total = sqrt(sum_i parts[i]);  coef = min(1, max_norm/(total+1e-6)) * extra_scale
//
// `clip_coef` folds its input on ONE thread. That was tolerable when the input
// was one f32 per parameter tensor (77 for a 6-layer GPT), but the cooperative
// grad-norm writes one partial per workgroup per tensor (~6 300 for the same
// model), and a serial walk over that is the very bug this pass removes.
// 64 threads accumulate a strided slice, ONE barrier (the CPU JIT's limit),
// then thread 0 folds the 64 and applies the clip formula.
//
// Uniform layout is IDENTICAL to `clip_coef.wgsl` (`n`, `max_norm`,
// `extra_scale`) so the optimiser reuses the same uniform buffer; only the
// meaning of `n` changes (partials, not tensors).
//
// Dispatch: 64 invocations (exactly one workgroup).

struct Params {
    n_parts: u32,
    max_norm: f32,
    extra_scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       parts: array<f32>;
@group(0) @binding(2) var<storage, read_write> coef:  array<f32>;

var<workgroup> psum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) li: vec3<u32>) {
    let t = li.x;
    var s = 0.0;
    for (var i = t; i < p.n_parts; i = i + 64u) {
        s = s + parts[i];
    }
    psum[t] = s;
    workgroupBarrier();
    if (t == 0u) {
        var tot = 0.0;
        for (var k = 0u; k < 64u; k = k + 1u) {
            tot = tot + psum[k];
        }
        let total = sqrt(tot);
        var c = p.max_norm / (total + 1e-6);
        if (c > 1.0) { c = 1.0; }
        coef[0] = c * p.extra_scale;
    }
}
