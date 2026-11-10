// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training CTC forward and backward variables in log space; loss = -log p(labels | x)
// @how   one thread per sequence (the recursion is sequential in the frames)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// ext is the extended label sequence blank, l1, blank, l2, ..., blank (s_len = 2 labels + 1). alpha and beta are
// [frames, s_len] scratch. "Minus infinity" is -1e30; sums with a state at or below -1e29 are skipped, as on the host.
// Params: frames, classes, s_len. Dispatch: 1 thread.

struct Params { frames: u32, classes: u32, s_len: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       lp: array<f32>;
@group(0) @binding(2) var<storage, read>       ext: array<u32>;
@group(0) @binding(3) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> loss: array<f32>;

fn log_add(a: f32, b: f32) -> f32 {
    if (a <= -1.0e29) { return b; }
    if (b <= -1.0e29) { return a; }
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= 1u) { return; }
    let neg = -1.0e30;
    let sl = p.s_len;
    for (var s = 0u; s < sl; s = s + 1u) { alpha[s] = neg; beta[(p.frames - 1u) * sl + s] = neg; }
    alpha[0] = lp[ext[0]];
    if (sl > 1u) { alpha[1] = lp[ext[1]]; }
    for (var t = 1u; t < p.frames; t = t + 1u) {
        for (var s = 0u; s < sl; s = s + 1u) {
            var a = alpha[(t - 1u) * sl + s];
            if (s >= 1u) { a = log_add(a, alpha[(t - 1u) * sl + s - 1u]); }
            if (s >= 2u && ext[s] != 0u && ext[s] != ext[s - 2u]) { a = log_add(a, alpha[(t - 1u) * sl + s - 2u]); }
            if (a <= -1.0e29) {
                alpha[t * sl + s] = neg;
            } else {
                alpha[t * sl + s] = a + lp[t * p.classes + ext[s]];
            }
        }
    }
    let last = (p.frames - 1u) * sl;
    var log_p = alpha[last];
    if (sl > 1u) { log_p = log_add(alpha[last + sl - 1u], alpha[last + sl - 2u]); }
    loss[0] = -log_p;
    beta[last + sl - 1u] = lp[(p.frames - 1u) * p.classes + ext[sl - 1u]];
    if (sl > 1u) { beta[last + sl - 2u] = lp[(p.frames - 1u) * p.classes + ext[sl - 2u]]; }
    for (var q = 1u; q < p.frames; q = q + 1u) {
        let t = p.frames - 1u - q;
        for (var s = 0u; s < sl; s = s + 1u) {
            var b = beta[(t + 1u) * sl + s];
            if (s + 1u < sl) { b = log_add(b, beta[(t + 1u) * sl + s + 1u]); }
            if (s + 2u < sl && ext[s + 2u] != 0u && ext[s + 2u] != ext[s]) { b = log_add(b, beta[(t + 1u) * sl + s + 2u]); }
            if (b <= -1.0e29) {
                beta[t * sl + s] = neg;
            } else {
                beta[t * sl + s] = b + lp[t * p.classes + ext[s]];
            }
        }
    }
}
