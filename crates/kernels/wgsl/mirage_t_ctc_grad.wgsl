// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training CTC logit gradient: softmax - occupancy, occupancy_k = sum_{s: ext[s] = k} alpha beta / (y_k P)
// @how   one thread per (frame, class), looping over the extended states
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Assigns d loss / d logits (the caller scales it by the loss gradient). Params: frames, classes, s_len.
// Dispatch: frames*classes threads.

struct Params { frames: u32, classes: u32, s_len: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       lp: array<f32>;
@group(0) @binding(2) var<storage, read>       ext: array<u32>;
@group(0) @binding(3) var<storage, read>       alpha: array<f32>;
@group(0) @binding(4) var<storage, read>       beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad: array<f32>;

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
    if (idx >= p.frames * p.classes) { return; }
    let k = idx % p.classes;
    let t = idx / p.classes;
    let sl = p.s_len;
    let last = (p.frames - 1u) * sl;
    var log_p = alpha[last];
    if (sl > 1u) { log_p = log_add(alpha[last + sl - 1u], alpha[last + sl - 2u]); }
    var occ = -1.0e30;
    for (var s = 0u; s < sl; s = s + 1u) {
        if (ext[s] == k) {
            let ab = alpha[t * sl + s] + beta[t * sl + s];
            if (ab > -5.0e29) { occ = log_add(occ, ab); }
        }
    }
    var occupancy = 0.0;
    if (occ > -5.0e29) { occupancy = exp(occ - lp[idx] - log_p); }
    grad[idx] = exp(lp[idx]) - occupancy;
}
