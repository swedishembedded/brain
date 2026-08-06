// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of dfl_loss w.r.t
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Gradient of dfl_loss w.r.t. logits. The per-side loss is cross-entropy of the
// softmax distribution against the two-hot soft target
//   target_tl = wl, target_tr = wr   (all other bins 0).
// For CE = -sum_k target_k * logsoftmax_k over a softmax, the logit gradient is
//   dlogit_j = softmax_j - target_j.
// (When tl is clamped == tr, the two weights collapse onto the same bin, giving
//  target = wl + wr = 1 there, which matches dfl_loss's clamped value loss.)
// One thread per (anchor, side); writes reg_max grads.

struct Params {
    A: u32,
    reg_max: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       tdist:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dlogit: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;                       // (anchor*4 + side)
    let total = p.A * 4u;
    if (idx >= total) { return; }

    let rmax = p.reg_max;
    let base = idx * rmax;

    var mx = -3.4e38;
    for (var i: u32 = 0u; i < rmax; i = i + 1u) {
        mx = max(mx, logits[base + i]);
    }
    var sum = 0.0;
    for (var i: u32 = 0u; i < rmax; i = i + 1u) {
        sum = sum + exp(logits[base + i] - mx);
    }

    // two-hot soft target. t >= 0 so i32 cast == floor; floor/clamp are not JIT
    // intrinsics, so use the int cast + min/max.
    let t = tdist[idx];
    let tl0 = i32(t);                      // floor(t) for t >= 0
    let tlf = f32(tl0);
    let wr = t - tlf;
    let wl = 1.0 - wr;
    let hi = i32(rmax) - 1;
    let tl = max(0, min(tl0, hi));
    let tr = max(0, min(tl0 + 1, hi));

    for (var j: u32 = 0u; j < rmax; j = j + 1u) {
        let sm = exp(logits[base + j] - mx) / sum;
        var tgt = 0.0;
        if (i32(j) == tl) { tgt = tgt + wl; }
        if (i32(j) == tr) { tgt = tgt + wr; }
        dlogit[base + j] = sm - tgt;
    }
}
