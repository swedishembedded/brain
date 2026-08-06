// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Distribution Focal Loss value per assigned anchor
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Distribution Focal Loss value per assigned anchor. For each of the 4 sides,
// the continuous target distance t splits across adjacent bins:
//   tl = floor(t), tr = tl + 1,  wr = t - tl,  wl = 1 - wr
// loss_side = wl * (-logsoftmax[tl]) + wr * (-logsoftmax[tr])
// where logsoftmax[k] = logits[k] - (mx + log(sum exp(logits - mx))).
// Output out[A] = sum over the 4 sides. One thread per anchor.
// tl/tr are clamped into [0, reg_max-1] so a target at the last bin is safe.

struct Params {
    A: u32,
    reg_max: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       tdist:  array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let a = gidx;
    if (a >= p.A) { return; }

    let rmax = p.reg_max;
    var loss = 0.0;
    for (var side: u32 = 0u; side < 4u; side = side + 1u) {
        let base = (a * 4u + side) * rmax;

        // logsumexp over the bins.
        var mx = -3.4e38;
        for (var i: u32 = 0u; i < rmax; i = i + 1u) {
            mx = max(mx, logits[base + i]);
        }
        var sum = 0.0;
        for (var i: u32 = 0u; i < rmax; i = i + 1u) {
            sum = sum + exp(logits[base + i] - mx);
        }
        let lse = mx + log(sum);

        // two-hot split of the continuous target. t >= 0, so truncation toward
        // zero (i32 cast) equals floor; floor/clamp are not JIT intrinsics so we
        // use the int cast + min/max.
        let t = tdist[a * 4u + side];
        let tl0 = i32(t);                 // floor(t) for t >= 0
        let tlf = f32(tl0);
        let wr = t - tlf;
        let wl = 1.0 - wr;
        let hi = i32(rmax) - 1;
        let tl = max(0, min(tl0, hi));
        let tr = max(0, min(tl0 + 1, hi));

        let ls_tl = logits[base + u32(tl)] - lse;
        let ls_tr = logits[base + u32(tr)] - lse;
        loss = loss + wl * (-ls_tl) + wr * (-ls_tr);
    }
    out[a] = loss;
}
