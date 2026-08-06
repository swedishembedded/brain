// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of bce_logits w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Gradient of bce_logits w.r.t. each logit z:
//   dlogit = sigmoid(z) - t
// sigmoid computed in the numerically stable two-branch form.
// One thread per element.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:    array<f32>;
@group(0) @binding(3) var<storage, read_write> dlogit: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let z = logits[idx];
    var s = 0.0;
    if (z >= 0.0) {
        let e = exp(-z);
        s = 1.0 / (1.0 + e);
    } else {
        let e = exp(z);
        s = e / (1.0 + e);
    }
    dlogit[idx] = s - tgt[idx];
}
