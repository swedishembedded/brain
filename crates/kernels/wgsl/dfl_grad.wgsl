// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  DFL decode gradient. Given upstream dE[A,4] = dL/dE for each expected distance E, produce logit grads
// @how   one thread per output element, 4 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// DFL decode gradient. Given upstream dE[A,4] = dL/dE for each expected
// distance E, produce logit grads. Softmax-expectation Jacobian:
//   E = sum_i i * p_i,   dE/dlogit_j = p_j * (j - E)
// so dL/dlogit_j = dE * p_j * (j - E).
// Recompute softmax + E locally. One thread per (anchor, side).

struct Params {
    A: u32,
    reg_max: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       dE:     array<f32>;
@group(0) @binding(3) var<storage, read_write> dlogit: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;                       // (anchor*4 + side)
    let total = p.A * 4u;
    if (idx >= total) { return; }

    let base = idx * p.reg_max;

    var pr: array<f32, 16>;
    var mx = -3.4e38;
    for (var i: u32 = 0u; i < p.reg_max; i = i + 1u) {
        mx = max(mx, logits[base + i]);
    }
    var sum = 0.0;
    for (var i: u32 = 0u; i < p.reg_max; i = i + 1u) {
        let e = exp(logits[base + i] - mx);
        pr[i] = e;
        sum = sum + e;
    }
    var ev = 0.0;
    for (var i: u32 = 0u; i < p.reg_max; i = i + 1u) {
        pr[i] = pr[i] / sum;
        ev = ev + f32(i) * pr[i];
    }

    let g = dE[idx];
    for (var j: u32 = 0u; j < p.reg_max; j = j + 1u) {
        dlogit[base + j] = g * pr[j] * (f32(j) - ev);
    }
}
