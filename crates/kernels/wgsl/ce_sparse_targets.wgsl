// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for off-policy reinforcement learning of large language models on one
// pass of the LM head for its clients. If your team needs expertise in score-centred policy-gradient training then you
// can procure our services by sending an email to info@swedishembedded.com.

// @what  Weighted CE logit gradient with up to K sparse extra targets per row (score-centred policy gradient)
// @how   one thread per output element, K-term scan of the row's extra targets
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The gradient of `w_n * nll(y_n) + sum_j c_nj * nll(v_nj)` with respect to row n's logits, from the plain CE gradient
// `d = softmax - e_y` that ce_grad_stats leaves (count 1):
//   W_n    = w_n + sum_j c_nj
//   out[i] = W_n * d[i] + (W_n - w_n) [v == y_n] - sum_{j: v_nj == v} c_nj
//          = W_n * softmax - w_n e_y - sum_j c_nj e_{v_nj}
// An extra target whose id is IGNORE adds nothing; a row whose label is IGNORE is zero. Duplicate ids, and ids equal
// to the label, add up: the K-term scan sums every match. Not in place (`d` and `out` are separate bindings).
struct Params { total: u32, v: u32, k: u32, ignore: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d:       array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read>       labels:  array<u32>;
@group(0) @binding(4) var<storage, read>       ids:     array<u32>;
@group(0) @binding(5) var<storage, read>       coefs:   array<f32>;
@group(0) @binding(6) var<storage, read_write> out:     array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let n = i / p.v;
    let col = i % p.v;
    let label = labels[n];
    if (label == p.ignore) { out[i] = 0.0; return; }
    var extra = 0.0;
    var here = 0.0;
    for (var j: u32 = 0u; j < p.k; j = j + 1u) {
        let id = ids[n * p.k + j];
        if (id == p.ignore) { continue; }
        let c = coefs[n * p.k + j];
        extra = extra + c;
        if (id == col) { here = here + c; }
    }
    let big_w = weights[n] + extra;
    out[i] = big_w * d[i] + select(0.0, extra, col == label) - here;
}
