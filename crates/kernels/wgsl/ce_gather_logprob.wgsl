// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for off-policy reinforcement learning of large language models on one
// pass of the LM head for its clients. If your team needs expertise in score-centred policy-gradient training then you
// can procure our services by sending an email to info@swedishembedded.com.

// @what  Log-probabilities of K extra ids per row from the row's logits, label logit and CE value (no re-reduction)
// @how   one thread per (row, id), O(1) each
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// ce_value leaves nll[n] = lse(logits[n]) - logits[n, y_n], so the row's log-sum-exp is logits[n, y_n] + nll[n] and
//   out[n*K + j] = logits[n, v_nj] - logits[n, y_n] - nll[n]
// with no second pass over the row. 0 where the id or the row's label is IGNORE.
struct Params { total: u32, v: u32, k: u32, ignore: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       labels: array<u32>;
@group(0) @binding(3) var<storage, read>       nll:    array<f32>;
@group(0) @binding(4) var<storage, read>       ids:    array<u32>;
@group(0) @binding(5) var<storage, read_write> out:    array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let n = i / p.k;
    let label = labels[n];
    let id = ids[i];
    if (label == p.ignore || id == p.ignore) { out[i] = 0.0; return; }
    out[i] = logits[n * p.v + id] - logits[n * p.v + label] - nll[n];
}
