// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-code average assignment and its entropy terms (LlamaGen's entropy loss: the codebook-usage half)
// @how   one thread per code, serial reduction over the tokens
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// With avg_k = mean_i exp(logp_ik) (the codebook's average assignment):
//   term_k = -avg_k * log(avg_k + 1e-5)       the host sums these: the average entropy
//   c_k    = log(avg_k + 1e-5) + avg_k / (avg_k + 1e-5)       d(-H_avg)/d avg_k, for the gradients
// 1e-5 is the reference's `log(avg_probs + 1e-5)`.
struct Params { R: u32, K: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logp: array<f32>;
@group(0) @binding(2) var<storage, read_write> c:    array<f32>;
@group(0) @binding(3) var<storage, read_write> term: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = gid.y * (nwg.x * 64u) + gid.x;
    if (k >= p.K) { return; }
    var avg = 0.0;
    for (var i: u32 = 0u; i < p.R; i = i + 1u) {
        avg = avg + exp(logp[i * p.K + k]);
    }
    avg = avg / f32(p.R);
    let l = log(avg + 1.0e-5);
    term[k] = -avg * l;
    c[k] = l + avg / (avg + 1.0e-5);
}
