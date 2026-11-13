// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-token log-softmax of the negative squared distances to a codebook, and each token's entropy (LlamaGen's entropy loss, forward)
// @how   one thread per token, three serial passes over the codes
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The forward of LlamaGen's `compute_entropy_loss(-d)` (see vqgan's `entropy.rs`).
// For token i with row z_i [D] and codebook rows e_k [D]:
//   a_ik    = -|z_i - e_k|^2 / tau
//   logp_ik = a_ik - max_k a_ik - log(sum_k exp(a_ik - max_k a_ik))      (log-softmax)
//   h_i     = -sum_k exp(logp_ik) * logp_ik                              (the token's entropy)
// `logp` is kept for `entropy_cols` and the two gradient kernels: a probability
// underflowing to 0 at a small tau must not turn into log(0), so the LOG is what
// is stored. The host takes the mean of `h`.
struct Params { R: u32, K: u32, D: u32, tau: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       z:    array<f32>;
@group(0) @binding(2) var<storage, read>       e:    array<f32>;
@group(0) @binding(3) var<storage, read_write> logp: array<f32>;
@group(0) @binding(4) var<storage, read_write> h:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.R) { return; }
    var m = -3.0e38;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        var dist = 0.0;
        for (var d: u32 = 0u; d < p.D; d = d + 1u) {
            let diff = z[row * p.D + d] - e[k * p.D + d];
            dist = dist + diff * diff;
        }
        let a = -dist / p.tau;
        logp[row * p.K + k] = a;
        m = max(m, a);
    }
    var s = 0.0;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        s = s + exp(logp[row * p.K + k] - m);
    }
    let log_sum = m + log(s);
    var ent = 0.0;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        let lp = logp[row * p.K + k] - log_sum;
        logp[row * p.K + k] = lp;
        ent = ent - exp(lp) * lp;
    }
    h[row] = ent;
}
