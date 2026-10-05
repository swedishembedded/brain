// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Heteroscedastic Gaussian negative log-likelihood with left- and right-censored (detection-limit) observations
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import normal
//
// Each row predicts a normal distribution (mu, log sigma) for one observed
// value y, which arrives in one of four states:
//   0  not scored                 out = 0
//   1  exact                      out = w * (0.5 z^2 + log sigma + 0.5 log 2pi)
//   2  below a limit (true <= y)  out = w * -log Phi(z)
//   3  above a limit (true >= y)  out = w * -log Phi(-z)
// with z = (y - mu) / sigma. States 2 and 3 are the Tobit terms: a value
// reported as "below the detection limit" is information about where it lies,
// not a missing value. `w` is the caller's per-row weight, already normalised
// so the host loss is a plain sum of `out`.
//
// Layout: pred [n, 2] row-major (mu, log sigma); y, w [n] f32; state [n] u32.
// Gradient: gauss_cens_nll_grad.

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:  array<f32>;
@group(0) @binding(2) var<storage, read>       y:     array<f32>;
@group(0) @binding(3) var<storage, read>       state: array<u32>;
@group(0) @binding(4) var<storage, read>       w:     array<f32>;
@group(0) @binding(5) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let s = state[i];
    if (s == 0u) { out[i] = 0.0; return; }
    let mu = pred[2u * i];
    let logsig = pred[2u * i + 1u];
    let z = (y[i] - mu) * exp(-logsig);
    var nll = 0.0;
    if (s == 1u) {
        nll = 0.5 * z * z + logsig + NORMAL_HALF_LOG_2PI;
    } else if (s == 2u) {
        nll = -normal_log_cdf(z);
    } else {
        nll = -normal_log_cdf(-z);
    }
    out[i] = w[i] * nll;
}
