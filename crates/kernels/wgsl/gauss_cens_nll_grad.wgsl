// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of gauss_cens_nll_value with respect to (mu, log sigma)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import normal
//
// With z = (y - mu) / sigma, dz/dmu = -1/sigma and dz/dlogsig = -z:
//   exact  d mu = -w z / sigma            d logsig = w (1 - z^2)
//   below  d mu =  w r(z) / sigma         d logsig = w r(z) z
//   above  d mu = -w r(-z) / sigma        d logsig = -w r(-z) z
// where r(x) = phi(x) / Phi(x) (the inverse Mills ratio, computed in log
// space by the `normal` library). State 0 writes zero. Same layout as
// gauss_cens_nll_value; writes (does not accumulate) d_pred [n, 2].

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>;
@group(0) @binding(2) var<storage, read>       y:      array<f32>;
@group(0) @binding(3) var<storage, read>       state:  array<u32>;
@group(0) @binding(4) var<storage, read>       w:      array<f32>;
@group(0) @binding(5) var<storage, read_write> d_pred: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let s = state[i];
    var dmu = 0.0;
    var dls = 0.0;
    if (s != 0u) {
        let mu = pred[2u * i];
        let logsig = pred[2u * i + 1u];
        let inv_sig = exp(-logsig);
        let z = (y[i] - mu) * inv_sig;
        let wi = w[i];
        if (s == 1u) {
            dmu = -wi * z * inv_sig;
            dls = wi * (1.0 - z * z);
        } else if (s == 2u) {
            let r = normal_mills(z);
            dmu = wi * r * inv_sig;
            dls = wi * r * z;
        } else {
            let r = normal_mills(-z);
            dmu = -wi * r * inv_sig;
            dls = -wi * r * z;
        }
    }
    d_pred[2u * i] = dmu;
    d_pred[2u * i + 1u] = dls;
}
