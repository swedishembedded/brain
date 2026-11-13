// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Entropy loss gradient wrt the token rows, and each token's mean c (LlamaGen's entropy loss, backward)
// @how   one thread per (token, dimension), two serial passes over the codes
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// With G_ik = exp(logp_ik) / R * (c_k - m_i - logp_ik - h_i), m_i = sum_j exp(logp_ij) c_j,
// the gradient of L = mean_i h_i - H_avg wrt the logits a_ik (see vqgan's `entropy.rs`), and
// a_ik = -|z_i - e_k|^2 / tau:
//   dL/dz_id = sum_k G_ik * 2 (e_kd - z_id) / tau
//   dL/de_kd = sum_i G_ik * 2 (z_id - e_kd) / tau
// both multiplied by `scale` (the loss ratio) and ASSIGNED, not accumulated.
// Every (i, d) thread recomputes m_i (the D threads of a token agree on it); the d = 0 thread
// stores it for `entropy_dcb`.
struct Params { R: u32, K: u32, D: u32, tau: f32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       z:     array<f32>;
@group(0) @binding(2) var<storage, read>       e:     array<f32>;
@group(0) @binding(3) var<storage, read>       logp:  array<f32>;
@group(0) @binding(4) var<storage, read>       c:     array<f32>;
@group(0) @binding(5) var<storage, read>       h:     array<f32>;
@group(0) @binding(6) var<storage, read_write> meanc: array<f32>;
@group(0) @binding(7) var<storage, read_write> dz:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.R * p.D) { return; }
    let i = idx / p.D;
    let d = idx % p.D;
    var m = 0.0;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        m = m + exp(logp[i * p.K + k]) * c[k];
    }
    if (d == 0u) { meanc[i] = m; }
    var acc = 0.0;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        let lp = logp[i * p.K + k];
        let g = exp(lp) / f32(p.R) * (c[k] - m - lp - h[i]);
        acc = acc + g * (e[k * p.D + d] - z[idx]);
    }
    dz[idx] = p.scale * 2.0 * acc / p.tau;
}
