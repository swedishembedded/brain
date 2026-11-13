// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Entropy loss gradient wrt the codebook rows (LlamaGen's entropy loss, backward)
// @how   one thread per (code, dimension), serial reduction over the tokens
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
// Reads the per-token mean `meanc` that `entropy_dz` wrote, so it is dispatched after it.
struct Params { R: u32, K: u32, D: u32, tau: f32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       z:     array<f32>;
@group(0) @binding(2) var<storage, read>       e:     array<f32>;
@group(0) @binding(3) var<storage, read>       logp:  array<f32>;
@group(0) @binding(4) var<storage, read>       c:     array<f32>;
@group(0) @binding(5) var<storage, read>       h:     array<f32>;
@group(0) @binding(6) var<storage, read>       meanc: array<f32>;
@group(0) @binding(7) var<storage, read_write> de:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.K * p.D) { return; }
    let k = idx / p.D;
    let d = idx % p.D;
    var acc = 0.0;
    for (var i: u32 = 0u; i < p.R; i = i + 1u) {
        let lp = logp[i * p.K + k];
        let g = exp(lp) / f32(p.R) * (c[k] - meanc[i] - lp - h[i]);
        acc = acc + g * (z[i * p.D + d] - e[idx]);
    }
    de[idx] = p.scale * 2.0 * acc / p.tau;
}
