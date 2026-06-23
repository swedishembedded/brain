// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Router backward: gradient w.r.t. the router logits, combining three paths.
// One invocation per token row. E <= 64.
//
//   p = softmax(logits)             (full distribution)
//   S = selected (top-k) experts,   gate_e = p_e / Z  for e in S,  Z = sum_{S} p
//
// 1. Combine path: given d_gate_e (e in S),
//      dp_f = d_gate_f / Z - (sum_{e in S} d_gate_e p_e) / Z^2     (f in S, else 0)
// 2. Load-balancing aux:  aux = aux_coef * E * sum_e f_e * mean_n p_{n,e}
//      => dp_e += aux_coef * E * f_e / n_rows     (all e)
// 3. Then softmax backward:  d_logit_i = p_i (dp_i - sum_j p_j dp_j)
// 4. Router z-loss:  z = z_coef * mean_n (logsumexp_n)^2
//      => d_logit_i += z_coef * 2 * lse / n_rows * p_i

struct Params {
    n_rows: u32,
    n_experts: u32,
    top_k: u32,
    _pad: u32,
    aux_coef: f32,
    z_coef: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       gate:    array<f32>;
@group(0) @binding(3) var<storage, read>       d_gate:  array<f32>;
@group(0) @binding(4) var<storage, read>       fe:      array<f32>;
@group(0) @binding(5) var<storage, read_write> dlogits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.n_rows) { return; }
    let E = p.n_experts;
    let base = t * E;
    let nrows = f32(p.n_rows);

    var pr: array<f32, 64>;
    var dp: array<f32, 64>;

    var mx = -3.4e38;
    for (var e: u32 = 0u; e < E; e = e + 1u) { mx = max(mx, logits[base + e]); }
    var sum = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) { sum = sum + exp(logits[base + e] - mx); }
    let lse = mx + log(sum);
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        pr[e] = exp(logits[base + e] - mx) / sum;
        dp[e] = 0.0;
    }

    // Z and sdp over the selected (gate>0) experts
    var z = 0.0;
    var sdp = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (gate[base + e] > 0.0) {
            z = z + pr[e];
            sdp = sdp + d_gate[base + e] * pr[e];
        }
    }
    let zz = max(z, 1e-9);

    // (1) combine path  +  (2) aux path
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (gate[base + e] > 0.0) {
            dp[e] = d_gate[base + e] / zz - sdp / (zz * zz);
        }
        dp[e] = dp[e] + p.aux_coef * f32(E) * fe[e] / nrows;
    }

    // (3) softmax backward  +  (4) z-loss
    var gpdot = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) { gpdot = gpdot + pr[e] * dp[e]; }
    let zterm = p.z_coef * 2.0 * lse / nrows;
    for (var i: u32 = 0u; i < E; i = i + 1u) {
        dlogits[base + i] = pr[i] * (dp[i] - gpdot) + zterm * pr[i];
    }
}
