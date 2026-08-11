// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Router backward: gradient w.r.t
// @how   one thread per output element, 5 nested serial reductions, array-free
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Router backward: gradient w.r.t. the router logits, combining three paths.
// One invocation per token row. No cap on E — `pr[e]`/`dp[e]` are recomputed
// from `logits`/`gate`/`d_gate` in each pass instead of cached in a
// fixed-size local array, mirroring `router_bwd_sigmoid.wgsl`'s style. This
// kernel used to hard-cap at `E <= 64` via `array<f32,64>` scratch — silent
// out-of-bounds writes above that (a failure shape seen before, recurring
// here because the earlier fix never reached this file).
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let t = gidx;
    if (t >= p.n_rows) { return; }
    let E = p.n_experts;
    let base = t * E;
    let nrows = f32(p.n_rows);

    // Pass 1+2: softmax normalisation constants (max, then sum of exp).
    var mx = -3.4e38;
    for (var e: u32 = 0u; e < E; e = e + 1u) { mx = max(mx, logits[base + e]); }
    var sum = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) { sum = sum + exp(logits[base + e] - mx); }
    let lse = mx + log(sum);

    // Pass 3: Z and sdp over the selected (gate>0) experts. `pr_e` recomputed,
    // not cached -- `mx`/`sum` already fix it, so re-deriving is one exp+div.
    var z = 0.0;
    var sdp = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (gate[base + e] > 0.0) {
            let pr_e = exp(logits[base + e] - mx) / sum;
            z = z + pr_e;
            sdp = sdp + d_gate[base + e] * pr_e;
        }
    }
    let zz = max(z, 1e-9);

    // Pass 4: gpdot = sum_e pr_e * dp_e. dp_e = (1) combine path (selected
    // only) + (2) aux path (all e), recomputed per e rather than cached.
    var gpdot = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        let pr_e = exp(logits[base + e] - mx) / sum;
        var dp_e = 0.0;
        if (gate[base + e] > 0.0) {
            dp_e = d_gate[base + e] / zz - sdp / (zz * zz);
        }
        dp_e = dp_e + p.aux_coef * f32(E) * fe[e] / nrows;
        gpdot = gpdot + pr_e * dp_e;
    }

    // Pass 5: (3) softmax backward + (4) z-loss, emitted per output element.
    let zterm = p.z_coef * 2.0 * lse / nrows;
    for (var i: u32 = 0u; i < E; i = i + 1u) {
        let pr_i = exp(logits[base + i] - mx) / sum;
        var dp_i = 0.0;
        if (gate[base + i] > 0.0) {
            dp_i = d_gate[base + i] / zz - sdp / (zz * zz);
        }
        dp_i = dp_i + p.aux_coef * f32(E) * fe[i] / nrows;
        dlogits[base + i] = pr_i * (dp_i - gpdot) + zterm * pr_i;
    }
}
