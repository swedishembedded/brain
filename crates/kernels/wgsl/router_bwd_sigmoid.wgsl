// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GLM/DeepSeek-V3 "noaux_tc" MoE router (backward). Grad w.r.t. the router
// logits through the sigmoid combine weights (NO aux/z-loss; the selection bias
// is not trained by backprop). One invocation per token row. E <= 64.
//
//   s_e = sigmoid(logit_e) ; selected set S = { e : gate_e > 0 }
//   no-norm:  w_e = scale * s_e
//   norm:     w_e = scale * s_e / Z ,   Z = sum_{e in S} s_e
// Given d_gate_e = dL/dw_e (e in S):
//   no-norm:  dL/ds_k = scale * d_gate_k                              (k in S)
//   norm:     dL/ds_k = scale * ( d_gate_k/Z - sdp/Z^2 ),  sdp = sum_{S} d_gate_e s_e
//   dL/dlogit_k = (dL/ds_k) * s_k (1 - s_k)     (k in S, else 0)

struct Params {
    n_rows: u32,
    n_experts: u32,
    top_k: u32,
    norm: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       gate:    array<f32>;
@group(0) @binding(3) var<storage, read>       d_gate:  array<f32>;
@group(0) @binding(4) var<storage, read_write> dlogits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let t = gidx;
    if (t >= p.n_rows) { return; }
    let E = p.n_experts;
    let base = t * E;

    var z = 0.0;
    var sdp = 0.0;
    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (gate[base + e] > 0.0) {
            let s = 1.0 / (1.0 + exp(-logits[base + e]));
            z = z + s;
            sdp = sdp + d_gate[base + e] * s;
        }
    }
    let zz = max(z, 1e-20);

    for (var e: u32 = 0u; e < E; e = e + 1u) {
        if (gate[base + e] > 0.0) {
            let s = 1.0 / (1.0 + exp(-logits[base + e]));
            var ds = p.scale * d_gate[base + e];
            if (p.norm != 0u) {
                ds = p.scale * (d_gate[base + e] / zz - sdp / (zz * zz));
            }
            dlogits[base + e] = ds * s * (1.0 - s);
        } else {
            dlogits[base + e] = 0.0;
        }
    }
}
