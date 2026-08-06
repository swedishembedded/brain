// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Cross-attention scores (materialised, for training)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
//
// Cross-attention scores (materialised, for training):
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,h,:]) / sqrt(head_dim)   for all j
// Non-causal, with TWO sequence lengths and TWO buffers (ADR 0001 §5.1):
//   * Q comes from the DECODER buffer `q` (fused QKV layout, stride q_stride =
//     3*d_model, q region at q_off=0), indexed by query position i in [0,T_dec).
//   * K comes from the ENCODER-MEMORY buffer `kv` (fused KV layout, stride
//     kv_stride = 2*d_model, k region at k_off=0), indexed by key position j in
//     [0,T_enc).
// scores layout: ((b*H + h)*T_dec + i)*T_enc + j. One invocation per (b,h,i,j).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,        // query length
    t_enc: u32,        // key/value length
    head_dim: u32,
    q_stride: u32,     // 3*d_model (decoder fused QKV)
    kv_stride: u32,    // 2*d_model (encoder fused KV)
    q_off: u32,        // 0
    k_off: u32,        // 0
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;  // decoder buffer
@group(0) @binding(2) var<storage, read>       kv:     array<f32>;  // encoder memory
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let total = p.bsz * p.n_heads * Tq * Tk;
    let idx = gidx;
    if (idx >= total) { return; }

    let j = idx % Tk;
    let r1 = idx / Tk;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let hd = p.head_dim;
    let q_base = (b * Tq + i) * p.q_stride + p.q_off + h * hd;
    let k_base = (b * Tk + j) * p.kv_stride + p.k_off + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * kv[k_base + d];
    }
    scores[idx] = s * inverseSqrt(f32(hd));
}
