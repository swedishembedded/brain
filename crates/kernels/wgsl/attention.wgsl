// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal multi-head attention with online (numerically stable) softmax
// @how   one thread per output element, 4 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Causal multi-head attention with online (numerically stable) softmax.
// One invocation per (token t, head h): it streams over keys s = 0..=t,
// accumulating the softmax-weighted sum of values. No score matrix is
// materialised and no atomics are used, so it runs everywhere.
//
// q/k/v are read from the fused qkv buffer (row_stride = 3*d_model); the
// result is written to a contiguous [seq_len, d_model] buffer.

const MAX_HEAD_DIM: u32 = 128u;

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.seq_len * p.n_heads;
    if (idx >= total) { return; }

    let h = idx % p.n_heads;
    let t = idx / p.n_heads;
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    var acc: array<f32, 128>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = 0.0; }

    var m = -3.4e38;   // running max logit
    var l = 0.0;       // running denominator

    let q_base = t * p.qkv_stride + p.q_off + h * hd;
    for (var s: u32 = 0u; s <= t; s = s + 1u) {
        let k_base = s * p.qkv_stride + p.k_off + h * hd;
        var score = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            score = score + qkv[q_base + d] * qkv[k_base + d];
        }
        score = score * scale;

        let new_m = max(m, score);
        let corr = exp(m - new_m);     // 0 on the first step (m = -inf)
        let pe = exp(score - new_m);
        l = l * corr + pe;

        let v_base = s * p.qkv_stride + p.v_off + h * hd;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            acc[d] = acc[d] * corr + pe * qkv[v_base + d];
        }
        m = new_m;
    }

    let o_base = t * p.d_model + h * hd;
    let inv_l = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        out[o_base + d] = acc[d] * inv_l;
    }
}
