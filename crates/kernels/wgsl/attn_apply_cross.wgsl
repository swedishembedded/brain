// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Cross-attention output: out[b,i,h,d] = sum_{j<T_enc} probs[b,h,i,j] * v[b,j,h,d].
// V comes from the ENCODER-MEMORY buffer `kv` (fused KV layout, stride kv_stride =
// 2*d_model, v region at v_off = d_model). out written contiguous [B*T_dec, d_model].
// probs layout: ((b*H + h)*T_dec + i)*T_enc + j. One invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    kv_stride: u32,    // 2*d_model
    v_off: u32,        // d_model
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       kv:    array<f32>;  // encoder memory
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * Tq * hd;
    let idx = gidx;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let p_base = ((b * p.n_heads + h) * Tq + i) * Tk;
    var acc = 0.0;
    for (var j: u32 = 0u; j < Tk; j = j + 1u) {
        let v = kv[(b * Tk + j) * p.kv_stride + p.v_off + h * hd + d];
        acc = acc + probs[p_base + j] * v;
    }
    out[(b * Tq + i) * p.d_model + h * hd + d] = acc;
}
