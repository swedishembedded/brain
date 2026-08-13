// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Attention output over ALL keys (non-causal), reading v from a SEPARATE value buffer (not a fused qkv)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Attention output over ALL keys (non-causal), reading v from a SEPARATE value
// buffer (not a fused qkv). Chronos-2 projects q/k/v into their own buffers.
//   out[b,i,h,d] = sum_{j in 0..T} probs[b,h,i,j] * v[b,j,h,d]
// v laid out [b, S, v_stride] with head h at channel offset h*head_dim; out laid
// out contiguous [b, S, d_model] (d_model = n_heads*head_dim = inner). One
// invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,       // T
    head_dim: u32,
    v_stride: u32,    // n_heads * head_dim (per-token channel stride of v)
    d_model: u32,     // = n_heads * head_dim (output stride)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       v:     array<f32>;
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * T * hd;
    if (gidx >= total) { return; }

    let d = gidx % hd;
    let r1 = gidx / hd;
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let p_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j < T; j = j + 1u) {
        let vv = v[(b * T + j) * p.v_stride + h * hd + d];
        acc = acc + probs[p_base + j] * vv;
    }
    out[(b * T + i) * p.d_model + h * hd + d] = acc;
}
