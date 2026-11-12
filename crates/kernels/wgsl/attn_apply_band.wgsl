// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Banded attention output: out[b,i,h,d] = sum over the 2*window+1 slots of probs * v
// @how   one thread per output element, serial inner reduction over the band
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Banded attention output, the twin of `attn_apply_cross` for the
// [rows, 2*window+1] slab `attn_scores_band` + `attn_softmax_cross` produce:
//   out[b,i,h,d] = sum_{s < 2*window+1, 0 <= j < t_enc} probs[b,h,i,s] * v[b,j,h,d],  j = q0 + i + s - window
// Slots whose key is outside the span carry probability 0 and are skipped
// rather than read. V comes from the fused buffer `kv` (v region at v_off);
// out is written contiguous [B*T_dec, d_model]. One invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    kv_stride: u32,
    v_off: u32,
    d_model: u32,
    q0: u32,           // this chunk's first query row, within the span
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       kv:    array<f32>;
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let W = 2u * p.window + 1u;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * Tq * hd;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let p_base = ((b * p.n_heads + h) * Tq + i) * W;
    let j_left = i32(p.q0 + i) - i32(p.window);
    var acc = 0.0;
    for (var s: u32 = 0u; s < W; s = s + 1u) {
        let j = j_left + i32(s);
        if (j < 0 || j >= i32(Tk)) { continue; }
        acc = acc + probs[p_base + s] * kv[(b * Tk + u32(j)) * p.kv_stride + p.v_off + h * hd + d];
    }
    out[(b * Tq + i) * p.d_model + h * hd + d] = acc;
}
