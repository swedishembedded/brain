// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Banded attention backward, step 2a - gradient w.r.t. k, accumulated over query chunks, only the keys a chunk can reach
// @how   one thread per output element, serial inner reduction over the chunk's rows that see the key
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Banded attention backward, the twin of `attn_bwd_dk_cross_acc`. A query
// chunk [q0, q0+t_dec) sees only keys [q0-window, q0+t_dec-1+window], so the
// caller dispatches just that key range: `j0` is its first key and `n_keys`
// its length, and the work per chunk is O((t_dec + 2*window) * t_dec) rather
// than O(t_enc * t_dec). For key j it sums the chunk rows i with
// |q0 + i - j| <= window:
//   d_k[b,j,h,d] += scale * sum_i d_score[b,h,i,s] * q[b,i,h,d],   s = j - q0 - i + window
// `fresh_from` replaces the dense kernel's assign-or-accumulate flag: a key
// below it was reached by an earlier chunk and accumulates; a key at or above
// it is reached for the first time and ASSIGNS, so the grad buffer needs no
// clear. The first chunk passes 0.

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    q_stride: u32,
    kv_stride: u32,
    q_off: u32,
    k_off: u32,
    q0: u32,
    window: u32,
    j0: u32,           // first key this dispatch covers (span-relative)
    n_keys: u32,       // number of keys this dispatch covers
    fresh_from: u32,   // keys j < fresh_from accumulate; j >= fresh_from assign
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       q:        array<f32>;
@group(0) @binding(3) var<storage, read_write> d_kv:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let W = 2u * p.window + 1u;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * p.n_keys * hd;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let jl = r1 % p.n_keys;
    let r2 = r1 / p.n_keys;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let j = p.j0 + jl;
    let scale = inverseSqrt(f32(hd));

    // Rows i of the chunk with |q0 + i - j| <= window.
    let i_lo = max(0, i32(j) - i32(p.window) - i32(p.q0));
    let i_hi = min(i32(Tq) - 1, i32(j) + i32(p.window) - i32(p.q0));
    var acc = 0.0;
    for (var i = i_lo; i <= i_hi; i = i + 1) {
        let s = u32(i32(j) - i32(p.q0) - i + i32(p.window));
        let sc = d_scores[((b * p.n_heads + h) * Tq + u32(i)) * W + s];
        acc = acc + sc * q[(b * Tq + u32(i)) * p.q_stride + p.q_off + h * hd + d];
    }
    let o = (b * Tk + j) * p.kv_stride + p.k_off + h * hd + d;
    let prev = select(0.0, d_kv[o], j < p.fresh_from);
    d_kv[o] = prev + acc * scale;
}
