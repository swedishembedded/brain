// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Cross-attention scores with a BIDIRECTIONAL sliding-window mask - `attn_scores_cross` plus `abs(i-j) <= window`
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Cross-attention scores with a BIDIRECTIONAL sliding-window mask - the
// non-causal twin of `gqa_scores_win`'s causal one, for models like
// ModernBERT that alternate full attention with a local window in BOTH
// directions (key `j` lives for query `i` whenever `|i-j| <= window`, not
// just when `j` precedes `i`):
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,h,:]) / sqrt(head_dim)   for |i-j| <= window
//                   = -inf                                          otherwise
// `window >= max(t_dec, t_enc)` degenerates to `attn_scores_cross`'s plain
// unwindowed mask exactly, so this is a strict generalization kept as a
// separate kernel rather than a parameter bolted onto the unwindowed one -
// same convention as `gqa_scores_win` vs `gqa_scores`.
//
// `i` is CHUNK-relative (queries `[q0, q0+t_dec)` of a caller's span); `j` is
// SPAN-relative (keys `[0, t_enc)` of that same span). The mask therefore
// compares the two GLOBAL positions `q0 + i` and `j`, which is why `q0`
// exists here and not on the unwindowed kernel. `attn_softmax_cross` and
// `attn_apply_cross` are reused unchanged: `-3.4e38` exponentiates to 0, and
// a symmetric window always keeps `j == q0+i` live, so no row is ever fully
// masked.
//
// Same fused-KV, key-major layout `attn_scores_cross` reads - see that
// kernel's own doc for why the key index runs fastest and why a caller with
// somewhere to put a transpose should prefer `attn_scores_cross_kt_win`
// instead.

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,        // query length (this chunk)
    t_enc: u32,        // key/value length (this span)
    head_dim: u32,
    q_stride: u32,     // 3*d_model (decoder fused QKV)
    kv_stride: u32,    // 2*d_model (encoder fused KV)
    q_off: u32,        // 0
    k_off: u32,        // 0
    q0: u32,           // this chunk's first query row, within the span
    window: u32,       // key j lives iff |(q0+i) - j| <= window
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

    let gi = p.q0 + i;
    let dist = select(gi - j, j - gi, j > gi);
    if (dist > p.window) { scores[idx] = -3.4e38; return; }

    let hd = p.head_dim;
    let q_base = (b * Tq + i) * p.q_stride + p.q_off + h * hd;
    let k_base = (b * Tk + j) * p.kv_stride + p.k_off + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * kv[k_base + d];
    }
    scores[idx] = s * inverseSqrt(f32(hd));
}
