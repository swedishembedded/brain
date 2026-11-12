// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Banded sliding-window attention scores over a [rows, 2*window+1] slab (slot s holds key q0+i+s-window)
// @how   one thread per slab element, serial inner reduction over head_dim
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Banded sliding-window attention scores for a BIDIRECTIONAL local window.
// Where `attn_scores_cross_win` writes a [rows, T_enc] slab and fills the
// outside of the band with a sentinel (so everything after it still touches
// T_enc entries per row), this writes only the 2*window+1 slots a row can see:
//   scores[b,h,i,s] = (q[b,i,h,:] . k[b,j,h,:]) / sqrt(head_dim),  j = q0 + i + s - window
// Slot `s` is the key's offset from the left edge of the band, so the slab is
// [bsz*n_heads*t_dec, 2*window+1] and a row never spans more than the window.
// A slot whose key falls outside [0, t_enc) (the first and last `window` rows
// of a span) reads -3.4e38, which `attn_softmax_cross` run with `t_enc =
// 2*window+1` exponentiates to 0; the centre slot s == window is always a live
// key, so no row is fully masked.
//
// `i` is CHUNK-relative (queries [q0, q0+t_dec) of a span), `j` SPAN-relative,
// the same convention as `attn_scores_cross_win`, whose output this equals on
// the live slots. Fused-KV, key-major layout as `attn_scores_cross`.

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,        // query rows in this chunk
    t_enc: u32,        // keys in this span
    head_dim: u32,
    q_stride: u32,     // 3*d_model (fused QKV)
    kv_stride: u32,    // 3*d_model (fused QKV)
    q_off: u32,
    k_off: u32,
    q0: u32,           // this chunk's first query row, within the span
    window: u32,       // per-side radius; the slab is 2*window+1 wide
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       kv:     array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let W = 2u * p.window + 1u;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let total = p.bsz * p.n_heads * Tq * W;
    if (idx >= total) { return; }

    let s = idx % W;
    let r1 = idx / W;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let j = i32(p.q0 + i) + i32(s) - i32(p.window);
    if (j < 0 || j >= i32(Tk)) { scores[idx] = -3.4e38; return; }

    let hd = p.head_dim;
    let q_base = (b * Tq + i) * p.q_stride + p.q_off + h * hd;
    let k_base = (b * Tk + u32(j)) * p.kv_stride + p.k_off + h * hd;
    var acc = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        acc = acc + q[q_base + d] * kv[k_base + d];
    }
    scores[idx] = acc * inverseSqrt(f32(hd));
}
