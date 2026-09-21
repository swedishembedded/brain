// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Windowed cross-attention scores against a key-minor K (`kv_k_headt` output) - same math as attn_scores_cross_win, coalesced
// @how   one thread per output element, serial inner reduction over head_dim
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The coalesced twin of `attn_scores_cross_win`, exactly as
// `attn_scores_cross_kt` is the coalesced twin of `attn_scores_cross`: same
// bidirectional sliding-window mask (`|q0+i - j| <= window`), same output
// values and layout, but K is read from the KEY-MINOR `[d_model, T_enc]`
// buffer `kv_k_headt.wgsl` produces instead of from the fused KV slab - see
// `attn_scores_cross_kt`'s own doc for why that matters for coalescing.
//
// This is the rung `crates/decide`'s own CPU-safe fallback dispatches
// through today (`Encoder::build_steps`'s `Err(None)` arm always passes
// `Some(&km)`), so for a model whose materialized rung is windowed, THIS
// kernel - not the plain fused-KV one - is the one actually exercised by a
// GPU-less test run.

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,        // query length (this chunk)
    t_enc: u32,        // key length (this span)
    head_dim: u32,
    q_stride: u32,     // 3*d_model (decoder fused QKV)
    q_off: u32,        // 0
    q0: u32,           // this chunk's first query row, within the span
    window: u32,       // key j lives iff |(q0+i) - j| <= window
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;  // decoder buffer
@group(0) @binding(2) var<storage, read>       kt:     array<f32>;  // [d_model, T_enc]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let total = p.bsz * p.n_heads * Tq * Tk;
    if (gidx >= total) { return; }

    let j = gidx % Tk;
    let r1 = gidx / Tk;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let gi = p.q0 + i;
    let dist = select(gi - j, j - gi, j > gi);
    if (dist > p.window) { scores[gidx] = -3.4e38; return; }

    let hd = p.head_dim;
    let q_base = (b * Tq + i) * p.q_stride + p.q_off + h * hd;
    // Row of `kt` this head's first component lives on; +1 row per d.
    var k_row = h * hd * Tk + j;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * kt[k_row];
        k_row = k_row + Tk;
    }
    scores[gidx] = s * inverseSqrt(f32(hd));
}
