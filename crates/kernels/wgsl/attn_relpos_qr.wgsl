// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decomposed relative-position bias, step 1 - the q·R hoist (SAM/ViTDet `add_decomposed_rel_pos`)
// @how   8-row register block per thread, coalesced on the table axis, serial reduction over head_dim
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Decomposed relative-position bias, step 1 - the q·R hoist.
//
// SAM's ViTDet attention adds a bias that DECOMPOSES over the two grid axes:
//   attn[b, (qh,qw), (kh,kw)] += q[b,(qh,qw),:] . Rh[qh,kh,:]
//                              + q[b,(qh,qw),:] . Rw[qw,kw,:]
// Both terms are functions of q, so materialising a full [heads, T, T] bias
// slab is never necessary - and at SAM ViT-B's global-attention shape
// (12 heads, 64x64 grid) it would be 805 MB. This kernel computes the two
// SMALL intermediates instead:
//   rel[h, i, k] = sum_c q[h, i, c] * R[panel(i), k, c]
// which at the same shape is 12*4096*64*4 B = 12.6 MB per axis. Fusing the
// q·R dot product into the per-(i,j) score loop would instead recompute each
// head_dim-length dot product k_ext times - a large, avoidable FLOP blowup.
//
// `axis` selects which of the two terms is being built:
//   axis = 0 (height): panel(i) = i / qw_ext   (rows sharing a panel are
//                      CONTIGUOUS, group size qw_ext), k_ext = kh_ext
//   axis = 1 (width):  panel(i) = i % qw_ext   (rows sharing a panel are
//                      STRIDED by qw_ext, group size qh_ext),  k_ext = kw_ext
// The window partition (`model::vit::WindowPlan`) already emits window-local
// ROW-MAJOR order, so `i -> (i / qw_ext, i % qw_ext)` needs no index buffer.
//
// q is read UNSCALED from the fused qkv buffer: `attn_scores_cross` applies
// `1/sqrt(head_dim)` itself, so nothing pre-scales the q region, and SAM's
// rel-pos term uses the unscaled q by definition.
//
// LAYOUTS
//   q    : fused [rows, q_stride], bound WHOLE; this span's q region starts at
//          the float offset `q_off` (= row0*q_stride + region offset), so a
//          ragged/short window needs no 256 B-aligned binding.
//   r_t  : [panel_ext, head_dim, k_ext] - the dense per-(panel,k) table
//          TRANSPOSED so `k` is the fastest axis. That is what makes the
//          dominant read here coalesced: consecutive invocations own
//          consecutive `k`. `r_t = nlc_nchw(r)` where `r` is the natural
//          [panel_ext*k_ext, head_dim] output of the interpolation gather; the
//          backward (`attn_relpos_dr`) writes `r`'s layout directly, so the
//          transpose is forward-only and needs no adjoint.
//   rel  : [heads, qh_ext*qw_ext, k_ext], SPAN-local rows.
//
// STRUCTURE. One workgroup owns (head, panel, an 8-row micro-tile of that
// panel's rows, a 64-wide tile of k). Invocation `t` owns k = kt*64 + t and
// carries EIGHT scalar accumulators - unrolled scalars, never `array<f32,8>`,
// because an indexed thread-private array is placed in local (VRAM-backed)
// memory and would run at spill bandwidth (the lesson `conv_act_reg` records).
// Per step of the head_dim loop: ONE coalesced `r_t` read feeds 8 FMAs, and
// the 8 `q` reads are workgroup-uniform broadcasts. So `r_t` is touched once
// per 8 rows and `q` once per k-tile, instead of once per output.
//
// Dispatch: heads * panel_ext * ceil(group_len/8) * ceil(k_ext/64) workgroups
// (thread count = that * 64).

struct Params {
    heads: u32,
    qh_ext: u32,       // query grid height of this span
    qw_ext: u32,       // query grid width of this span
    k_ext: u32,        // kh_ext on axis 0, kw_ext on axis 1
    head_dim: u32,
    q_stride: u32,     // fused qkv row stride (3*C for a ViT block)
    q_off: u32,        // float offset of this span's q region in `q`
    axis: u32,         // 0 = height, 1 = width
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       r_t: array<f32>;
@group(0) @binding(3) var<storage, read_write> rel: array<f32>;

const BM: u32 = 8u;   // rows per invocation (the register block)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch), split back
    // into (workgroup, lane). The tile below is a REGISTER block, not a shared-
    // memory one, so nothing here needs `workgroup_id`/`local_invocation_id` -
    // and the CPU JIT's independent-invocation path (which is what a
    // barrier-free kernel takes) only provides `global_invocation_id`.
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let wg = gidx / 64u;
    let t = gidx % 64u;

    let panel_ext = select(p.qw_ext, p.qh_ext, p.axis == 0u);
    let group_len = select(p.qh_ext, p.qw_ext, p.axis == 0u);
    let ktiles = (p.k_ext + 63u) / 64u;
    let rtiles = (group_len + BM - 1u) / BM;

    let kt = wg % ktiles;
    let w1 = wg / ktiles;
    let rt = w1 % rtiles;
    let w2 = w1 / rtiles;
    let pnl = w2 % panel_ext;
    let h = w2 / panel_ext;
    if (h >= p.heads) { return; }

    let k = kt * 64u + t;
    if (k >= p.k_ext) { return; }

    let hd = p.head_dim;
    let g0 = rt * BM;
    let gl1 = group_len - 1u;
    // Row of member `m`: axis 0 walks a contiguous run, axis 1 a strided one.
    let rb = select(pnl, pnl * p.qw_ext, p.axis == 0u);
    let rs = select(p.qw_ext, 1u, p.axis == 0u);
    // Out-of-range members are CLAMPED (they compute a duplicate row) rather
    // than branched, and their stores are guarded below.
    let b0 = p.q_off + (rb + min(g0 + 0u, gl1) * rs) * p.q_stride + h * hd;
    let b1 = p.q_off + (rb + min(g0 + 1u, gl1) * rs) * p.q_stride + h * hd;
    let b2 = p.q_off + (rb + min(g0 + 2u, gl1) * rs) * p.q_stride + h * hd;
    let b3 = p.q_off + (rb + min(g0 + 3u, gl1) * rs) * p.q_stride + h * hd;
    let b4 = p.q_off + (rb + min(g0 + 4u, gl1) * rs) * p.q_stride + h * hd;
    let b5 = p.q_off + (rb + min(g0 + 5u, gl1) * rs) * p.q_stride + h * hd;
    let b6 = p.q_off + (rb + min(g0 + 6u, gl1) * rs) * p.q_stride + h * hd;
    let b7 = p.q_off + (rb + min(g0 + 7u, gl1) * rs) * p.q_stride + h * hd;

    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
    let rbase = pnl * hd * p.k_ext + k;
    for (var c: u32 = 0u; c < hd; c = c + 1u) {
        let rv = r_t[rbase + c * p.k_ext];
        a0 = a0 + q[b0 + c] * rv;
        a1 = a1 + q[b1 + c] * rv;
        a2 = a2 + q[b2 + c] * rv;
        a3 = a3 + q[b3 + c] * rv;
        a4 = a4 + q[b4 + c] * rv;
        a5 = a5 + q[b5 + c] * rv;
        a6 = a6 + q[b6 + c] * rv;
        a7 = a7 + q[b7 + c] * rv;
    }

    let obase = h * p.qh_ext * p.qw_ext;
    if (g0 + 0u < group_len) { rel[(obase + rb + (g0 + 0u) * rs) * p.k_ext + k] = a0; }
    if (g0 + 1u < group_len) { rel[(obase + rb + (g0 + 1u) * rs) * p.k_ext + k] = a1; }
    if (g0 + 2u < group_len) { rel[(obase + rb + (g0 + 2u) * rs) * p.k_ext + k] = a2; }
    if (g0 + 3u < group_len) { rel[(obase + rb + (g0 + 3u) * rs) * p.k_ext + k] = a3; }
    if (g0 + 4u < group_len) { rel[(obase + rb + (g0 + 4u) * rs) * p.k_ext + k] = a4; }
    if (g0 + 5u < group_len) { rel[(obase + rb + (g0 + 5u) * rs) * p.k_ext + k] = a5; }
    if (g0 + 6u < group_len) { rel[(obase + rb + (g0 + 6u) * rs) * p.k_ext + k] = a6; }
    if (g0 + 7u < group_len) { rel[(obase + rb + (g0 + 7u) * rs) * p.k_ext + k] = a7; }
}
