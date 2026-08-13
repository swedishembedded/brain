// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decomposed relative-position bias backward - the d(q) contribution, ACCUMULATED into the fused d_qkv
// @how   8-row register block per thread, coalesced on head_dim, serial reduction over k_ext
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Decomposed relative-position bias backward - the d(q) contribution.
//
// The GEMM-transpose of `attn_relpos_qr`: where the forward contracted `q`
// against the table over head_dim, this contracts the intermediate's gradient
// against the same table over k_ext.
//
//   d_q[h, i, c] += sum_k d_rel[h, i, k] * R[panel(i), k, c]
//
// Run it TWICE per span, `axis = 0` with (d_rel_h, Rh) and `axis = 1` with
// (d_rel_w, Rw); the two contributions sum into the same q region.
//
// ALWAYS ACCUMULATES (`+=`), and has no flag to do otherwise. The plain
// `q@k^T` backward (`attn_bwd_dq_cross`) ASSIGNS the same rows and runs first;
// this is the second term of the same derivative, not an alternative to it.
// A caller that dispatched this without the base term first would be adding to
// whatever the buffer already held - the q region of `d_qkv` is therefore a
// hard ordering precondition, not a zero-clear one.
//
// LAYOUTS
//   d_rel : [heads, qh_ext*qw_ext, k_ext] - `attn_relpos_qr`'s output shape,
//           filled by `attn_relpos_drh`/`_drw`. Dispatch this once per SPAN
//           after the query-chunk loop has filled every row.
//   r     : [panel_ext, k_ext, head_dim] - the NATURAL (untransposed) dense
//           table, so consecutive invocations (consecutive `c`) read
//           consecutive addresses. The forward reads the transposed twin
//           `r_t = nlc_nchw(r)` for the same reason on its own fastest axis;
//           holding both costs one small extra copy of a table that is at most
//           [64, 64, 64] at SAM ViT-B's global-attention shape.
//   d_qkv : fused [rows, q_stride], bound WHOLE, q region at float offset
//           `q_off` (= row0*q_stride + region offset).
//
// STRUCTURE mirrors `attn_relpos_qr`: one workgroup owns (head, panel, an
// 8-row micro-tile, a 64-wide tile of head_dim); invocation `t` owns
// c = ct*64 + t and carries eight UNROLLED scalar accumulators (never an
// indexed private array - that lands in local memory and runs at spill
// bandwidth). Per step of the k_ext loop: one coalesced `r` read feeds 8 FMAs,
// and the 8 `d_rel` reads are workgroup-uniform broadcasts.
//
// Dispatch: heads * panel_ext * ceil(group_len/8) * ceil(head_dim/64)
// workgroups (thread count = that * 64).

struct Params {
    heads: u32,
    qh_ext: u32,
    qw_ext: u32,
    k_ext: u32,        // kh_ext on axis 0, kw_ext on axis 1
    head_dim: u32,
    q_stride: u32,
    q_off: u32,
    axis: u32,         // 0 = height, 1 = width
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_rel: array<f32>;
@group(0) @binding(2) var<storage, read>       r:     array<f32>;
@group(0) @binding(3) var<storage, read_write> d_qkv: array<f32>;

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
    let hd = p.head_dim;
    let ctiles = (hd + 63u) / 64u;
    let rtiles = (group_len + BM - 1u) / BM;

    let ct = wg % ctiles;
    let w1 = wg / ctiles;
    let rt = w1 % rtiles;
    let w2 = w1 / rtiles;
    let pnl = w2 % panel_ext;
    let h = w2 / panel_ext;
    if (h >= p.heads) { return; }

    let c = ct * 64u + t;
    if (c >= hd) { return; }

    let g0 = rt * BM;
    let gl1 = group_len - 1u;
    let rb = select(pnl, pnl * p.qw_ext, p.axis == 0u);
    let rs = select(p.qw_ext, 1u, p.axis == 0u);
    let obase = h * p.qh_ext * p.qw_ext;
    // Out-of-range members are CLAMPED (duplicate row) and their stores guarded.
    let d0 = (obase + rb + min(g0 + 0u, gl1) * rs) * p.k_ext;
    let d1 = (obase + rb + min(g0 + 1u, gl1) * rs) * p.k_ext;
    let d2 = (obase + rb + min(g0 + 2u, gl1) * rs) * p.k_ext;
    let d3 = (obase + rb + min(g0 + 3u, gl1) * rs) * p.k_ext;
    let d4 = (obase + rb + min(g0 + 4u, gl1) * rs) * p.k_ext;
    let d5 = (obase + rb + min(g0 + 5u, gl1) * rs) * p.k_ext;
    let d6 = (obase + rb + min(g0 + 6u, gl1) * rs) * p.k_ext;
    let d7 = (obase + rb + min(g0 + 7u, gl1) * rs) * p.k_ext;

    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
    let rbase = pnl * p.k_ext * hd + c;
    for (var k: u32 = 0u; k < p.k_ext; k = k + 1u) {
        let rv = r[rbase + k * hd];
        a0 = a0 + d_rel[d0 + k] * rv;
        a1 = a1 + d_rel[d1 + k] * rv;
        a2 = a2 + d_rel[d2 + k] * rv;
        a3 = a3 + d_rel[d3 + k] * rv;
        a4 = a4 + d_rel[d4 + k] * rv;
        a5 = a5 + d_rel[d5 + k] * rv;
        a6 = a6 + d_rel[d6 + k] * rv;
        a7 = a7 + d_rel[d7 + k] * rv;
    }

    let qb = p.q_off + h * hd + c;
    if (g0 + 0u < group_len) { let o = qb + (rb + (g0 + 0u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a0; }
    if (g0 + 1u < group_len) { let o = qb + (rb + (g0 + 1u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a1; }
    if (g0 + 2u < group_len) { let o = qb + (rb + (g0 + 2u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a2; }
    if (g0 + 3u < group_len) { let o = qb + (rb + (g0 + 3u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a3; }
    if (g0 + 4u < group_len) { let o = qb + (rb + (g0 + 4u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a4; }
    if (g0 + 5u < group_len) { let o = qb + (rb + (g0 + 5u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a5; }
    if (g0 + 6u < group_len) { let o = qb + (rb + (g0 + 6u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a6; }
    if (g0 + 7u < group_len) { let o = qb + (rb + (g0 + 7u) * rs) * p.q_stride; d_qkv[o] = d_qkv[o] + a7; }
}
