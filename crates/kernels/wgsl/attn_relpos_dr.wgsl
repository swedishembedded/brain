// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decomposed relative-position bias backward - the dense-table adjoint, accumulated over heads/rows/spans
// @how   8-column register block per thread, coalesced on head_dim, 2 nested serial reductions
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Decomposed relative-position bias backward - the DENSE TABLE adjoint.
//
//   d_R[panel, k, c] += sum_h sum_{i : panel(i) == panel} d_rel[h, i, k] * q[h, i, c]
//
// This is the one piece of the rel-pos backward that genuinely needs its own
// kernel rather than a composition. Expressed with the generic `matmul_dw` it
// is one tiny GEMM PER PANEL - 64 dispatches per axis per block at SAM ViT-B's
// global-attention shape, 12 288 per forward across the tower - which is the
// dispatch explosion this repo's kernel rules name as sufficient justification.
// Here one dispatch covers every panel.
//
// ACCUMULATES under `acc`, following the repo's `_acc` convention: `0` assigns
// (the first span), `1` adds. Unlike the two intermediate adjoints
// (`attn_relpos_drh`/`_drw`, which assign because query chunks partition the
// rows), this output IS shared - every window of a windowed block reads the
// SAME table, so every span, every head and every query row folds into it. A
// dropped accumulate here is a *partial* gradient error, the shape a directional
// finite-difference check is measurably blind to; `gradcheck::deepseekocr`'s
// per-ENTRY check is what covers it.
//
// LAYOUTS
//   d_rel : [heads, qh_ext*qw_ext, k_ext] (`attn_relpos_qr`'s output shape).
//   q     : fused [rows, q_stride], bound WHOLE, q region at float offset
//           `q_off`. UNSCALED, as in the forward.
//   d_r   : [panel_ext, k_ext, head_dim] - the NATURAL layout, i.e. exactly
//           [panel_ext*k_ext, head_dim] rows of the interpolation gather, so
//           the remaining chain to the learned table is
//           `scale_row` + `emb_bwd` with no permute and no adjoint of the
//           forward's `nlc_nchw` transpose.
//
// STRUCTURE. One workgroup owns (panel, an 8-wide micro-tile of k, a 64-wide
// tile of head_dim). Invocation `t` owns c = ct*64 + t and carries eight
// UNROLLED scalar accumulators (an indexed private array would be placed in
// local memory and run at spill bandwidth). The (head, row) loop reads ONE
// coalesced `q` value per step and broadcasts 8 `d_rel` values, so `q` is
// touched once per k-tile instead of once per output.
//
// Dispatch: panel_ext * ceil(k_ext/8) * ceil(head_dim/64) workgroups
// (thread count = that * 64).

struct Params {
    heads: u32,
    qh_ext: u32,
    qw_ext: u32,
    k_ext: u32,        // kh_ext on axis 0, kw_ext on axis 1
    head_dim: u32,
    q_stride: u32,
    q_off: u32,
    axis: u32,         // 0 = height, 1 = width
    acc: u32,          // 0 = assign (first span), 1 = accumulate
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_rel: array<f32>;
@group(0) @binding(2) var<storage, read>       q:     array<f32>;
@group(0) @binding(3) var<storage, read_write> d_r:   array<f32>;

const BK: u32 = 8u;   // table rows per invocation (the register block)

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
    let ktiles = (p.k_ext + BK - 1u) / BK;

    let ct = wg % ctiles;
    let w1 = wg / ctiles;
    let kt = w1 % ktiles;
    let pnl = w1 / ktiles;
    if (pnl >= panel_ext) { return; }

    let c = ct * 64u + t;
    if (c >= hd) { return; }

    let k0 = kt * BK;
    let ke1 = p.k_ext - 1u;
    let e0 = min(k0 + 0u, ke1); let e1 = min(k0 + 1u, ke1);
    let e2 = min(k0 + 2u, ke1); let e3 = min(k0 + 3u, ke1);
    let e4 = min(k0 + 4u, ke1); let e5 = min(k0 + 5u, ke1);
    let e6 = min(k0 + 6u, ke1); let e7 = min(k0 + 7u, ke1);

    let rb = select(pnl, pnl * p.qw_ext, p.axis == 0u);
    let rs = select(p.qw_ext, 1u, p.axis == 0u);
    let qn = p.qh_ext * p.qw_ext;

    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
    for (var h: u32 = 0u; h < p.heads; h = h + 1u) {
        for (var g: u32 = 0u; g < group_len; g = g + 1u) {
            let row = rb + g * rs;
            let qv = q[p.q_off + row * p.q_stride + h * hd + c];
            let db = (h * qn + row) * p.k_ext;
            a0 = a0 + d_rel[db + e0] * qv;
            a1 = a1 + d_rel[db + e1] * qv;
            a2 = a2 + d_rel[db + e2] * qv;
            a3 = a3 + d_rel[db + e3] * qv;
            a4 = a4 + d_rel[db + e4] * qv;
            a5 = a5 + d_rel[db + e5] * qv;
            a6 = a6 + d_rel[db + e6] * qv;
            a7 = a7 + d_rel[db + e7] * qv;
        }
    }

    let ob = pnl * p.k_ext * hd + c;
    if (k0 + 0u < p.k_ext) { let o = ob + (k0 + 0u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a0; }
    if (k0 + 1u < p.k_ext) { let o = ob + (k0 + 1u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a1; }
    if (k0 + 2u < p.k_ext) { let o = ob + (k0 + 2u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a2; }
    if (k0 + 3u < p.k_ext) { let o = ob + (k0 + 3u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a3; }
    if (k0 + 4u < p.k_ext) { let o = ob + (k0 + 4u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a4; }
    if (k0 + 5u < p.k_ext) { let o = ob + (k0 + 5u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a5; }
    if (k0 + 6u < p.k_ext) { let o = ob + (k0 + 6u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a6; }
    if (k0 + 7u < p.k_ext) { let o = ob + (k0 + 7u) * hd; d_r[o] = select(0.0, d_r[o], p.acc == 1u) + a7; }
}
