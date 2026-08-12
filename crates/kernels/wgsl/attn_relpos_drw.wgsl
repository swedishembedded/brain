// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decomposed relative-position bias backward - d(rel_w), the STRIDED column sums of d_scores
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Decomposed relative-position bias backward - d(rel_w).
//
// The bias is purely ADDITIVE (`attn_relpos_add`), so the adjoint of the fold
// is just a pair of partial sums of `d_scores`, with no softmax/scale factor:
//
//   d_rel_w[h, q0+i, kw] = sum_kh d_scores[h, i, kh*kw_ext + kw]
//
// i.e. the COLUMN sums of each query row's [kh_ext, kw_ext] block. One thread
// per output, inner loop over `kh` with stride `kw_ext`: consecutive
// invocations own consecutive `kw` and therefore read consecutive addresses at
// every step of the reduction, so this is coalesced without a workgroup tile.
// (Its sibling `attn_relpos_drh` sums the OTHER axis, whose segments are
// contiguous, and needs a cooperative reduction to stay coalesced.)
//
// ASSIGNS, and deliberately has no `acc` flag. `d_rel_w[h, row, :]` depends on
// exactly one query row, and query CHUNKS partition the rows - so every chunk
// writes rows no other chunk touches, and spans each own their own slab. An
// `acc` flag here would be an unused parameter that the next reader has to
// prove is dead. (The table adjoint `attn_relpos_dr`, which DOES sum over every
// head, row and span, is where the accumulate flag lives.)
//
// d_scores layout: ((b*H + h)*qn + i)*kn + j, b == 1, j = kh*kw_ext + kw.
// d_rel_w  layout: (h*span_qn + row)*kw_ext + kw.

struct Params {
    heads: u32,
    qn: u32,           // this chunk's query rows
    kn: u32,           // span key count == kh_ext*kw_ext
    q0: u32,           // chunk's span-local first query row
    span_qn: u32,      // d_rel_w row stride (the span's full query rows)
    kh_ext: u32,
    kw_ext: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_rel_w:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.heads * p.qn * p.kw_ext;
    if (idx >= total) { return; }

    let kw = idx % p.kw_ext;
    let r1 = idx / p.kw_ext;
    let i = r1 % p.qn;
    let h = r1 / p.qn;

    let base = (h * p.qn + i) * p.kn + kw;
    var acc = 0.0;
    for (var kh: u32 = 0u; kh < p.kh_ext; kh = kh + 1u) {
        acc = acc + d_scores[base + kh * p.kw_ext];
    }
    d_rel_w[(h * p.span_qn + p.q0 + i) * p.kw_ext + kw] = acc;
}
