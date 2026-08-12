// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decomposed relative-position bias, step 2 - fold the two q·R intermediates into the score slab
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Decomposed relative-position bias, step 2 - fold the hoisted intermediates
// into an already-computed score slab, IN PLACE:
//
//   scores[h, i, j] += rel_h[h, q0+i, j / kw_ext] + rel_w[h, q0+i, j % kw_ext]
//
// `rel_h`/`rel_w` come from `attn_relpos_qr` (axis 0 / axis 1) and cover the
// WHOLE span; `scores` covers one query CHUNK of it, so `q0` (the chunk's
// span-local first query row) and `span_qn` (the rel intermediates' own row
// stride) are what tie the two together. That is the only extra parameter a
// query-chunked dispatch needs - the window partition emits window-local
// row-major order, so the grid position of a span-local row is arithmetic.
//
// SAM's ordering: `scores` already holds `scale*(q@k^T)` (the `1/sqrt(head_dim)`
// is applied by `attn_scores_cross`), and the rel-pos term added here is built
// from UNSCALED q. Run this between the scores kernel and the softmax.
//
// Bandwidth-bound and fully coalesced: consecutive invocations own consecutive
// `j`, so both the read-modify-write of `scores` and the `rel_w` read are
// contiguous, and the `rel_h` read is a broadcast over each run of kw_ext
// invocations. One thread per score, no inner loop.
//
// scores layout: ((b*H + h)*qn + i)*kn + j, b == 1 - the same slab
// `attn_scores_cross` writes, bound at offset 0.
// rel_h layout:  (h*span_qn + row)*kh_ext + kh
// rel_w layout:  (h*span_qn + row)*kw_ext + kw
// Requires kn == kh_ext*kw_ext (the span's keys in grid row-major order).

struct Params {
    heads: u32,
    qn: u32,           // this chunk's query rows
    kn: u32,           // span key count == kh_ext*kw_ext
    q0: u32,           // chunk's span-local first query row
    span_qn: u32,      // rel_h/rel_w row stride (the span's full query rows)
    kh_ext: u32,
    kw_ext: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       rel_h:  array<f32>;
@group(0) @binding(2) var<storage, read>       rel_w:  array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.heads * p.qn * p.kn;
    if (idx >= total) { return; }

    let j = idx % p.kn;
    let r1 = idx / p.kn;
    let i = r1 % p.qn;
    let h = r1 / p.qn;

    let row = h * p.span_qn + p.q0 + i;
    let kh = j / p.kw_ext;
    let kw = j % p.kw_ext;
    scores[idx] = scores[idx] + rel_h[row * p.kh_ext + kh] + rel_w[row * p.kw_ext + kw];
}
