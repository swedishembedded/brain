// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decomposed relative-position bias, step 2 - fold the two q·R intermediates into the score slab
// @how   JB=8 keys per thread (register block), coalesced within the block
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
// ONE THREAD PER SCORE was the original shape, and at DeepSeek-OCR's real
// global-attention shape it dispatched an oversized grid: `heads=12`, the
// whole-span key count `kn=4096`, and a chunked query count `qn=256` give
// `heads*qn*kn = 12 582 912` threads, past `backend_api::MAX_GROUPS_PER_DIM`'s
// 65 535-workgroup-per-dimension limit at `@workgroup_size(64)` - so the
// dispatch tiled into a 2D grid (`gid.y*(nwg.x*64u)+gid.x`), and a chunked
// global block issued 16 such tiled dispatches to this SAME read-write
// `scores` buffer inside one flush. Blocking `JB` keys per invocation below
// divides the dispatch width by `JB`, which keeps every SAM-1 config this
// repo defines under the tiling threshold entirely.
//
// This is a MITIGATION, not a confirmed fix, and the difference matters: the
// SAM-1 tower's wgpu backend still corrupts unrelated device buffers at this
// shape intermittently (not on every run), with every corrupted run landing
// on the SAME specific wrong value rather than varying noise - the signature
// of a missing synchronization barrier somewhere in how many large dispatches
// queued in one submit get scheduled, not of an index formula that is simply
// wrong. Narrowing this dispatch's width changed the failure rate but did not
// eliminate it (`crates/sam1/tests/wgpu_block_count_corruption.rs` measured
// 2/5 clean runs after this change landed). `backend-cpu` has no per-dimension
// dispatch-count limit, never took the tiled path, and has never shown the
// defect in any run - consistent with, but not sole proof of, a wgpu/driver
// scheduling race rather than a kernel-side bug. The real synchronization gap
// is still unidentified; see that test file for the measured evidence.
//
// Bandwidth-bound: each invocation's own `JB` keys are read/written
// contiguously (`scores`, `rel_w`, `rel_h`). Consecutive invocations own
// consecutive KEY BLOCKS, so the access pattern across a workgroup remains
// coalesced at `JB`-element granularity.
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

// Keys per invocation. `crate::vit::RelPos::ADD_JB` on the host side derives
// the matching dispatch width (`ceil(kn/ADD_JB)` key-blocks) - the two MUST
// agree, since the dispatch bounds check below (`idx >= total`) is computed
// from the same division.
const JB: u32 = 8u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let kn_blocks = (p.kn + JB - 1u) / JB;
    let total = p.heads * p.qn * kn_blocks;
    if (idx >= total) { return; }

    let jt = idx % kn_blocks;
    let r1 = idx / kn_blocks;
    let i = r1 % p.qn;
    let h = r1 / p.qn;

    let row = h * p.span_qn + p.q0 + i;
    let base = (h * p.qn + i) * p.kn;
    let j0 = jt * JB;

    for (var m: u32 = 0u; m < JB; m = m + 1u) {
        let j = j0 + m;
        if (j >= p.kn) { break; }
        let kh = j / p.kw_ext;
        let kw = j % p.kw_ext;
        let s = base + j;
        scores[s] = scores[s] + rel_h[row * p.kh_ext + kh] + rel_w[row * p.kw_ext + kw];
    }
}
