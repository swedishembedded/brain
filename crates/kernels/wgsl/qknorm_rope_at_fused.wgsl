// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused per-head RMSNorm + RoPE at an EXPLICIT absolute position over Q or K rows, single-token decode - M4.6
// @how   64-thread workgroup per head row, 1 barrier, re-read after the reduction instead of a register array
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Fuses `rmsnorm_rows` and `rope_at` into ONE dispatch - the incremental-
// decode-step sibling of `qknorm_rope_base_fused.wgsl` (M4.5, built for a
// plain forward's PREFILL calling convention, where every row's RoPE
// position is derived from its own row index because one dispatch spans
// many tokens at once). A cached-tape decode step (`qwen3tts`'s
// `TalkerGen::step`) dispatches exactly ONE new token per call - `rope_at.
// wgsl`'s own contract - so every row in this call (one row per head) shares
// the SAME absolute position, supplied as an explicit `pos_base` field
// instead of derived from the row index. `pos_base` is the one field this
// kernel's caller rewrites between decode-tape reuses, via the same
// caller-owned-uniform-buffer mechanism (`Gpu::step_buf`) `rope_at.wgsl`'s
// own `pos_base` already uses for that cached tape - see `TalkerGen::
// build_dec_cache` and `PosUniform`.
//
//   x  : [rows, head_dim]   out: [rows, head_dim]   w: [head_dim]
//   params: rows (heads in THIS buffer - n_heads for Q, n_kv_heads for K,
//           since a decode step's row count is always its head count: one
//           token, one row per head), head_dim, eps, rope_base, pos_base
//           (the absolute position of the single new token, `rope_at.wgsl`'s
//           own `pos_base` field - every row here shares it, unlike
//           `qknorm_rope_base_fused.wgsl`'s per-row-derived `pos`).
//
// Same 64-thread-workgroup-per-row / one-barrier / re-read-after-reduction
// shape as `qknorm_rope_base_fused.wgsl` - see that kernel's own header for
// the full derivation of the normalization half, unchanged here; only the
// RoPE half's position source differs (`rope_at.wgsl`'s explicit absolute
// position rather than `rope_base.wgsl`'s row-derived one).
//
// **Requires `caps.workgroup_reductions`** - the same correctness gate every
// sibling in this family carries: the caller only dispatches this kernel
// when that capability holds and falls back to the original unfused
// `rmsnorm_rows` + `rope_at` pair otherwise.
//
// Bit agreement vs the two-dispatch path: same operations, same order
// (normalize each element, then rotate the normalized pair) as
// `rmsnorm_rows.wgsl` followed by `rope_at.wgsl` - not a reassociation, so
// this is bit-identical, not merely "agrees within tolerance".

struct Params {
    rows: u32,
    head_dim: u32,
    eps: f32,
    rope_base: f32,
    pos_base: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let d = p.head_dim;
    let half = d / 2u;
    let base = row * d;
    var acc = 0.0;
    for (var c = t; c < d; c = c + 64u) {
        let v = x[base + c];
        acc = acc + v * v;
    }
    partial[t] = acc;
    workgroupBarrier();
    var ss = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        ss = ss + partial[i];
    }
    let inv = 1.0 / sqrt(ss / f32(d) + p.eps);
    let pos = p.pos_base;
    for (var m = t; m < half; m = m + 64u) {
        let x0 = x[base + m]        * inv * w[m];
        let x1 = x[base + m + half] * inv * w[m + half];
        let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(d));
        let cs = cos(angle);
        let sn = sin(angle);
        out[base + m]        = x0 * cs - x1 * sn;
        out[base + m + half] = x1 * cs + x0 * sn;
    }
}
