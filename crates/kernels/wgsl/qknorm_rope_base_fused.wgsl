// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused per-head RMSNorm + batched RoPE (half-split, configurable base theta) over Q or K rows, non-paged - M4.5
// @how   64-thread workgroup per (token, head) row, 1 barrier, re-read after the reduction instead of a register array
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Fuses `rmsnorm_rows` and `rope_base` into ONE dispatch - the non-paged
// sibling of `qknorm_rope_fused.wgsl` (M4.2, built for `qwen3::serve`'s
// paged continuous-batching engine, which reads one absolute position per
// row from a `positions` buffer). A plain generation engine like
// `qwen3tts`'s Talker has no such buffer: its unfused `rmsnorm_fwd` +
// `rope_base.wgsl` pair computes each row's position directly from its own
// row index (`pos = row % tcols`, `rope_base.wgsl`'s own contract) instead
// of a lookup, so this fused sibling reproduces that arithmetic exactly
// rather than requiring a `positions` buffer this caller does not have.
//
//   x  : [rows, head_dim]   out: [rows, head_dim]   w: [head_dim]
//   params: rows, heads (rows-per-token, i.e. n_heads or n_kv_heads - the
//           SAME (token, head) flattening `rmsnorm_rows`'s own row count
//           already assumes), head_dim, eps, rope_base, tcols (`rope_base.
//           wgsl`'s own modulo divisor, the caller's own `t` argument;
//           `pos = (row / heads) % tcols` is bit-identical to that kernel's
//           `pos = row % tcols` over its wider `[n_rows, heads*head_dim]`
//           addressing of the SAME underlying bytes - `row` here is already
//           per-head-flattened, `row / heads` recovers the token index
//           `rope_base.wgsl` calls `row`).
//
// Same 64-thread-workgroup-per-row / one-barrier / re-read-after-reduction
// shape as `qknorm_rope_fused.wgsl` - see that kernel's own header for the
// full derivation of the normalization half, unchanged here.
//
// **Requires `caps.workgroup_reductions`** - the same correctness gate
// every sibling in this family carries: the split-at-barrier CPU JIT
// mis-executes this cooperative-reduction shape, so a caller only dispatches
// this kernel when that capability holds and falls back to the original
// unfused `rmsnorm_rows` + `rope_base` pair otherwise.
//
// Bit agreement vs the two-dispatch path: same operations, same order
// (normalize each element, then rotate the normalized pair) as
// `rmsnorm_rows.wgsl` followed by `rope_base.wgsl` - not a reassociation,
// so this is bit-identical, not merely "agrees within tolerance".

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
    eps: f32,
    rope_base: f32,
    tcols: u32,
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
    let tok = row / p.heads;
    let pos = tok % p.tcols;
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
