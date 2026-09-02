// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused per-head RMSNorm + paged RoPE (half-split, base theta) over Q or K rows - M4.2
// @how   64-thread workgroup per (batch, head) row, 1 barrier, re-read after the reduction instead of a register array
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Fuses `rmsnorm_rows` and `rope_paged` into ONE dispatch over the same
// per-head row: `qwen3::serve`'s QK-norm and RoPE ran as four separate
// dispatches (rms(q), rms(k), rope(q), rope(k)) over data that never leaves
// this kernel's own workgroup - RoPE always rotates a PAIR (m, m+half)
// drawn from the SAME already-normalized row RMSNorm just produced, so
// splitting them across two dispatches pays for a round trip through
// device memory that buys nothing.
//
//   x  : [rows, head_dim]   out: [rows, head_dim]   w: [head_dim]
//   positions: [rows / heads]   (one absolute position per BATCH item)
//   params: rows, heads (rows-per-batch item, i.e. n_heads or n_kv_heads,
//           used only to recover the batch index as `row / heads` - the
//           SAME (batch, head) flattening `qwen3::serve::Engine::rms`'s own
//           `b * nh` / `b * nkv` row counts already assume), head_dim, eps,
//           rope_base.
//
// One workgroup per row. Phase 1 (pre-barrier): each thread strides over
// `head_dim` accumulating a partial sum of squares, exactly like
// `rmsnorm_rows` (coalesced: thread t's stride-64 walk keeps a warp's loads
// contiguous). ONE barrier folds the 64 partials redundantly on every
// thread (the CPU JIT's own single-top-level-barrier limit - see
// `rmsnorm_rows.wgsl`'s own doc for why the fold is redundant rather than a
// second reduction). Phase 2 (post-barrier): each thread re-reads its OWN
// pair `(x[m], x[m+half])` from global memory - a cache-warm re-read, not a
// second pass over the whole row - applies the RMSNorm scale and the RoPE
// rotation in one step, and writes the rotated pair out. Re-reading instead
// of caching the pair in a `var<function>` local avoids a per-thread array
// sized off a runtime `head_dim` - such an array is backed by local (i.e.
// global-backed) memory rather than registers unless the compiler can
// unroll every index, which a runtime bound prevents - and generalizes to
// any `head_dim` the RMS accumulation loop already handles, not only
// `head_dim <= 128` (`half <= 64`, one pair per thread).
//
// **Requires `caps.workgroup_reductions`**, the same correctness gate
// `rmsnorm_rows` itself carries: the split-at-barrier CPU JIT mis-executes
// this cooperative-reduction shape, so `qwen3::serve::Engine` only dispatches
// this kernel when that capability holds and falls back to the original
// unfused `rms` + `ROPE_PAGED` pair otherwise - never a raw `caps` check
// bypassing the existing selector precedent, an unconditional dispatch here
// would reproduce that exact defect class.
//
// Bit agreement vs the two-dispatch path: same operations, same order
// (normalize each element, then rotate the normalized pair) as
// `rmsnorm_rows.wgsl` followed by `rope_paged.wgsl` - not a reassociation,
// so this is bit-identical, not merely "agrees within tolerance".

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
    eps: f32,
    rope_base: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:         array<f32>;
@group(0) @binding(2) var<storage, read>       w:         array<f32>;
@group(0) @binding(3) var<storage, read>       positions: array<u32>;
@group(0) @binding(4) var<storage, read_write> out:       array<f32>;

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
    let batch = row / p.heads;
    let pos = positions[batch];
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
