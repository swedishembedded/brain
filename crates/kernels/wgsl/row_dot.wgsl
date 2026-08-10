// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Generic per-row dot product: out[row] = alpha * sum_d a[row,d]*b[row,d]
// @how   one thread per output row, serial reduction over a Params-bounded axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Generic per-row dot product, with an independent flat offset into each
// input and an `alpha` scale:
//
//   out[row] = alpha * sum_{k=0}^{d-1} a[a_off + row*d + k] * b[b_off + row*d + k]
//
// A reusable primitive, not GDN-specific (hence no `gdn_` prefix, matching
// `row_dot`'s sibling `scale_row.wgsl`/`bmm.wgsl` naming) — checked against
// `crates/kernels/wgsl/` before adding (per `docs/kernel-checklist.md` §A) and
// none of the existing per-row reductions (`argmax_row`, `bn_stats`,
// `ce_stats`, ...) compute a plain two-operand row dot product with
// independent offsets. First needed by `model::gdn::gdn_chunk_bwd` (Gated
// DeltaNet backward): every `y = x * s` row-broadcast-scale in the forward
// (`gdn_row_scale_off.wgsl`/`scale_row.wgsl`) has a row-scale-factor gradient
// `ds[row] = sum_d dy[row,d] * x[row,d]`, needed at FOUR call sites
// (`d_exp_g_cs` from both the per-chunk `q_scaled` row-scale and the
// whole-tensor `k_cumdecay` row-scale, `d_decay_scale`, and `d_beta` from
// both `v_beta`'s and `k_beta`'s row-scales).
//
// Always OVERWRITES its own dedicated output row (never accumulates) — this
// kernel does exactly one thing (a row reduction into a fresh buffer); where a
// caller needs the result ADDED into an already-populated gradient buffer
// (e.g. `d_exp_g_cs`'s second contribution, `d_beta`'s two contributions), it
// composes this kernel's dense output with `splice_add.wgsl` (`dst[base+i] +=
// src[i]`), which already exists for exactly this "commit a freshly computed
// dense result into a pre-zeroed accumulator, possibly at an offset" pattern.
// This keeps `row_dot` itself trivial (no output offset, no accumulate flag)
// at the cost of one extra dispatch per accumulating use — irrelevant at the
// dispatch counts GDN's chunk recurrence runs at (`docs/kernel-checklist.md`
// §E's own dispatch-overhead measurement: ~0.03%).

struct Params { rows: u32, d: u32, a_off: u32, b_off: u32, alpha: f32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.rows) { return; }
    let a_base = p.a_off + row * p.d;
    let b_base = p.b_off + row * p.d;
    var acc = 0.0;
    var k: u32 = 0u;
    loop {
        if (k >= p.d) { break; }
        acc = acc + a[a_base + k] * b[b_base + k];
        k = k + 1u;
    }
    out[row] = p.alpha * acc;
}
