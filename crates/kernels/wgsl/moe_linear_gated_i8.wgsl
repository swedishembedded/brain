// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE expert linear, int8 (DP4A): moe_linear_gated.wgsl's row skip, packed weights
// @how   DP4A packed int8, one thread per output element, serial inner reduction, early exit
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
//
// The int8 counterpart of `moe_linear_gated.wgsl`, in the SAME naive tier as
// that kernel (one thread per output element, no workgroup tiling) rather
// than `matmul_i8_dyn`'s 256-thread register-tiled shape. That is deliberate,
// not an oversight: `matmul_i8_dyn`/`matmul_i8_gemv` stage rows into
// WORKGROUP-SHARED memory across a barrier, and WGSL requires every thread in
// a workgroup to reach a `workgroupBarrier()` uniformly — a per-thread early
// return for a non-routed row would make that undefined behaviour. Gating
// safely there needs row COMPACTION (feed the tile only the routed rows) or a
// whole-tile skip decision, either of which needs a gather/prefix-sum pass
// this workstream deferred (see `moe_linear_gated.wgsl`'s own doc). This
// kernel accepts a tiled kernel's
// throughput ceiling in exchange for a row-level skip that is trivially safe
// (an ordinary `return`, no barrier in this kernel at all) — the same trade
// `moe_linear_gated.wgsl` already makes for the fp32 tier. Promoting both to
// a tiled/gathered design is one future change, not two.
//
// Same dequant contract as `matmul_i8_dyn`: dynamic per-token activation
// scale (`sx`, from `model::int8::quant_rows_steps`) times per-channel weight
// scale (`sw`, from `model::int8::quantize_weight`).
//
//   x_q  : [m, k/4] u32    4 int8 activations packed along K per u32
//   w_q  : [n, k/4] u32    4 int8 weights    packed along K per u32
//   sx   : [m]      f32    per-token activation scale
//   sw   : [n]      f32    per-channel weight scale
//   gate : [m, n_experts]  dense per-token-per-expert weight (0 = not routed)
//   out  : [m, n]   f32    out[row,:] = 0 for a non-routed row

struct Params {
    m: u32,
    kg: u32,   // k/4
    n: u32,
    n_experts: u32,
    e_idx: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:   array<u32>;
@group(0) @binding(2) var<storage, read>       wq:   array<u32>;
@group(0) @binding(3) var<storage, read>       sx:   array<f32>;
@group(0) @binding(4) var<storage, read>       sw:   array<f32>;
@group(0) @binding(5) var<storage, read>       gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.m * p.n;
    if (idx >= total) { return; }
    let row = idx / p.n;
    let col = idx % p.n;
    if (gate[row * p.n_experts + p.e_idx] <= 0.0) {
        out[idx] = 0.0;
        return;
    }
    let x_base = row * p.kg;
    let w_base = col * p.kg;
    var acc = 0i;
    for (var g: u32 = 0u; g < p.kg; g = g + 1u) {
        acc = acc + dot4I8Packed(xq[x_base + g], wq[w_base + g]);
    }
    out[idx] = f32(acc) * sx[row] * sw[col];
}
