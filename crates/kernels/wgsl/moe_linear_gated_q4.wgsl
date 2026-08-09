// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE expert linear, W4A8 q4: moe_linear_gated_i8.wgsl's row skip, int4-packed weights
// @how   one thread per output element, serial inner reduction, early exit
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant q4
//
// The q4 counterpart of `moe_linear_gated_i8.wgsl`, in the SAME naive tier as
// that kernel (one thread per output element, no workgroup tiling, no
// register block) -- for exactly the reason `moe_linear_gated_i8.wgsl`'s own
// header gives: a per-thread early `return` for a non-routed row is only
// safe without a `workgroupBarrier()` in the kernel at all, and the tiled
// GEMMs (`matmul_i8_dyn`, and any future `matmul_q4_dyn` register-tiled
// sibling) stage rows into workgroup-shared memory across a barrier that
// every thread must reach uniformly. Gating safely there needs row
// COMPACTION, which this workstream has not built for int8 either -- see
// that file's doc. Promoting both int8 and q4 sparse-expert kernels to a
// tiled/gathered design is one future change, not two.
//
// Same dequant contract as `matmul_q4_dyn`: dynamic per-token activation
// scale (`sx`, from `model::int8::quant_rows_steps` -- q4 is W4A8, activations
// stay int8) times per-channel weight scale (`sw`, from
// `model::int4::quantize_weight_q4`).
//
//   x_q  : [m, k/4] u32    4 int8 activations packed along K per u32
//   w_q  : [n, k/8] u32    8 int4 weights    packed along K per u32
//   sx   : [m]      f32    per-token activation scale
//   sw   : [n]      f32    per-channel weight scale
//   gate : [m, n_experts]  dense per-token-per-expert weight (0 = not routed)
//   out  : [m, n]   f32    out[row,:] = 0 for a non-routed row
//   params: m, k (LOGICAL K, a multiple of 8, NOT pre-divided -- x and w have
//   different words/row for the same K), n, n_experts, e_idx.
//
// Nibble/byte sign extension uses `shl` + arithmetic `shr`, not `extractBits`
// -- see `matmul_q4_dyn.wgsl`'s header for why (no CPU-JIT lowering). Inlined
// into `main`, not a helper `fn` -- the CPU JIT has no lowering for calling a
// user-defined WGSL function, and no other kernel in this tree defines one.

struct Params {
    m: u32,
    k: u32,
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
    let kgx = p.k / 4u; // x words per row (int8 packing)
    let kgw = p.k / 8u; // w words per row (int4 packing)
    let x_base = row * kgx;
    let w_base = col * kgw;
    var acc = 0i;
    for (var g: u32 = 0u; g < kgw; g = g + 1u) {
        let ww = wq[w_base + g];
        let xw0 = xq[x_base + 2u * g];
        let xw1 = xq[x_base + 2u * g + 1u];
        for (var b: u32 = 0u; b < 8u; b = b + 1u) {
            let wn = bitcast<i32>(ww << (28u - 4u * b)) >> 28u;
            var xb: i32;
            if (b < 4u) {
                xb = bitcast<i32>(xw0 << (24u - 8u * b)) >> 24u;
            } else {
                let bb = b - 4u;
                xb = bitcast<i32>(xw1 << (24u - 8u * bb)) >> 24u;
            }
            acc = acc + wn * xb;
        }
    }
    out[idx] = f32(acc) * sx[row] * sw[col];
}
