// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Naive W4A8 matmul (out = dequant(x_q8 @ w_q4ᵀ)) -- the correct-first, non-tiled q4 GEMM
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant q4
//
// Naive W4A8 matmul: 8-bit dynamic per-token activations times 4-bit
// per-channel weights, out = dequant(x_q8 @ w_q4^T). The int4-weight sibling
// of `matmul.wgsl` / `matmul_i8_dyn.wgsl`'s naive tier -- deliberately NOT
// register-tiled. `.agents/rules/porting.md` §10 rule 1: get it correct, then
// freeze it. A register-tiled `matmul_q4_reg`/`matmul_q4_dyn` mirroring
// `matmul_i8_dyn.wgsl`'s 128x128 interleaved-tile shape is the documented
// follow-on optimization once a real model dispatches this kernel enough to
// need it -- not attempted here.
//
//   x_q : [M, K/4] u32  -- 4 int8 activations packed along K per u32 (model::int8 / quant_pack.wgsl)
//   w_q : [N, K/8] u32  -- 8 int4 weights    packed along K per u32 (model::int4::quantize_weight_q4)
//   sx  : [M] per-token activation scale     sw : [N] per-channel weight scale
//   out : [M, N] f32    -- out[m,n] = acc_i32 * sx[m] * sw[n]
//   params: m, k (LOGICAL K, a multiple of 8), n. `k` is passed un-divided
//   because x and w have DIFFERENT word densities for the same K (4 vs 8
//   values/u32) -- a single shared `kg` the way the int8 family uses would be
//   ambiguous about which operand it counts.
//
// W4A8 (int8 activation, int4 weight), not W4A4: this is the ONLY new
// device-side math the q4 tier needs. Activations stay on the existing int8
// dynamic-quant path (`max_abs_row` -> `quant_pack`, `model::int8::
// quant_rows_steps`) completely unchanged -- a model already wired for int8
// activations adopts q4 weights by relinking this GEMM, not by building a
// second activation-quant tier.
//
// Nibble/byte sign extension uses `shl` + arithmetic `shr` (the same trick
// `dot4I8Packed`'s own CPU-JIT lowering uses for its byte sign-extension:
// shift the field's sign bit up to bit 31, then arithmetic-shift back down),
// NOT the WGSL builtin `extractBits`: `extractBits` (`MathFunction::
// ExtractBits`) has no lowering in this repo's CPU Cranelift JIT
// (`crates/wgsl-cpu/src/lib.rs::math()` has no `ExtractBits` arm -- it hits
// the catch-all "unsupported math fn" and the kernel would silently never
// run on `BRAIN_DEVICE=cpu`), so it is avoided here even though it is valid
// WGSL and would work on the GPU backend alone.
//
// Inlined, not a helper `fn`: `crates/wgsl-cpu`'s WGSL->Cranelift JIT has no
// lowering for calling a user-defined WGSL function at all (only the entry
// point's own statements) -- every kernel in this tree already inlines its
// math into `main` for exactly this reason, confirmed by grepping
// `crates/kernels/wgsl/*.wgsl` for a top-level `fn` other than `main`: none
// exist. A helper `fn` compiles fine on the GPU (naga/wgpu) but panics the
// CPU JIT with "unsupported statement Call", so `@cpu yes` above would be a
// lie if this used one.

struct Params { m: u32, k: u32, n: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, k/4]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, k/8]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

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
    let kgx = p.k / 4u; // x words per row (int8 packing)
    let kgw = p.k / 8u; // w words per row (int4 packing)
    let x_base = row * kgx;
    let w_base = col * kgw;
    var acc = 0i;
    for (var g: u32 = 0u; g < kgw; g = g + 1u) {
        let ww = wq[w_base + g];
        // One octet (8 logical K values) of w == exactly two u32 of x.
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
