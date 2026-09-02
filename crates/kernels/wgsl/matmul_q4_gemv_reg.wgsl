// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M W4A8 matmul (out = dequant(x_q8 @ w_q4ᵀ)), one WORKGROUP per output COLUMN, REGISTER accumulators - the GPU decode-regime q4 GEMM
// @how   nibble-unpack into DP4A packed int8, register block per thread, 64-thread workgroup tile, 1 barrier
// @opt   5
// @cpu   no
// @gpu   yes
// @npu   yes
// @quant int8
// @dtype f32
// @tpl   MREG -> rows carried in registers (kernels::template::interned)
//
// `@quant int8`, not `q4`, even though the WEIGHT is int4-packed: the
// catalogue's `@quant` field tracks which packed-integer arithmetic
// primitive the kernel's compute actually uses (`dot4I8Packed`), not the
// storage width of an operand - the same rule `scripts/build/gen-kernel-
// table.py`'s cross-check enforces uniformly, matching every other
// `dot4I8Packed`-calling kernel in this tree. `matmul_q4_gemv.wgsl`'s own
// `q4` declaration stays correct for IT specifically because it never calls
// `dot4I8Packed` at all (plain scalar nibble MACs) - this kernel is the one
// that changes that, for exactly this Params/binding pair.
//
// The GPU-only sibling of `matmul_q4_gemv.wgsl`, standing to it exactly as
// `matmul_i8_gemv_reg.wgsl` stands to `matmul_i8_gemv.wgsl` - **edit the two
// together**: same `Params` layout (`m, k, n` - the RAW logical `k`, not a
// packed word count, matching `matmul_q4_gemv.wgsl`'s own contract), same
// binding order, same 64-thread workgroup-per-column tile, same single
// top-level barrier, so `gpu_core::upgrade` can substitute this one for
// `matmul_q4_gemv` at dispatch with no caller-visible change at all.
//
//   x_q : [M, K/4] u32  -- 4 int8 activations packed along K per u32
//   w_q : [N, K/8] u32  -- 8 int4 weights    packed along K per u32
//   sx  : [M] per-token activation scale
//   sw  : [N, K/32] GROUP-WISE weight scale (`model::int8::GROUP`, shared with
//         the int8 tier). Four w words per group.
//   out : [M, N] f32    -- out[m,n] = sx[m] * sum_g acc[m,n,g] * sw[n,g]
//   params: m, k, n. REQUIRES m <= MREG and K a multiple of 32.
//
// ## Why this exists
//
// `matmul_q4_gemv` carries the SAME two costs `matmul_i8_gemv_reg`'s header
// diagnoses for its int8 twin: an `array<f32, 2048>` workgroup accumulator
// sized for the worst case (`m = 32`) whatever `m` actually is - 8 KB/
// workgroup, capping residency at 12 of 32 workgroups/SM on a GP102 - and a
// read-modify-WRITE into that array per `(k, m)`, a serial shared-memory
// dependency chain. `MREG` moves the accumulators into a function-local
// `array<f32, MREG>` (registers, compile-time-bounded loop) and shrinks
// `partial` to `MREG * 64`, exactly as the int8 fix does.
//
// ## The SECOND fix: `dot4I8Packed` instead of 8 scalar MACs per word
//
// `matmul_q4_gemv`'s inner loop extracts each of a weight word's 8 nibbles
// and each matching activation byte one at a time (8 sign-extends, 8
// multiplies, 8 adds). The nibbles of ONE weight word `wv` are shared by
// every row `m` this workgroup serves, so they are unpacked exactly ONCE
// per `(thread, g)` - not once per `(thread, g, m)` - into TWO packed-int8
// words (`wlo` = nibbles 0..3, `whi` = nibbles 4..7), each byte holding the
// nibble's value sign-extended to a full int8 in the SAME bit pattern
// `dot4I8Packed` expects (two's-complement, so the low byte of the
// arithmetic-shifted i32 IS the correctly-signed int8 byte - the identical
// trick `matmul_q4_dyn.wgsl`/`matmul_q4_gemv.wgsl` already use to extract a
// nibble, just kept in the u32 rather than converted to `f32` immediately).
// The per-row cost then drops to two `dot4I8Packed` calls (four MACs each,
// eight total, matching the eight nibbles) instead of eight scalar
// mul-adds - the same width-matching argument that justifies the tiled
// int8 GEMM's own DP4A-vs-scalar switch, applied to the decode-shaped
// kernel here.
//
// ## Bit-identity with `matmul_q4_gemv`
//
// Not exact: `matmul_q4_gemv` computes each nibble's product with plain i32
// multiplication in the compiler's own order (`wn * xb`, summed with the
// other seven of the word via `local = local + wn * xb`), while this kernel
// computes the SAME eight products via two `dot4I8Packed` calls - a
// different, but equally valid, order of the identical 8 integer products
// within one word. All 8 products are exact integers and their sum fits
// comfortably inside `dot4I8Packed`'s i32 accumulator (max magnitude
// `8 * 127 * 7`), so the per-word integer sum is IDENTICAL regardless of
// grouping order (integer addition is associative and lossless at this
// magnitude) - gated by an exact (not tolerance) comparison against
// `matmul_q4_gemv` in `crates/model/tests/matmul_q4_gemm.rs`. The f32
// accumulation ACROSS words (`acc[m] += f32(...) * s`) is unchanged in
// order from `matmul_q4_gemv`'s own `partial[...] + f32(local) * s`, so the
// two kernels are bit-identical end to end.
//
// ## `MREG` and rows past `p.m`
//
// Same convention as `matmul_i8_gemv_reg`: `MREG` is the `kernels::template`
// knob (one specialised variant per bucket, so the accumulator count is a
// compile-time constant), and a row past `p.m` reads `x_q` row 0 (already
// hot) rather than out of bounds - the caller's contract never depends on
// what those discarded rows compute.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, k/4]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, k/8]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N, k/32]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

// Packed u32 words of w per weight-scale group: GROUP(32 int4) / 8 nibbles/word.
const WPG4: u32 = 4u;

/// Rows this specialisation carries in registers. `kernels::template`
/// rewrites this declaration; `gpu_core::upgrade`'s bucket ladder picks
/// which rewrite runs for a given `p.m`.
const MREG: u32 = 32u;

// Only the cross-thread FOLD needs shared memory - `MREG * 64` f32s, not the
// worst-case 2048 (matmul_q4_gemv's `array<f32, 2048>`).
var<workgroup> partial: array<f32, MREG * 64u>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n || p.m == 0u) { return; }

    let kgx = p.k / 4u; // x words per row (int8 packing)
    let kgw = p.k / 8u; // w words per row (int4 packing)

    var acc: array<f32, MREG>;
    var xoff: array<u32, MREG>;
    for (var m = 0u; m < MREG; m = m + 1u) {
        acc[m] = 0.0;
        // Rows past `p.m` are computed and thrown away; point them at row 0
        // (already hot) instead of past the binding.
        xoff[m] = select(0u, m * kgx, m < p.m);
    }

    let wbase = col * kgw;
    let swbase = col * (kgw / WPG4);
    for (var g = t; g < kgw; g = g + 64u) {
        let wv = wq[wbase + g];
        let s = sw[swbase + g / WPG4];

        // Unpack `wv`'s 8 nibbles ONCE (shared by every row `m` below) into
        // two DP4A-packed int8 words: `wlo` holds nibbles 0..3 (the same
        // logical K values as `xq`'s word `2*g`), `whi` holds nibbles 4..7
        // (`xq`'s word `2*g + 1`). Each byte is the nibble sign-extended to
        // a full int8 via the arithmetic-shift trick every other q4 kernel
        // in this tree already uses, kept as a u32 bit pattern instead of
        // being converted to `f32` immediately.
        var wlo = 0u;
        var whi = 0u;
        for (var b: u32 = 0u; b < 4u; b = b + 1u) {
            let byte_lo = bitcast<u32>(bitcast<i32>(wv << (28u - 4u * b)) >> 28u) & 0xffu;
            let byte_hi = bitcast<u32>(bitcast<i32>(wv << (28u - 4u * (b + 4u))) >> 28u) & 0xffu;
            wlo = wlo | (byte_lo << (8u * b));
            whi = whi | (byte_hi << (8u * b));
        }

        for (var m = 0u; m < MREG; m = m + 1u) {
            let xbase = xoff[m] + 2u * g;
            let xw0 = xq[xbase];
            let xw1 = xq[xbase + 1u];
            acc[m] = acc[m] + f32(dot4I8Packed(xw0, wlo) + dot4I8Packed(xw1, whi)) * s;
        }
    }

    for (var m = 0u; m < MREG; m = m + 1u) {
        partial[m * 64u + t] = acc[m];
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials and apply the per-token
    // activation scale - the same ascending order `matmul_q4_gemv` folds in.
    //
    // `min(p.m, MREG)`, not `p.m`, is a memory-safety floor: `partial` is
    // only `MREG * 64` long, so a caller handing this variant more rows than
    // it was specialised for would fold PAST the workgroup array. Equal
    // under the contract, so this costs nothing and cannot change a correct
    // result.
    if (t < min(p.m, MREG)) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s * sx[t];
    }
}
