// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M AFFINE K-quant (Q4_K/Q5_K) matmul, one WORKGROUP per output COLUMN, REGISTER accumulators - the GPU decode-regime affine GEMM
// @how   DP4A packed int8, register block per thread, 64-thread workgroup tile, staging-time code unpack, per-word group dequant + one-thread-per-group min correction, 1 barrier
// @opt   5
// @cpu   no
// @gpu   yes
// @npu   yes
// @quant int8
// @dtype f32
// @tpl   MREG -> rows carried in registers (kernels::template::interned)
//
// Swedish Embedded AB implements quantized inference kernels for edge and
// embedded GPUs for its clients. If your team needs expertise in shipping
// affine K-quant (GGUF Q4_K/Q5_K-class) inference on commodity GPU hardware
// without an intermediate fp32 detour then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// The GPU-only sibling of `matmul_kq_gemv.wgsl`, standing to it exactly as
// `matmul_i8_gemv_reg` stands to `matmul_i8_gemv`. **Edit the two `matmul_kq_*`
// GEMV kernels together**: the contract, the k-stride order, the staging-time
// code unpack and the min-correction guard below are copied from
// `matmul_kq_gemv.wgsl` deliberately, and `gpu_core::upgrade` substitutes this
// one for it at dispatch, so a change to either that is not mirrored in the
// other silently changes results on exactly one backend.
//
//   xq  : [M, k/4]              u32 - 4 int8 activations packed along K per u32
//   wq  : [N, k*CODE_BITS/32]   u32 - K-CONTIGUOUS unsigned codes, `32/CODE_BITS` codes per word, low bits first
//   sx  : [M]                   f32 - per-token activation scale
//   wsm : [N, ceil(k/32/2)]     u32 - (M14) packed per-group (sc, m) sub-scale byte pairs - see matmul_kq_dyn.wgsl's own header for the exact bit layout
//   wd  : [N, ceil(k/32/GPS)]   u32 - (M14) packed per-super-block (d, dmin) f16 bit-pattern pairs, GPS=8 groups/super-block
//   xgs : [M, k/32]             f32 - activation group sums (quant_group_sum.wgsl)
//   out : [M, N]                f32 - out[m,n] = sx[m] * Σ_g( ds[n,g]*A[m,n,g] - dm[n,g]*S[m,g] )
//   params: m, k (RAW LOGICAL K), n. REQUIRES m <= MREG and k a multiple of 32.
//   Dispatch: n * 64 invocations - the SAME thread count `matmul_kq_gemv` takes,
//   so a caller needs no change at all.
//
// ## Why this exists
//
// `matmul_kq_gemv` accumulates in WORKGROUP memory (`partial[m * 64 + t]`,
// `array<f32, 2048>`), which costs it exactly what `matmul_i8_gemv` was
// rescued from (see that kernel's `_reg` sibling's own header): the array is
// sized for the worst case (`m = 32`) so every workgroup reserves 8 KB of
// shared memory whatever `m` is, and the inner loop's read-modify-WRITE per
// `(k, m)` into that array serialises each accumulator through a
// shared-memory round trip per k-step. Both fixes come from the same `MREG`
// constant `matmul_i8_gemv_reg` already established: the accumulators become
// a function-local `array<f32, MREG>` whose every index is a compile-time-
// bounded loop, so they land in registers, and `partial` is written ONCE per
// row at the end and sized `MREG * 64`.
//
// `@cpu no`, for the identical reason `matmul_i8_gemv_reg` states: the CPU
// JIT rejects a function-local array in a work-group kernel outright, which
// is the second structural reason (alongside the barrier count) a
// register-accumulator kernel is a GPU-only SIBLING, not a template variant
// of the portable `matmul_kq_gemv`. `backend-cpu` reports
// `workgroup_reductions: false` and never selects either.
//
// ## Bit-identity with `matmul_kq_gemv`
//
// f32 addition is not associative, so bit-identity is a property of the two
// kernels performing the identical operations in the identical order, not a
// free consequence of the arithmetic. The k-stride is the same (`g = t; g +=
// 64`), the same staging-time code unpack produces the same DP4A operand per
// quad, the same `f32(dot4I8Packed(...)) * ds - dm * S` term is formed per
// quad with the SAME one-thread-per-group min-correction guard (`(g % WPGK)
// == 0u`, see `matmul_kq_gemv.wgsl`'s own header for why this guard exists at
// all - the naive per-quad application would land the correction 8 times),
// each output keeps its own accumulator, and the final fold sums the same 64
// partials in the same ascending order. Gated on the raw bits - which is
// exactly why the two must be edited together.
//
// ## `MREG` and rows past `p.m`
//
// `MREG` is the `kernels::template` knob: one specialised variant per bucket,
// so the accumulator count is a compile-time constant. `m` values between two
// buckets use the next bucket up and simply compute rows they discard - those
// rows are pointed at row 0 of `xq`/`xgs` (`xoff`/`goff` below) rather than
// read out of bounds, because bounds behaviour past the end of a storage
// binding is a per-backend policy, not something to rely on.

struct Params {
    m: u32,
    k: u32,  // RAW LOGICAL K
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, k/4]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, k*CODE_BITS/32]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       wsm: array<u32>;  // [N, ceil(k/32/2)] packed (sc, m) pairs
@group(0) @binding(5) var<storage, read>       wd:  array<u32>;  // [N, ceil(k/32/GPS)] packed (d, dmin) f16 pairs
@group(0) @binding(6) var<storage, read>       xgs: array<f32>;  // [M, k/32]
@group(0) @binding(7) var<storage, read_write> out: array<f32>;  // [M, N]

const CODE_BITS: u32 = 8u;  // template knob: 4 (Q4_K) or 8 (Q5_K)

/// Rows this specialisation carries in registers. `kernels::template`
/// rewrites this declaration; `gpu_core::upgrade`'s bucket ladder is what
/// picks which rewrite runs for a given `p.m`.
const MREG: u32 = 32u;

/// Quads (4-element units) per weight-scale group (32 elements / 4).
const WPGK: u32 = 8u;

// Groups per `wd` super-block entry (M14) - see matmul_kq_dyn.wgsl's own
// header for the full derivation.
const GPS: u32 = 8u;

// The same magic-multiply/FTZ-safe f16 decode `matmul_kq_dyn.wgsl`/
// `matmul_kq_gemv.wgsl` carry - duplicated per this codebase's "every kernel
// is self-contained WGSL text" convention; edited together with
// `matmul_kq_gemv.wgsl`'s copy per this file's own header.
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1Fu;
    let subnormal = bitcast<f32>(0x38800000u | ((h & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
    let normal = bitcast<f32>((h & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
    let inf_or_nan = bitcast<f32>(0x7F800000u | ((h & 0x3FFu) << 13u));
    let mag = select(select(subnormal, normal, exp != 0u), inf_or_nan, exp == 31u);
    return bitcast<f32>(bitcast<u32>(mag) | sign);
}

// Only the cross-thread FOLD needs shared memory now, so this is `MREG * 64`
// f32s rather than the worst case - matching `matmul_i8_gemv_reg`'s own
// occupancy fix.
var<workgroup> partial: array<f32, MREG * 64u>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n || p.m == 0u) { return; }

    let kgx = p.k / 4u;                       // quads per row (xq's own word density)
    let ng = p.k / 32u;                       // weight-scale groups per row
    let wq_row_words = p.k * CODE_BITS / 32u; // wq words per row
    let qpwq = (32u / CODE_BITS) / 4u;        // quads packed per raw wq word
    let cmask = (1u << CODE_BITS) - 1u;       // low-CODE_BITS mask
    let wsm_per_row = (ng + 1u) / 2u;         // wsm words per row (M14)
    let wd_per_row = (ng + GPS - 1u) / GPS;   // wd words per row (M14)

    var acc: array<f32, MREG>;
    var xoff: array<u32, MREG>;
    var goff: array<u32, MREG>;
    for (var m = 0u; m < MREG; m = m + 1u) {
        acc[m] = 0.0;
        // Rows past `p.m` are computed and thrown away (see the header);
        // point them at row 0, which is already hot, instead of past the
        // binding.
        xoff[m] = select(0u, m * kgx, m < p.m);
        goff[m] = select(0u, m * ng, m < p.m);
    }

    let wbase = col * wq_row_words;
    let wsmbase = col * wsm_per_row;
    let wdbase = col * wd_per_row;

    for (var g = t; g < kgx; g = g + 64u) {
        let word_idx = g / qpwq;
        let qoff = g % qpwq;
        let src = wq[wbase + word_idx];
        let base_bit = qoff * 4u * CODE_BITS;
        let u0 = (src >> (base_bit + 0u * CODE_BITS)) & cmask;
        let u1 = (src >> (base_bit + 1u * CODE_BITS)) & cmask;
        let u2 = (src >> (base_bit + 2u * CODE_BITS)) & cmask;
        let u3 = (src >> (base_bit + 3u * CODE_BITS)) & cmask;
        let wv = u0 | (u1 << 8u) | (u2 << 16u) | (u3 << 24u);

        let grp = g / WPGK;
        let is_lead = (g % WPGK) == 0u;
        // Same redundant-`ds`/lead-gated-`dmv` reasoning as `matmul_kq_gemv.
        // wgsl`'s own copy of this block (edited together, see this file's
        // header) - kept BYTE-IDENTICAL to that kernel's arithmetic.
        let wword = wd[wdbase + grp / GPS];
        let sword = wsm[wsmbase + grp / 2u];
        let shift = select(0u, 16u, (grp % 2u) == 1u);
        let ds = f16_to_f32(wword & 0xFFFFu) * f32((sword >> shift) & 0xFFu);
        let dmv = select(0.0, f16_to_f32(wword >> 16u) * f32((sword >> (shift + 8u)) & 0xFFu), is_lead);

        for (var m = 0u; m < MREG; m = m + 1u) {
            let xw = xq[xoff[m] + g];
            var contrib = f32(dot4I8Packed(xw, wv)) * ds;
            contrib = contrib - dmv * xgs[goff[m] + grp];
            acc[m] = acc[m] + contrib;
        }
    }

    for (var m = 0u; m < MREG; m = m + 1u) {
        partial[m * 64u + t] = acc[m];
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials and apply the per-token
    // activation scale (the only scale left - the weight side is already in),
    // the same ascending order `matmul_kq_gemv` folds in.
    //
    // `min(p.m, MREG)`, not `p.m`, is a memory-safety floor, not a feature -
    // see `matmul_i8_gemv_reg.wgsl`'s own comment on the identical line.
    if (t < min(p.m, MREG)) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s * sx[t];
    }
}
