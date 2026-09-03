// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M AFFINE K-quant (Q4_K/Q5_K) matmul, one WORKGROUP per output COLUMN - the decode-regime GEMM for the two GGUF quant types no existing kernel can dequantize losslessly
// @how   DP4A packed int8, 64-thread workgroup tile, staging-time code unpack, per-word group dequant + one-thread-per-group min correction, 1 barrier
// @opt   5
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant int8
// @dtype f32
//
// Swedish Embedded AB implements quantized inference kernels for edge and
// embedded GPUs for its clients. If your team needs expertise in shipping
// affine K-quant (GGUF Q4_K/Q5_K-class) inference on commodity GPU hardware
// without an intermediate fp32 detour then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// `matmul_i8_gemv`'s AFFINE sibling, for the decode regime (`m` small) the
// same way `matmul_kq_dyn` is `matmul_i8_dyn`'s. Same two deltas as that
// kernel - see its header for the full rationale, restated briefly here:
//
//  1. the weight word read UNPACKS a `CODE_BITS`-wide unsigned code (4 for
//     Q4_K, 8 for Q5_K) into a DP4A-ready packed word, instead of reading a
//     ready-made symmetric int8 layout directly;
//  2. the accumulation gains the affine `- dm[n,g]*S[m,g]` correction term
//     alongside the usual `ds[n,g]*dot4I8Packed(...)` term.
//
//   xq  : [M, k/4]              u32 - 4 int8 activations packed along K per u32
//   wq  : [N, k*CODE_BITS/32]   u32 - K-CONTIGUOUS unsigned codes, `32/CODE_BITS` codes per word, low bits first
//   sx  : [M]                   f32 - per-token activation scale
//   wsm : [N, ceil(k/32/2)]     u32 - (M14) packed per-group (sc, m) sub-scale byte pairs - see matmul_kq_dyn.wgsl's own header for the exact bit layout
//   wd  : [N, ceil(k/32/GPS)]   u32 - (M14) packed per-super-block (d, dmin) f16 bit-pattern pairs, GPS=8 groups/super-block
//   xgs : [M, k/32]             f32 - activation group sums (quant_group_sum.wgsl)
//   out : [M, N]                f32 - out[m,n] = sx[m] * Σ_g( ds[n,g]*A[m,n,g] - dm[n,g]*S[m,g] )
//   params: m, k (RAW LOGICAL K - see matmul_kq_dyn's header for why this
//   cannot be a shared packed-word count), n. REQUIRES m <= 32 and k a
//   multiple of 32.
//
// Same shape rationale as `matmul_i8_gemv`: at decode `m` is small and a
// 128x128 tile is mostly idle, so this kernel gives the affine tier the same
// `matmul_gemv` shape - 64 threads split K, each reads its slice of the
// weight row ONCE and applies it to all `m` rows, one barrier, threads
// `0..m` fold the partials.
//
// ## The one-thread-per-group min-correction guard
//
// This kernel's k-stride is 64 QUADS (a "quad" is 4 elements, the same unit
// `xq`'s own packing uses - see below for why quads, not raw words, are the
// per-thread stride unit here). A weight-scale group is 8 quads (32
// elements), so 64 threads span 8 groups per stride and consecutive quads of
// ONE group land on 8 DIFFERENT threads (quad `8g+i` is visited by thread
// `(8g+i) mod 64`, and 8 consecutive integers mod 64 are always 8 distinct
// threads). If every thread that touches a word of a group applied that
// group's `dm*S` correction, it would land 8 TIMES instead of once - a real
// bug, not a cosmetic one, since `dm*S` is a per-GROUP quantity, not a
// per-quad one. The guard is `(g % WPGK) == 0u`: only the thread visiting a
// group's FIRST quad (`g` a multiple of `WPGK = 8`) applies the correction,
// via `select(0.0, dm, is_lead)` so the multiply-by-zero is unconditional
// (no branch) rather than skipped - correctness-equivalent, and avoids a
// per-quad branch in the hot loop. Every group's first quad is visited by
// exactly one thread across the WHOLE stride (not just one stride step), so
// this fires exactly once per (row, group) pair over the kernel's full run.
//
// ## Why quads, not raw `wq` words, are the per-thread stride unit
//
// `xq` is always 4 int8/word; `wq` is `32/CODE_BITS` codes/word, which
// DIFFERS from `xq`'s density at `CODE_BITS=4` (Q4_K: 8 codes/word). A
// stride over raw `wq` words would desynchronize which `xq` word a thread
// reads and which weight quad it applies to it. Striding over QUADS (a
// `xq`-word-sized unit for BOTH operands) keeps the two in lockstep
// regardless of `CODE_BITS`, at the cost of a staging-time unpack per
// weight quad (below) rather than a direct `wq` read - exactly
// `matmul_kq_dyn`'s own reason for taking raw logical `k`, restated for the
// GEMV's per-thread stride instead of a tile's chunk loop.
//
// Single top-level barrier + no atomics. Dispatch: n * 64 invocations (one
// workgroup per column).

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

// Quads (4-element units) per weight-scale group (32 elements / 4).
const WPGK: u32 = 8u;

// Groups per `wd` super-block entry (M14 - see matmul_kq_dyn.wgsl's own
// header for the full derivation and why this is a fixed compile-time
// constant, not a per-dispatch parameter).
const GPS: u32 = 8u;

// The same magic-multiply/FTZ-safe f16 decode `matmul_kq_dyn.wgsl` carries,
// duplicated per this codebase's "every kernel is self-contained WGSL text"
// convention - but INLINED as a `let` sequence rather than a callable `fn`
// here (see the two `{X}h_*` blocks in `main` below): this kernel is `@cpu
// yes` (the CPU JIT must compile it), and `wgsl_cpu::Jit` does not support
// a `Call` statement to a user-defined function - only builtins - so a
// separate `f16_to_f32` function (as `matmul_kq_dyn.wgsl`/`matmul_kq_gemv_
// reg.wgsl` use, both `@cpu no`) fails CPU JIT compilation here with
// "unsupported statement Call". Inlining is exactly how `kernels::template::
// f16_decode_expr` already handles the identical constraint for the bf16/
// f16 WEIGHT STORAGE tier.

// f32 accumulators in workgroup memory (indexed [m*64 + t]) - same layout as
// matmul_i8_gemv, same CPU-JIT-compatible single-barrier shape.
var<workgroup> partial: array<f32, 2048>; // up to 32 rows x 64 threads

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n) { return; }
    for (var m = 0u; m < p.m; m = m + 1u) {
        partial[m * 64u + t] = 0.0;
    }

    let kgx = p.k / 4u;                       // quads per row (xq's own word density)
    let ng = p.k / 32u;                       // weight-scale groups per row
    let wq_row_words = p.k * CODE_BITS / 32u; // wq words per row
    let qpwq = (32u / CODE_BITS) / 4u;        // quads packed per raw wq word
    let cmask = (1u << CODE_BITS) - 1u;       // low-CODE_BITS mask
    let wsm_per_row = (ng + 1u) / 2u;         // wsm words per row (M14)
    let wd_per_row = (ng + GPS - 1u) / GPS;   // wd words per row (M14)

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
        // `ds` is read (and re-decoded) by every one of the 8 threads that
        // share this group - correctness-equivalent to the OLD flat-`wsz`
        // read, which was already redundant across those 8 threads the same
        // way (`ds` MULTIPLIES each quad's own contribution, so redundant
        // reads are harmless; only `dmv`, an ADDITIVE per-group term, must
        // fire exactly once - see this kernel's own header for the guard).
        let wword = wd[wdbase + grp / GPS];
        let sword = wsm[wsmbase + grp / 2u];
        let shift = select(0u, 16u, (grp % 2u) == 1u);

        // Inline f16 decode of `d` (low half of `wword`) - see this file's
        // own note above `main` for why this is a `let` sequence, not a
        // callable function.
        let dh = wword & 0xFFFFu;
        let dh_exp = (dh >> 10u) & 0x1Fu;
        let dh_sub = bitcast<f32>(0x38800000u | ((dh & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
        let dh_norm = bitcast<f32>((dh & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
        let dh_infnan = bitcast<f32>(0x7F800000u | ((dh & 0x3FFu) << 13u));
        let dh_mag = select(select(dh_sub, dh_norm, dh_exp != 0u), dh_infnan, dh_exp == 31u);
        let dh_f32 = bitcast<f32>(bitcast<u32>(dh_mag) | ((dh & 0x8000u) << 16u));

        // Inline f16 decode of `dmin` (high half of `wword`) - identical
        // construction, high half.
        let dmh = wword >> 16u;
        let dmh_exp = (dmh >> 10u) & 0x1Fu;
        let dmh_sub = bitcast<f32>(0x38800000u | ((dmh & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
        let dmh_norm = bitcast<f32>((dmh & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
        let dmh_infnan = bitcast<f32>(0x7F800000u | ((dmh & 0x3FFu) << 13u));
        let dmh_mag = select(select(dmh_sub, dmh_norm, dmh_exp != 0u), dmh_infnan, dmh_exp == 31u);
        let dmh_f32 = bitcast<f32>(bitcast<u32>(dmh_mag) | ((dmh & 0x8000u) << 16u));

        let ds = dh_f32 * f32((sword >> shift) & 0xFFu);
        let dmv = select(0.0, dmh_f32 * f32((sword >> (shift + 8u)) & 0xFFu), is_lead);

        for (var m = 0u; m < p.m; m = m + 1u) {
            let xw = xq[m * kgx + g];
            var contrib = f32(dot4I8Packed(xw, wv)) * ds;
            contrib = contrib - dmv * xgs[m * ng + grp];
            partial[m * 64u + t] = partial[m * 64u + t] + contrib;
        }
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials and apply the per-token
    // activation scale (the only scale left - the weight side is already in).
    if (t < p.m) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s * sx[t];
    }
}
