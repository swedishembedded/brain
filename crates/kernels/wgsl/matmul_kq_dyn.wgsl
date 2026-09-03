// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled AFFINE K-quant (Q4_K/Q5_K) GEMM with a DYNAMIC per-token activation scale and the affine min-correction term - the prefill/DiT GEMM for the two GGUF quant types no existing kernel can dequantize losslessly
// @how   DP4A packed int8, vec4 shared tiles, register block per thread, staging-time code unpack, per-k-chunk group dequant + min correction, 256-thread workgroup tile, 3 barriers
// @opt   5
// @cpu   no
// @gpu   yes-wg256
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
// `matmul_i8_dyn`'s AFFINE sibling: the weight is not `q[n,k]` stored as
// ready-to-DP4A signed int8 words, it is an UNSIGNED code at a
// template-chosen bit width (`CODE_BITS`, 4 for Q4_K or 8 for Q5_K) that
// reconstructs as `ds*code - dm` per weight-scale group rather than the
// symmetric family's `ds*code`. Two deltas from `matmul_i8_dyn`, nothing
// else about the tile/register/pipelining structure changes:
//
//  1. the weight staging load UNPACKS `CODE_BITS`-wide codes into DP4A-ready
//     packed words instead of reading a ready-made symmetric int8 layout -
//     a staging-time unpack, not a runtime dequant (the code's numeric VALUE
//     never changes, only its BIT POSITION);
//  2. the per-register-block fold gains a second reduction, the affine
//     min-correction term, alongside the usual int8 dot product.
//
//   xq  : [M, k/4]              u32 - 4 int8 activations packed along K per u32 (model::int8)
//   wq  : [N, k*CODE_BITS/32]   u32 - K-CONTIGUOUS unsigned codes, `32/CODE_BITS` codes per word, LOW BITS FIRST (code `b` of word `w` occupies bits `[CODE_BITS*b, CODE_BITS*b+CODE_BITS)` and covers element `w*(32/CODE_BITS)+b`)
//   sx  : [M]                   f32 - per-token activation scale
//   wsm : [N, ceil(k/32/2)]     u32 - (M14) PACKED per-32-element-group (sc, m) sub-scale byte pair, two groups per word: group `g`'s pair sits in bits `[0,16)` of word `g/2` (g even) or `[16,32)` (g odd); within a group's own 16-bit half, `sc` is the LOW byte, `m` the HIGH byte
//   wd  : [N, ceil(k/32/GPS)]   u32 - (M14) PACKED per-super-block (d, dmin) f16 bit-pattern pair, one word per GPS=8 consecutive groups (the GGUF Q4_K/Q5_K super-block, 256 elements): `d`'s raw f16 bits in bits `[0,16)`, `dmin`'s in `[16,32)`. `ds = f16_to_f32(d_bits) * f32(sc)`, `dm = f16_to_f32(dmin_bits) * f32(m)` - the IDENTICAL `d*sc`/`dmin*m` fp32 expressions checkpoint::gguf's own Q4_K/Q5_K decoder uses, since `wd` stores the format's own raw f16 bit pattern unrounded (see gguf::kquant's own module doc comment)
//   xgs : [M, k/32]             f32 - Σ_{j in group} xq[m,j], the activation-only prepass term (quant_group_sum.wgsl)
//   out : [M, N]                f32 - out[m,n] = sx[m] * Σ_g( ds[n,g]*A[m,n,g] - dm[n,g]*S[m,g] )
//
// A[m,n,g] = Σ_{k in g}( code[n,k]*xq[m,k] ) is the ordinary DP4A dot product
// every int8 GEMM in this tree already computes. S[m,g] = xgs[m,g] is
// activation-only (independent of n) and MUST come from the int8 activation,
// never a f32 one - mixing them is a systematic bias proportional to `dm`,
// not a rounding difference, because the correction has to match exactly
// what the A term consumes. `sx` factors out of BOTH terms, so the epilogue
// applying only the per-token scale is unchanged from `matmul_i8_dyn`.
//
// `k` (the params field, and the "K" in every shape above) is the RAW
// LOGICAL reduction length, NOT a packed-word count: `xq` and `wq` have
// DIFFERENT word densities for the same `k` (4 codes/word for `xq` always,
// `32/CODE_BITS` codes/word for `wq` - 8 at CODE_BITS=4, 4 at CODE_BITS=8),
// so a single shared word-count parameter the way the symmetric int8 family
// uses (`kg = K/4`) would be ambiguous about which operand it counts. `k`
// must be a multiple of 32 (one weight-scale group), matching every other
// kernel in this family's group contract.
//
// ## The staging-time code unpack
//
// A "quad" is 4 consecutive K elements - the granularity `xq`'s own packing
// already uses, and the granularity every k-group-minor `vec4<u32>` shared
// load in this tile family reads. `wq` packs `32/CODE_BITS` codes per word,
// so at CODE_BITS=8 (Q5_K) a quad is exactly one `wq` word (the code's 5-bit
// VALUE sits in a full 8-bit SLOT, per the device layout's own contract, so
// no bits above the value are ever set and the word is already a valid DP4A
// operand - the unpack below reduces to reading `wq[word_idx]` unchanged).
// At CODE_BITS=4 (Q4_K) two quads share one `wq` word (8 nibbles), so the
// unpack extracts 4 `CODE_BITS`-wide fields starting at bit
// `(quad_offset_within_word * 4 * CODE_BITS)` and repacks them one per BYTE
// (`u0 | u1<<8 | u2<<16 | u3<<24`) - a pure bit-shuffle, never a multiply or
// an added/subtracted bias, which is exactly why this is a staging-time
// unpack and not a "runtime dequant": the affine codes stay their raw
// unsigned value all the way into the DP4A operand, and `ds`/`dm` are only
// ever applied once, in the group fold below, never per-element.
//
// Every code value is `< 2^CODE_BITS <= 256` and always packed at the LOW
// end of its byte slot (top bits zero), so its signed-int8 reinterpretation
// (what `dot4I8Packed` performs) is IDENTICAL to its unsigned value for
// every code this format can produce (max 31, Q5_K) - the sign bit (bit 7)
// is never set, so there is no unsigned-code-read-as-signed hazard to guard
// against on the weight side. This is a real, gated case (adversarial case
// 3 pairs a mixed-sign ACTIVATION with these always-nonnegative weight
// codes) - the hazard `dot4I8Packed` exists to catch is on the activation
// operand, not this one.
//
// ## Where the affine correction is applied, and what it costs
//
// Same fold point as `matmul_i8_dyn`'s weight-scale dequant: every `QPG`-th
// quad (`QPG=2`, fixed - this kernel's weight-scale group is always 32
// elements, unlike the symmetric family's Q6_K special case), the live
// 64-accumulator integer block folds into the running f32 totals through
// `ds`, MINUS `dm[n,g]*S[m,g]`. `dm` varies per COLUMN (one value shared by
// all 8 rows of a fold), `S` varies per ROW (one value shared by all 8
// columns) - an 8+8 read per fold instead of `matmul_i8_dyn`'s 8, and the
// two combine as an outer product across the 8x8 register block, exactly
// mirroring how the existing `ds` term already broadcasts one per-column
// value across 8 rows.
//
// ## Adversarial cases this kernel is gated against
//
// Mutation-verified (the specific term was temporarily zeroed/broken in this
// file, the corresponding test confirmed red, then the break was reverted):
// dropping the min-correction term entirely is invisible when `dmin == 0`
// (a real, common case) but produces a large, systematic error once `dmin`
// is nonzero and every value in a group sits far from zero - so the gate
// pairs a `dmin == 0` case against a `dmin != 0` case and asserts the two
// disagree, proving the correction term is load-bearing rather than
// coincidentally near-zero on the test data. Sub-block scale variation
// across a super-block's 8 groups catches a hoisted-out-of-the-k-loop
// scale/offset read. Mixed positive/negative ACTIVATION values catch a
// sign-extension bug on the `xq` side and, independently, would catch an
// unsigned weight code accidentally read as signed (structurally impossible
// here per the note above, gated anyway). An all-zero sub-block catches a
// div-by-zero/NaN from the host relayout or an uninitialized-scale read
// reaching the output. A genuine sub-rectangle (`r0 != 0` AND `c0 != 0`) at
// two super-blocks' worth of `k` catches a hoisted row/column offset AND
// exercises two genuinely different `d` values in one dispatch. Ragged
// tiles (M/N not multiples of 128) exercise the guarded-store epilogue. A
// `k` where CODE_BITS=4's word density (8 codes/word) genuinely differs from
// `xq`'s (4/word) catches a stride mix-up between the two operands that a
// coincidentally-equal word count could hide.
//
// Register-block ownership, bank padding, software pipelining and the
// epilogue are copied from `matmul_i8_dyn` unchanged - see that kernel's own
// header for why the shared tiles are `vec4<u32>` and k-group-minor, why
// only one operand side is hoisted into registers, and why the register
// block is interleaved.
//
// ## M14: the packed `(wsm, wd)` scale plane and its in-shader f16 decode
//
// `wsm`/`wd` replace the old flat `wsz: [N, 2*k/32] f32` interleaved plane
// (M8-M13) with the SAME two-piece product every one of Q4_K/Q5_K's own GGUF
// blocks already carries: a per-group `(sc, m)` sub-scale byte (`wsm`) times
// a per-super-block `(d, dmin)` f16 pair shared by `GPS=8` consecutive groups
// (`wd`) - `256/32`, the GGUF Q4_K/Q5_K super-block's own element count over
// this kernel's fixed 32-element group. `f16_to_f32` is the same magic-
// multiply/FTZ-safe-subnormal/inf-nan construction `kernels::template::
// f16_decode_expr` already generates for the bf16/f16 WEIGHT STORAGE tier
// (`crates/kernels/src/template.rs`) - reused here verbatim (same magic
// constants `0x77800000`/`0x38800000`/`0x7F800000`) rather than reinvented,
// since this repo has no native f16 WGSL type and no `unpack2x16float`/
// `extractBits` lowering on its CPU JIT. `kq_ds_dm` decodes once per (column,
// group) fold - the SAME frequency the old flat `wsz` read already ran at,
// so no additional hoisting is needed for correctness; a further hoist
// across several folds sharing one `wd` word is a possible follow-up, not
// attempted here (see the M14 ledger entry for why: this fold point is
// already the natural "once per several elements" granularity, and a deeper
// hoist would need restructuring this kernel's software-pipelined chunk loop
// for a benefit this file's own gating did not require).
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1Fu;
    // FTZ-safe subnormal (magic bias subtract) and normal (magic multiply)
    // branches - see `kernels::template::f16_decode_expr`'s doc comment for
    // the full derivation of both magic constants.
    let subnormal = bitcast<f32>(0x38800000u | ((h & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
    let normal = bitcast<f32>((h & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
    let inf_or_nan = bitcast<f32>(0x7F800000u | ((h & 0x3FFu) << 13u));
    let mag = select(select(subnormal, normal, exp != 0u), inf_or_nan, exp == 31u);
    return bitcast<f32>(bitcast<u32>(mag) | sign);
}

// One weight-scale group `sg` of output column `col`'s `(ds, dm)` pair,
// reading `wsm`/`wd` directly - `vec2(0,0)` for a column past `ncols` (the
// same ragged-tile OOB guard the old flat-`wsz` read's `select(0.0, ..)`
// performed). `GPS` groups share one `wd` word (`sblk = sg/GPS`).
const GPS: u32 = 8u; // groups per wd super-block entry: 256 (Q4_K/Q5_K super-block) / 32 (this kernel's group)
fn kq_ds_dm(col: u32, ncols: u32, sg: u32, wsm_per_row: u32, wd_per_row: u32) -> vec2<f32> {
    if (col >= ncols) { return vec2<f32>(0.0, 0.0); }
    let wword = wd[col * wd_per_row + sg / GPS];
    let d = f16_to_f32(wword & 0xFFFFu);
    let dmin = f16_to_f32(wword >> 16u);
    let sword = wsm[col * wsm_per_row + sg / 2u];
    let shift = select(0u, 16u, (sg % 2u) == 1u);
    let scv = f32((sword >> shift) & 0xFFu);
    let mv = f32((sword >> (shift + 8u)) & 0xFFu);
    return vec2<f32>(d * scv, dmin * mv);
}

// @workgroup_size(256). Not CPU-JIT'able (multi-barrier work-group); the CPU
// int8 reference lives in the validation test, so parity is still gated.

struct Params { m: u32, k: u32, n: u32 };  // k = RAW LOGICAL K (see header)

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, k/4]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, k*CODE_BITS/32] unsigned codes
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M] per-token activation scale
@group(0) @binding(4) var<storage, read>       wsm: array<u32>;  // [N, ceil(k/32/2)] packed (sc, m) pairs
@group(0) @binding(5) var<storage, read>       wd:  array<u32>;  // [N, ceil(k/32/GPS)] packed (d, dmin) f16 pairs
@group(0) @binding(6) var<storage, read>       xgs: array<f32>;  // [M, k/32] activation group sums
@group(0) @binding(7) var<storage, read_write> out: array<f32>;  // [M, N]

const CODE_BITS: u32 = 8u;  // template knob: 4 (Q4_K) or 8 (Q5_K)

const BM: u32 = 128u;
const BN: u32 = 128u;
const BKG: u32 = 8u;    // quads (4-element units, matching xq's word density) per chunk = one weight-scale group (32 elements)
const BKQ: u32 = 2u;    // BKG / 4 - vec4-of-quads loads per row per chunk
const QPG: u32 = 2u;    // quads per weight-scale group. FIXED at 2 (group=32) - this
                         // kernel serves only Q4_K/Q5_K, both group=32; Q6_K's group=16
                         // reaches the device through the EXISTING symmetric kernels'
                         // own QPG knob instead, not this one.
const SP4: u32 = 3u;    // padded shared stride in vec4s (BKQ + 1), bank-spread
const LN: u32 = 16u;    // lane grid: 16 x 16 threads, stride-16 interleave
const RS: u32 = 48u;    // LN * SP4 - vec4 step between a thread's own rows

var<workgroup> As: array<vec4<u32>, 384>;  // BM * SP4, row-major: As[r*SP4 + q]
var<workgroup> Bs: array<vec4<u32>, 384>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = lid.x;
    let ty = tid / LN;
    let tx = tid % LN;
    let wg = wgid.y * nwg.x + wgid.x;
    let tiles_n = (p.n + BN - 1u) / BN;
    let row0 = (wg / tiles_n) * BM;
    let col0 = (wg % tiles_n) * BN;

    let kgx = p.k / 4u;                       // quads per row (xq's own word density)
    let ng = p.k / 32u;                       // weight-scale groups per row
    let wq_row_words = p.k * CODE_BITS / 32u; // wq words per row
    let qpwq = (32u / CODE_BITS) / 4u;        // quads packed per raw wq word
    let cmask = (1u << CODE_BITS) - 1u;       // low-CODE_BITS mask
    let wsm_per_row = (ng + 1u) / 2u;         // wsm words per row (M14)
    let wd_per_row = (ng + GPS - 1u) / GPS;   // wd words per row (M14)

    // Staging assignment: one vec4 of A and one of B per thread, same
    // row/quad split matmul_i8_dyn uses.
    let sr = tid / BKQ;
    let sq = tid % BKQ;
    let arow = row0 + sr;
    let brow = col0 + sr;
    let a_ok = arow < p.m;
    let b_ok = brow < p.n;
    let a_base = arow * kgx;
    let w_base = brow * wq_row_words;
    let sh_idx = sr * SP4 + sq;

    // 64 int32 accumulators.
    var c00 = 0i; var c01 = 0i; var c02 = 0i; var c03 = 0i; var c04 = 0i; var c05 = 0i; var c06 = 0i; var c07 = 0i;
    var c10 = 0i; var c11 = 0i; var c12 = 0i; var c13 = 0i; var c14 = 0i; var c15 = 0i; var c16 = 0i; var c17 = 0i;
    var c20 = 0i; var c21 = 0i; var c22 = 0i; var c23 = 0i; var c24 = 0i; var c25 = 0i; var c26 = 0i; var c27 = 0i;
    var c30 = 0i; var c31 = 0i; var c32 = 0i; var c33 = 0i; var c34 = 0i; var c35 = 0i; var c36 = 0i; var c37 = 0i;
    var c40 = 0i; var c41 = 0i; var c42 = 0i; var c43 = 0i; var c44 = 0i; var c45 = 0i; var c46 = 0i; var c47 = 0i;
    var c50 = 0i; var c51 = 0i; var c52 = 0i; var c53 = 0i; var c54 = 0i; var c55 = 0i; var c56 = 0i; var c57 = 0i;
    var c60 = 0i; var c61 = 0i; var c62 = 0i; var c63 = 0i; var c64 = 0i; var c65 = 0i; var c66 = 0i; var c67 = 0i;
    var c70 = 0i; var c71 = 0i; var c72 = 0i; var c73 = 0i; var c74 = 0i; var c75 = 0i; var c76 = 0i; var c77 = 0i;

    // 64 f32 running totals.
    var d00 = 0.0; var d01 = 0.0; var d02 = 0.0; var d03 = 0.0; var d04 = 0.0; var d05 = 0.0; var d06 = 0.0; var d07 = 0.0;
    var d10 = 0.0; var d11 = 0.0; var d12 = 0.0; var d13 = 0.0; var d14 = 0.0; var d15 = 0.0; var d16 = 0.0; var d17 = 0.0;
    var d20 = 0.0; var d21 = 0.0; var d22 = 0.0; var d23 = 0.0; var d24 = 0.0; var d25 = 0.0; var d26 = 0.0; var d27 = 0.0;
    var d30 = 0.0; var d31 = 0.0; var d32 = 0.0; var d33 = 0.0; var d34 = 0.0; var d35 = 0.0; var d36 = 0.0; var d37 = 0.0;
    var d40 = 0.0; var d41 = 0.0; var d42 = 0.0; var d43 = 0.0; var d44 = 0.0; var d45 = 0.0; var d46 = 0.0; var d47 = 0.0;
    var d50 = 0.0; var d51 = 0.0; var d52 = 0.0; var d53 = 0.0; var d54 = 0.0; var d55 = 0.0; var d56 = 0.0; var d57 = 0.0;
    var d60 = 0.0; var d61 = 0.0; var d62 = 0.0; var d63 = 0.0; var d64 = 0.0; var d65 = 0.0; var d66 = 0.0; var d67 = 0.0;
    var d70 = 0.0; var d71 = 0.0; var d72 = 0.0; var d73 = 0.0; var d74 = 0.0; var d75 = 0.0; var d76 = 0.0; var d77 = 0.0;

    var rA: vec4<u32>;
    var rB: vec4<u32>;

    let nchunks = (kgx + BKG - 1u) / BKG;
    // This thread's eight output rows/columns, in the stride-16 interleave
    // the epilogue AND the correction-term fold both read.
    let cc = col0 + tx;
    let m0 = row0 + ty + 0u;   let m1 = row0 + ty + 16u;  let m2 = row0 + ty + 32u;  let m3 = row0 + ty + 48u;
    let m4 = row0 + ty + 64u;  let m5 = row0 + ty + 80u;  let m6 = row0 + ty + 96u;  let m7 = row0 + ty + 112u;

    // Prime chunk 0.
    {
        let g0 = sq * 4u;
        var av = vec4<u32>(0u, 0u, 0u, 0u);
        if (a_ok && g0 + 3u < kgx) {
            av = vec4<u32>(xq[a_base + g0], xq[a_base + g0 + 1u], xq[a_base + g0 + 2u], xq[a_base + g0 + 3u]);
        } else if (a_ok) {
            if (g0 + 0u < kgx) { av.x = xq[a_base + g0]; }
            if (g0 + 1u < kgx) { av.y = xq[a_base + g0 + 1u]; }
            if (g0 + 2u < kgx) { av.z = xq[a_base + g0 + 2u]; }
        }
        var bv = vec4<u32>(0u, 0u, 0u, 0u);
        if (b_ok) {
            for (var li: u32 = 0u; li < 4u; li = li + 1u) {
                let gq = g0 + li;
                if (gq < kgx) {
                    let word_idx = gq / qpwq;
                    let qoff = gq % qpwq;
                    let src = wq[w_base + word_idx];
                    let base_bit = qoff * 4u * CODE_BITS;
                    let u0 = (src >> (base_bit + 0u * CODE_BITS)) & cmask;
                    let u1 = (src >> (base_bit + 1u * CODE_BITS)) & cmask;
                    let u2 = (src >> (base_bit + 2u * CODE_BITS)) & cmask;
                    let u3 = (src >> (base_bit + 3u * CODE_BITS)) & cmask;
                    bv[li] = u0 | (u1 << 8u) | (u2 << 16u) | (u3 << 24u);
                }
            }
        }
        As[sh_idx] = av;
        Bs[sh_idx] = bv;
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let g1 = (c + 1u) * BKG + sq * 4u;
            rA = vec4<u32>(0u, 0u, 0u, 0u);
            if (a_ok && g1 + 3u < kgx) {
                rA = vec4<u32>(xq[a_base + g1], xq[a_base + g1 + 1u], xq[a_base + g1 + 2u], xq[a_base + g1 + 3u]);
            } else if (a_ok) {
                if (g1 + 0u < kgx) { rA.x = xq[a_base + g1]; }
                if (g1 + 1u < kgx) { rA.y = xq[a_base + g1 + 1u]; }
                if (g1 + 2u < kgx) { rA.z = xq[a_base + g1 + 2u]; }
            }
            rB = vec4<u32>(0u, 0u, 0u, 0u);
            if (b_ok) {
                for (var li: u32 = 0u; li < 4u; li = li + 1u) {
                    let gq = g1 + li;
                    if (gq < kgx) {
                        let word_idx = gq / qpwq;
                        let qoff = gq % qpwq;
                        let src = wq[w_base + word_idx];
                        let base_bit = qoff * 4u * CODE_BITS;
                        let u0 = (src >> (base_bit + 0u * CODE_BITS)) & cmask;
                        let u1 = (src >> (base_bit + 1u * CODE_BITS)) & cmask;
                        let u2 = (src >> (base_bit + 2u * CODE_BITS)) & cmask;
                        let u3 = (src >> (base_bit + 3u * CODE_BITS)) & cmask;
                        rB[li] = u0 | (u1 << 8u) | (u2 << 16u) | (u3 << 24u);
                    }
                }
            }
        }
        for (var q = 0u; q < BKQ; q = q + 1u) {
            let ao = ty * SP4 + q;
            let bo = tx * SP4 + q;
            let a0 = As[ao];
            let a1 = As[ao + RS];
            let a2 = As[ao + 2u * RS];
            let a3 = As[ao + 3u * RS];
            let a4 = As[ao + 4u * RS];
            let a5 = As[ao + 5u * RS];
            let a6 = As[ao + 6u * RS];
            let a7 = As[ao + 7u * RS];
            {
                let b = Bs[bo];
                c00 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c10 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c20 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c30 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c40 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c50 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c60 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c70 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            {
                let b = Bs[bo + RS];
                c01 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c11 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c21 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c31 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c41 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c51 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c61 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c71 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            {
                let b = Bs[bo + 2u * RS];
                c02 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c12 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c22 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c32 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c42 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c52 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c62 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c72 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            {
                let b = Bs[bo + 3u * RS];
                c03 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c13 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c23 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c33 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c43 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c53 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c63 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c73 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            {
                let b = Bs[bo + 4u * RS];
                c04 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c14 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c24 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c34 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c44 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c54 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c64 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c74 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            {
                let b = Bs[bo + 5u * RS];
                c05 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c15 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c25 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c35 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c45 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c55 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c65 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c75 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            {
                let b = Bs[bo + 6u * RS];
                c06 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c16 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c26 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c36 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c46 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c56 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c66 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c76 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            {
                let b = Bs[bo + 7u * RS];
                c07 += dot4I8Packed(a0.x, b.x) + dot4I8Packed(a0.y, b.y) + dot4I8Packed(a0.z, b.z) + dot4I8Packed(a0.w, b.w);
                c17 += dot4I8Packed(a1.x, b.x) + dot4I8Packed(a1.y, b.y) + dot4I8Packed(a1.z, b.z) + dot4I8Packed(a1.w, b.w);
                c27 += dot4I8Packed(a2.x, b.x) + dot4I8Packed(a2.y, b.y) + dot4I8Packed(a2.z, b.z) + dot4I8Packed(a2.w, b.w);
                c37 += dot4I8Packed(a3.x, b.x) + dot4I8Packed(a3.y, b.y) + dot4I8Packed(a3.z, b.z) + dot4I8Packed(a3.w, b.w);
                c47 += dot4I8Packed(a4.x, b.x) + dot4I8Packed(a4.y, b.y) + dot4I8Packed(a4.z, b.z) + dot4I8Packed(a4.w, b.w);
                c57 += dot4I8Packed(a5.x, b.x) + dot4I8Packed(a5.y, b.y) + dot4I8Packed(a5.z, b.z) + dot4I8Packed(a5.w, b.w);
                c67 += dot4I8Packed(a6.x, b.x) + dot4I8Packed(a6.y, b.y) + dot4I8Packed(a6.z, b.z) + dot4I8Packed(a6.w, b.w);
                c77 += dot4I8Packed(a7.x, b.x) + dot4I8Packed(a7.y, b.y) + dot4I8Packed(a7.z, b.z) + dot4I8Packed(a7.w, b.w);
            }
            // Every QPG-th quad completes one weight-scale group: fold the
            // live integer sums into the f32 running totals through `ds`,
            // MINUS the affine `dm[n,g]*S[m,g]` correction, then clear for
            // the next group. `dm`/`e` vary per COLUMN (one read per fold,
            // broadcast across all 8 rows); `s` (the activation group sum)
            // varies per ROW (one read per fold, broadcast across all 8
            // columns) - the two combine as an outer product over the 8x8
            // block, exactly the affine correction's own `Σ_g(... - dm*S)`
            // structure.
            if ((q + 1u) % QPG == 0u) {
                let sgu = (c * BKQ + q) / QPG;
                let sg = select(0u, sgu, sgu < ng);
                let v0 = kq_ds_dm(cc +   0u, p.n, sg, wsm_per_row, wd_per_row);
                let v1 = kq_ds_dm(cc +  16u, p.n, sg, wsm_per_row, wd_per_row);
                let v2 = kq_ds_dm(cc +  32u, p.n, sg, wsm_per_row, wd_per_row);
                let v3 = kq_ds_dm(cc +  48u, p.n, sg, wsm_per_row, wd_per_row);
                let v4 = kq_ds_dm(cc +  64u, p.n, sg, wsm_per_row, wd_per_row);
                let v5 = kq_ds_dm(cc +  80u, p.n, sg, wsm_per_row, wd_per_row);
                let v6 = kq_ds_dm(cc +  96u, p.n, sg, wsm_per_row, wd_per_row);
                let v7 = kq_ds_dm(cc + 112u, p.n, sg, wsm_per_row, wd_per_row);
                let e0 = v0.x; let dm0 = v0.y;
                let e1 = v1.x; let dm1 = v1.y;
                let e2 = v2.x; let dm2 = v2.y;
                let e3 = v3.x; let dm3 = v3.y;
                let e4 = v4.x; let dm4 = v4.y;
                let e5 = v5.x; let dm5 = v5.y;
                let e6 = v6.x; let dm6 = v6.y;
                let e7 = v7.x; let dm7 = v7.y;
                let s0 = select(0.0, xgs[m0 * ng + sg], m0 < p.m);
                let s1 = select(0.0, xgs[m1 * ng + sg], m1 < p.m);
                let s2 = select(0.0, xgs[m2 * ng + sg], m2 < p.m);
                let s3 = select(0.0, xgs[m3 * ng + sg], m3 < p.m);
                let s4 = select(0.0, xgs[m4 * ng + sg], m4 < p.m);
                let s5 = select(0.0, xgs[m5 * ng + sg], m5 < p.m);
                let s6 = select(0.0, xgs[m6 * ng + sg], m6 < p.m);
                let s7 = select(0.0, xgs[m7 * ng + sg], m7 < p.m);
                d00 += f32(c00) * e0 - dm0 * s0; d01 += f32(c01) * e1 - dm1 * s0; d02 += f32(c02) * e2 - dm2 * s0; d03 += f32(c03) * e3 - dm3 * s0; d04 += f32(c04) * e4 - dm4 * s0; d05 += f32(c05) * e5 - dm5 * s0; d06 += f32(c06) * e6 - dm6 * s0; d07 += f32(c07) * e7 - dm7 * s0;
                d10 += f32(c10) * e0 - dm0 * s1; d11 += f32(c11) * e1 - dm1 * s1; d12 += f32(c12) * e2 - dm2 * s1; d13 += f32(c13) * e3 - dm3 * s1; d14 += f32(c14) * e4 - dm4 * s1; d15 += f32(c15) * e5 - dm5 * s1; d16 += f32(c16) * e6 - dm6 * s1; d17 += f32(c17) * e7 - dm7 * s1;
                d20 += f32(c20) * e0 - dm0 * s2; d21 += f32(c21) * e1 - dm1 * s2; d22 += f32(c22) * e2 - dm2 * s2; d23 += f32(c23) * e3 - dm3 * s2; d24 += f32(c24) * e4 - dm4 * s2; d25 += f32(c25) * e5 - dm5 * s2; d26 += f32(c26) * e6 - dm6 * s2; d27 += f32(c27) * e7 - dm7 * s2;
                d30 += f32(c30) * e0 - dm0 * s3; d31 += f32(c31) * e1 - dm1 * s3; d32 += f32(c32) * e2 - dm2 * s3; d33 += f32(c33) * e3 - dm3 * s3; d34 += f32(c34) * e4 - dm4 * s3; d35 += f32(c35) * e5 - dm5 * s3; d36 += f32(c36) * e6 - dm6 * s3; d37 += f32(c37) * e7 - dm7 * s3;
                d40 += f32(c40) * e0 - dm0 * s4; d41 += f32(c41) * e1 - dm1 * s4; d42 += f32(c42) * e2 - dm2 * s4; d43 += f32(c43) * e3 - dm3 * s4; d44 += f32(c44) * e4 - dm4 * s4; d45 += f32(c45) * e5 - dm5 * s4; d46 += f32(c46) * e6 - dm6 * s4; d47 += f32(c47) * e7 - dm7 * s4;
                d50 += f32(c50) * e0 - dm0 * s5; d51 += f32(c51) * e1 - dm1 * s5; d52 += f32(c52) * e2 - dm2 * s5; d53 += f32(c53) * e3 - dm3 * s5; d54 += f32(c54) * e4 - dm4 * s5; d55 += f32(c55) * e5 - dm5 * s5; d56 += f32(c56) * e6 - dm6 * s5; d57 += f32(c57) * e7 - dm7 * s5;
                d60 += f32(c60) * e0 - dm0 * s6; d61 += f32(c61) * e1 - dm1 * s6; d62 += f32(c62) * e2 - dm2 * s6; d63 += f32(c63) * e3 - dm3 * s6; d64 += f32(c64) * e4 - dm4 * s6; d65 += f32(c65) * e5 - dm5 * s6; d66 += f32(c66) * e6 - dm6 * s6; d67 += f32(c67) * e7 - dm7 * s6;
                d70 += f32(c70) * e0 - dm0 * s7; d71 += f32(c71) * e1 - dm1 * s7; d72 += f32(c72) * e2 - dm2 * s7; d73 += f32(c73) * e3 - dm3 * s7; d74 += f32(c74) * e4 - dm4 * s7; d75 += f32(c75) * e5 - dm5 * s7; d76 += f32(c76) * e6 - dm6 * s7; d77 += f32(c77) * e7 - dm7 * s7;
                c00 = 0i; c01 = 0i; c02 = 0i; c03 = 0i; c04 = 0i; c05 = 0i; c06 = 0i; c07 = 0i;
                c10 = 0i; c11 = 0i; c12 = 0i; c13 = 0i; c14 = 0i; c15 = 0i; c16 = 0i; c17 = 0i;
                c20 = 0i; c21 = 0i; c22 = 0i; c23 = 0i; c24 = 0i; c25 = 0i; c26 = 0i; c27 = 0i;
                c30 = 0i; c31 = 0i; c32 = 0i; c33 = 0i; c34 = 0i; c35 = 0i; c36 = 0i; c37 = 0i;
                c40 = 0i; c41 = 0i; c42 = 0i; c43 = 0i; c44 = 0i; c45 = 0i; c46 = 0i; c47 = 0i;
                c50 = 0i; c51 = 0i; c52 = 0i; c53 = 0i; c54 = 0i; c55 = 0i; c56 = 0i; c57 = 0i;
                c60 = 0i; c61 = 0i; c62 = 0i; c63 = 0i; c64 = 0i; c65 = 0i; c66 = 0i; c67 = 0i;
                c70 = 0i; c71 = 0i; c72 = 0i; c73 = 0i; c74 = 0i; c75 = 0i; c76 = 0i; c77 = 0i;
            }
        }
        workgroupBarrier();
        if (has_next) {
            As[sh_idx] = rA;
            Bs[sh_idx] = rB;
        }
        workgroupBarrier();
    }

    // Guarded stores: thread (ty,tx) owns rows ty+16i and columns tx+16j. The
    // weight scale AND the min correction are already inside `dXY` (applied
    // per k-chunk above), so the epilogue only has the per-token activation
    // scale left to apply.
    let sv0 = select(0.0, sx[m0], m0 < p.m); let sv1 = select(0.0, sx[m1], m1 < p.m);
    let sv2 = select(0.0, sx[m2], m2 < p.m); let sv3 = select(0.0, sx[m3], m3 < p.m);
    let sv4 = select(0.0, sx[m4], m4 < p.m); let sv5 = select(0.0, sx[m5], m5 < p.m);
    let sv6 = select(0.0, sx[m6], m6 < p.m); let sv7 = select(0.0, sx[m7], m7 < p.m);

    if (m0 < p.m) {
        let r0 = m0 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r0 + 0u]   = d00 * sv0; }
        if (col0 + tx + 16u < p.n) { out[r0 + 16u]  = d01 * sv0; }
        if (col0 + tx + 32u < p.n) { out[r0 + 32u]  = d02 * sv0; }
        if (col0 + tx + 48u < p.n) { out[r0 + 48u]  = d03 * sv0; }
        if (col0 + tx + 64u < p.n) { out[r0 + 64u]  = d04 * sv0; }
        if (col0 + tx + 80u < p.n) { out[r0 + 80u]  = d05 * sv0; }
        if (col0 + tx + 96u < p.n) { out[r0 + 96u]  = d06 * sv0; }
        if (col0 + tx + 112u < p.n) { out[r0 + 112u] = d07 * sv0; }
    }
    if (m1 < p.m) {
        let r1 = m1 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r1 + 0u]   = d10 * sv1; }
        if (col0 + tx + 16u < p.n) { out[r1 + 16u]  = d11 * sv1; }
        if (col0 + tx + 32u < p.n) { out[r1 + 32u]  = d12 * sv1; }
        if (col0 + tx + 48u < p.n) { out[r1 + 48u]  = d13 * sv1; }
        if (col0 + tx + 64u < p.n) { out[r1 + 64u]  = d14 * sv1; }
        if (col0 + tx + 80u < p.n) { out[r1 + 80u]  = d15 * sv1; }
        if (col0 + tx + 96u < p.n) { out[r1 + 96u]  = d16 * sv1; }
        if (col0 + tx + 112u < p.n) { out[r1 + 112u] = d17 * sv1; }
    }
    if (m2 < p.m) {
        let r2 = m2 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r2 + 0u]   = d20 * sv2; }
        if (col0 + tx + 16u < p.n) { out[r2 + 16u]  = d21 * sv2; }
        if (col0 + tx + 32u < p.n) { out[r2 + 32u]  = d22 * sv2; }
        if (col0 + tx + 48u < p.n) { out[r2 + 48u]  = d23 * sv2; }
        if (col0 + tx + 64u < p.n) { out[r2 + 64u]  = d24 * sv2; }
        if (col0 + tx + 80u < p.n) { out[r2 + 80u]  = d25 * sv2; }
        if (col0 + tx + 96u < p.n) { out[r2 + 96u]  = d26 * sv2; }
        if (col0 + tx + 112u < p.n) { out[r2 + 112u] = d27 * sv2; }
    }
    if (m3 < p.m) {
        let r3 = m3 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r3 + 0u]   = d30 * sv3; }
        if (col0 + tx + 16u < p.n) { out[r3 + 16u]  = d31 * sv3; }
        if (col0 + tx + 32u < p.n) { out[r3 + 32u]  = d32 * sv3; }
        if (col0 + tx + 48u < p.n) { out[r3 + 48u]  = d33 * sv3; }
        if (col0 + tx + 64u < p.n) { out[r3 + 64u]  = d34 * sv3; }
        if (col0 + tx + 80u < p.n) { out[r3 + 80u]  = d35 * sv3; }
        if (col0 + tx + 96u < p.n) { out[r3 + 96u]  = d36 * sv3; }
        if (col0 + tx + 112u < p.n) { out[r3 + 112u] = d37 * sv3; }
    }
    if (m4 < p.m) {
        let r4 = m4 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r4 + 0u]   = d40 * sv4; }
        if (col0 + tx + 16u < p.n) { out[r4 + 16u]  = d41 * sv4; }
        if (col0 + tx + 32u < p.n) { out[r4 + 32u]  = d42 * sv4; }
        if (col0 + tx + 48u < p.n) { out[r4 + 48u]  = d43 * sv4; }
        if (col0 + tx + 64u < p.n) { out[r4 + 64u]  = d44 * sv4; }
        if (col0 + tx + 80u < p.n) { out[r4 + 80u]  = d45 * sv4; }
        if (col0 + tx + 96u < p.n) { out[r4 + 96u]  = d46 * sv4; }
        if (col0 + tx + 112u < p.n) { out[r4 + 112u] = d47 * sv4; }
    }
    if (m5 < p.m) {
        let r5 = m5 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r5 + 0u]   = d50 * sv5; }
        if (col0 + tx + 16u < p.n) { out[r5 + 16u]  = d51 * sv5; }
        if (col0 + tx + 32u < p.n) { out[r5 + 32u]  = d52 * sv5; }
        if (col0 + tx + 48u < p.n) { out[r5 + 48u]  = d53 * sv5; }
        if (col0 + tx + 64u < p.n) { out[r5 + 64u]  = d54 * sv5; }
        if (col0 + tx + 80u < p.n) { out[r5 + 80u]  = d55 * sv5; }
        if (col0 + tx + 96u < p.n) { out[r5 + 96u]  = d56 * sv5; }
        if (col0 + tx + 112u < p.n) { out[r5 + 112u] = d57 * sv5; }
    }
    if (m6 < p.m) {
        let r6 = m6 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r6 + 0u]   = d60 * sv6; }
        if (col0 + tx + 16u < p.n) { out[r6 + 16u]  = d61 * sv6; }
        if (col0 + tx + 32u < p.n) { out[r6 + 32u]  = d62 * sv6; }
        if (col0 + tx + 48u < p.n) { out[r6 + 48u]  = d63 * sv6; }
        if (col0 + tx + 64u < p.n) { out[r6 + 64u]  = d64 * sv6; }
        if (col0 + tx + 80u < p.n) { out[r6 + 80u]  = d65 * sv6; }
        if (col0 + tx + 96u < p.n) { out[r6 + 96u]  = d66 * sv6; }
        if (col0 + tx + 112u < p.n) { out[r6 + 112u] = d67 * sv6; }
    }
    if (m7 < p.m) {
        let r7 = m7 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n)   { out[r7 + 0u]   = d70 * sv7; }
        if (col0 + tx + 16u < p.n) { out[r7 + 16u]  = d71 * sv7; }
        if (col0 + tx + 32u < p.n) { out[r7 + 32u]  = d72 * sv7; }
        if (col0 + tx + 48u < p.n) { out[r7 + 48u]  = d73 * sv7; }
        if (col0 + tx + 64u < p.n) { out[r7 + 64u]  = d74 * sv7; }
        if (col0 + tx + 80u < p.n) { out[r7 + 80u]  = d75 * sv7; }
        if (col0 + tx + 96u < p.n) { out[r7 + 96u]  = d76 * sv7; }
        if (col0 + tx + 112u < p.n) { out[r7 + 112u] = d77 * sv7; }
    }
}
