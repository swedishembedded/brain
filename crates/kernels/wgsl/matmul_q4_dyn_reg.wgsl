// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled W4A8 GEMM with a DYNAMIC per-token activation scale and a GROUP-WISE (32-element) weight scale - the prefill/DiT q4 GEMM, register-tiled + DP4A
// @how   nibble-unpack into DP4A packed int8, k-major shared tiles, register block per thread, per-chunk group dequant, 256-thread workgroup tile, 3 barriers
// @opt   5
// @cpu   no
// @gpu   yes-wg256
// @npu   yes
// @quant int8
// @dtype f32
//
// `@quant int8`, not `q4` - see `matmul_q4_gemv_reg.wgsl`'s header for why:
// the field tracks the packed-integer arithmetic primitive in use
// (`dot4I8Packed`), not an operand's storage width, and this is the q4
// kernel that adopts it.
//
// The register-tiled sibling of `matmul_q4_dyn.wgsl` - that kernel is
// deliberately naive (one thread per output element, 8 scalar nibble MACs
// per weight word), documented there as "correct first, then freeze" with
// this kernel as "the documented follow-on optimization". Mirrors
// `matmul_i8_dyn.wgsl`'s 128x128 register-tiled shape (same `BM`/`BN`, same
// interleaved register-block ownership, same per-chunk group-wise dequant
// fold), with `dot4I8Packed` fed by nibble-unpacked weight bytes instead of
// int8 bytes read directly off the wire.
//
//   x_q : [M, K/4] u32   - 4 int8 activations packed along K per u32 (row-major)
//   w_q : [N, K/8] u32   - 8 int4 weights    packed along K per u32 (row-major)
//   sx  : [M]      f32   - per-token activation scale
//   sw  : [N, K/32] f32  - GROUP-WISE weight scale (`model::int8::GROUP` = 32,
//         shared with the int8 tier - Q4_0's block is Q8_0's)
//   out : [M, N]   f32   - out[m,n] = sx[m] * sum_g acc_i32[m,n,g] * sw[n,g]
//
// K must be a multiple of 32 (the group), which subsumes the multiple-of-8
// (weight packing) and multiple-of-4 (activation packing) both operands need.
//
// ## Why one chunk is FOUR weight words, not eight
//
// `matmul_i8_dyn` chunks by `BKG = 8` packed int8 words because 8 packed
// int8 words is exactly one 32-element weight-scale group (4 int8/word).
// Q4 packs TWICE as many values per word (8 nibbles/word), so a q4
// weight-scale group is only `BKGW = 4` packed q4 words - and since the
// activation stays int8 (W4A8: only the weight narrows further, see
// `model::int4`'s module doc), the SAME 32-element chunk spans `BKGX = 8`
// packed activation words, TWICE the weight side. The two operands are
// therefore staged into separately-shaped tiles (`As` sized `BKGX * SP`,
// `Bs` sized `BKGW * SP`) rather than the single symmetric shape
// `matmul_i8_dyn` uses for both - the price of one operand being twice as
// dense as the other, not a design choice.
//
// One q4 weight word `wv` at chunk-relative position `kw` (0..BKGW) pairs
// with exactly two activation words at positions `2*kw` (nibbles 0..3) and
// `2*kw + 1` (nibbles 4..7) of the SAME chunk - the identical pairing
// `matmul_q4_gemv_reg.wgsl` uses, and for the same reason: nibble `b` of
// `wv` is one logical K value, and K values `[8*kw, 8*kw+4)` are exactly the
// four int8 activations packed into word `2*kw`.
//
// ## k-major, scalar shared loads - NOT the vec4/k-group-minor layout
//
// `matmul_i8_dyn`'s own header explains why ITS tile reads four k-groups per
// shared-memory load instruction instead of one: DP4A retires four times the
// MACs per instruction that FFMA does, so a k-major tile with one scalar
// load per DP4A leaves the load-store pipe as the limiter rather than the
// arithmetic pipe. That fix is a property of the ARITHMETIC WIDTH of the
// instruction, not of this kernel's tile geometry, so it applies here too in
// principle - but this kernel keeps the SIMPLER, `matmul_reg3`/
// `matmul_i8.wgsl`-style k-major scalar-load tile deliberately, as a
// well-understood, orthogonal follow-on left for once this shape is
// profiled at real model scale rather than guessed at up front for a kernel
// no production model dispatches yet.
//
// Register-block ownership is INTERLEAVED exactly as `matmul_i8_dyn`'s:
// thread ty/tx owns rows/cols {ty, ty+16, ty+32, ...} so the 16 threads of a
// tx-group read 16 CONSECUTIVE shared words (no bank conflict) and the
// epilogue's global stores are 16 consecutive elements per instruction.
// Padded stride `SP = 129` (`BM+1`/`BN+1`), the same fix `matmul_i8.wgsl`
// documents for the identical reason.
//
// ## Where the group scale is applied
//
// One chunk == one weight-scale group by construction (`BKGW` words = 32
// nibbles = 32 logical K = one group), so the dequantize lands on the chunk
// boundary exactly as `matmul_i8_dyn`'s does: 64 i32 accumulators sum a
// chunk exactly (integer, exact, re-orderable), then fold into 64 f32
// running totals scaled by that chunk's `sw[n, c]`. The per-token `sx[m]`
// scale is applied once, in the epilogue.
//
// @workgroup_size(256). Not CPU-JIT'able (multi-barrier work-group); the
// naive `matmul_q4_dyn.wgsl` is the CPU-side / correctness-oracle kernel this
// one is checked against, not a fallback this kernel itself provides.

struct Params { m: u32, k: u32, n: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, k/4]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, k/8]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N, k/32]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

const BM: u32 = 128u;
const BN: u32 = 128u;
const BKGW: u32 = 4u;  // packed q4-weight words per chunk (= one weight-scale group)
const BKGX: u32 = 8u;  // packed x (int8) words per chunk (= 2 * BKGW, same logical K span)
const SP: u32 = 129u;  // padded shared stride (BM+1 / BN+1)
const WG: u32 = 256u;
const LN: u32 = 16u;   // lane grid: 16 x 16 threads, stride-16 interleave

var<workgroup> As: array<u32, 1032>;  // BKGX * SP, k-major: As[kx*SP + r]
var<workgroup> Bs: array<u32, 516>;   // BKGW * SP, k-major: Bs[kw*SP + r]

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

    let kgx = p.k / 4u; // x words per row (int8 packing)
    let kgw = p.k / 8u; // w words per row (int4 packing)

    // Staging assignment for the A (activation) tile: BM*BKGX/WG = 4 words/thread.
    var sra: array<u32, 4>;
    var skx: array<u32, 4>;
    var arow_g: array<u32, 4>;
    for (var e = 0u; e < 4u; e = e + 1u) {
        let idx = tid + e * WG;
        sra[e] = idx / BKGX;
        skx[e] = idx % BKGX;
        arow_g[e] = row0 + sra[e];
    }
    // Staging assignment for the B (weight) tile: BM*BKGW/WG = 2 words/thread.
    var srb: array<u32, 2>;
    var skw: array<u32, 2>;
    var brow_g: array<u32, 2>;
    for (var e = 0u; e < 2u; e = e + 1u) {
        let idx = tid + e * WG;
        srb[e] = idx / BKGW;
        skw[e] = idx % BKGW;
        brow_g[e] = col0 + srb[e];
    }

    // 64 int32 accumulators - ONE weight-scale group's worth (one chunk).
    var c00 = 0i; var c01 = 0i; var c02 = 0i; var c03 = 0i; var c04 = 0i; var c05 = 0i; var c06 = 0i; var c07 = 0i;
    var c10 = 0i; var c11 = 0i; var c12 = 0i; var c13 = 0i; var c14 = 0i; var c15 = 0i; var c16 = 0i; var c17 = 0i;
    var c20 = 0i; var c21 = 0i; var c22 = 0i; var c23 = 0i; var c24 = 0i; var c25 = 0i; var c26 = 0i; var c27 = 0i;
    var c30 = 0i; var c31 = 0i; var c32 = 0i; var c33 = 0i; var c34 = 0i; var c35 = 0i; var c36 = 0i; var c37 = 0i;
    var c40 = 0i; var c41 = 0i; var c42 = 0i; var c43 = 0i; var c44 = 0i; var c45 = 0i; var c46 = 0i; var c47 = 0i;
    var c50 = 0i; var c51 = 0i; var c52 = 0i; var c53 = 0i; var c54 = 0i; var c55 = 0i; var c56 = 0i; var c57 = 0i;
    var c60 = 0i; var c61 = 0i; var c62 = 0i; var c63 = 0i; var c64 = 0i; var c65 = 0i; var c66 = 0i; var c67 = 0i;
    var c70 = 0i; var c71 = 0i; var c72 = 0i; var c73 = 0i; var c74 = 0i; var c75 = 0i; var c76 = 0i; var c77 = 0i;

    // 64 f32 running totals - cross-chunk sum, dequantized per chunk.
    var d00 = 0.0; var d01 = 0.0; var d02 = 0.0; var d03 = 0.0; var d04 = 0.0; var d05 = 0.0; var d06 = 0.0; var d07 = 0.0;
    var d10 = 0.0; var d11 = 0.0; var d12 = 0.0; var d13 = 0.0; var d14 = 0.0; var d15 = 0.0; var d16 = 0.0; var d17 = 0.0;
    var d20 = 0.0; var d21 = 0.0; var d22 = 0.0; var d23 = 0.0; var d24 = 0.0; var d25 = 0.0; var d26 = 0.0; var d27 = 0.0;
    var d30 = 0.0; var d31 = 0.0; var d32 = 0.0; var d33 = 0.0; var d34 = 0.0; var d35 = 0.0; var d36 = 0.0; var d37 = 0.0;
    var d40 = 0.0; var d41 = 0.0; var d42 = 0.0; var d43 = 0.0; var d44 = 0.0; var d45 = 0.0; var d46 = 0.0; var d47 = 0.0;
    var d50 = 0.0; var d51 = 0.0; var d52 = 0.0; var d53 = 0.0; var d54 = 0.0; var d55 = 0.0; var d56 = 0.0; var d57 = 0.0;
    var d60 = 0.0; var d61 = 0.0; var d62 = 0.0; var d63 = 0.0; var d64 = 0.0; var d65 = 0.0; var d66 = 0.0; var d67 = 0.0;
    var d70 = 0.0; var d71 = 0.0; var d72 = 0.0; var d73 = 0.0; var d74 = 0.0; var d75 = 0.0; var d76 = 0.0; var d77 = 0.0;

    var rA: array<u32, 4>;
    var rB: array<u32, 2>;

    let nchunks = (kgw + BKGW - 1u) / BKGW;
    // Weight-scale groups per row - equal to `nchunks` under the contract
    // (K a multiple of 32 => kgw a multiple of BKGW).
    let ng = kgw / BKGW;
    let cc = col0 + tx;

    // Prime chunk 0.
    for (var e = 0u; e < 4u; e = e + 1u) {
        let gk = skx[e];
        if (arow_g[e] < p.m && gk < kgx) { As[skx[e] * SP + sra[e]] = xq[arow_g[e] * kgx + gk]; }
        else { As[skx[e] * SP + sra[e]] = 0u; }
    }
    for (var e = 0u; e < 2u; e = e + 1u) {
        let gk = skw[e];
        if (brow_g[e] < p.n && gk < kgw) { Bs[skw[e] * SP + srb[e]] = wq[brow_g[e] * kgw + gk]; }
        else { Bs[skw[e] * SP + srb[e]] = 0u; }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let kx1 = (c + 1u) * BKGX;
            for (var e = 0u; e < 4u; e = e + 1u) {
                let gk = kx1 + skx[e];
                if (arow_g[e] < p.m && gk < kgx) { rA[e] = xq[arow_g[e] * kgx + gk]; } else { rA[e] = 0u; }
            }
            let kw1 = (c + 1u) * BKGW;
            for (var e = 0u; e < 2u; e = e + 1u) {
                let gk = kw1 + skw[e];
                if (brow_g[e] < p.n && gk < kgw) { rB[e] = wq[brow_g[e] * kgw + gk]; } else { rB[e] = 0u; }
            }
        }

        for (var kw = 0u; kw < BKGW; kw = kw + 1u) {
            let bo = kw * SP + tx;
            let wv0 = Bs[bo + 0u];
            let wv1 = Bs[bo + 16u];
            let wv2 = Bs[bo + 32u];
            let wv3 = Bs[bo + 48u];
            let wv4 = Bs[bo + 64u];
            let wv5 = Bs[bo + 80u];
            let wv6 = Bs[bo + 96u];
            let wv7 = Bs[bo + 112u];

            // Unpack each column's weight word ONCE into two DP4A-packed
            // int8 words: `lo` holds nibbles 0..3 (paired with As row
            // `2*kw`), `hi` holds nibbles 4..7 (paired with `2*kw + 1`).
            var wlo0 = 0u; var whi0 = 0u; var wlo1 = 0u; var whi1 = 0u;
            var wlo2 = 0u; var whi2 = 0u; var wlo3 = 0u; var whi3 = 0u;
            var wlo4 = 0u; var whi4 = 0u; var wlo5 = 0u; var whi5 = 0u;
            var wlo6 = 0u; var whi6 = 0u; var wlo7 = 0u; var whi7 = 0u;
            for (var b: u32 = 0u; b < 4u; b = b + 1u) {
                let sh_lo = 28u - 4u * b;
                let sh_hi = 28u - 4u * (b + 4u);
                let shift = 8u * b;
                wlo0 = wlo0 | ((bitcast<u32>(bitcast<i32>(wv0 << sh_lo) >> 28u) & 0xffu) << shift);
                whi0 = whi0 | ((bitcast<u32>(bitcast<i32>(wv0 << sh_hi) >> 28u) & 0xffu) << shift);
                wlo1 = wlo1 | ((bitcast<u32>(bitcast<i32>(wv1 << sh_lo) >> 28u) & 0xffu) << shift);
                whi1 = whi1 | ((bitcast<u32>(bitcast<i32>(wv1 << sh_hi) >> 28u) & 0xffu) << shift);
                wlo2 = wlo2 | ((bitcast<u32>(bitcast<i32>(wv2 << sh_lo) >> 28u) & 0xffu) << shift);
                whi2 = whi2 | ((bitcast<u32>(bitcast<i32>(wv2 << sh_hi) >> 28u) & 0xffu) << shift);
                wlo3 = wlo3 | ((bitcast<u32>(bitcast<i32>(wv3 << sh_lo) >> 28u) & 0xffu) << shift);
                whi3 = whi3 | ((bitcast<u32>(bitcast<i32>(wv3 << sh_hi) >> 28u) & 0xffu) << shift);
                wlo4 = wlo4 | ((bitcast<u32>(bitcast<i32>(wv4 << sh_lo) >> 28u) & 0xffu) << shift);
                whi4 = whi4 | ((bitcast<u32>(bitcast<i32>(wv4 << sh_hi) >> 28u) & 0xffu) << shift);
                wlo5 = wlo5 | ((bitcast<u32>(bitcast<i32>(wv5 << sh_lo) >> 28u) & 0xffu) << shift);
                whi5 = whi5 | ((bitcast<u32>(bitcast<i32>(wv5 << sh_hi) >> 28u) & 0xffu) << shift);
                wlo6 = wlo6 | ((bitcast<u32>(bitcast<i32>(wv6 << sh_lo) >> 28u) & 0xffu) << shift);
                whi6 = whi6 | ((bitcast<u32>(bitcast<i32>(wv6 << sh_hi) >> 28u) & 0xffu) << shift);
                wlo7 = wlo7 | ((bitcast<u32>(bitcast<i32>(wv7 << sh_lo) >> 28u) & 0xffu) << shift);
                whi7 = whi7 | ((bitcast<u32>(bitcast<i32>(wv7 << sh_hi) >> 28u) & 0xffu) << shift);
            }

            let ao_lo = (2u * kw) * SP + ty;
            let ao_hi = (2u * kw + 1u) * SP + ty;
            let a0l = As[ao_lo + 0u];   let a0h = As[ao_hi + 0u];
            let a1l = As[ao_lo + 16u];  let a1h = As[ao_hi + 16u];
            let a2l = As[ao_lo + 32u];  let a2h = As[ao_hi + 32u];
            let a3l = As[ao_lo + 48u];  let a3h = As[ao_hi + 48u];
            let a4l = As[ao_lo + 64u];  let a4h = As[ao_hi + 64u];
            let a5l = As[ao_lo + 80u];  let a5h = As[ao_hi + 80u];
            let a6l = As[ao_lo + 96u];  let a6h = As[ao_hi + 96u];
            let a7l = As[ao_lo + 112u]; let a7h = As[ao_hi + 112u];

            c00 += dot4I8Packed(a0l, wlo0) + dot4I8Packed(a0h, whi0); c01 += dot4I8Packed(a0l, wlo1) + dot4I8Packed(a0h, whi1); c02 += dot4I8Packed(a0l, wlo2) + dot4I8Packed(a0h, whi2); c03 += dot4I8Packed(a0l, wlo3) + dot4I8Packed(a0h, whi3); c04 += dot4I8Packed(a0l, wlo4) + dot4I8Packed(a0h, whi4); c05 += dot4I8Packed(a0l, wlo5) + dot4I8Packed(a0h, whi5); c06 += dot4I8Packed(a0l, wlo6) + dot4I8Packed(a0h, whi6); c07 += dot4I8Packed(a0l, wlo7) + dot4I8Packed(a0h, whi7);
            c10 += dot4I8Packed(a1l, wlo0) + dot4I8Packed(a1h, whi0); c11 += dot4I8Packed(a1l, wlo1) + dot4I8Packed(a1h, whi1); c12 += dot4I8Packed(a1l, wlo2) + dot4I8Packed(a1h, whi2); c13 += dot4I8Packed(a1l, wlo3) + dot4I8Packed(a1h, whi3); c14 += dot4I8Packed(a1l, wlo4) + dot4I8Packed(a1h, whi4); c15 += dot4I8Packed(a1l, wlo5) + dot4I8Packed(a1h, whi5); c16 += dot4I8Packed(a1l, wlo6) + dot4I8Packed(a1h, whi6); c17 += dot4I8Packed(a1l, wlo7) + dot4I8Packed(a1h, whi7);
            c20 += dot4I8Packed(a2l, wlo0) + dot4I8Packed(a2h, whi0); c21 += dot4I8Packed(a2l, wlo1) + dot4I8Packed(a2h, whi1); c22 += dot4I8Packed(a2l, wlo2) + dot4I8Packed(a2h, whi2); c23 += dot4I8Packed(a2l, wlo3) + dot4I8Packed(a2h, whi3); c24 += dot4I8Packed(a2l, wlo4) + dot4I8Packed(a2h, whi4); c25 += dot4I8Packed(a2l, wlo5) + dot4I8Packed(a2h, whi5); c26 += dot4I8Packed(a2l, wlo6) + dot4I8Packed(a2h, whi6); c27 += dot4I8Packed(a2l, wlo7) + dot4I8Packed(a2h, whi7);
            c30 += dot4I8Packed(a3l, wlo0) + dot4I8Packed(a3h, whi0); c31 += dot4I8Packed(a3l, wlo1) + dot4I8Packed(a3h, whi1); c32 += dot4I8Packed(a3l, wlo2) + dot4I8Packed(a3h, whi2); c33 += dot4I8Packed(a3l, wlo3) + dot4I8Packed(a3h, whi3); c34 += dot4I8Packed(a3l, wlo4) + dot4I8Packed(a3h, whi4); c35 += dot4I8Packed(a3l, wlo5) + dot4I8Packed(a3h, whi5); c36 += dot4I8Packed(a3l, wlo6) + dot4I8Packed(a3h, whi6); c37 += dot4I8Packed(a3l, wlo7) + dot4I8Packed(a3h, whi7);
            c40 += dot4I8Packed(a4l, wlo0) + dot4I8Packed(a4h, whi0); c41 += dot4I8Packed(a4l, wlo1) + dot4I8Packed(a4h, whi1); c42 += dot4I8Packed(a4l, wlo2) + dot4I8Packed(a4h, whi2); c43 += dot4I8Packed(a4l, wlo3) + dot4I8Packed(a4h, whi3); c44 += dot4I8Packed(a4l, wlo4) + dot4I8Packed(a4h, whi4); c45 += dot4I8Packed(a4l, wlo5) + dot4I8Packed(a4h, whi5); c46 += dot4I8Packed(a4l, wlo6) + dot4I8Packed(a4h, whi6); c47 += dot4I8Packed(a4l, wlo7) + dot4I8Packed(a4h, whi7);
            c50 += dot4I8Packed(a5l, wlo0) + dot4I8Packed(a5h, whi0); c51 += dot4I8Packed(a5l, wlo1) + dot4I8Packed(a5h, whi1); c52 += dot4I8Packed(a5l, wlo2) + dot4I8Packed(a5h, whi2); c53 += dot4I8Packed(a5l, wlo3) + dot4I8Packed(a5h, whi3); c54 += dot4I8Packed(a5l, wlo4) + dot4I8Packed(a5h, whi4); c55 += dot4I8Packed(a5l, wlo5) + dot4I8Packed(a5h, whi5); c56 += dot4I8Packed(a5l, wlo6) + dot4I8Packed(a5h, whi6); c57 += dot4I8Packed(a5l, wlo7) + dot4I8Packed(a5h, whi7);
            c60 += dot4I8Packed(a6l, wlo0) + dot4I8Packed(a6h, whi0); c61 += dot4I8Packed(a6l, wlo1) + dot4I8Packed(a6h, whi1); c62 += dot4I8Packed(a6l, wlo2) + dot4I8Packed(a6h, whi2); c63 += dot4I8Packed(a6l, wlo3) + dot4I8Packed(a6h, whi3); c64 += dot4I8Packed(a6l, wlo4) + dot4I8Packed(a6h, whi4); c65 += dot4I8Packed(a6l, wlo5) + dot4I8Packed(a6h, whi5); c66 += dot4I8Packed(a6l, wlo6) + dot4I8Packed(a6h, whi6); c67 += dot4I8Packed(a6l, wlo7) + dot4I8Packed(a6h, whi7);
            c70 += dot4I8Packed(a7l, wlo0) + dot4I8Packed(a7h, whi0); c71 += dot4I8Packed(a7l, wlo1) + dot4I8Packed(a7h, whi1); c72 += dot4I8Packed(a7l, wlo2) + dot4I8Packed(a7h, whi2); c73 += dot4I8Packed(a7l, wlo3) + dot4I8Packed(a7h, whi3); c74 += dot4I8Packed(a7l, wlo4) + dot4I8Packed(a7h, whi4); c75 += dot4I8Packed(a7l, wlo5) + dot4I8Packed(a7h, whi5); c76 += dot4I8Packed(a7l, wlo6) + dot4I8Packed(a7h, whi6); c77 += dot4I8Packed(a7l, wlo7) + dot4I8Packed(a7h, whi7);
        }

        // One chunk == one weight-scale group: fold this group's exact
        // integer sums into the f32 running totals through the group's own
        // scale, then clear them for the next group.
        {
            let sg = select(0u, c, c < ng);
            let e0 = select(0.0, sw[(cc +   0u) * ng + sg], cc +   0u < p.n);
            let e1 = select(0.0, sw[(cc +  16u) * ng + sg], cc +  16u < p.n);
            let e2 = select(0.0, sw[(cc +  32u) * ng + sg], cc +  32u < p.n);
            let e3 = select(0.0, sw[(cc +  48u) * ng + sg], cc +  48u < p.n);
            let e4 = select(0.0, sw[(cc +  64u) * ng + sg], cc +  64u < p.n);
            let e5 = select(0.0, sw[(cc +  80u) * ng + sg], cc +  80u < p.n);
            let e6 = select(0.0, sw[(cc +  96u) * ng + sg], cc +  96u < p.n);
            let e7 = select(0.0, sw[(cc + 112u) * ng + sg], cc + 112u < p.n);
            d00 += f32(c00) * e0; d01 += f32(c01) * e1; d02 += f32(c02) * e2; d03 += f32(c03) * e3; d04 += f32(c04) * e4; d05 += f32(c05) * e5; d06 += f32(c06) * e6; d07 += f32(c07) * e7;
            d10 += f32(c10) * e0; d11 += f32(c11) * e1; d12 += f32(c12) * e2; d13 += f32(c13) * e3; d14 += f32(c14) * e4; d15 += f32(c15) * e5; d16 += f32(c16) * e6; d17 += f32(c17) * e7;
            d20 += f32(c20) * e0; d21 += f32(c21) * e1; d22 += f32(c22) * e2; d23 += f32(c23) * e3; d24 += f32(c24) * e4; d25 += f32(c25) * e5; d26 += f32(c26) * e6; d27 += f32(c27) * e7;
            d30 += f32(c30) * e0; d31 += f32(c31) * e1; d32 += f32(c32) * e2; d33 += f32(c33) * e3; d34 += f32(c34) * e4; d35 += f32(c35) * e5; d36 += f32(c36) * e6; d37 += f32(c37) * e7;
            d40 += f32(c40) * e0; d41 += f32(c41) * e1; d42 += f32(c42) * e2; d43 += f32(c43) * e3; d44 += f32(c44) * e4; d45 += f32(c45) * e5; d46 += f32(c46) * e6; d47 += f32(c47) * e7;
            d50 += f32(c50) * e0; d51 += f32(c51) * e1; d52 += f32(c52) * e2; d53 += f32(c53) * e3; d54 += f32(c54) * e4; d55 += f32(c55) * e5; d56 += f32(c56) * e6; d57 += f32(c57) * e7;
            d60 += f32(c60) * e0; d61 += f32(c61) * e1; d62 += f32(c62) * e2; d63 += f32(c63) * e3; d64 += f32(c64) * e4; d65 += f32(c65) * e5; d66 += f32(c66) * e6; d67 += f32(c67) * e7;
            d70 += f32(c70) * e0; d71 += f32(c71) * e1; d72 += f32(c72) * e2; d73 += f32(c73) * e3; d74 += f32(c74) * e4; d75 += f32(c75) * e5; d76 += f32(c76) * e6; d77 += f32(c77) * e7;
            c00 = 0i; c01 = 0i; c02 = 0i; c03 = 0i; c04 = 0i; c05 = 0i; c06 = 0i; c07 = 0i;
            c10 = 0i; c11 = 0i; c12 = 0i; c13 = 0i; c14 = 0i; c15 = 0i; c16 = 0i; c17 = 0i;
            c20 = 0i; c21 = 0i; c22 = 0i; c23 = 0i; c24 = 0i; c25 = 0i; c26 = 0i; c27 = 0i;
            c30 = 0i; c31 = 0i; c32 = 0i; c33 = 0i; c34 = 0i; c35 = 0i; c36 = 0i; c37 = 0i;
            c40 = 0i; c41 = 0i; c42 = 0i; c43 = 0i; c44 = 0i; c45 = 0i; c46 = 0i; c47 = 0i;
            c50 = 0i; c51 = 0i; c52 = 0i; c53 = 0i; c54 = 0i; c55 = 0i; c56 = 0i; c57 = 0i;
            c60 = 0i; c61 = 0i; c62 = 0i; c63 = 0i; c64 = 0i; c65 = 0i; c66 = 0i; c67 = 0i;
            c70 = 0i; c71 = 0i; c72 = 0i; c73 = 0i; c74 = 0i; c75 = 0i; c76 = 0i; c77 = 0i;
        }
        workgroupBarrier();
        if (has_next) {
            for (var e = 0u; e < 4u; e = e + 1u) { As[skx[e] * SP + sra[e]] = rA[e]; }
            for (var e = 0u; e < 2u; e = e + 1u) { Bs[skw[e] * SP + srb[e]] = rB[e]; }
        }
        workgroupBarrier();
    }

    // Guarded stores: thread (ty,tx) owns rows ty+16i and columns tx+16j. The
    // weight scale is already inside `dXY` (applied per k-chunk above), so
    // the epilogue only has the per-token activation scale left to apply.
    let m0 = row0 + ty + 0u;
    let m1 = row0 + ty + 16u;
    let m2 = row0 + ty + 32u;
    let m3 = row0 + ty + 48u;
    let m4 = row0 + ty + 64u;
    let m5 = row0 + ty + 80u;
    let m6 = row0 + ty + 96u;
    let m7 = row0 + ty + 112u;
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
