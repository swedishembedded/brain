// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled int8 (DP4A) GEMM with a DYNAMIC per-token activation scale and a GROUP-WISE (32-element) weight scale, both read from buffers - the prefill/DiT int8 GEMM
// @how   DP4A packed int8, vec4 shared tiles, register block per thread, per-k-chunk group dequant, 256-thread workgroup tile, 3 barriers
// @opt   5
// @cpu   no
// @gpu   yes-wg256
// @npu   yes
// @quant int8
// @dtype f32
//
// `matmul_i8`'s dynamic-scale sibling: both scales come from buffers rather
// than from the uniform, so one build serves every shape and the activation
// scale can be recomputed per forward.
//
//   x_q : [M, K/4] u32   - 4 int8 activations packed along K per u32 (row-major)
//   w_q : [N, K/4] u32   - 4 int8 weights    packed along K per u32 (row-major)
//   sx  : [M]      f32   - per-token activation scale
//   sw  : [N, K/32] f32  - GROUP-WISE weight scale (`model::int8::GROUP` = 32)
//   out : [M, N]   f32   - out[m,n] = sx[m] * Σ_g acc_i32[m,n,g] * sw[n,g]
//
// K must be a multiple of 32 (the group), which subsumes the multiple of 4 the
// packing needs.
//
// This is the P40's fastest inference path. DP4A (`dot4I8Packed`) does four
// int8 multiply-accumulates in one instruction, four times the MACs of an fp32
// FMA, which is the hardware `crates/vulkan/tests/peak_flops.rs` demonstrates.
// int8 weights also
// move 1/4 the bytes of fp32, so the memory side wins too.
//
// 128x128 output tile, 8x8 int32 register block per thread, 256 threads on a
// 16x16 lane grid, k-chunk of BKG packed groups (= 4*BKG int8 along K),
// software-pipelined through registers so the next chunk's global loads are in
// flight while the current one is consumed.
//
// ## Why the shared tiles are `vec4<u32>` and k-group-MINOR
//
// The staging arrays hold `[row][k-group]` with the k-groups CONTIGUOUS, four
// to a `vec4<u32>`, rather than the k-major `[k-group][row]` a textbook tile
// uses. That is a THROUGHPUT decision, not a layout preference: with a k-major
// scalar tile the inner loop retires four DP4A per shared-memory load
// instruction, and a Pascal SM issues four times as many integer/FMA lanes per
// clock as load-store lanes - so the shared traffic and the arithmetic are the
// same order and the kernel runs against its load-store issue rate rather than
// against DP4A. Reading four k-groups per load instruction raises that ratio
// four-fold and moves the limiter onto the arithmetic. The measured effect is
// in this repo's int8 roadmap ledger, reproduced by `qwen_bench gemm8`.
//
// Two consequences the layout has to pay for, both handled here:
//
//  1. The staging LOAD becomes four consecutive `x`/`w` words per thread -
//     the same 32-byte-per-eight-threads global pattern the k-major form had,
//     since the k-group axis is already the fastest-varying axis of both
//     operands. Nothing is transposed on the way in.
//  2. The shared STRIDE is padded to `SP4` vec4s per row (one more than the
//     `BKG/4` actually used), so the 16 lanes of a tx-group read 16 addresses
//     whose bank indices are distinct. An unpadded stride puts them on a
//     quarter of the banks and costs a four-way conflict on every B read.
//
// The A operands for a whole k-group quad are hoisted into registers once and
// reused across the eight B columns, so only ONE B vec4 is live at a time;
// hoisting both sides would push the register block past the point where two
// workgroups still fit on an SM, which is the occupancy the double-buffered
// staging depends on.
//
// Register-block ownership is INTERLEAVED: thread ty/tx owns rows/cols
// {ty, ty+16, ty+32, …} instead of {8*ty … 8*ty+7}, so the 16 threads of a
// tx-group read 16 consecutive shared rows and the epilogue's global stores
// become 16 consecutive elements per instruction instead of a stride-8 scatter.
//
// ## Where the group scale is applied, and what it costs
//
// `BKG` is 8 packed groups = 32 int8 along K, which is EXACTLY one weight-scale
// group. So one k-chunk of this kernel is one quantization group, and the
// dequantize lands on the chunk boundary: the 64 integer accumulators sum a
// chunk exactly as before, then fold into 64 f32 running totals scaled by that
// chunk's `sw[n, c]`. The integer sum inside a group is still exact and
// re-orderable; only the 64 cross-group folds per chunk are floating point.
//
// That costs 64 FMAs per 512 DP4A - about 12% of the inner loop's issue slots -
// and, more importantly, DOUBLES the accumulator register block (64 i32 for
// the live group plus 64 f32 for the running sum). On a GP102 that pushes the
// kernel from two resident 256-thread workgroups per SM to one, so some of the
// latency hiding the double-buffered staging was written for now has to come
// from the software pipeline alone. That is the price of a weight scale that
// does not span 17408 elements of K, and it is a price this repo has already
// decided to pay - see `model::int8`'s module doc for the measurement.
//
// The alternative that keeps 64 registers - convert and scale every individual
// DP4A result - was rejected: it triples the inner loop's instruction count
// (convert + FMA per dot4 instead of one integer add) in a kernel whose own
// header explains it is arithmetic-limited, not occupancy-limited.
//
// The accumulation WITHIN a group is INTEGER, so re-ordering the k-axis sum
// inside a chunk is exact; the cross-chunk sum is f32 and runs in ascending
// chunk order.
//
// @workgroup_size(256). Not CPU-JIT'able (multi-barrier work-group); the CPU
// int8 reference lives in the validation test, so parity is still gated.

struct Params { m: u32, kg: u32, n: u32 };  // dynamic sx + group-wise sw, kg = K/4

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<u32>;  // [M, kg]
@group(0) @binding(2) var<storage, read>       w:   array<u32>;  // [N, kg]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M] per-token activation scale
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N, kg/BKG] group-wise weight scale
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

const BM: u32 = 128u;
const BN: u32 = 128u;
const BKG: u32 = 8u;    // packed K-groups per chunk (= 32 int8 along K)
const BKQ: u32 = 2u;    // BKG / 4 - vec4 quads of k-groups per row per chunk
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

    // Staging assignment: one vec4 of A and one of B per thread. Threads
    // 2r and 2r+1 cover row r's two quads, i.e. eight consecutive words of
    // that row - the same global access pattern the scalar form produced.
    let sr = tid / BKQ;      // staged row within the tile
    let sq = tid % BKQ;      // which k-group quad of it
    let arow = row0 + sr;
    let brow = col0 + sr;
    let a_ok = arow < p.m;
    let b_ok = brow < p.n;
    let a_base = arow * p.kg;
    let b_base = brow * p.kg;
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

    // 64 f32 running totals: the integer block above holds ONE weight-scale
    // group (one k-chunk); these carry the dequantized sum across groups.
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

    let nchunks = (p.kg + BKG - 1u) / BKG;
    // Weight-scale groups per row. Equal to `nchunks` under the contract
    // (K a multiple of 32 => kg a multiple of BKG); computed separately, and
    // the group index clamped below, so a caller that broke the contract gets
    // a wrong number rather than a read past the binding.
    let ng = p.kg / BKG;
    // This thread's eight output columns, in the same stride-16 interleave the
    // epilogue stores with.
    let cc = col0 + tx;

    // Prime chunk 0. The whole-quad case is one uniform branch and four
    // consecutive loads; the ragged tail (`kg` not a multiple of BKG, or a
    // partial row block) falls into the per-lane guards, so no lane ever
    // forms an out-of-range index.
    {
        let g0 = sq * 4u;
        var av = vec4<u32>(0u, 0u, 0u, 0u);
        if (a_ok && g0 + 3u < p.kg) {
            av = vec4<u32>(x[a_base + g0], x[a_base + g0 + 1u], x[a_base + g0 + 2u], x[a_base + g0 + 3u]);
        } else if (a_ok) {
            if (g0 + 0u < p.kg) { av.x = x[a_base + g0]; }
            if (g0 + 1u < p.kg) { av.y = x[a_base + g0 + 1u]; }
            if (g0 + 2u < p.kg) { av.z = x[a_base + g0 + 2u]; }
        }
        var bv = vec4<u32>(0u, 0u, 0u, 0u);
        if (b_ok && g0 + 3u < p.kg) {
            bv = vec4<u32>(w[b_base + g0], w[b_base + g0 + 1u], w[b_base + g0 + 2u], w[b_base + g0 + 3u]);
        } else if (b_ok) {
            if (g0 + 0u < p.kg) { bv.x = w[b_base + g0]; }
            if (g0 + 1u < p.kg) { bv.y = w[b_base + g0 + 1u]; }
            if (g0 + 2u < p.kg) { bv.z = w[b_base + g0 + 2u]; }
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
            if (a_ok && g1 + 3u < p.kg) {
                rA = vec4<u32>(x[a_base + g1], x[a_base + g1 + 1u], x[a_base + g1 + 2u], x[a_base + g1 + 3u]);
            } else if (a_ok) {
                if (g1 + 0u < p.kg) { rA.x = x[a_base + g1]; }
                if (g1 + 1u < p.kg) { rA.y = x[a_base + g1 + 1u]; }
                if (g1 + 2u < p.kg) { rA.z = x[a_base + g1 + 2u]; }
            }
            rB = vec4<u32>(0u, 0u, 0u, 0u);
            if (b_ok && g1 + 3u < p.kg) {
                rB = vec4<u32>(w[b_base + g1], w[b_base + g1 + 1u], w[b_base + g1 + 2u], w[b_base + g1 + 3u]);
            } else if (b_ok) {
                if (g1 + 0u < p.kg) { rB.x = w[b_base + g1]; }
                if (g1 + 1u < p.kg) { rB.y = w[b_base + g1 + 1u]; }
                if (g1 + 2u < p.kg) { rB.z = w[b_base + g1 + 2u]; }
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
        }
        // One chunk == one weight-scale group: fold this group's exact integer
        // sums into the f32 running totals through the group's own scale, then
        // clear them for the next group.
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
            As[sh_idx] = rA;
            Bs[sh_idx] = rB;
        }
        workgroupBarrier();
    }

    // Guarded stores: thread (ty,tx) owns rows ty+16i and columns tx+16j. The
    // weight scale is already inside `dXY` (applied per k-chunk above), so the
    // epilogue only has the per-token activation scale left to apply.
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
