// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Register-tiled matmul (out = x @ Wᵀ) with vec4 shared-memory reads and a hand-unrolled K-step - matmul_reg3's tiling re-laid-out to spend fewer issue slots per FMA
// @how   register block per thread, 256-thread workgroup tile, 3 barriers
// @opt   5
// @cpu   native-only
// @gpu   yes-wg256
// @npu   yes
// @quant none
// @dtype f32
//
// Register-tiled matmul (out = x @ Wᵀ). Same 128x128 workgroup tile, same 8x8
// per-thread register block, same software pipelining and same Params as
// matmul_reg3 - what changes is the shared-memory layout, chosen so every
// inner-loop shared read is a `vec4<f32>`.
//
// WHY (Pascal, GP102). matmul_reg3's inner loop issues 16 scalar shared loads
// (8 A + 8 B) per 64 FMAs. Vectorizing them into 4 vec4 reads buys ISSUE SLOTS,
// NOT BANDWIDTH - and the distinction matters, because the bandwidth story is
// wrong and easy to talk oneself into:
//
//   A Pascal SM has 32 LD/ST units (8 per processing block x 4 blocks) and 32
//   shared-memory banks x 4 B = 128 B/clk. Those are BALANCED BY DESIGN: a
//   warp-wide 32-bit shared load is 128 B = one wavefront = one clock of bank
//   bandwidth, and the SM can issue one such warp instruction per clock. A
//   warp-wide 128-bit load moves 512 B but costs FOUR wavefronts - the same
//   bytes per clock. Vectorizing therefore does NOT raise shared-memory
//   throughput. (Measured: a GTX 1080 reaches 93.4% of the 128 B/clk figure
//   using scalar 32-bit accesses, and "using 64-bit data types did not make a
//   significant difference"; on GP104 Tesla P4 it is 90.7%. On Fermi, 128-bit
//   shared loads were outright SLOWER than 32-bit ones.)
//
// What a vec4 read does buy is 1 issue slot instead of 4, 4x fewer address
// IADDs and 4x fewer dependency barriers - i.e. a larger fraction of the
// instruction stream is FFMA. Lai & Seznec (CGO'13) quantify exactly this for
// SGEMM on NVIDIA: the FFMA fraction rises 75% (scalar LDS) -> 85.7% (LDS.64)
// -> 92.3% (LDS.128). Scott Gray's maxas SGEMM likewise never justifies
// LDS.128 by bandwidth ("all memory operations are dual issued in our main loop
// and don't factor into the flops calculation at all").
//
// Measured here on a real P40 (fenced, min-of-N, arms interleaved), m=n=k=2048
// under native Vulkan: 4023 (matmul_reg3) -> 4455 GFLOP/s. That +10.7% is the
// issue-slot effect, and it is the size Lai & Seznec's instruction-mix
// accounting predicts - not the 4x a bandwidth argument would have predicted.
//
// This is the "vectorize SMEM access" step of the standard CUDA SGEMM
// progression (Goto & van de Geijn's register-block micro-kernel, then float4
// loads as in siboehm's cuBLAS-parity walkthrough).
//
// WHY THE REGISTER TILE IS 8x8 AND NOT SMALLER. Shared-memory bandwidth sets a
// hard floor on the tile: 128 B/clk/SM x 30 SM x 1.531 GHz = 5.88 TB/s against
// 11.76 TFLOP/s is 0.5 B/flop of budget. A TxT register tile reads 2T floats
// per k-step to do 2T^2 flops = 4/T B/flop, so 4/T <= 0.5 forces T >= 8. A 4x4
// tile needs 1.0 B/flop, twice the budget, and is capped near 50% of peak no
// matter how well it is scheduled. 8x8 sits exactly on the bound (Volkov, GTC
// 2010, makes the same argument for Fermi).
//
// HOW THE LAYOUT MAKES THAT POSSIBLE. The tile is k-major with the ROW axis
// grouped into quads: As[kk][r] lives at quad `kk*SQ + r/4`, component `r%4`.
// A thread reads rows {4ty..4ty+3} and {64+4ty..64+4ty+3} as two vec4s, and
// columns {4tx..4tx+3} and {64+4tx..64+4tx+3} likewise - still 8 rows x 8
// columns = the same 64 accumulators, just gathered 4 at a time.
//
// BANK CONFLICTS (32 banks x 4 B):
//
//  * The B read `Bs[kk*SQ + tx]` is a 128-bit load; the hardware splits a
//    warp-wide 128-bit LDS into phases of 8 threads, and 8 threads x 16 B =
//    128 B = each of the 32 banks exactly once. Conflict-free.
//  * The A read `As[kk*SQ + ty]` takes only 2 distinct addresses per warp
//    (ty = tid/16), which is a broadcast, not a conflict.
//  * The staging STORE is scalar (the global->shared transpose is inherently a
//    scatter). The padded quad stride SQ = 33 makes the float stride 132, and
//    132 = 4 (mod 32), so the bank index is (4*kk + r) mod 32; a warp covers
//    r = 0..3 x kk = 0..7, giving 32 distinct banks. Conflict-free.
//
//    This is a real fix over matmul_reg3, whose stride of 129 floats yields
//    bank (kk + r) mod 32 - that maps the same warp's 32 lanes onto only 11
//    banks, i.e. up to a 4-way conflict its own comment did not account for.
//    reg3 removed reg2's 8-way store conflict but not all of it.
//
// HAND-UNROLLED INNER LOOP, because nothing in this toolchain will do
// it. naga is a translation library with no optimization passes, and it
// lowers EVERY WGSL `for` into a `while` with the induction variable and
// comparison as explicit body statements (gfx-rs/wgpu#6521: "naga
// exclusively generates `while` loops ... causes suboptimal code
// generation on downstream compilers, again because loops cannot be
// unrolled"). The driver's SPIR-V compiler therefore cannot see a
// known-trip-count loop: the compare/increment/branch stay in the inner
// loop stealing issue slots from FFMA, and no cross-iteration scheduling
// interleaves step k+1's loads with step k's FMAs. WGSL has no
// `#pragma unroll` and no `__launch_bounds__`, so writing the BK=8 steps
// out is the only way to express it. Measured here on a P40 (fenced,
// min-of-N, interleaved): 4455 -> 5088 GFLOP/s at m=n=k=2048.
//
// WHEN TO PREFER THIS OVER matmul_reg3. Measured on a P40 at m=1024, n=4096,
// sweeping K (fenced, min-of-N, arms interleaved, GFLOP/s reg3 -> this):
//
//   K= 4096  4536 -> 5399   K=10240  5136 -> 6126
//   K= 6144  5111 -> 6029   K=12288  4865 -> 4529   <- crosses over
//   K= 8192  5177 -> 6132   K=14336  4867 -> 4488
//
// So this kernel is the faster one by 4-18% for K up to ~10k and LOSES to
// matmul_reg3 beyond ~12k. The crossover is sharp rather than gradual, which
// points at a discrete threshold (occupancy or cache residency) rather than a
// gradual bandwidth effect; the mechanism was NOT isolated. What was ruled out
// by measurement: it is not the workgroup count / available parallelism - at a
// fixed K=14336 the ordering is identical at 256, 512 and 1024 workgroups, and
// at a fixed 256 workgroups the ordering flips with K alone. Anything
// dispatching K >= 12288 should select matmul_reg3 until this is understood.
//
// The epilogue's global stores become stride-4 rather than stride-16 scatters,
// which is a wash: the epilogue is O(M*N) against the loop's O(M*N*K).
//
// Shared use: 1 * 2 * 8 * 33 * 16 = 8448 B. fp32 only, one bind group,
// 3 storage buffers, no atomics/subgroups/f16.

struct Params { m: u32, k: u32, n: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;
const SQ: u32 = 33u;   // padded shared stride in vec4 quads (32 live + 1 pad)
const NQ: u32 = 264u;  // quads per tile = BK * SQ
const WG: u32 = 256u;
const LN: u32 = 16u;   // lane grid: 16 x 16 threads, each owning 2 row/col quads

// Quad q, component c holds row 4*q + c.
var<workgroup> As: array<vec4<f32>, 264>;
var<workgroup> Bs: array<vec4<f32>, 264>;

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

    // Each thread stages 4 A and 4 B elements; only the k-offset moves per
    // chunk. Same index map as matmul_reg3 - it is the fully-coalesced one
    // (a warp reads 4 rows x 8 consecutive k = 4 whole 32 B sectors).
    var sq: array<u32, 4>;   // destination quad
    var sc: array<u32, 4>;   // destination component within the quad
    var skk: array<u32, 4>;
    var arow_g: array<u32, 4>;
    var brow_g: array<u32, 4>;
    for (var e = 0u; e < 4u; e = e + 1u) {
        let idx = tid + e * WG;   // 0..1023
        let r = idx / BK;         // 0..127
        let kk = idx % BK;        // 0..7
        skk[e] = kk;
        sq[e] = kk * SQ + (r >> 2u);
        sc[e] = r & 3u;
        arow_g[e] = row0 + r;
        brow_g[e] = col0 + r;
    }

    // 64 scalar-register accumulators (unrolled -> real registers, not spill).
    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0; var c04 = 0.0; var c05 = 0.0; var c06 = 0.0; var c07 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0; var c14 = 0.0; var c15 = 0.0; var c16 = 0.0; var c17 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0; var c24 = 0.0; var c25 = 0.0; var c26 = 0.0; var c27 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0; var c34 = 0.0; var c35 = 0.0; var c36 = 0.0; var c37 = 0.0;
    var c40 = 0.0; var c41 = 0.0; var c42 = 0.0; var c43 = 0.0; var c44 = 0.0; var c45 = 0.0; var c46 = 0.0; var c47 = 0.0;
    var c50 = 0.0; var c51 = 0.0; var c52 = 0.0; var c53 = 0.0; var c54 = 0.0; var c55 = 0.0; var c56 = 0.0; var c57 = 0.0;
    var c60 = 0.0; var c61 = 0.0; var c62 = 0.0; var c63 = 0.0; var c64 = 0.0; var c65 = 0.0; var c66 = 0.0; var c67 = 0.0;
    var c70 = 0.0; var c71 = 0.0; var c72 = 0.0; var c73 = 0.0; var c74 = 0.0; var c75 = 0.0; var c76 = 0.0; var c77 = 0.0;

    var rA: array<f32, 4>;
    var rB: array<f32, 4>;

    let nchunks = (p.k + BK - 1u) / BK;

    // Prime: load chunk 0 into shared.
    for (var e = 0u; e < 4u; e = e + 1u) {
        let gk = skk[e];
        if (arow_g[e] < p.m && gk < p.k) { As[sq[e]][sc[e]] = x[arow_g[e] * p.k + gk]; }
        else                             { As[sq[e]][sc[e]] = 0.0; }
        if (brow_g[e] < p.n && gk < p.k) { let wi = brow_g[e] * p.k + gk; Bs[sq[e]][sc[e]] = w[wi]; }
        else                             { Bs[sq[e]][sc[e]] = 0.0; }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        let cur = 0u;
        let nxt = 0u;
        if (has_next) {
            let k1 = (c + 1u) * BK;
            for (var e = 0u; e < 4u; e = e + 1u) {
                let gk = k1 + skk[e];
                if (arow_g[e] < p.m && gk < p.k) { rA[e] = x[arow_g[e] * p.k + gk]; } else { rA[e] = 0.0; }
                if (brow_g[e] < p.n && gk < p.k) { let wi = brow_g[e] * p.k + gk; rB[e] = w[wi]; } else { rB[e] = 0.0; }
            }
        }
        let base_0 = cur + 0u * SQ;
        let av0_0 = As[base_0 + ty];
        let av1_0 = As[base_0 + LN + ty];
        let bv0_0 = Bs[base_0 + tx];
        let bv1_0 = Bs[base_0 + LN + tx];
        let a0_0 = av0_0.x;
        let a1_0 = av0_0.y;
        let a2_0 = av0_0.z;
        let a3_0 = av0_0.w;
        let a4_0 = av1_0.x;
        let a5_0 = av1_0.y;
        let a6_0 = av1_0.z;
        let a7_0 = av1_0.w;
        let b0_0 = bv0_0.x;
        let b1_0 = bv0_0.y;
        let b2_0 = bv0_0.z;
        let b3_0 = bv0_0.w;
        let b4_0 = bv1_0.x;
        let b5_0 = bv1_0.y;
        let b6_0 = bv1_0.z;
        let b7_0 = bv1_0.w;
        c00 += a0_0 * b0_0; c01 += a0_0 * b1_0; c02 += a0_0 * b2_0; c03 += a0_0 * b3_0; c04 += a0_0 * b4_0; c05 += a0_0 * b5_0; c06 += a0_0 * b6_0; c07 += a0_0 * b7_0;
        c10 += a1_0 * b0_0; c11 += a1_0 * b1_0; c12 += a1_0 * b2_0; c13 += a1_0 * b3_0; c14 += a1_0 * b4_0; c15 += a1_0 * b5_0; c16 += a1_0 * b6_0; c17 += a1_0 * b7_0;
        c20 += a2_0 * b0_0; c21 += a2_0 * b1_0; c22 += a2_0 * b2_0; c23 += a2_0 * b3_0; c24 += a2_0 * b4_0; c25 += a2_0 * b5_0; c26 += a2_0 * b6_0; c27 += a2_0 * b7_0;
        c30 += a3_0 * b0_0; c31 += a3_0 * b1_0; c32 += a3_0 * b2_0; c33 += a3_0 * b3_0; c34 += a3_0 * b4_0; c35 += a3_0 * b5_0; c36 += a3_0 * b6_0; c37 += a3_0 * b7_0;
        c40 += a4_0 * b0_0; c41 += a4_0 * b1_0; c42 += a4_0 * b2_0; c43 += a4_0 * b3_0; c44 += a4_0 * b4_0; c45 += a4_0 * b5_0; c46 += a4_0 * b6_0; c47 += a4_0 * b7_0;
        c50 += a5_0 * b0_0; c51 += a5_0 * b1_0; c52 += a5_0 * b2_0; c53 += a5_0 * b3_0; c54 += a5_0 * b4_0; c55 += a5_0 * b5_0; c56 += a5_0 * b6_0; c57 += a5_0 * b7_0;
        c60 += a6_0 * b0_0; c61 += a6_0 * b1_0; c62 += a6_0 * b2_0; c63 += a6_0 * b3_0; c64 += a6_0 * b4_0; c65 += a6_0 * b5_0; c66 += a6_0 * b6_0; c67 += a6_0 * b7_0;
        c70 += a7_0 * b0_0; c71 += a7_0 * b1_0; c72 += a7_0 * b2_0; c73 += a7_0 * b3_0; c74 += a7_0 * b4_0; c75 += a7_0 * b5_0; c76 += a7_0 * b6_0; c77 += a7_0 * b7_0;
        let base_1 = cur + 1u * SQ;
        let av0_1 = As[base_1 + ty];
        let av1_1 = As[base_1 + LN + ty];
        let bv0_1 = Bs[base_1 + tx];
        let bv1_1 = Bs[base_1 + LN + tx];
        let a0_1 = av0_1.x;
        let a1_1 = av0_1.y;
        let a2_1 = av0_1.z;
        let a3_1 = av0_1.w;
        let a4_1 = av1_1.x;
        let a5_1 = av1_1.y;
        let a6_1 = av1_1.z;
        let a7_1 = av1_1.w;
        let b0_1 = bv0_1.x;
        let b1_1 = bv0_1.y;
        let b2_1 = bv0_1.z;
        let b3_1 = bv0_1.w;
        let b4_1 = bv1_1.x;
        let b5_1 = bv1_1.y;
        let b6_1 = bv1_1.z;
        let b7_1 = bv1_1.w;
        c00 += a0_1 * b0_1; c01 += a0_1 * b1_1; c02 += a0_1 * b2_1; c03 += a0_1 * b3_1; c04 += a0_1 * b4_1; c05 += a0_1 * b5_1; c06 += a0_1 * b6_1; c07 += a0_1 * b7_1;
        c10 += a1_1 * b0_1; c11 += a1_1 * b1_1; c12 += a1_1 * b2_1; c13 += a1_1 * b3_1; c14 += a1_1 * b4_1; c15 += a1_1 * b5_1; c16 += a1_1 * b6_1; c17 += a1_1 * b7_1;
        c20 += a2_1 * b0_1; c21 += a2_1 * b1_1; c22 += a2_1 * b2_1; c23 += a2_1 * b3_1; c24 += a2_1 * b4_1; c25 += a2_1 * b5_1; c26 += a2_1 * b6_1; c27 += a2_1 * b7_1;
        c30 += a3_1 * b0_1; c31 += a3_1 * b1_1; c32 += a3_1 * b2_1; c33 += a3_1 * b3_1; c34 += a3_1 * b4_1; c35 += a3_1 * b5_1; c36 += a3_1 * b6_1; c37 += a3_1 * b7_1;
        c40 += a4_1 * b0_1; c41 += a4_1 * b1_1; c42 += a4_1 * b2_1; c43 += a4_1 * b3_1; c44 += a4_1 * b4_1; c45 += a4_1 * b5_1; c46 += a4_1 * b6_1; c47 += a4_1 * b7_1;
        c50 += a5_1 * b0_1; c51 += a5_1 * b1_1; c52 += a5_1 * b2_1; c53 += a5_1 * b3_1; c54 += a5_1 * b4_1; c55 += a5_1 * b5_1; c56 += a5_1 * b6_1; c57 += a5_1 * b7_1;
        c60 += a6_1 * b0_1; c61 += a6_1 * b1_1; c62 += a6_1 * b2_1; c63 += a6_1 * b3_1; c64 += a6_1 * b4_1; c65 += a6_1 * b5_1; c66 += a6_1 * b6_1; c67 += a6_1 * b7_1;
        c70 += a7_1 * b0_1; c71 += a7_1 * b1_1; c72 += a7_1 * b2_1; c73 += a7_1 * b3_1; c74 += a7_1 * b4_1; c75 += a7_1 * b5_1; c76 += a7_1 * b6_1; c77 += a7_1 * b7_1;
        let base_2 = cur + 2u * SQ;
        let av0_2 = As[base_2 + ty];
        let av1_2 = As[base_2 + LN + ty];
        let bv0_2 = Bs[base_2 + tx];
        let bv1_2 = Bs[base_2 + LN + tx];
        let a0_2 = av0_2.x;
        let a1_2 = av0_2.y;
        let a2_2 = av0_2.z;
        let a3_2 = av0_2.w;
        let a4_2 = av1_2.x;
        let a5_2 = av1_2.y;
        let a6_2 = av1_2.z;
        let a7_2 = av1_2.w;
        let b0_2 = bv0_2.x;
        let b1_2 = bv0_2.y;
        let b2_2 = bv0_2.z;
        let b3_2 = bv0_2.w;
        let b4_2 = bv1_2.x;
        let b5_2 = bv1_2.y;
        let b6_2 = bv1_2.z;
        let b7_2 = bv1_2.w;
        c00 += a0_2 * b0_2; c01 += a0_2 * b1_2; c02 += a0_2 * b2_2; c03 += a0_2 * b3_2; c04 += a0_2 * b4_2; c05 += a0_2 * b5_2; c06 += a0_2 * b6_2; c07 += a0_2 * b7_2;
        c10 += a1_2 * b0_2; c11 += a1_2 * b1_2; c12 += a1_2 * b2_2; c13 += a1_2 * b3_2; c14 += a1_2 * b4_2; c15 += a1_2 * b5_2; c16 += a1_2 * b6_2; c17 += a1_2 * b7_2;
        c20 += a2_2 * b0_2; c21 += a2_2 * b1_2; c22 += a2_2 * b2_2; c23 += a2_2 * b3_2; c24 += a2_2 * b4_2; c25 += a2_2 * b5_2; c26 += a2_2 * b6_2; c27 += a2_2 * b7_2;
        c30 += a3_2 * b0_2; c31 += a3_2 * b1_2; c32 += a3_2 * b2_2; c33 += a3_2 * b3_2; c34 += a3_2 * b4_2; c35 += a3_2 * b5_2; c36 += a3_2 * b6_2; c37 += a3_2 * b7_2;
        c40 += a4_2 * b0_2; c41 += a4_2 * b1_2; c42 += a4_2 * b2_2; c43 += a4_2 * b3_2; c44 += a4_2 * b4_2; c45 += a4_2 * b5_2; c46 += a4_2 * b6_2; c47 += a4_2 * b7_2;
        c50 += a5_2 * b0_2; c51 += a5_2 * b1_2; c52 += a5_2 * b2_2; c53 += a5_2 * b3_2; c54 += a5_2 * b4_2; c55 += a5_2 * b5_2; c56 += a5_2 * b6_2; c57 += a5_2 * b7_2;
        c60 += a6_2 * b0_2; c61 += a6_2 * b1_2; c62 += a6_2 * b2_2; c63 += a6_2 * b3_2; c64 += a6_2 * b4_2; c65 += a6_2 * b5_2; c66 += a6_2 * b6_2; c67 += a6_2 * b7_2;
        c70 += a7_2 * b0_2; c71 += a7_2 * b1_2; c72 += a7_2 * b2_2; c73 += a7_2 * b3_2; c74 += a7_2 * b4_2; c75 += a7_2 * b5_2; c76 += a7_2 * b6_2; c77 += a7_2 * b7_2;
        let base_3 = cur + 3u * SQ;
        let av0_3 = As[base_3 + ty];
        let av1_3 = As[base_3 + LN + ty];
        let bv0_3 = Bs[base_3 + tx];
        let bv1_3 = Bs[base_3 + LN + tx];
        let a0_3 = av0_3.x;
        let a1_3 = av0_3.y;
        let a2_3 = av0_3.z;
        let a3_3 = av0_3.w;
        let a4_3 = av1_3.x;
        let a5_3 = av1_3.y;
        let a6_3 = av1_3.z;
        let a7_3 = av1_3.w;
        let b0_3 = bv0_3.x;
        let b1_3 = bv0_3.y;
        let b2_3 = bv0_3.z;
        let b3_3 = bv0_3.w;
        let b4_3 = bv1_3.x;
        let b5_3 = bv1_3.y;
        let b6_3 = bv1_3.z;
        let b7_3 = bv1_3.w;
        c00 += a0_3 * b0_3; c01 += a0_3 * b1_3; c02 += a0_3 * b2_3; c03 += a0_3 * b3_3; c04 += a0_3 * b4_3; c05 += a0_3 * b5_3; c06 += a0_3 * b6_3; c07 += a0_3 * b7_3;
        c10 += a1_3 * b0_3; c11 += a1_3 * b1_3; c12 += a1_3 * b2_3; c13 += a1_3 * b3_3; c14 += a1_3 * b4_3; c15 += a1_3 * b5_3; c16 += a1_3 * b6_3; c17 += a1_3 * b7_3;
        c20 += a2_3 * b0_3; c21 += a2_3 * b1_3; c22 += a2_3 * b2_3; c23 += a2_3 * b3_3; c24 += a2_3 * b4_3; c25 += a2_3 * b5_3; c26 += a2_3 * b6_3; c27 += a2_3 * b7_3;
        c30 += a3_3 * b0_3; c31 += a3_3 * b1_3; c32 += a3_3 * b2_3; c33 += a3_3 * b3_3; c34 += a3_3 * b4_3; c35 += a3_3 * b5_3; c36 += a3_3 * b6_3; c37 += a3_3 * b7_3;
        c40 += a4_3 * b0_3; c41 += a4_3 * b1_3; c42 += a4_3 * b2_3; c43 += a4_3 * b3_3; c44 += a4_3 * b4_3; c45 += a4_3 * b5_3; c46 += a4_3 * b6_3; c47 += a4_3 * b7_3;
        c50 += a5_3 * b0_3; c51 += a5_3 * b1_3; c52 += a5_3 * b2_3; c53 += a5_3 * b3_3; c54 += a5_3 * b4_3; c55 += a5_3 * b5_3; c56 += a5_3 * b6_3; c57 += a5_3 * b7_3;
        c60 += a6_3 * b0_3; c61 += a6_3 * b1_3; c62 += a6_3 * b2_3; c63 += a6_3 * b3_3; c64 += a6_3 * b4_3; c65 += a6_3 * b5_3; c66 += a6_3 * b6_3; c67 += a6_3 * b7_3;
        c70 += a7_3 * b0_3; c71 += a7_3 * b1_3; c72 += a7_3 * b2_3; c73 += a7_3 * b3_3; c74 += a7_3 * b4_3; c75 += a7_3 * b5_3; c76 += a7_3 * b6_3; c77 += a7_3 * b7_3;
        let base_4 = cur + 4u * SQ;
        let av0_4 = As[base_4 + ty];
        let av1_4 = As[base_4 + LN + ty];
        let bv0_4 = Bs[base_4 + tx];
        let bv1_4 = Bs[base_4 + LN + tx];
        let a0_4 = av0_4.x;
        let a1_4 = av0_4.y;
        let a2_4 = av0_4.z;
        let a3_4 = av0_4.w;
        let a4_4 = av1_4.x;
        let a5_4 = av1_4.y;
        let a6_4 = av1_4.z;
        let a7_4 = av1_4.w;
        let b0_4 = bv0_4.x;
        let b1_4 = bv0_4.y;
        let b2_4 = bv0_4.z;
        let b3_4 = bv0_4.w;
        let b4_4 = bv1_4.x;
        let b5_4 = bv1_4.y;
        let b6_4 = bv1_4.z;
        let b7_4 = bv1_4.w;
        c00 += a0_4 * b0_4; c01 += a0_4 * b1_4; c02 += a0_4 * b2_4; c03 += a0_4 * b3_4; c04 += a0_4 * b4_4; c05 += a0_4 * b5_4; c06 += a0_4 * b6_4; c07 += a0_4 * b7_4;
        c10 += a1_4 * b0_4; c11 += a1_4 * b1_4; c12 += a1_4 * b2_4; c13 += a1_4 * b3_4; c14 += a1_4 * b4_4; c15 += a1_4 * b5_4; c16 += a1_4 * b6_4; c17 += a1_4 * b7_4;
        c20 += a2_4 * b0_4; c21 += a2_4 * b1_4; c22 += a2_4 * b2_4; c23 += a2_4 * b3_4; c24 += a2_4 * b4_4; c25 += a2_4 * b5_4; c26 += a2_4 * b6_4; c27 += a2_4 * b7_4;
        c30 += a3_4 * b0_4; c31 += a3_4 * b1_4; c32 += a3_4 * b2_4; c33 += a3_4 * b3_4; c34 += a3_4 * b4_4; c35 += a3_4 * b5_4; c36 += a3_4 * b6_4; c37 += a3_4 * b7_4;
        c40 += a4_4 * b0_4; c41 += a4_4 * b1_4; c42 += a4_4 * b2_4; c43 += a4_4 * b3_4; c44 += a4_4 * b4_4; c45 += a4_4 * b5_4; c46 += a4_4 * b6_4; c47 += a4_4 * b7_4;
        c50 += a5_4 * b0_4; c51 += a5_4 * b1_4; c52 += a5_4 * b2_4; c53 += a5_4 * b3_4; c54 += a5_4 * b4_4; c55 += a5_4 * b5_4; c56 += a5_4 * b6_4; c57 += a5_4 * b7_4;
        c60 += a6_4 * b0_4; c61 += a6_4 * b1_4; c62 += a6_4 * b2_4; c63 += a6_4 * b3_4; c64 += a6_4 * b4_4; c65 += a6_4 * b5_4; c66 += a6_4 * b6_4; c67 += a6_4 * b7_4;
        c70 += a7_4 * b0_4; c71 += a7_4 * b1_4; c72 += a7_4 * b2_4; c73 += a7_4 * b3_4; c74 += a7_4 * b4_4; c75 += a7_4 * b5_4; c76 += a7_4 * b6_4; c77 += a7_4 * b7_4;
        let base_5 = cur + 5u * SQ;
        let av0_5 = As[base_5 + ty];
        let av1_5 = As[base_5 + LN + ty];
        let bv0_5 = Bs[base_5 + tx];
        let bv1_5 = Bs[base_5 + LN + tx];
        let a0_5 = av0_5.x;
        let a1_5 = av0_5.y;
        let a2_5 = av0_5.z;
        let a3_5 = av0_5.w;
        let a4_5 = av1_5.x;
        let a5_5 = av1_5.y;
        let a6_5 = av1_5.z;
        let a7_5 = av1_5.w;
        let b0_5 = bv0_5.x;
        let b1_5 = bv0_5.y;
        let b2_5 = bv0_5.z;
        let b3_5 = bv0_5.w;
        let b4_5 = bv1_5.x;
        let b5_5 = bv1_5.y;
        let b6_5 = bv1_5.z;
        let b7_5 = bv1_5.w;
        c00 += a0_5 * b0_5; c01 += a0_5 * b1_5; c02 += a0_5 * b2_5; c03 += a0_5 * b3_5; c04 += a0_5 * b4_5; c05 += a0_5 * b5_5; c06 += a0_5 * b6_5; c07 += a0_5 * b7_5;
        c10 += a1_5 * b0_5; c11 += a1_5 * b1_5; c12 += a1_5 * b2_5; c13 += a1_5 * b3_5; c14 += a1_5 * b4_5; c15 += a1_5 * b5_5; c16 += a1_5 * b6_5; c17 += a1_5 * b7_5;
        c20 += a2_5 * b0_5; c21 += a2_5 * b1_5; c22 += a2_5 * b2_5; c23 += a2_5 * b3_5; c24 += a2_5 * b4_5; c25 += a2_5 * b5_5; c26 += a2_5 * b6_5; c27 += a2_5 * b7_5;
        c30 += a3_5 * b0_5; c31 += a3_5 * b1_5; c32 += a3_5 * b2_5; c33 += a3_5 * b3_5; c34 += a3_5 * b4_5; c35 += a3_5 * b5_5; c36 += a3_5 * b6_5; c37 += a3_5 * b7_5;
        c40 += a4_5 * b0_5; c41 += a4_5 * b1_5; c42 += a4_5 * b2_5; c43 += a4_5 * b3_5; c44 += a4_5 * b4_5; c45 += a4_5 * b5_5; c46 += a4_5 * b6_5; c47 += a4_5 * b7_5;
        c50 += a5_5 * b0_5; c51 += a5_5 * b1_5; c52 += a5_5 * b2_5; c53 += a5_5 * b3_5; c54 += a5_5 * b4_5; c55 += a5_5 * b5_5; c56 += a5_5 * b6_5; c57 += a5_5 * b7_5;
        c60 += a6_5 * b0_5; c61 += a6_5 * b1_5; c62 += a6_5 * b2_5; c63 += a6_5 * b3_5; c64 += a6_5 * b4_5; c65 += a6_5 * b5_5; c66 += a6_5 * b6_5; c67 += a6_5 * b7_5;
        c70 += a7_5 * b0_5; c71 += a7_5 * b1_5; c72 += a7_5 * b2_5; c73 += a7_5 * b3_5; c74 += a7_5 * b4_5; c75 += a7_5 * b5_5; c76 += a7_5 * b6_5; c77 += a7_5 * b7_5;
        let base_6 = cur + 6u * SQ;
        let av0_6 = As[base_6 + ty];
        let av1_6 = As[base_6 + LN + ty];
        let bv0_6 = Bs[base_6 + tx];
        let bv1_6 = Bs[base_6 + LN + tx];
        let a0_6 = av0_6.x;
        let a1_6 = av0_6.y;
        let a2_6 = av0_6.z;
        let a3_6 = av0_6.w;
        let a4_6 = av1_6.x;
        let a5_6 = av1_6.y;
        let a6_6 = av1_6.z;
        let a7_6 = av1_6.w;
        let b0_6 = bv0_6.x;
        let b1_6 = bv0_6.y;
        let b2_6 = bv0_6.z;
        let b3_6 = bv0_6.w;
        let b4_6 = bv1_6.x;
        let b5_6 = bv1_6.y;
        let b6_6 = bv1_6.z;
        let b7_6 = bv1_6.w;
        c00 += a0_6 * b0_6; c01 += a0_6 * b1_6; c02 += a0_6 * b2_6; c03 += a0_6 * b3_6; c04 += a0_6 * b4_6; c05 += a0_6 * b5_6; c06 += a0_6 * b6_6; c07 += a0_6 * b7_6;
        c10 += a1_6 * b0_6; c11 += a1_6 * b1_6; c12 += a1_6 * b2_6; c13 += a1_6 * b3_6; c14 += a1_6 * b4_6; c15 += a1_6 * b5_6; c16 += a1_6 * b6_6; c17 += a1_6 * b7_6;
        c20 += a2_6 * b0_6; c21 += a2_6 * b1_6; c22 += a2_6 * b2_6; c23 += a2_6 * b3_6; c24 += a2_6 * b4_6; c25 += a2_6 * b5_6; c26 += a2_6 * b6_6; c27 += a2_6 * b7_6;
        c30 += a3_6 * b0_6; c31 += a3_6 * b1_6; c32 += a3_6 * b2_6; c33 += a3_6 * b3_6; c34 += a3_6 * b4_6; c35 += a3_6 * b5_6; c36 += a3_6 * b6_6; c37 += a3_6 * b7_6;
        c40 += a4_6 * b0_6; c41 += a4_6 * b1_6; c42 += a4_6 * b2_6; c43 += a4_6 * b3_6; c44 += a4_6 * b4_6; c45 += a4_6 * b5_6; c46 += a4_6 * b6_6; c47 += a4_6 * b7_6;
        c50 += a5_6 * b0_6; c51 += a5_6 * b1_6; c52 += a5_6 * b2_6; c53 += a5_6 * b3_6; c54 += a5_6 * b4_6; c55 += a5_6 * b5_6; c56 += a5_6 * b6_6; c57 += a5_6 * b7_6;
        c60 += a6_6 * b0_6; c61 += a6_6 * b1_6; c62 += a6_6 * b2_6; c63 += a6_6 * b3_6; c64 += a6_6 * b4_6; c65 += a6_6 * b5_6; c66 += a6_6 * b6_6; c67 += a6_6 * b7_6;
        c70 += a7_6 * b0_6; c71 += a7_6 * b1_6; c72 += a7_6 * b2_6; c73 += a7_6 * b3_6; c74 += a7_6 * b4_6; c75 += a7_6 * b5_6; c76 += a7_6 * b6_6; c77 += a7_6 * b7_6;
        let base_7 = cur + 7u * SQ;
        let av0_7 = As[base_7 + ty];
        let av1_7 = As[base_7 + LN + ty];
        let bv0_7 = Bs[base_7 + tx];
        let bv1_7 = Bs[base_7 + LN + tx];
        let a0_7 = av0_7.x;
        let a1_7 = av0_7.y;
        let a2_7 = av0_7.z;
        let a3_7 = av0_7.w;
        let a4_7 = av1_7.x;
        let a5_7 = av1_7.y;
        let a6_7 = av1_7.z;
        let a7_7 = av1_7.w;
        let b0_7 = bv0_7.x;
        let b1_7 = bv0_7.y;
        let b2_7 = bv0_7.z;
        let b3_7 = bv0_7.w;
        let b4_7 = bv1_7.x;
        let b5_7 = bv1_7.y;
        let b6_7 = bv1_7.z;
        let b7_7 = bv1_7.w;
        c00 += a0_7 * b0_7; c01 += a0_7 * b1_7; c02 += a0_7 * b2_7; c03 += a0_7 * b3_7; c04 += a0_7 * b4_7; c05 += a0_7 * b5_7; c06 += a0_7 * b6_7; c07 += a0_7 * b7_7;
        c10 += a1_7 * b0_7; c11 += a1_7 * b1_7; c12 += a1_7 * b2_7; c13 += a1_7 * b3_7; c14 += a1_7 * b4_7; c15 += a1_7 * b5_7; c16 += a1_7 * b6_7; c17 += a1_7 * b7_7;
        c20 += a2_7 * b0_7; c21 += a2_7 * b1_7; c22 += a2_7 * b2_7; c23 += a2_7 * b3_7; c24 += a2_7 * b4_7; c25 += a2_7 * b5_7; c26 += a2_7 * b6_7; c27 += a2_7 * b7_7;
        c30 += a3_7 * b0_7; c31 += a3_7 * b1_7; c32 += a3_7 * b2_7; c33 += a3_7 * b3_7; c34 += a3_7 * b4_7; c35 += a3_7 * b5_7; c36 += a3_7 * b6_7; c37 += a3_7 * b7_7;
        c40 += a4_7 * b0_7; c41 += a4_7 * b1_7; c42 += a4_7 * b2_7; c43 += a4_7 * b3_7; c44 += a4_7 * b4_7; c45 += a4_7 * b5_7; c46 += a4_7 * b6_7; c47 += a4_7 * b7_7;
        c50 += a5_7 * b0_7; c51 += a5_7 * b1_7; c52 += a5_7 * b2_7; c53 += a5_7 * b3_7; c54 += a5_7 * b4_7; c55 += a5_7 * b5_7; c56 += a5_7 * b6_7; c57 += a5_7 * b7_7;
        c60 += a6_7 * b0_7; c61 += a6_7 * b1_7; c62 += a6_7 * b2_7; c63 += a6_7 * b3_7; c64 += a6_7 * b4_7; c65 += a6_7 * b5_7; c66 += a6_7 * b6_7; c67 += a6_7 * b7_7;
        c70 += a7_7 * b0_7; c71 += a7_7 * b1_7; c72 += a7_7 * b2_7; c73 += a7_7 * b3_7; c74 += a7_7 * b4_7; c75 += a7_7 * b5_7; c76 += a7_7 * b6_7; c77 += a7_7 * b7_7;
        workgroupBarrier();
        if (has_next) {
            for (var e = 0u; e < 4u; e = e + 1u) {
                As[nxt + sq[e]][sc[e]] = rA[e];
                Bs[nxt + sq[e]][sc[e]] = rB[e];
            }
        }
        workgroupBarrier();
    }

    // Thread (ty,tx) owns rows {4ty+i} and {64+4ty+i}, columns {4tx+j} and
    // {64+4tx+j}, i,j in 0..3 - the quad grouping the vec4 reads imply.
    let m0 = row0 + 4u * ty + 0u;
    let m1 = row0 + 4u * ty + 1u;
    let m2 = row0 + 4u * ty + 2u;
    let m3 = row0 + 4u * ty + 3u;
    let m4 = row0 + 4u * ty + 64u;
    let m5 = row0 + 4u * ty + 65u;
    let m6 = row0 + 4u * ty + 66u;
    let m7 = row0 + 4u * ty + 67u;

    if (m0 < p.m) {
        let r0 = m0 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r0 + 0u] = c00; }
        if (col0 + 4u * tx + 1u < p.n) { out[r0 + 1u] = c01; }
        if (col0 + 4u * tx + 2u < p.n) { out[r0 + 2u] = c02; }
        if (col0 + 4u * tx + 3u < p.n) { out[r0 + 3u] = c03; }
        if (col0 + 4u * tx + 64u < p.n) { out[r0 + 64u] = c04; }
        if (col0 + 4u * tx + 65u < p.n) { out[r0 + 65u] = c05; }
        if (col0 + 4u * tx + 66u < p.n) { out[r0 + 66u] = c06; }
        if (col0 + 4u * tx + 67u < p.n) { out[r0 + 67u] = c07; }
    }
    if (m1 < p.m) {
        let r1 = m1 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r1 + 0u] = c10; }
        if (col0 + 4u * tx + 1u < p.n) { out[r1 + 1u] = c11; }
        if (col0 + 4u * tx + 2u < p.n) { out[r1 + 2u] = c12; }
        if (col0 + 4u * tx + 3u < p.n) { out[r1 + 3u] = c13; }
        if (col0 + 4u * tx + 64u < p.n) { out[r1 + 64u] = c14; }
        if (col0 + 4u * tx + 65u < p.n) { out[r1 + 65u] = c15; }
        if (col0 + 4u * tx + 66u < p.n) { out[r1 + 66u] = c16; }
        if (col0 + 4u * tx + 67u < p.n) { out[r1 + 67u] = c17; }
    }
    if (m2 < p.m) {
        let r2 = m2 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r2 + 0u] = c20; }
        if (col0 + 4u * tx + 1u < p.n) { out[r2 + 1u] = c21; }
        if (col0 + 4u * tx + 2u < p.n) { out[r2 + 2u] = c22; }
        if (col0 + 4u * tx + 3u < p.n) { out[r2 + 3u] = c23; }
        if (col0 + 4u * tx + 64u < p.n) { out[r2 + 64u] = c24; }
        if (col0 + 4u * tx + 65u < p.n) { out[r2 + 65u] = c25; }
        if (col0 + 4u * tx + 66u < p.n) { out[r2 + 66u] = c26; }
        if (col0 + 4u * tx + 67u < p.n) { out[r2 + 67u] = c27; }
    }
    if (m3 < p.m) {
        let r3 = m3 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r3 + 0u] = c30; }
        if (col0 + 4u * tx + 1u < p.n) { out[r3 + 1u] = c31; }
        if (col0 + 4u * tx + 2u < p.n) { out[r3 + 2u] = c32; }
        if (col0 + 4u * tx + 3u < p.n) { out[r3 + 3u] = c33; }
        if (col0 + 4u * tx + 64u < p.n) { out[r3 + 64u] = c34; }
        if (col0 + 4u * tx + 65u < p.n) { out[r3 + 65u] = c35; }
        if (col0 + 4u * tx + 66u < p.n) { out[r3 + 66u] = c36; }
        if (col0 + 4u * tx + 67u < p.n) { out[r3 + 67u] = c37; }
    }
    if (m4 < p.m) {
        let r4 = m4 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r4 + 0u] = c40; }
        if (col0 + 4u * tx + 1u < p.n) { out[r4 + 1u] = c41; }
        if (col0 + 4u * tx + 2u < p.n) { out[r4 + 2u] = c42; }
        if (col0 + 4u * tx + 3u < p.n) { out[r4 + 3u] = c43; }
        if (col0 + 4u * tx + 64u < p.n) { out[r4 + 64u] = c44; }
        if (col0 + 4u * tx + 65u < p.n) { out[r4 + 65u] = c45; }
        if (col0 + 4u * tx + 66u < p.n) { out[r4 + 66u] = c46; }
        if (col0 + 4u * tx + 67u < p.n) { out[r4 + 67u] = c47; }
    }
    if (m5 < p.m) {
        let r5 = m5 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r5 + 0u] = c50; }
        if (col0 + 4u * tx + 1u < p.n) { out[r5 + 1u] = c51; }
        if (col0 + 4u * tx + 2u < p.n) { out[r5 + 2u] = c52; }
        if (col0 + 4u * tx + 3u < p.n) { out[r5 + 3u] = c53; }
        if (col0 + 4u * tx + 64u < p.n) { out[r5 + 64u] = c54; }
        if (col0 + 4u * tx + 65u < p.n) { out[r5 + 65u] = c55; }
        if (col0 + 4u * tx + 66u < p.n) { out[r5 + 66u] = c56; }
        if (col0 + 4u * tx + 67u < p.n) { out[r5 + 67u] = c57; }
    }
    if (m6 < p.m) {
        let r6 = m6 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r6 + 0u] = c60; }
        if (col0 + 4u * tx + 1u < p.n) { out[r6 + 1u] = c61; }
        if (col0 + 4u * tx + 2u < p.n) { out[r6 + 2u] = c62; }
        if (col0 + 4u * tx + 3u < p.n) { out[r6 + 3u] = c63; }
        if (col0 + 4u * tx + 64u < p.n) { out[r6 + 64u] = c64; }
        if (col0 + 4u * tx + 65u < p.n) { out[r6 + 65u] = c65; }
        if (col0 + 4u * tx + 66u < p.n) { out[r6 + 66u] = c66; }
        if (col0 + 4u * tx + 67u < p.n) { out[r6 + 67u] = c67; }
    }
    if (m7 < p.m) {
        let r7 = m7 * p.n + col0 + 4u * tx;
        if (col0 + 4u * tx + 0u < p.n) { out[r7 + 0u] = c70; }
        if (col0 + 4u * tx + 1u < p.n) { out[r7 + 1u] = c71; }
        if (col0 + 4u * tx + 2u < p.n) { out[r7 + 2u] = c72; }
        if (col0 + 4u * tx + 3u < p.n) { out[r7 + 3u] = c73; }
        if (col0 + 4u * tx + 64u < p.n) { out[r7 + 64u] = c74; }
        if (col0 + 4u * tx + 65u < p.n) { out[r7 + 65u] = c75; }
        if (col0 + 4u * tx + 66u < p.n) { out[r7 + 66u] = c76; }
        if (col0 + 4u * tx + 67u < p.n) { out[r7 + 67u] = c77; }
    }
}
