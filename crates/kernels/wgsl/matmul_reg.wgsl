// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Register-tiled matmul - same math as matmul.wgsl (out = x @ Wᵀ), sized for a COMPUTE-bound discrete GPU
// @how   register block per thread, 256-thread workgroup tile, 2 barriers
// @opt   5
// @cpu   native-only
// @gpu   yes-wg256
// @npu   yes
// @quant none
// @dtype f32
//
// Register-tiled matmul — same math as matmul.wgsl (out = x @ Wᵀ), sized for a
// COMPUTE-bound discrete GPU.
//   x  : [M, K]  row-major (activations)
//   W  : [N, K]  row-major (W[n,k] = weight row n = output feature n)
//   out: [M, N]  row-major,  out[m,n] = sum_k x[m,k] * W[n,k]
//
// Why a third matmul. The ratio that decides a GEMM's ceiling is arithmetic
// intensity — FLOPs performed per byte read from device memory:
//
//   matmul.wgsl        1 output/invocation, K-loop        0.25 FLOP/byte
//   matmul_tiled.wgsl  32x32 tile, 64 invocations, 4x4    8    FLOP/byte
//   this kernel        128x128 tile, 256 invocations, 8x8  32  FLOP/byte
//
// A card is arithmetic-limited only above `peak_flops / bandwidth` - a ratio in
// the tens for a Tesla P40 (its datasheet fp32 peak over its datasheet
// bandwidth), higher still on an A100. Below that ratio the arithmetic units
// idle waiting on memory no matter how many there are, which is why the naive
// kernel cannot get within a hundredth of peak on a P40, and why the 32x32 tile
// stalls around a tenth of the card.
//
// Structure: each workgroup owns a BM x BN output tile; its 256 invocations form
// a 16x16 grid, each holding a TM x TN (8x8) micro-tile. Per K-chunk the
// workgroup cooperatively stages As[BK][BM] and Bs[BK][BN] into workgroup memory
// — k-major, so a fixed `kk` reads contiguously — then accumulates 64 products
// per invocation. Global memory is touched once per tile instead of once per
// output.
//
// The 64 partial sums are SCALAR registers, fully unrolled (c{i}{j}, row i,
// col j), not `array<f32, 64>`. This is the same lesson conv_act_reg records:
// an indexed local array is placed in *local memory* (a VRAM-backed spill
// region), and a GEMM whose accumulators live there runs at spill bandwidth
// no matter how good its tiling is. Unrolled scalars are what makes the
// register tile an actual register tile.
//
// Workgroup memory: (128+128) * 8 * 4 B = 8 KiB, inside the 16 KiB downlevel
// floor, so occupancy is bounded by registers (~96/invocation => 2 workgroups
// per SM on a P40), not by shared memory.
//
// @workgroup_size is 256, NOT the engine's usual 64: 64 invocations cannot hold
// a 128x128 tile. `backend_api::workgroup_size_of` reads the attribute below and
// every backend lays the grid out with it, so the flat id reconstruction here
// uses 256u to match.
//
// Dispatch: ceil(M/BM)*ceil(N/BN) workgroups (thread count = that * 256).
//
// The CPU JIT skips this kernel (multi-barrier work-group structure); on CPU it
// routes to the same AVX2 GEMM as `matmul`/`matmul_tiled` — one graph, three
// backends, same math.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;
const WG: u32 = 256u;

// k-major staging: As[kk * BM + r], Bs[kk * BN + r].
var<workgroup> As: array<f32, 1024>;  // BK*BM
var<workgroup> Bs: array<f32, 1024>;  // BK*BN

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = lid.x;           // 0..255
    let ty = tid / 16u;        // 0..15  (row group)
    let tx = tid % 16u;        // 0..15  (col group)
    let wg = wgid.y * nwg.x + wgid.x;
    let tiles_n = (p.n + BN - 1u) / BN;
    let row0 = (wg / tiles_n) * BM;
    let col0 = (wg % tiles_n) * BN;
    let arow = ty * 8u;        // first row of this invocation's micro-tile
    let bcol = tx * 8u;        // first col

    // 64 partial sums, one per (row i, col j) of the 8x8 micro-tile.
    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0; var c04 = 0.0; var c05 = 0.0; var c06 = 0.0; var c07 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0; var c14 = 0.0; var c15 = 0.0; var c16 = 0.0; var c17 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0; var c24 = 0.0; var c25 = 0.0; var c26 = 0.0; var c27 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0; var c34 = 0.0; var c35 = 0.0; var c36 = 0.0; var c37 = 0.0;
    var c40 = 0.0; var c41 = 0.0; var c42 = 0.0; var c43 = 0.0; var c44 = 0.0; var c45 = 0.0; var c46 = 0.0; var c47 = 0.0;
    var c50 = 0.0; var c51 = 0.0; var c52 = 0.0; var c53 = 0.0; var c54 = 0.0; var c55 = 0.0; var c56 = 0.0; var c57 = 0.0;
    var c60 = 0.0; var c61 = 0.0; var c62 = 0.0; var c63 = 0.0; var c64 = 0.0; var c65 = 0.0; var c66 = 0.0; var c67 = 0.0;
    var c70 = 0.0; var c71 = 0.0; var c72 = 0.0; var c73 = 0.0; var c74 = 0.0; var c75 = 0.0; var c76 = 0.0; var c77 = 0.0;

    let nchunks = (p.k + BK - 1u) / BK;
    for (var c = 0u; c < nchunks; c = c + 1u) {
        let k0 = c * BK;
        // Cooperative stage: 1024 As + 1024 Bs elements, 4 of each per
        // invocation. `idx % BK` is the fastest-varying term, so consecutive
        // invocations read consecutive addresses of x/w (coalesced); the
        // transpose into k-major happens on the workgroup-memory store.
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = tid + e * WG;    // 0..1023
            let r = idx / BK;          // 0..127 (row within tile)
            let kk = idx % BK;         // 0..7   (k within chunk)
            let gk = k0 + kk;
            let ar = row0 + r;
            if (ar < p.m && gk < p.k) { As[kk * BM + r] = x[ar * p.k + gk]; }
            else                      { As[kk * BM + r] = 0.0; }
            let br = col0 + r;
            if (br < p.n && gk < p.k) { Bs[kk * BN + r] = w[br * p.k + gk]; }
            else                      { Bs[kk * BN + r] = 0.0; }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let ao = kk * BM + arow;
            let bo = kk * BN + bcol;
            // 16 workgroup-memory reads feed 64 fused multiply-adds.
            let a0 = As[ao + 0u]; let a1 = As[ao + 1u]; let a2 = As[ao + 2u]; let a3 = As[ao + 3u]; let a4 = As[ao + 4u]; let a5 = As[ao + 5u]; let a6 = As[ao + 6u]; let a7 = As[ao + 7u];
            let b0 = Bs[bo + 0u]; let b1 = Bs[bo + 1u]; let b2 = Bs[bo + 2u]; let b3 = Bs[bo + 3u]; let b4 = Bs[bo + 4u]; let b5 = Bs[bo + 5u]; let b6 = Bs[bo + 6u]; let b7 = Bs[bo + 7u];
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3; c04 += a0 * b4; c05 += a0 * b5; c06 += a0 * b6; c07 += a0 * b7;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3; c14 += a1 * b4; c15 += a1 * b5; c16 += a1 * b6; c17 += a1 * b7;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3; c24 += a2 * b4; c25 += a2 * b5; c26 += a2 * b6; c27 += a2 * b7;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3; c34 += a3 * b4; c35 += a3 * b5; c36 += a3 * b6; c37 += a3 * b7;
            c40 += a4 * b0; c41 += a4 * b1; c42 += a4 * b2; c43 += a4 * b3; c44 += a4 * b4; c45 += a4 * b5; c46 += a4 * b6; c47 += a4 * b7;
            c50 += a5 * b0; c51 += a5 * b1; c52 += a5 * b2; c53 += a5 * b3; c54 += a5 * b4; c55 += a5 * b5; c56 += a5 * b6; c57 += a5 * b7;
            c60 += a6 * b0; c61 += a6 * b1; c62 += a6 * b2; c63 += a6 * b3; c64 += a6 * b4; c65 += a6 * b5; c66 += a6 * b6; c67 += a6 * b7;
            c70 += a7 * b0; c71 += a7 * b1; c72 += a7 * b2; c73 += a7 * b3; c74 += a7 * b4; c75 += a7 * b5; c76 += a7 * b6; c77 += a7 * b7;
        }
        workgroupBarrier();
    }

    // Guarded stores: M and N need not be multiples of the tile.
    let m0 = row0 + arow + 0u;
    let m1 = row0 + arow + 1u;
    let m2 = row0 + arow + 2u;
    let m3 = row0 + arow + 3u;
    let m4 = row0 + arow + 4u;
    let m5 = row0 + arow + 5u;
    let m6 = row0 + arow + 6u;
    let m7 = row0 + arow + 7u;

    if (m0 < p.m) {
        let r0 = m0 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r0 + 0u] = c00; }
        if (col0 + bcol + 1u < p.n) { out[r0 + 1u] = c01; }
        if (col0 + bcol + 2u < p.n) { out[r0 + 2u] = c02; }
        if (col0 + bcol + 3u < p.n) { out[r0 + 3u] = c03; }
        if (col0 + bcol + 4u < p.n) { out[r0 + 4u] = c04; }
        if (col0 + bcol + 5u < p.n) { out[r0 + 5u] = c05; }
        if (col0 + bcol + 6u < p.n) { out[r0 + 6u] = c06; }
        if (col0 + bcol + 7u < p.n) { out[r0 + 7u] = c07; }
    }
    if (m1 < p.m) {
        let r1 = m1 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r1 + 0u] = c10; }
        if (col0 + bcol + 1u < p.n) { out[r1 + 1u] = c11; }
        if (col0 + bcol + 2u < p.n) { out[r1 + 2u] = c12; }
        if (col0 + bcol + 3u < p.n) { out[r1 + 3u] = c13; }
        if (col0 + bcol + 4u < p.n) { out[r1 + 4u] = c14; }
        if (col0 + bcol + 5u < p.n) { out[r1 + 5u] = c15; }
        if (col0 + bcol + 6u < p.n) { out[r1 + 6u] = c16; }
        if (col0 + bcol + 7u < p.n) { out[r1 + 7u] = c17; }
    }
    if (m2 < p.m) {
        let r2 = m2 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r2 + 0u] = c20; }
        if (col0 + bcol + 1u < p.n) { out[r2 + 1u] = c21; }
        if (col0 + bcol + 2u < p.n) { out[r2 + 2u] = c22; }
        if (col0 + bcol + 3u < p.n) { out[r2 + 3u] = c23; }
        if (col0 + bcol + 4u < p.n) { out[r2 + 4u] = c24; }
        if (col0 + bcol + 5u < p.n) { out[r2 + 5u] = c25; }
        if (col0 + bcol + 6u < p.n) { out[r2 + 6u] = c26; }
        if (col0 + bcol + 7u < p.n) { out[r2 + 7u] = c27; }
    }
    if (m3 < p.m) {
        let r3 = m3 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r3 + 0u] = c30; }
        if (col0 + bcol + 1u < p.n) { out[r3 + 1u] = c31; }
        if (col0 + bcol + 2u < p.n) { out[r3 + 2u] = c32; }
        if (col0 + bcol + 3u < p.n) { out[r3 + 3u] = c33; }
        if (col0 + bcol + 4u < p.n) { out[r3 + 4u] = c34; }
        if (col0 + bcol + 5u < p.n) { out[r3 + 5u] = c35; }
        if (col0 + bcol + 6u < p.n) { out[r3 + 6u] = c36; }
        if (col0 + bcol + 7u < p.n) { out[r3 + 7u] = c37; }
    }
    if (m4 < p.m) {
        let r4 = m4 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r4 + 0u] = c40; }
        if (col0 + bcol + 1u < p.n) { out[r4 + 1u] = c41; }
        if (col0 + bcol + 2u < p.n) { out[r4 + 2u] = c42; }
        if (col0 + bcol + 3u < p.n) { out[r4 + 3u] = c43; }
        if (col0 + bcol + 4u < p.n) { out[r4 + 4u] = c44; }
        if (col0 + bcol + 5u < p.n) { out[r4 + 5u] = c45; }
        if (col0 + bcol + 6u < p.n) { out[r4 + 6u] = c46; }
        if (col0 + bcol + 7u < p.n) { out[r4 + 7u] = c47; }
    }
    if (m5 < p.m) {
        let r5 = m5 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r5 + 0u] = c50; }
        if (col0 + bcol + 1u < p.n) { out[r5 + 1u] = c51; }
        if (col0 + bcol + 2u < p.n) { out[r5 + 2u] = c52; }
        if (col0 + bcol + 3u < p.n) { out[r5 + 3u] = c53; }
        if (col0 + bcol + 4u < p.n) { out[r5 + 4u] = c54; }
        if (col0 + bcol + 5u < p.n) { out[r5 + 5u] = c55; }
        if (col0 + bcol + 6u < p.n) { out[r5 + 6u] = c56; }
        if (col0 + bcol + 7u < p.n) { out[r5 + 7u] = c57; }
    }
    if (m6 < p.m) {
        let r6 = m6 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r6 + 0u] = c60; }
        if (col0 + bcol + 1u < p.n) { out[r6 + 1u] = c61; }
        if (col0 + bcol + 2u < p.n) { out[r6 + 2u] = c62; }
        if (col0 + bcol + 3u < p.n) { out[r6 + 3u] = c63; }
        if (col0 + bcol + 4u < p.n) { out[r6 + 4u] = c64; }
        if (col0 + bcol + 5u < p.n) { out[r6 + 5u] = c65; }
        if (col0 + bcol + 6u < p.n) { out[r6 + 6u] = c66; }
        if (col0 + bcol + 7u < p.n) { out[r6 + 7u] = c67; }
    }
    if (m7 < p.m) {
        let r7 = m7 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r7 + 0u] = c70; }
        if (col0 + bcol + 1u < p.n) { out[r7 + 1u] = c71; }
        if (col0 + bcol + 2u < p.n) { out[r7 + 2u] = c72; }
        if (col0 + bcol + 3u < p.n) { out[r7 + 3u] = c73; }
        if (col0 + bcol + 4u < p.n) { out[r7 + 4u] = c74; }
        if (col0 + bcol + 5u < p.n) { out[r7 + 5u] = c75; }
        if (col0 + bcol + 6u < p.n) { out[r7 + 6u] = c76; }
        if (col0 + bcol + 7u < p.n) { out[r7 + 7u] = c77; }
    }
}
