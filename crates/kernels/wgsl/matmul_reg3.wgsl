// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Register-tiled matmul (out = x @ Wᵀ), matmul_reg2's tiling with its two
// shared-memory bank-conflict patterns removed. Same 128x128 tile, same 8x8
// per-thread register block, same software pipelining (next K-chunk's global
// loads issued into rA/rB before the current chunk's FMAs), same Params,
// same @workgroup_size(256) — a drop-in alternative selected per shape.
//
// WHAT WAS CONFLICTING IN matmul_reg2 (32 banks, 4-byte words, Pascal):
//
//  1. THE B-TILE READ. Thread tx owns 8 CONTIGUOUS columns, so it reads
//     Bs[kk*128 + tx*8 + j]. Across a warp tx = 0..15, and (8*tx) mod 32 takes
//     only 4 distinct values -> 16 addresses land on 4 banks = a 4-WAY CONFLICT
//     on 64 of the 128 shared loads each thread issues per K-chunk.
//  2. THE SHARED STORE. Staging maps idx -> (r = idx/8, kk = idx%8) and writes
//     As[kk*128 + r]; a warp covers r = 0..3 x kk = 0..7, all on 4 banks =
//     an 8-WAY CONFLICT.
//
// THE FIXES, both layout-only (the arithmetic is untouched):
//
//  1. INTERLEAVED register tiling: thread ty/tx owns rows/cols
//     {ty, ty+16, ty+32, …} instead of {8*ty … 8*ty+7}. The 16 threads of a
//     tx-group then read 16 CONSECUTIVE shared words — one per bank, no
//     conflict — and the epilogue's global stores become 16 consecutive floats
//     per instruction instead of a stride-8 scatter.
//  2. PADDED tile stride 129 instead of 128, so the staging store's bank index
//     becomes (kk + r) mod 32 rather than r mod 32.
//
// Shared use: 2 * 8 * 129 * 4 = 8256 B (vs 8192 B) — occupancy unchanged.
// fp32 only, one bind group, 3 storage buffers, no atomics/subgroups/f16.

struct Params { m: u32, k: u32, n: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;
const SP: u32 = 129u;  // padded shared stride (BM + 1)
const WG: u32 = 256u;
const LN: u32 = 16u;   // lane grid: 16 x 16 threads, stride-16 interleave

var<workgroup> As: array<f32, 1032>;  // BK*SP, k-major: As[kk*SP + r]
var<workgroup> Bs: array<f32, 1032>;

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

    // Each thread stages 4 A and 4 B elements; only the k-offset moves per chunk.
    var sr: array<u32, 4>;
    var skk: array<u32, 4>;
    var arow_g: array<u32, 4>;
    var brow_g: array<u32, 4>;
    for (var e = 0u; e < 4u; e = e + 1u) {
        let idx = tid + e * WG;   // 0..1023
        let r = idx / BK;         // 0..127
        let kk = idx % BK;        // 0..7
        sr[e] = r; skk[e] = kk;
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
        if (arow_g[e] < p.m && gk < p.k) { As[skk[e] * SP + sr[e]] = x[arow_g[e] * p.k + gk]; }
        else                             { As[skk[e] * SP + sr[e]] = 0.0; }
        if (brow_g[e] < p.n && gk < p.k) { Bs[skk[e] * SP + sr[e]] = w[brow_g[e] * p.k + gk]; }
        else                             { Bs[skk[e] * SP + sr[e]] = 0.0; }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let k1 = (c + 1u) * BK;
            for (var e = 0u; e < 4u; e = e + 1u) {
                let gk = k1 + skk[e];
                if (arow_g[e] < p.m && gk < p.k) { rA[e] = x[arow_g[e] * p.k + gk]; } else { rA[e] = 0.0; }
                if (brow_g[e] < p.n && gk < p.k) { rB[e] = w[brow_g[e] * p.k + gk]; } else { rB[e] = 0.0; }
            }
        }
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let ao = kk * SP + ty;
            let bo = kk * SP + tx;
            let a0 = As[ao + 0u];
            let a1 = As[ao + 16u];
            let a2 = As[ao + 32u];
            let a3 = As[ao + 48u];
            let a4 = As[ao + 64u];
            let a5 = As[ao + 80u];
            let a6 = As[ao + 96u];
            let a7 = As[ao + 112u];
            let b0 = Bs[bo + 0u];
            let b1 = Bs[bo + 16u];
            let b2 = Bs[bo + 32u];
            let b3 = Bs[bo + 48u];
            let b4 = Bs[bo + 64u];
            let b5 = Bs[bo + 80u];
            let b6 = Bs[bo + 96u];
            let b7 = Bs[bo + 112u];
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
        if (has_next) {
            for (var e = 0u; e < 4u; e = e + 1u) {
                As[skk[e] * SP + sr[e]] = rA[e];
                Bs[skk[e] * SP + sr[e]] = rB[e];
            }
        }
        workgroupBarrier();
    }

    // Guarded stores: thread (ty,tx) owns rows ty+16i and columns tx+16j.
    let m0 = row0 + ty + 0u;
    let m1 = row0 + ty + 16u;
    let m2 = row0 + ty + 32u;
    let m3 = row0 + ty + 48u;
    let m4 = row0 + ty + 64u;
    let m5 = row0 + ty + 80u;
    let m6 = row0 + ty + 96u;
    let m7 = row0 + ty + 112u;

    if (m0 < p.m) {
        let r0 = m0 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r0 + 0u] = c00; }
        if (col0 + tx + 16u < p.n) { out[r0 + 16u] = c01; }
        if (col0 + tx + 32u < p.n) { out[r0 + 32u] = c02; }
        if (col0 + tx + 48u < p.n) { out[r0 + 48u] = c03; }
        if (col0 + tx + 64u < p.n) { out[r0 + 64u] = c04; }
        if (col0 + tx + 80u < p.n) { out[r0 + 80u] = c05; }
        if (col0 + tx + 96u < p.n) { out[r0 + 96u] = c06; }
        if (col0 + tx + 112u < p.n) { out[r0 + 112u] = c07; }
    }
    if (m1 < p.m) {
        let r1 = m1 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r1 + 0u] = c10; }
        if (col0 + tx + 16u < p.n) { out[r1 + 16u] = c11; }
        if (col0 + tx + 32u < p.n) { out[r1 + 32u] = c12; }
        if (col0 + tx + 48u < p.n) { out[r1 + 48u] = c13; }
        if (col0 + tx + 64u < p.n) { out[r1 + 64u] = c14; }
        if (col0 + tx + 80u < p.n) { out[r1 + 80u] = c15; }
        if (col0 + tx + 96u < p.n) { out[r1 + 96u] = c16; }
        if (col0 + tx + 112u < p.n) { out[r1 + 112u] = c17; }
    }
    if (m2 < p.m) {
        let r2 = m2 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r2 + 0u] = c20; }
        if (col0 + tx + 16u < p.n) { out[r2 + 16u] = c21; }
        if (col0 + tx + 32u < p.n) { out[r2 + 32u] = c22; }
        if (col0 + tx + 48u < p.n) { out[r2 + 48u] = c23; }
        if (col0 + tx + 64u < p.n) { out[r2 + 64u] = c24; }
        if (col0 + tx + 80u < p.n) { out[r2 + 80u] = c25; }
        if (col0 + tx + 96u < p.n) { out[r2 + 96u] = c26; }
        if (col0 + tx + 112u < p.n) { out[r2 + 112u] = c27; }
    }
    if (m3 < p.m) {
        let r3 = m3 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r3 + 0u] = c30; }
        if (col0 + tx + 16u < p.n) { out[r3 + 16u] = c31; }
        if (col0 + tx + 32u < p.n) { out[r3 + 32u] = c32; }
        if (col0 + tx + 48u < p.n) { out[r3 + 48u] = c33; }
        if (col0 + tx + 64u < p.n) { out[r3 + 64u] = c34; }
        if (col0 + tx + 80u < p.n) { out[r3 + 80u] = c35; }
        if (col0 + tx + 96u < p.n) { out[r3 + 96u] = c36; }
        if (col0 + tx + 112u < p.n) { out[r3 + 112u] = c37; }
    }
    if (m4 < p.m) {
        let r4 = m4 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r4 + 0u] = c40; }
        if (col0 + tx + 16u < p.n) { out[r4 + 16u] = c41; }
        if (col0 + tx + 32u < p.n) { out[r4 + 32u] = c42; }
        if (col0 + tx + 48u < p.n) { out[r4 + 48u] = c43; }
        if (col0 + tx + 64u < p.n) { out[r4 + 64u] = c44; }
        if (col0 + tx + 80u < p.n) { out[r4 + 80u] = c45; }
        if (col0 + tx + 96u < p.n) { out[r4 + 96u] = c46; }
        if (col0 + tx + 112u < p.n) { out[r4 + 112u] = c47; }
    }
    if (m5 < p.m) {
        let r5 = m5 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r5 + 0u] = c50; }
        if (col0 + tx + 16u < p.n) { out[r5 + 16u] = c51; }
        if (col0 + tx + 32u < p.n) { out[r5 + 32u] = c52; }
        if (col0 + tx + 48u < p.n) { out[r5 + 48u] = c53; }
        if (col0 + tx + 64u < p.n) { out[r5 + 64u] = c54; }
        if (col0 + tx + 80u < p.n) { out[r5 + 80u] = c55; }
        if (col0 + tx + 96u < p.n) { out[r5 + 96u] = c56; }
        if (col0 + tx + 112u < p.n) { out[r5 + 112u] = c57; }
    }
    if (m6 < p.m) {
        let r6 = m6 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r6 + 0u] = c60; }
        if (col0 + tx + 16u < p.n) { out[r6 + 16u] = c61; }
        if (col0 + tx + 32u < p.n) { out[r6 + 32u] = c62; }
        if (col0 + tx + 48u < p.n) { out[r6 + 48u] = c63; }
        if (col0 + tx + 64u < p.n) { out[r6 + 64u] = c64; }
        if (col0 + tx + 80u < p.n) { out[r6 + 80u] = c65; }
        if (col0 + tx + 96u < p.n) { out[r6 + 96u] = c66; }
        if (col0 + tx + 112u < p.n) { out[r6 + 112u] = c67; }
    }
    if (m7 < p.m) {
        let r7 = m7 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r7 + 0u] = c70; }
        if (col0 + tx + 16u < p.n) { out[r7 + 16u] = c71; }
        if (col0 + tx + 32u < p.n) { out[r7 + 32u] = c72; }
        if (col0 + tx + 48u < p.n) { out[r7 + 48u] = c73; }
        if (col0 + tx + 64u < p.n) { out[r7 + 64u] = c74; }
        if (col0 + tx + 80u < p.n) { out[r7 + 80u] = c75; }
        if (col0 + tx + 96u < p.n) { out[r7 + 96u] = c76; }
        if (col0 + tx + 112u < p.n) { out[r7 + 112u] = c77; }
    }
}
