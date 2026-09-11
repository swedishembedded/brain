// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Register-tiled matmul (out = x @ Wᵀ), matmul_reg3's exact structure retiled to a 64x64 output tile
// @how   register block per thread, 256-thread workgroup tile (64x64), 3 barriers
// @opt   5
// @cpu   no
// @gpu   yes-wg256
// @npu   yes
// @quant none
// @dtype f32
//
// Register-tiled matmul (out = x @ Wᵀ) - `matmul_reg3.wgsl`'s IDENTICAL
// structure (same Params, same @workgroup_size(256), same padded-shared /
// interleaved-register layout that kills matmul_reg2's bank conflicts),
// mechanically retiled from a 128x128 output tile to a 64x64 one (M8.15).
// Every constant that scales with BM/BN shrinks in step; nothing about the
// ALGORITHM changes, only how much of the output one workgroup covers:
//
//   BM=BN=64 (was 128) => each thread's register block is 4x4=16
//   accumulators (was 8x8=64) - LN stays 16 (the 16x16 thread grid), so
//   BM/LN = 64/16 = 4 elements per thread per dimension, same interleave-by-
//   LN pattern `matmul_reg3.wgsl` uses at BM/LN=8.
//
//   BM*BK/WG = 64*8/256 = 2 elements staged per thread per K-chunk (was 4).
//
// WHY THIS EXISTS (M8.15): `matmul_reg3`'s 128x128 tile grid is
// `ceil(m/128)*ceil(n/128)` workgroups - for an M just above the decode
// regime (e.g. a 48-row prefill chunk), a 128-row tile computes 128 rows of
// output but only 48 are real: 80 rows of pure waste, and the SAME waste
// again on the N axis whenever N is not a multiple of 128. A 64-row/64-col
// tile halves that waste on each axis AND roughly quadruples the tile count
// for the same shape, trading more (cheaper, 16-accumulator) workgroups for
// fewer wasted lanes - a real, orthogonal-to-split-K axis the schedule
// autotuner (`backend_api::select::{Schedule,TileShape}`) now searches,
// wired in only where it measures a real win (see `qwen3::serve::Engine::
// tune_splitk`'s own doc for the measured numbers this was and was not
// wired in for).
//
// Shared use: 2 * 8 * 65 * 4 = 4160 B (vs matmul_reg3's 8256 B) - well under
// this box's 32 KiB shared-memory ceiling (Intel Arc Meteor Lake Xe-LPG),
// same as the 128-tile kernel. fp32 only, one bind group, 3 storage
// buffers, no atomics/subgroups/f16.

struct Params { m: u32, k: u32, n: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 8u;
const SP: u32 = 65u;   // padded shared stride (BM + 1)
const WG: u32 = 256u;
const LN: u32 = 16u;   // lane grid: 16 x 16 threads, stride-16 interleave

var<workgroup> As: array<f32, 520>;  // BK*SP, k-major: As[kk*SP + r]
var<workgroup> Bs: array<f32, 520>;

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

    // Each thread stages 2 A and 2 B elements; only the k-offset moves per chunk.
    var sr: array<u32, 2>;
    var skk: array<u32, 2>;
    var arow_g: array<u32, 2>;
    var brow_g: array<u32, 2>;
    for (var e = 0u; e < 2u; e = e + 1u) {
        let idx = tid + e * WG;   // 0..511
        let r = idx / BK;         // 0..63
        let kk = idx % BK;        // 0..7
        sr[e] = r; skk[e] = kk;
        arow_g[e] = row0 + r;
        brow_g[e] = col0 + r;
    }

    // 16 scalar-register accumulators (unrolled -> real registers, not spill).
    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0;

    var rA: array<f32, 2>;
    var rB: array<f32, 2>;

    let nchunks = (p.k + BK - 1u) / BK;

    // Prime: load chunk 0 into shared.
    for (var e = 0u; e < 2u; e = e + 1u) {
        let gk = skk[e];
        if (arow_g[e] < p.m && gk < p.k) { As[skk[e] * SP + sr[e]] = x[arow_g[e] * p.k + gk]; }
        else                             { As[skk[e] * SP + sr[e]] = 0.0; }
        // Hoisted to a bare identifier (B4) -- see matmul.wgsl's comment on
        // the same pattern: `dtype_variant`'s bf16 decode reads `wi` twice.
        if (brow_g[e] < p.n && gk < p.k) { let wi = brow_g[e] * p.k + gk; Bs[skk[e] * SP + sr[e]] = w[wi]; }
        else                             { Bs[skk[e] * SP + sr[e]] = 0.0; }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let k1 = (c + 1u) * BK;
            for (var e = 0u; e < 2u; e = e + 1u) {
                let gk = k1 + skk[e];
                if (arow_g[e] < p.m && gk < p.k) { rA[e] = x[arow_g[e] * p.k + gk]; } else { rA[e] = 0.0; }
                if (brow_g[e] < p.n && gk < p.k) { let wi = brow_g[e] * p.k + gk; rB[e] = w[wi]; } else { rB[e] = 0.0; }
            }
        }
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let ao = kk * SP + ty;
            let bo = kk * SP + tx;
            let a0 = As[ao + 0u];
            let a1 = As[ao + 16u];
            let a2 = As[ao + 32u];
            let a3 = As[ao + 48u];
            let b0 = Bs[bo + 0u];
            let b1 = Bs[bo + 16u];
            let b2 = Bs[bo + 32u];
            let b3 = Bs[bo + 48u];
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }
        workgroupBarrier();
        if (has_next) {
            for (var e = 0u; e < 2u; e = e + 1u) {
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

    if (m0 < p.m) {
        let r0 = m0 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r0 + 0u] = c00; }
        if (col0 + tx + 16u < p.n) { out[r0 + 16u] = c01; }
        if (col0 + tx + 32u < p.n) { out[r0 + 32u] = c02; }
        if (col0 + tx + 48u < p.n) { out[r0 + 48u] = c03; }
    }
    if (m1 < p.m) {
        let r1 = m1 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r1 + 0u] = c10; }
        if (col0 + tx + 16u < p.n) { out[r1 + 16u] = c11; }
        if (col0 + tx + 32u < p.n) { out[r1 + 32u] = c12; }
        if (col0 + tx + 48u < p.n) { out[r1 + 48u] = c13; }
    }
    if (m2 < p.m) {
        let r2 = m2 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r2 + 0u] = c20; }
        if (col0 + tx + 16u < p.n) { out[r2 + 16u] = c21; }
        if (col0 + tx + 32u < p.n) { out[r2 + 32u] = c22; }
        if (col0 + tx + 48u < p.n) { out[r2 + 48u] = c23; }
    }
    if (m3 < p.m) {
        let r3 = m3 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r3 + 0u] = c30; }
        if (col0 + tx + 16u < p.n) { out[r3 + 16u] = c31; }
        if (col0 + tx + 32u < p.n) { out[r3 + 32u] = c32; }
        if (col0 + tx + 48u < p.n) { out[r3 + 48u] = c33; }
    }
}
