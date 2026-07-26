// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// matmul_i8 with a DYNAMIC per-tensor activation scale (sx from a buffer, sw a
//
//   x_q : [M, K/4] u32  — 4 int8 activations packed along K per u32 (row-major)
//   w_q : [N, K/4] u32  — 4 int8 weights    packed along K per u32 (row-major)
//   out : [M, N]  f32   — dequantized:  out[m,n] = acc_i32 * sx * sw
//
// This is the P40's fastest inference path. DP4A (`dot4I8Packed`) does four
// int8 multiply-accumulates in one instruction — 4x the MACs of an fp32 FMA —
// which is the 47-TOPS hardware the peak bench demonstrated. int8 weights also
// move 1/4 the bytes of fp32, so the memory side wins too.
//
// Structure mirrors matmul_reg2: 128x128 output tile, 256 threads, 8x8 register
// micro-tile, software-pipelined (prefetch the next K-chunk into registers to
// hide global-load latency). The only differences from the fp32 kernel: staged
// values are u32 (int8x4), the inner op is `acc += dot4I8Packed(a, b)` (WGSL has
// no fused-accumulate dot, so a 32-bit add follows each dot), accumulators are
// i32, and the epilogue multiplies by the per-tensor scales sx*sw.
//
// K must be a multiple of 4 (packing). Per-tensor scales here; per-row (x) /
// per-column (w) scales are the production refinement — same kernel, scales
// indexed by m / n in the epilogue.
//
// @workgroup_size(256). Not CPU-JIT'able (multi-barrier work-group); the CPU
// int8 reference lives in the validation test, so parity is still gated.

struct Params { m: u32, kg: u32, n: u32 };  // dynamic sx + per-channel sw, kg = K/4

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<u32>;  // [M, kg]
@group(0) @binding(2) var<storage, read>       w:   array<u32>;  // [N, kg]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [1] dynamic activation scale
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N] per-channel weight scale
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

const BM: u32 = 128u;
const BN: u32 = 128u;
const BKG: u32 = 8u;   // packed K-groups per chunk (= 32 int8 along K)
const WG: u32 = 256u;

var<workgroup> As: array<u32, 1024>;  // BKG*BM
var<workgroup> Bs: array<u32, 1024>;  // BKG*BN

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = lid.x;
    let ty = tid / 16u;
    let tx = tid % 16u;
    let wg = wgid.y * nwg.x + wgid.x;
    let tiles_n = (p.n + BN - 1u) / BN;
    let row0 = (wg / tiles_n) * BM;
    let col0 = (wg % tiles_n) * BN;
    let arow = ty * 8u;
    let bcol = tx * 8u;

    var sr: array<u32, 4>;
    var skk: array<u32, 4>;
    var arow_g: array<u32, 4>;
    var brow_g: array<u32, 4>;
    for (var e = 0u; e < 4u; e = e + 1u) {
        let idx = tid + e * WG;
        let r = idx / BKG;
        let kk = idx % BKG;
        sr[e] = r; skk[e] = kk;
        arow_g[e] = row0 + r;
        brow_g[e] = col0 + r;
    }

    // 64 int32 accumulators.
    var c00 = 0i; var c01 = 0i; var c02 = 0i; var c03 = 0i; var c04 = 0i; var c05 = 0i; var c06 = 0i; var c07 = 0i;
    var c10 = 0i; var c11 = 0i; var c12 = 0i; var c13 = 0i; var c14 = 0i; var c15 = 0i; var c16 = 0i; var c17 = 0i;
    var c20 = 0i; var c21 = 0i; var c22 = 0i; var c23 = 0i; var c24 = 0i; var c25 = 0i; var c26 = 0i; var c27 = 0i;
    var c30 = 0i; var c31 = 0i; var c32 = 0i; var c33 = 0i; var c34 = 0i; var c35 = 0i; var c36 = 0i; var c37 = 0i;
    var c40 = 0i; var c41 = 0i; var c42 = 0i; var c43 = 0i; var c44 = 0i; var c45 = 0i; var c46 = 0i; var c47 = 0i;
    var c50 = 0i; var c51 = 0i; var c52 = 0i; var c53 = 0i; var c54 = 0i; var c55 = 0i; var c56 = 0i; var c57 = 0i;
    var c60 = 0i; var c61 = 0i; var c62 = 0i; var c63 = 0i; var c64 = 0i; var c65 = 0i; var c66 = 0i; var c67 = 0i;
    var c70 = 0i; var c71 = 0i; var c72 = 0i; var c73 = 0i; var c74 = 0i; var c75 = 0i; var c76 = 0i; var c77 = 0i;

    var rA: array<u32, 4>;
    var rB: array<u32, 4>;

    let nchunks = (p.kg + BKG - 1u) / BKG;

    // Prime chunk 0.
    for (var e = 0u; e < 4u; e = e + 1u) {
        let gk = skk[e];
        if (arow_g[e] < p.m && gk < p.kg) { As[skk[e] * BM + sr[e]] = x[arow_g[e] * p.kg + gk]; }
        else                              { As[skk[e] * BM + sr[e]] = 0u; }
        if (brow_g[e] < p.n && gk < p.kg) { Bs[skk[e] * BN + sr[e]] = w[brow_g[e] * p.kg + gk]; }
        else                              { Bs[skk[e] * BN + sr[e]] = 0u; }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let k1 = (c + 1u) * BKG;
            for (var e = 0u; e < 4u; e = e + 1u) {
                let gk = k1 + skk[e];
                if (arow_g[e] < p.m && gk < p.kg) { rA[e] = x[arow_g[e] * p.kg + gk]; } else { rA[e] = 0u; }
                if (brow_g[e] < p.n && gk < p.kg) { rB[e] = w[brow_g[e] * p.kg + gk]; } else { rB[e] = 0u; }
            }
        }
        for (var kk = 0u; kk < BKG; kk = kk + 1u) {
            let ao = kk * BM + arow;
            let bo = kk * BN + bcol;
            let a0 = As[ao + 0u]; let a1 = As[ao + 1u]; let a2 = As[ao + 2u]; let a3 = As[ao + 3u]; let a4 = As[ao + 4u]; let a5 = As[ao + 5u]; let a6 = As[ao + 6u]; let a7 = As[ao + 7u];
            let b0 = Bs[bo + 0u]; let b1 = Bs[bo + 1u]; let b2 = Bs[bo + 2u]; let b3 = Bs[bo + 3u]; let b4 = Bs[bo + 4u]; let b5 = Bs[bo + 5u]; let b6 = Bs[bo + 6u]; let b7 = Bs[bo + 7u];
            c00 += dot4I8Packed(a0, b0); c01 += dot4I8Packed(a0, b1); c02 += dot4I8Packed(a0, b2); c03 += dot4I8Packed(a0, b3); c04 += dot4I8Packed(a0, b4); c05 += dot4I8Packed(a0, b5); c06 += dot4I8Packed(a0, b6); c07 += dot4I8Packed(a0, b7);
            c10 += dot4I8Packed(a1, b0); c11 += dot4I8Packed(a1, b1); c12 += dot4I8Packed(a1, b2); c13 += dot4I8Packed(a1, b3); c14 += dot4I8Packed(a1, b4); c15 += dot4I8Packed(a1, b5); c16 += dot4I8Packed(a1, b6); c17 += dot4I8Packed(a1, b7);
            c20 += dot4I8Packed(a2, b0); c21 += dot4I8Packed(a2, b1); c22 += dot4I8Packed(a2, b2); c23 += dot4I8Packed(a2, b3); c24 += dot4I8Packed(a2, b4); c25 += dot4I8Packed(a2, b5); c26 += dot4I8Packed(a2, b6); c27 += dot4I8Packed(a2, b7);
            c30 += dot4I8Packed(a3, b0); c31 += dot4I8Packed(a3, b1); c32 += dot4I8Packed(a3, b2); c33 += dot4I8Packed(a3, b3); c34 += dot4I8Packed(a3, b4); c35 += dot4I8Packed(a3, b5); c36 += dot4I8Packed(a3, b6); c37 += dot4I8Packed(a3, b7);
            c40 += dot4I8Packed(a4, b0); c41 += dot4I8Packed(a4, b1); c42 += dot4I8Packed(a4, b2); c43 += dot4I8Packed(a4, b3); c44 += dot4I8Packed(a4, b4); c45 += dot4I8Packed(a4, b5); c46 += dot4I8Packed(a4, b6); c47 += dot4I8Packed(a4, b7);
            c50 += dot4I8Packed(a5, b0); c51 += dot4I8Packed(a5, b1); c52 += dot4I8Packed(a5, b2); c53 += dot4I8Packed(a5, b3); c54 += dot4I8Packed(a5, b4); c55 += dot4I8Packed(a5, b5); c56 += dot4I8Packed(a5, b6); c57 += dot4I8Packed(a5, b7);
            c60 += dot4I8Packed(a6, b0); c61 += dot4I8Packed(a6, b1); c62 += dot4I8Packed(a6, b2); c63 += dot4I8Packed(a6, b3); c64 += dot4I8Packed(a6, b4); c65 += dot4I8Packed(a6, b5); c66 += dot4I8Packed(a6, b6); c67 += dot4I8Packed(a6, b7);
            c70 += dot4I8Packed(a7, b0); c71 += dot4I8Packed(a7, b1); c72 += dot4I8Packed(a7, b2); c73 += dot4I8Packed(a7, b3); c74 += dot4I8Packed(a7, b4); c75 += dot4I8Packed(a7, b5); c76 += dot4I8Packed(a7, b6); c77 += dot4I8Packed(a7, b7);
        }
        workgroupBarrier();
        if (has_next) {
            for (var e = 0u; e < 4u; e = e + 1u) {
                As[skk[e] * BM + sr[e]] = rA[e];
                Bs[skk[e] * BN + sr[e]] = rB[e];
            }
        }
        workgroupBarrier();
    }


    let c0 = col0 + bcol;
    var swc: array<f32, 8>;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) { let cc = c0 + j; swc[j] = select(0.0, sw[cc], cc < p.n); }
    let m0 = row0 + arow + 0u;
    let m1 = row0 + arow + 1u;
    let m2 = row0 + arow + 2u;
    let m3 = row0 + arow + 3u;
    let m4 = row0 + arow + 4u;
    let m5 = row0 + arow + 5u;
    let m6 = row0 + arow + 6u;
    let m7 = row0 + arow + 7u;
    let sv0 = select(0.0, sx[m0], m0 < p.m); let sv1 = select(0.0, sx[m1], m1 < p.m);
    let sv2 = select(0.0, sx[m2], m2 < p.m); let sv3 = select(0.0, sx[m3], m3 < p.m);
    let sv4 = select(0.0, sx[m4], m4 < p.m); let sv5 = select(0.0, sx[m5], m5 < p.m);
    let sv6 = select(0.0, sx[m6], m6 < p.m); let sv7 = select(0.0, sx[m7], m7 < p.m);

    if (m0 < p.m) {
        let r0 = m0 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r0 + 0u] = f32(c00) * sv0 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r0 + 1u] = f32(c01) * sv0 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r0 + 2u] = f32(c02) * sv0 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r0 + 3u] = f32(c03) * sv0 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r0 + 4u] = f32(c04) * sv0 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r0 + 5u] = f32(c05) * sv0 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r0 + 6u] = f32(c06) * sv0 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r0 + 7u] = f32(c07) * sv0 * swc[7]; }
    }
    if (m1 < p.m) {
        let r1 = m1 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r1 + 0u] = f32(c10) * sv1 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r1 + 1u] = f32(c11) * sv1 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r1 + 2u] = f32(c12) * sv1 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r1 + 3u] = f32(c13) * sv1 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r1 + 4u] = f32(c14) * sv1 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r1 + 5u] = f32(c15) * sv1 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r1 + 6u] = f32(c16) * sv1 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r1 + 7u] = f32(c17) * sv1 * swc[7]; }
    }
    if (m2 < p.m) {
        let r2 = m2 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r2 + 0u] = f32(c20) * sv2 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r2 + 1u] = f32(c21) * sv2 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r2 + 2u] = f32(c22) * sv2 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r2 + 3u] = f32(c23) * sv2 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r2 + 4u] = f32(c24) * sv2 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r2 + 5u] = f32(c25) * sv2 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r2 + 6u] = f32(c26) * sv2 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r2 + 7u] = f32(c27) * sv2 * swc[7]; }
    }
    if (m3 < p.m) {
        let r3 = m3 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r3 + 0u] = f32(c30) * sv3 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r3 + 1u] = f32(c31) * sv3 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r3 + 2u] = f32(c32) * sv3 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r3 + 3u] = f32(c33) * sv3 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r3 + 4u] = f32(c34) * sv3 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r3 + 5u] = f32(c35) * sv3 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r3 + 6u] = f32(c36) * sv3 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r3 + 7u] = f32(c37) * sv3 * swc[7]; }
    }
    if (m4 < p.m) {
        let r4 = m4 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r4 + 0u] = f32(c40) * sv4 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r4 + 1u] = f32(c41) * sv4 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r4 + 2u] = f32(c42) * sv4 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r4 + 3u] = f32(c43) * sv4 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r4 + 4u] = f32(c44) * sv4 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r4 + 5u] = f32(c45) * sv4 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r4 + 6u] = f32(c46) * sv4 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r4 + 7u] = f32(c47) * sv4 * swc[7]; }
    }
    if (m5 < p.m) {
        let r5 = m5 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r5 + 0u] = f32(c50) * sv5 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r5 + 1u] = f32(c51) * sv5 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r5 + 2u] = f32(c52) * sv5 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r5 + 3u] = f32(c53) * sv5 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r5 + 4u] = f32(c54) * sv5 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r5 + 5u] = f32(c55) * sv5 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r5 + 6u] = f32(c56) * sv5 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r5 + 7u] = f32(c57) * sv5 * swc[7]; }
    }
    if (m6 < p.m) {
        let r6 = m6 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r6 + 0u] = f32(c60) * sv6 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r6 + 1u] = f32(c61) * sv6 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r6 + 2u] = f32(c62) * sv6 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r6 + 3u] = f32(c63) * sv6 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r6 + 4u] = f32(c64) * sv6 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r6 + 5u] = f32(c65) * sv6 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r6 + 6u] = f32(c66) * sv6 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r6 + 7u] = f32(c67) * sv6 * swc[7]; }
    }
    if (m7 < p.m) {
        let r7 = m7 * p.n + col0 + bcol;
        if (col0 + bcol + 0u < p.n) { out[r7 + 0u] = f32(c70) * sv7 * swc[0]; }
        if (col0 + bcol + 1u < p.n) { out[r7 + 1u] = f32(c71) * sv7 * swc[1]; }
        if (col0 + bcol + 2u < p.n) { out[r7 + 2u] = f32(c72) * sv7 * swc[2]; }
        if (col0 + bcol + 3u < p.n) { out[r7 + 3u] = f32(c73) * sv7 * swc[3]; }
        if (col0 + bcol + 4u < p.n) { out[r7 + 4u] = f32(c74) * sv7 * swc[4]; }
        if (col0 + bcol + 5u < p.n) { out[r7 + 5u] = f32(c75) * sv7 * swc[5]; }
        if (col0 + bcol + 6u < p.n) { out[r7 + 6u] = f32(c76) * sv7 * swc[6]; }
        if (col0 + bcol + 7u < p.n) { out[r7 + 7u] = f32(c77) * sv7 * swc[7]; }
    }
}
