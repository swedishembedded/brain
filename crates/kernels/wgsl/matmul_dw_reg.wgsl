// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Backward of out = x @ W^T w.r.t. W (tiled):  dW[n,k] += sum_m dY[m,n]*X[m,k].
//
// Tiled + software-pipelined (matmul_reg2 structure: 128x128 output tile, 256
// threads, 8x8 register micro-tile, prefetch the next contraction chunk to hide
// global-load latency). The forward `matmul_reg2` runs ~34% of the P40's peak;
// this brings the backward GEMMs — every training step's dominant cost — to the
// same regime instead of the naive 0.5%. CPU routes to the AVX2 gemm.
//
// @workgroup_size(256).

struct Params { m: u32, k: u32, n: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;
const WG: u32 = 256u;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = lid.x;
    let ty = tid / 16u;
    let tx = tid % 16u;
    let wg = wgid.y * nwg.x + wgid.x;
    let R = p.n;
    let C = p.k;
    let L = p.m;
    let tiles_c = (C + BN - 1u) / BN;
    let row0 = (wg / tiles_c) * BM;   // output row block (0..R)
    let col0 = (wg % tiles_c) * BN;   // output col block (0..C)
    let arow = ty * 8u;
    let bcol = tx * 8u;

    var sr: array<u32, 4>;
    var skk: array<u32, 4>;
    var arow_g: array<u32, 4>;
    var brow_g: array<u32, 4>;
    for (var e = 0u; e < 4u; e = e + 1u) {
        let idx = tid + e * WG;
        let r = idx / BK;
        let kk = idx % BK;
        sr[e] = r; skk[e] = kk;
        arow_g[e] = row0 + r;
        brow_g[e] = col0 + r;
    }

    // 64 scalar-register accumulators.
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

    let nchunks = (L + BK - 1u) / BK;

    // prime chunk 0
    for (var e = 0u; e < 4u; e = e + 1u) {
        let gk = skk[e];
        { let ar = arow_g[e]; if (ar < R && gk < L) { As[skk[e] * BM + sr[e]] = a[gk * p.n + ar]; } else { As[skk[e] * BM + sr[e]] = 0.0; } }
        { let br = brow_g[e]; if (br < C && gk < L) { Bs[skk[e] * BN + sr[e]] = b[gk * p.k + br]; } else { Bs[skk[e] * BN + sr[e]] = 0.0; } }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let k1 = (c + 1u) * BK;
            for (var e = 0u; e < 4u; e = e + 1u) {
                let gk = k1 + skk[e];
                { let ar = arow_g[e]; if (ar < R && gk < L) { rA[e] = a[gk * p.n + ar]; } else { rA[e] = 0.0; } }
                { let br = brow_g[e]; if (br < C && gk < L) { rB[e] = b[gk * p.k + br]; } else { rB[e] = 0.0; } }
            }
        }
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let ao = kk * BM + arow;
            let bo = kk * BN + bcol;
            let av0 = As[ao + 0u]; let av1 = As[ao + 1u]; let av2 = As[ao + 2u]; let av3 = As[ao + 3u]; let av4 = As[ao + 4u]; let av5 = As[ao + 5u]; let av6 = As[ao + 6u]; let av7 = As[ao + 7u];
            let bv0 = Bs[bo + 0u]; let bv1 = Bs[bo + 1u]; let bv2 = Bs[bo + 2u]; let bv3 = Bs[bo + 3u]; let bv4 = Bs[bo + 4u]; let bv5 = Bs[bo + 5u]; let bv6 = Bs[bo + 6u]; let bv7 = Bs[bo + 7u];
            c00 += av0 * bv0; c01 += av0 * bv1; c02 += av0 * bv2; c03 += av0 * bv3; c04 += av0 * bv4; c05 += av0 * bv5; c06 += av0 * bv6; c07 += av0 * bv7;
            c10 += av1 * bv0; c11 += av1 * bv1; c12 += av1 * bv2; c13 += av1 * bv3; c14 += av1 * bv4; c15 += av1 * bv5; c16 += av1 * bv6; c17 += av1 * bv7;
            c20 += av2 * bv0; c21 += av2 * bv1; c22 += av2 * bv2; c23 += av2 * bv3; c24 += av2 * bv4; c25 += av2 * bv5; c26 += av2 * bv6; c27 += av2 * bv7;
            c30 += av3 * bv0; c31 += av3 * bv1; c32 += av3 * bv2; c33 += av3 * bv3; c34 += av3 * bv4; c35 += av3 * bv5; c36 += av3 * bv6; c37 += av3 * bv7;
            c40 += av4 * bv0; c41 += av4 * bv1; c42 += av4 * bv2; c43 += av4 * bv3; c44 += av4 * bv4; c45 += av4 * bv5; c46 += av4 * bv6; c47 += av4 * bv7;
            c50 += av5 * bv0; c51 += av5 * bv1; c52 += av5 * bv2; c53 += av5 * bv3; c54 += av5 * bv4; c55 += av5 * bv5; c56 += av5 * bv6; c57 += av5 * bv7;
            c60 += av6 * bv0; c61 += av6 * bv1; c62 += av6 * bv2; c63 += av6 * bv3; c64 += av6 * bv4; c65 += av6 * bv5; c66 += av6 * bv6; c67 += av6 * bv7;
            c70 += av7 * bv0; c71 += av7 * bv1; c72 += av7 * bv2; c73 += av7 * bv3; c74 += av7 * bv4; c75 += av7 * bv5; c76 += av7 * bv6; c77 += av7 * bv7;
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

    let orow0 = row0 + arow + 0u;
    let orow1 = row0 + arow + 1u;
    let orow2 = row0 + arow + 2u;
    let orow3 = row0 + arow + 3u;
    let orow4 = row0 + arow + 4u;
    let orow5 = row0 + arow + 5u;
    let orow6 = row0 + arow + 6u;
    let orow7 = row0 + arow + 7u;

    if (orow0 < R) {
        let base0 = orow0 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base0 + 0u] = out[base0 + 0u] + c00; }
        if (col0 + bcol + 1u < C) { out[base0 + 1u] = out[base0 + 1u] + c01; }
        if (col0 + bcol + 2u < C) { out[base0 + 2u] = out[base0 + 2u] + c02; }
        if (col0 + bcol + 3u < C) { out[base0 + 3u] = out[base0 + 3u] + c03; }
        if (col0 + bcol + 4u < C) { out[base0 + 4u] = out[base0 + 4u] + c04; }
        if (col0 + bcol + 5u < C) { out[base0 + 5u] = out[base0 + 5u] + c05; }
        if (col0 + bcol + 6u < C) { out[base0 + 6u] = out[base0 + 6u] + c06; }
        if (col0 + bcol + 7u < C) { out[base0 + 7u] = out[base0 + 7u] + c07; }
    }
    if (orow1 < R) {
        let base1 = orow1 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base1 + 0u] = out[base1 + 0u] + c10; }
        if (col0 + bcol + 1u < C) { out[base1 + 1u] = out[base1 + 1u] + c11; }
        if (col0 + bcol + 2u < C) { out[base1 + 2u] = out[base1 + 2u] + c12; }
        if (col0 + bcol + 3u < C) { out[base1 + 3u] = out[base1 + 3u] + c13; }
        if (col0 + bcol + 4u < C) { out[base1 + 4u] = out[base1 + 4u] + c14; }
        if (col0 + bcol + 5u < C) { out[base1 + 5u] = out[base1 + 5u] + c15; }
        if (col0 + bcol + 6u < C) { out[base1 + 6u] = out[base1 + 6u] + c16; }
        if (col0 + bcol + 7u < C) { out[base1 + 7u] = out[base1 + 7u] + c17; }
    }
    if (orow2 < R) {
        let base2 = orow2 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base2 + 0u] = out[base2 + 0u] + c20; }
        if (col0 + bcol + 1u < C) { out[base2 + 1u] = out[base2 + 1u] + c21; }
        if (col0 + bcol + 2u < C) { out[base2 + 2u] = out[base2 + 2u] + c22; }
        if (col0 + bcol + 3u < C) { out[base2 + 3u] = out[base2 + 3u] + c23; }
        if (col0 + bcol + 4u < C) { out[base2 + 4u] = out[base2 + 4u] + c24; }
        if (col0 + bcol + 5u < C) { out[base2 + 5u] = out[base2 + 5u] + c25; }
        if (col0 + bcol + 6u < C) { out[base2 + 6u] = out[base2 + 6u] + c26; }
        if (col0 + bcol + 7u < C) { out[base2 + 7u] = out[base2 + 7u] + c27; }
    }
    if (orow3 < R) {
        let base3 = orow3 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base3 + 0u] = out[base3 + 0u] + c30; }
        if (col0 + bcol + 1u < C) { out[base3 + 1u] = out[base3 + 1u] + c31; }
        if (col0 + bcol + 2u < C) { out[base3 + 2u] = out[base3 + 2u] + c32; }
        if (col0 + bcol + 3u < C) { out[base3 + 3u] = out[base3 + 3u] + c33; }
        if (col0 + bcol + 4u < C) { out[base3 + 4u] = out[base3 + 4u] + c34; }
        if (col0 + bcol + 5u < C) { out[base3 + 5u] = out[base3 + 5u] + c35; }
        if (col0 + bcol + 6u < C) { out[base3 + 6u] = out[base3 + 6u] + c36; }
        if (col0 + bcol + 7u < C) { out[base3 + 7u] = out[base3 + 7u] + c37; }
    }
    if (orow4 < R) {
        let base4 = orow4 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base4 + 0u] = out[base4 + 0u] + c40; }
        if (col0 + bcol + 1u < C) { out[base4 + 1u] = out[base4 + 1u] + c41; }
        if (col0 + bcol + 2u < C) { out[base4 + 2u] = out[base4 + 2u] + c42; }
        if (col0 + bcol + 3u < C) { out[base4 + 3u] = out[base4 + 3u] + c43; }
        if (col0 + bcol + 4u < C) { out[base4 + 4u] = out[base4 + 4u] + c44; }
        if (col0 + bcol + 5u < C) { out[base4 + 5u] = out[base4 + 5u] + c45; }
        if (col0 + bcol + 6u < C) { out[base4 + 6u] = out[base4 + 6u] + c46; }
        if (col0 + bcol + 7u < C) { out[base4 + 7u] = out[base4 + 7u] + c47; }
    }
    if (orow5 < R) {
        let base5 = orow5 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base5 + 0u] = out[base5 + 0u] + c50; }
        if (col0 + bcol + 1u < C) { out[base5 + 1u] = out[base5 + 1u] + c51; }
        if (col0 + bcol + 2u < C) { out[base5 + 2u] = out[base5 + 2u] + c52; }
        if (col0 + bcol + 3u < C) { out[base5 + 3u] = out[base5 + 3u] + c53; }
        if (col0 + bcol + 4u < C) { out[base5 + 4u] = out[base5 + 4u] + c54; }
        if (col0 + bcol + 5u < C) { out[base5 + 5u] = out[base5 + 5u] + c55; }
        if (col0 + bcol + 6u < C) { out[base5 + 6u] = out[base5 + 6u] + c56; }
        if (col0 + bcol + 7u < C) { out[base5 + 7u] = out[base5 + 7u] + c57; }
    }
    if (orow6 < R) {
        let base6 = orow6 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base6 + 0u] = out[base6 + 0u] + c60; }
        if (col0 + bcol + 1u < C) { out[base6 + 1u] = out[base6 + 1u] + c61; }
        if (col0 + bcol + 2u < C) { out[base6 + 2u] = out[base6 + 2u] + c62; }
        if (col0 + bcol + 3u < C) { out[base6 + 3u] = out[base6 + 3u] + c63; }
        if (col0 + bcol + 4u < C) { out[base6 + 4u] = out[base6 + 4u] + c64; }
        if (col0 + bcol + 5u < C) { out[base6 + 5u] = out[base6 + 5u] + c65; }
        if (col0 + bcol + 6u < C) { out[base6 + 6u] = out[base6 + 6u] + c66; }
        if (col0 + bcol + 7u < C) { out[base6 + 7u] = out[base6 + 7u] + c67; }
    }
    if (orow7 < R) {
        let base7 = orow7 * C + col0 + bcol;
        if (col0 + bcol + 0u < C) { out[base7 + 0u] = out[base7 + 0u] + c70; }
        if (col0 + bcol + 1u < C) { out[base7 + 1u] = out[base7 + 1u] + c71; }
        if (col0 + bcol + 2u < C) { out[base7 + 2u] = out[base7 + 2u] + c72; }
        if (col0 + bcol + 3u < C) { out[base7 + 3u] = out[base7 + 3u] + c73; }
        if (col0 + bcol + 4u < C) { out[base7 + 4u] = out[base7 + 4u] + c74; }
        if (col0 + bcol + 5u < C) { out[base7 + 5u] = out[base7 + 5u] + c75; }
        if (col0 + bcol + 6u < C) { out[base7 + 6u] = out[base7 + 6u] + c76; }
        if (col0 + bcol + 7u < C) { out[base7 + 7u] = out[base7 + 7u] + c77; }
    }
}
