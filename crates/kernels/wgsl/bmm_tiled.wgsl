// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched matmul, overwrite or accumulate: out[b,m,n] (+)= alpha * sum_k A[b,·]·B[b,·], 64x64 tile per workgroup - the fast form of `bmm.wgsl` / `bmm_acc.wgsl`
// @how   register block per thread, 256-thread workgroup tile, 2 barriers
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// `bmm.wgsl` and `bmm_acc.wgsl` give one thread one output element and walk
// `k` from global memory: every operand element is re-read once per output it
// feeds (64x here), so the kernel runs at a few percent of the card. This one
// stages 64x16 tiles of both operands through workgroup memory and gives each
// thread a 4x4 register block, so each staged value is reused four times from
// registers and global traffic drops by the tile width.
//
// Same contract as the pair it replaces, plus one trailing flag:
//   params: batch, m, k, n, trans_a, trans_b, alpha, a_off, b_off, out_off, acc
//   out[b,m,n]  =  alpha * sum_k Â[b,m,k] * B̂[b,k,n]          (acc == 0, `bmm`)
//   out[b,m,n] +=  alpha * sum_k Â[b,m,k] * B̂[b,k,n]          (acc != 0, `bmm_acc`)
// with `trans_a`/`trans_b` and the flat element offsets exactly as `bmm.wgsl`
// documents them.
//
// NUMERICS: bit-identical to the pair it replaces. Each output is one fp32
// accumulator walked over `k` in ASCENDING order, `acc = acc + a * b`, then
// scaled by `alpha` (and added to the old output when accumulating) - the
// tiling changes where the operands are READ from, never the order they are
// summed in. Out-of-range lanes of a tile are zero-filled, and `acc + 0 * 0`
// is exact, so a `k` that is not a multiple of the staged depth is not a
// source of drift.
//
// Dispatch: ONE WORKGROUP per `(batch, 64-row tile, 64-column tile)`,
// `batch * ceil(m/64) * ceil(n/64)` workgroups, tile-column fastest.
//
// Staging is by the operand's own contiguous axis (rows of `k` for a
// non-transposed A / transposed B, rows of `m`/`n` otherwise), so every global
// load is coalesced whichever transposes the caller asks for. The staged tile
// is stored `[k][row]` with one pad word per row so the transposing stores do
// not serialise on a bank.

struct Params {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    trans_a: u32,
    trans_b: u32,
    alpha: f32,
    a_off: u32,
    b_off: u32,
    out_off: u32,
    acc: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const PAD: u32 = 65u;   // BM + 1: staged row stride in words

var<workgroup> As: array<f32, 1040>;  // BK * PAD, [kk][row]
var<workgroup> Bs: array<f32, 1040>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = lid.x;
    let ty = tid / 16u;
    let tx = tid % 16u;
    let w = wg.y * nwg.x + wg.x;
    let tiles_n = (p.n + BN - 1u) / BN;
    let tiles_m = (p.m + BM - 1u) / BM;
    let bi = w / (tiles_m * tiles_n);
    // Uniform across the workgroup, so returning before a barrier is legal.
    if (bi >= p.batch) { return; }
    let rest = w % (tiles_m * tiles_n);
    let row0 = (rest / tiles_n) * BM;
    let col0 = (rest % tiles_n) * BN;

    let a_base = p.a_off + bi * (p.m * p.k);
    let b_base = p.b_off + bi * (p.k * p.n);

    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0;

    for (var k0 = 0u; k0 < p.k; k0 = k0 + BK) {
        // Stage A[row0.., k0..] and B[k0.., col0..]: 1024 words each, four per
        // thread, contiguous axis fastest across the workgroup.
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = tid + e * 256u;
            // A
            var r: u32;
            var kk: u32;
            if (p.trans_a == 0u) { kk = idx % BK; r = idx / BK; } else { r = idx % BM; kk = idx / BM; }
            var av = 0.0;
            if (row0 + r < p.m && k0 + kk < p.k) {
                if (p.trans_a == 0u) { av = a[a_base + (row0 + r) * p.k + k0 + kk]; }
                else                 { av = a[a_base + (k0 + kk) * p.m + row0 + r]; }
            }
            As[kk * PAD + r] = av;
            // B
            var c: u32;
            var kb: u32;
            if (p.trans_b == 0u) { c = idx % BN; kb = idx / BN; } else { kb = idx % BK; c = idx / BK; }
            var bv = 0.0;
            if (col0 + c < p.n && k0 + kb < p.k) {
                if (p.trans_b == 0u) { bv = b[b_base + (k0 + kb) * p.n + col0 + c]; }
                else                 { bv = b[b_base + (col0 + c) * p.k + k0 + kb]; }
            }
            Bs[kb * PAD + c] = bv;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let ao = kk * PAD + ty * 4u;
            let bo = kk * PAD + tx * 4u;
            let a0 = As[ao]; let a1 = As[ao + 1u]; let a2 = As[ao + 2u]; let a3 = As[ao + 3u];
            let b0 = Bs[bo]; let b1 = Bs[bo + 1u]; let b2 = Bs[bo + 2u]; let b3 = Bs[bo + 3u];
            c00 = c00 + a0 * b0; c01 = c01 + a0 * b1; c02 = c02 + a0 * b2; c03 = c03 + a0 * b3;
            c10 = c10 + a1 * b0; c11 = c11 + a1 * b1; c12 = c12 + a1 * b2; c13 = c13 + a1 * b3;
            c20 = c20 + a2 * b0; c21 = c21 + a2 * b1; c22 = c22 + a2 * b2; c23 = c23 + a2 * b3;
            c30 = c30 + a3 * b0; c31 = c31 + a3 * b1; c32 = c32 + a3 * b2; c33 = c33 + a3 * b3;
        }
        workgroupBarrier();
    }

    let o_base = p.out_off + bi * (p.m * p.n);
    let r0 = row0 + ty * 4u;
    let cc = col0 + tx * 4u;
    // Row-major 4x4 block; guarded per element.
    for (var i = 0u; i < 4u; i = i + 1u) {
        let r = r0 + i;
        if (r >= p.m) { continue; }
        var v0 = c00; var v1 = c01; var v2 = c02; var v3 = c03;
        if (i == 1u) { v0 = c10; v1 = c11; v2 = c12; v3 = c13; }
        if (i == 2u) { v0 = c20; v1 = c21; v2 = c22; v3 = c23; }
        if (i == 3u) { v0 = c30; v1 = c31; v2 = c32; v3 = c33; }
        let o = o_base + r * p.n + cc;
        if (cc + 0u < p.n) { if (p.acc == 0u) { out[o]      = p.alpha * v0; } else { out[o]      = out[o]      + p.alpha * v0; } }
        if (cc + 1u < p.n) { if (p.acc == 0u) { out[o + 1u] = p.alpha * v1; } else { out[o + 1u] = out[o + 1u] + p.alpha * v1; } }
        if (cc + 2u < p.n) { if (p.acc == 0u) { out[o + 2u] = p.alpha * v2; } else { out[o + 2u] = out[o + 2u] + p.alpha * v2; } }
        if (cc + 3u < p.n) { if (p.acc == 0u) { out[o + 3u] = p.alpha * v3; } else { out[o + 3u] = out[o + 3u] + p.alpha * v3; } }
    }
}
