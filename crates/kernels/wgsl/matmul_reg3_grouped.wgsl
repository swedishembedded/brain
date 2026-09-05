// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped variant of matmul_reg3: one dispatch computes EVERY expert's compacted-batch GEMM
// @how   register block per thread, 256-thread workgroup tile, 3 barriers, per-workgroup expert-group lookup
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Grouped MoE, stage 3 (the GEMM itself): `matmul_reg3.wgsl`'s exact tiling,
// register block, shared-memory layout and 3-barrier software pipeline,
// UNCHANGED, dispatched ONCE across every expert's compacted row range
// instead of once per expert. This is the primitive M5.4 (`kernel-
// performance.md`) called "not buildable" for want of an indirect-dispatch
// primitive - see that entry's correction addendum for why a worst-case
// HOST-COMPUTABLE grid makes indirect dispatch unnecessary here: every row
// selects exactly `top_k` experts, so `rows*top_k` is a host-known constant
// before any device work runs, which bounds both the total compacted row
// count and (via `ceil(count_e/BM) <= count_e/BM + 1` summed over experts)
// the worst-case tile grid `n_experts + ceil(rows*top_k/BM)`.
//
// THE ONLY CHANGE from `matmul_reg3.wgsl`: the single tile-to-row mapping
// `row0 = (wg/tiles_n)*BM` becomes a per-workgroup LOOKUP. `group_tile_start`
// (host-worst-case-sized, device-scanned exclusive prefix of each expert's
// own `ceil(count_e/BM)`) tells a workgroup which expert group its
// `row_tile_idx = wg/tiles_n` falls in; `group_row_start`/`group_row_count`
// (the same per-expert scan `moe_group_perm_emit.wgsl` already consumes)
// give that group's row range in the COMPACTED `x`/`out` buffers. A
// workgroup whose `row_tile_idx` lands past every real expert's tiles (the
// slack between the exact tile total and the host's worst-case upper bound)
// resolves to the LAST expert group with `row0 >= m_end`, so every one of
// its bound checks below fails and it is a pure no-op - safe padding, not a
// special case.
//
// BARRIER UNIFORMITY, preserved on purpose: every `workgroupBarrier()` below
// still gates ONLY on `nchunks` (derived from `p.k`, a Params UNIFORM
// identical for every expert's gate/up/down projection - only the row count
// `M` varies per expert, never `K` or `N`). The expert-group lookup and the
// `m_end`/`row0`/`w_base` values it produces are used ONLY in per-thread
// LOAD/STORE bound checks, exactly where `matmul_reg3.wgsl` already used
// `p.m` - never to gate a barrier. A storage-derived barrier condition is
// what naga's uniformity analysis (and the CPU JIT's one-barrier-per-kernel
// limit) rejects; this kernel never introduces one.
//
// WEIGHT BASE OFFSET is a STORAGE-buffer read's own arithmetic
// (`e * p.k * p.n`), not a Params-uniform field: `e` is resolved INSIDE this
// kernel per workgroup (from `wg`, not known at dispatch time), and a single
// Params uniform is the same for every workgroup in the dispatch, so it has
// nowhere to carry a per-workgroup-varying offset. Because every expert's
// gate/up/down weight matrix is the IDENTICAL `(k, n)` shape (this module's
// own doc), that offset is plain per-expert stride arithmetic against the
// CONCATENATED `w` buffer - no separate offset table needed at all, which
// also sidesteps the 256-byte `min_storage_buffer_offset_alignment` padding
// M5.4's compact path had to pad around: this is a storage-array READ
// inside the kernel, not a `step_sliced` buffer-view offset.
//
// `@cpu no`: unlike `matmul_reg3` (whose 3 barriers are why IT needs a
// hand-written native CPU path, `backend_cpu::FastIdx::matmul_reg3`, rather
// than the generic WGSL CPU JIT), this kernel is a NEW name the CPU backend
// has no native override for, so it would fall through to the JIT and fail
// its one-top-level-barrier limit. No native CPU port is attempted this
// session - a deliberately deferred follow-up, not a silent gap: this
// kernel is GPU-only until one lands.

struct Params { k: u32, n: u32, n_experts: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:                array<f32>; // [compacted_rows, k]
@group(0) @binding(2) var<storage, read>       w:                array<f32>; // [n_experts, n, k] concatenated
@group(0) @binding(3) var<storage, read_write> out:              array<f32>; // [compacted_rows, n]
@group(0) @binding(4) var<storage, read>       group_row_start:  array<u32>; // n_experts
@group(0) @binding(5) var<storage, read>       group_row_count:  array<u32>; // n_experts
@group(0) @binding(6) var<storage, read>       group_tile_start: array<u32>; // n_experts

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
    let row_tile_idx = wg / tiles_n;
    let col0 = (wg % tiles_n) * BN;

    // Which expert group `row_tile_idx` falls in: the largest `e` with
    // `group_tile_start[e] <= row_tile_idx` (non-decreasing by construction,
    // so a linear scan can stop the moment it overshoots).
    var e: u32 = 0u;
    for (var i: u32 = 0u; i < p.n_experts; i = i + 1u) {
        if (group_tile_start[i] <= row_tile_idx) { e = i; } else { break; }
    }
    let local_tile = row_tile_idx - group_tile_start[e];
    let row0 = group_row_start[e] + local_tile * BM;
    let m_end = group_row_start[e] + group_row_count[e];
    let w_base = e * p.k * p.n;

    // Each thread stages 4 A and 4 B elements; only the k-offset moves per chunk.
    var sr: array<u32, 4>;
    var skk: array<u32, 4>;
    var arow_g: array<u32, 4>;
    var brow_g: array<u32, 4>;
    for (var ee = 0u; ee < 4u; ee = ee + 1u) {
        let idx = tid + ee * WG;   // 0..1023
        let r = idx / BK;         // 0..127
        let kk = idx % BK;        // 0..7
        sr[ee] = r; skk[ee] = kk;
        arow_g[ee] = row0 + r;
        brow_g[ee] = col0 + r;
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
    for (var ee = 0u; ee < 4u; ee = ee + 1u) {
        let gk = skk[ee];
        if (arow_g[ee] < m_end && gk < p.k) { As[skk[ee] * SP + sr[ee]] = x[arow_g[ee] * p.k + gk]; }
        else                                { As[skk[ee] * SP + sr[ee]] = 0.0; }
        if (brow_g[ee] < p.n && gk < p.k) { let wi = w_base + brow_g[ee] * p.k + gk; Bs[skk[ee] * SP + sr[ee]] = w[wi]; }
        else                              { Bs[skk[ee] * SP + sr[ee]] = 0.0; }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let k1 = (c + 1u) * BK;
            for (var ee = 0u; ee < 4u; ee = ee + 1u) {
                let gk = k1 + skk[ee];
                if (arow_g[ee] < m_end && gk < p.k) { rA[ee] = x[arow_g[ee] * p.k + gk]; } else { rA[ee] = 0.0; }
                if (brow_g[ee] < p.n && gk < p.k) { let wi = w_base + brow_g[ee] * p.k + gk; rB[ee] = w[wi]; } else { rB[ee] = 0.0; }
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
            for (var ee = 0u; ee < 4u; ee = ee + 1u) {
                As[skk[ee] * SP + sr[ee]] = rA[ee];
                Bs[skk[ee] * SP + sr[ee]] = rB[ee];
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

    if (m0 < m_end) {
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
    if (m1 < m_end) {
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
    if (m2 < m_end) {
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
    if (m3 < m_end) {
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
    if (m4 < m_end) {
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
    if (m5 < m_end) {
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
    if (m6 < m_end) {
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
    if (m7 < m_end) {
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
