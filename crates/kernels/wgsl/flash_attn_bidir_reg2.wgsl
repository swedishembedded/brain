// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Flash attention (bidirectional self-attention), two-query-row register block with a software-pipelined K/V tile
// @how   256-thread workgroup tile, 128 query rows, vec4 shared tiles, prefetch into registers, 3 barriers (1 prologue + 2 per K/V tile)
// @opt   5
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Flash attention (bidirectional self-attention), two-query-row register block
// with a SOFTWARE-PIPELINED K/V tile. Same math, same Params and same output
// layout as `flash_attn_bidir_split`; a workgroup owns BR = 128 query rows, so
// the caller must size the grid from this kernel's own BR (which is what
// `model::block::flash_bidir_variant` returns alongside the pipeline index).
//
// This is `flash_attn_bidir_reg`'s lane split plus the three things `matmul_reg3`
// does around its inner loop:
//
//  1. THE NEXT TILE'S GLOBAL LOADS ARE ISSUED FIRST, into registers, and only
//     copied into shared once the current tile has been consumed. The loads
//     therefore overlap ~2000 fused multiply-adds of work per thread instead
//     of stalling a barrier. This matters more here than in a GEMM because the
//     two-query-row block leaves only ONE workgroup resident per SM, so there
//     are just 8 warps to hide a global round-trip with.
//
//  2. K AND V ARE RE-STAGED SEPARATELY, at the two points where each becomes
//     dead. `ksh` is finished with as soon as the partial dot products are in
//     `part`, and `vsh` as soon as the output accumulators are updated, so the
//     write-after-read hazards can share the barriers the algorithm already
//     needs. That is what takes the kernel from THREE barriers per K/V tile to
//     TWO - the ordering is load-bearing, and the hazard argument for it is:
//
//         read ksh (partial dots)         -- tile kt
//         BARRIER 1
//         write ksh                       -- tile kt+1, safe: reads above done
//         read part, softmax, read vsh    -- tile kt
//         BARRIER 2
//         write vsh                       -- tile kt+1, safe: reads above done
//         (next iteration reads ksh, written before BARRIER 2, and reads vsh
//          and writes part, both fenced by the next iteration's BARRIER 1)
//
// Two barriers is still more than the ONE the Cranelift CPU backend can split
// a kernel body at, so this kernel remains GPU-only, exactly like every other
// member of the family.
//
// LANE OWNERSHIP AND BANKS, extending `flash_attn_bidir_reg`'s: thread `lt`
// owns row group `rg = lt/LANES` and vec4-slot phase `lane = lt%LANES`; its two
// query rows are `rg` and `rg+64` of the tile; its channels are the interleaved
// vec4 slots {lane, lane+4, …}, so for a fixed (key, slot) the four lanes read
// four consecutive vec4s = 16 consecutive words = 16 distinct banks, broadcast
// across the eight row groups in a warp. The cross-lane partials are stored as
// `vec4<f32>` indexed [j][rg] and each lane writes its own COMPONENT, so the
// re-sum is one 128-bit load rather than four 32-bit ones, and a warp's stores
// span all 32 banks exactly once.
//
// The two query rows carry INDEPENDENT online-softmax state (m, l), as they
// must: they are different queries.
//
// TILES ARE ALWAYS HD=128 WIDE, zero-filled past `head_dim`, so every channel
// loop's trip count is a compile-time constant - the property that keeps
// `q0/q1/o0/o1` in registers. `head_dim` need not be a multiple of 4: staging
// and the epilogue guard per element, the loops do not.
//
// @workgroup_size(256) = (BR/2 row groups) x LANES(4), checked against
// `DeviceCaps::max_workgroup_size` (queried, never assumed). Shared use is
// 48 KiB (ksh 8 + vsh 8 + part0 16 + part1 16), which is the Vulkan/NVIDIA
// 49152-byte compute limit exactly and far above the 16 KiB a Vulkan
// implementation is only REQUIRED to offer - so selection must also check
// `DeviceCaps::workgroup_mem_bytes`, and `flash_attn_bidir_reg` is the 16 KiB
// fallback. Pascal-friendly: no subgroups, no atomics, no f16, one bind group,
// 2 storage buffers.

const BR: u32 = 128u;    // query rows per workgroup
const RG: u32 = 64u;     // row groups (BR / 2 query rows per thread)
const BC: u32 = 16u;     // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const V4: u32 = 8u;      // vec4 slots per lane (V4*4*LANES == HD)
const HDV: u32 = 32u;    // vec4 slots per key row (HD/4)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide
const STG: u32 = 2u;     // vec4 staged per thread per tile (BC*HDV / 256)

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,        // T
    head_dim: u32,     // <= 128
    qkv_stride: u32,   // 3 * d_model
    q_off: u32,        // 0
    k_off: u32,        // d_model
    v_off: u32,        // 2 * d_model
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> ksh:   array<vec4<f32>, 512>;  // BC*HDV  ->  8 KiB
var<workgroup> vsh:   array<vec4<f32>, 512>;  // BC*HDV  ->  8 KiB
var<workgroup> part0: array<vec4<f32>, 1024>; // BC*RG   -> 16 KiB
var<workgroup> part1: array<vec4<f32>, 1024>; // BC*RG   -> 16 KiB

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let T = p.tcols;
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (b, h, query-tile).
    let wg = wgid.y * nwg.x + wgid.x;
    let ntiles_q = (T + BR - 1u) / BR;
    let qt = wg % ntiles_q;
    let r = wg / ntiles_q;
    let h = r % p.n_heads;
    let b = r / p.n_heads;
    if (b >= p.bsz) { return; }

    let lt = lid.x;                 // 0..255
    let rg = lt / LANES;            // 0..63  -> row group within the tile
    let lane = lt % LANES;          // 0..3   -> vec4-slot phase
    let i0 = qt * BR + rg;          // this thread's first query row
    let i1 = i0 + RG;               // ... and its second
    let live0 = i0 < T;
    let live1 = i1 < T;

    // Two query rows' q, and their output accumulators. Slot c holds channels
    // 4*(c*LANES + lane) .. +3.
    var q0: array<vec4<f32>, 8>;
    var q1: array<vec4<f32>, 8>;
    var o0: array<vec4<f32>, 8>;
    var o1: array<vec4<f32>, 8>;
    let qb0 = (b * T + i0) * p.qkv_stride + p.q_off + h * hd;
    let qb1 = (b * T + i1) * p.qkv_stride + p.q_off + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        var v0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var v1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (live0) {
            if (d0 + 0u < hd) { v0.x = qkv[qb0 + d0 + 0u]; }
            if (d0 + 1u < hd) { v0.y = qkv[qb0 + d0 + 1u]; }
            if (d0 + 2u < hd) { v0.z = qkv[qb0 + d0 + 2u]; }
            if (d0 + 3u < hd) { v0.w = qkv[qb0 + d0 + 3u]; }
        }
        if (live1) {
            if (d0 + 0u < hd) { v1.x = qkv[qb1 + d0 + 0u]; }
            if (d0 + 1u < hd) { v1.y = qkv[qb1 + d0 + 1u]; }
            if (d0 + 2u < hd) { v1.z = qkv[qb1 + d0 + 2u]; }
            if (d0 + 3u < hd) { v1.w = qkv[qb1 + d0 + 3u]; }
        }
        q0[c] = v0;
        q1[c] = v1;
        o0[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        o1[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Where in `qkv` this thread's STG staged vec4s live, as an offset from a
    // tile's first row - loop-invariant, so the pipelined loads below are two
    // adds and a bounds check.
    var srow: array<u32, 2>;   // key row within the tile
    var sd0:  array<u32, 2>;   // first channel of the staged vec4
    for (var s = 0u; s < STG; s = s + 1u) {
        let e = lt + s * 256u;
        srow[s] = e / HDV;
        sd0[s] = (e % HDV) * 4u;
    }

    var rk: array<vec4<f32>, 2>;
    var rv: array<vec4<f32>, 2>;

    let ntiles_k = (T + BC - 1u) / BC;

    // Prime the pipeline: tile 0 straight into shared.
    for (var s = 0u; s < STG; s = s + 1u) {
        let j = srow[s];
        let d0 = sd0[s];
        var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (j < T) {
            let base = (b * T + j) * p.qkv_stride + h * hd + d0;
            let bk = base + p.k_off;
            let bv = base + p.v_off;
            if (d0 + 3u < hd) {
                kv = vec4<f32>(qkv[bk], qkv[bk + 1u], qkv[bk + 2u], qkv[bk + 3u]);
                vv = vec4<f32>(qkv[bv], qkv[bv + 1u], qkv[bv + 2u], qkv[bv + 3u]);
            } else {
                if (d0 + 0u < hd) { kv.x = qkv[bk + 0u]; vv.x = qkv[bv + 0u]; }
                if (d0 + 1u < hd) { kv.y = qkv[bk + 1u]; vv.y = qkv[bv + 1u]; }
                if (d0 + 2u < hd) { kv.z = qkv[bk + 2u]; vv.z = qkv[bv + 2u]; }
            }
        }
        ksh[lt + s * 256u] = kv;
        vsh[lt + s * 256u] = vv;
    }
    workgroupBarrier();

    var m0 = -3.4e38;   // running max, row i0
    var l0 = 0.0;       // running sum of exp, row i0
    var m1 = -3.4e38;
    var l1 = 0.0;

    var pj0: array<f32, 16>;
    var pj1: array<f32, 16>;

    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        // 1. Issue the NEXT tile's global loads into registers, first, so they
        //    overlap this tile's ~2000 multiply-adds instead of a barrier.
        let has_next = kt + 1u < ntiles_k;
        if (has_next) {
            let j0 = (kt + 1u) * BC;
            for (var s = 0u; s < STG; s = s + 1u) {
                let j = j0 + srow[s];
                let d0 = sd0[s];
                var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                if (j < T) {
                    let base = (b * T + j) * p.qkv_stride + h * hd + d0;
                    let bk = base + p.k_off;
                    let bv = base + p.v_off;
                    if (d0 + 3u < hd) {
                        kv = vec4<f32>(qkv[bk], qkv[bk + 1u], qkv[bk + 2u], qkv[bk + 3u]);
                        vv = vec4<f32>(qkv[bv], qkv[bv + 1u], qkv[bv + 2u], qkv[bv + 3u]);
                    } else {
                        if (d0 + 0u < hd) { kv.x = qkv[bk + 0u]; vv.x = qkv[bv + 0u]; }
                        if (d0 + 1u < hd) { kv.y = qkv[bk + 1u]; vv.y = qkv[bv + 1u]; }
                        if (d0 + 2u < hd) { kv.z = qkv[bk + 2u]; vv.z = qkv[bv + 2u]; }
                    }
                }
                rk[s] = kv;
                rv[s] = vv;
            }
        }

        // 2. Partial dot products for BOTH query rows off one tile read.
        for (var j = 0u; j < BC; j = j + 1u) {
            var a0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var a1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            let ko = j * HDV + lane;
            for (var c = 0u; c < V4; c = c + 1u) {
                let kv = ksh[ko + c * LANES];
                a0 = a0 + q0[c] * kv;
                a1 = a1 + q1[c] * kv;
            }
            let po = j * RG + rg;
            part0[po][lane] = a0.x + a0.y + a0.z + a0.w;
            part1[po][lane] = a1.x + a1.y + a1.z + a1.w;
        }
        workgroupBarrier();

        // 3. `ksh` is dead now - re-stage it under the barrier just taken.
        if (has_next) {
            ksh[lt] = rk[0];
            ksh[lt + 256u] = rk[1];
        }

        var krows = BC;
        let rem = T - kt * BC;
        if (rem < BC) { krows = rem; }
        var tmax0 = -3.4e38;
        var tmax1 = -3.4e38;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s0 = -3.4e38;
            var s1 = -3.4e38;
            if (j < krows) {
                let po = j * RG + rg;
                let v0 = part0[po];
                let v1 = part1[po];
                s0 = (v0.x + v0.y + v0.z + v0.w) * scale;
                s1 = (v1.x + v1.y + v1.z + v1.w) * scale;
            }
            pj0[j] = s0;
            pj1[j] = s1;
            tmax0 = max(tmax0, s0);
            tmax1 = max(tmax1, s1);
        }
        let mn0 = max(m0, tmax0);
        let mn1 = max(m1, tmax1);
        let corr0 = exp(m0 - mn0);
        let corr1 = exp(m1 - mn1);
        var ls0 = 0.0;
        var ls1 = 0.0;
        for (var j = 0u; j < BC; j = j + 1u) {
            let e0 = exp(pj0[j] - mn0);
            let e1 = exp(pj1[j] - mn1);
            pj0[j] = e0;
            pj1[j] = e1;
            ls0 = ls0 + e0;
            ls1 = ls1 + e1;
        }
        l0 = l0 * corr0 + ls0;
        l1 = l1 * corr1 + ls1;
        m0 = mn0;
        m1 = mn1;

        // 4. Both rows' o accumulate off one vsh read.
        for (var c = 0u; c < V4; c = c + 1u) {
            let vo = c * LANES + lane;
            var a0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var a1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            for (var j = 0u; j < BC; j = j + 1u) {
                let vv = vsh[j * HDV + vo];
                a0 = a0 + pj0[j] * vv;
                a1 = a1 + pj1[j] * vv;
            }
            o0[c] = o0[c] * corr0 + a0;
            o1[c] = o1[c] * corr1 + a1;
        }
        workgroupBarrier();

        // 5. `vsh` is dead now, and `ksh` for the next tile is already visible
        //    through the barrier above.
        if (has_next) {
            vsh[lt] = rv[0];
            vsh[lt + 256u] = rv[1];
        }
    }

    let inv0 = 1.0 / l0;
    let inv1 = 1.0 / l1;
    let ob0 = (b * T + i0) * p.d_model + h * hd;
    let ob1 = (b * T + i1) * p.d_model + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        if (live0) {
            let v = o0[c] * inv0;
            if (d0 + 0u < hd) { out[ob0 + d0 + 0u] = v.x; }
            if (d0 + 1u < hd) { out[ob0 + d0 + 1u] = v.y; }
            if (d0 + 2u < hd) { out[ob0 + d0 + 2u] = v.z; }
            if (d0 + 3u < hd) { out[ob0 + d0 + 3u] = v.w; }
        }
        if (live1) {
            let v = o1[c] * inv1;
            if (d0 + 0u < hd) { out[ob1 + d0 + 0u] = v.x; }
            if (d0 + 1u < hd) { out[ob1 + d0 + 1u] = v.y; }
            if (d0 + 2u < hd) { out[ob1 + d0 + 2u] = v.z; }
            if (d0 + 3u < hd) { out[ob1 + d0 + 3u] = v.w; }
        }
    }
}
