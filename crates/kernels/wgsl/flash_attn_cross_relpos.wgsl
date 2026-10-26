// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Flash attention over spans with a decomposed relative-position bias (SAM/ViTDet), two-query-row register block with a software-pipelined K/V tile
// @how   flash_attn_cross_reg2's tiling with `rel_h[i][j / kw] + rel_w[i][j % kw]` added to each scaled score; spans are the batch axis
// @opt   5
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in fused attention for ViT encoders with relative-position bias then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// Flash attention with SAM's DECOMPOSED RELATIVE-POSITION BIAS: for span `b`, head `h`,
//   out[b,i,h,:] = softmax_j( q[b,i,h,:].k[b,j,h,:] / sqrt(hd) + rel_h[b,h,i,j / kw] + rel_w[b,h,i,j % kw] )
//                  . v[b,j,h,:]
// where the keys of a span are a `kh x kw` grid in row-major order and `rel_h`/`rel_w` are the
// hoisted `q . R` tables `attn_relpos_qr` builds (`[spans, heads, nq, kh]` and `[spans, heads, nq, kw]`),
// so the `[heads, nq, nk]` score slab, its bias and its probabilities are never materialized.
// A ViT block's windows are the spans (`bsz` = window count, each `nq = nk = window^2` rows of the
// window-partitioned slab), a global block is one span over the whole grid.
//
// It is `flash_attn_cross_reg2` with one added term per score. The bias is read, not recomputed: two loads per (row, key) from tables that are small against the
// score slab they replace (at SAM ViT-B's global shape 12.6 MB each against 805 MB).
//
// Everything else is `flash_attn_cross_reg2`'s - the kernel `flash_attn_bidir_reg2` generalised to separate
// q/k/v buffers - whose header derives the tiling, the two-query-row register block, the two-barrier software
// pipeline and the lane/bank ownership. The one invariant worth repeating: TILES ARE ALWAYS HD=128 WIDE,
// zero-filled past `head_dim` (SAM's is 64), so every channel loop's trip count is a compile-time constant
// and `q0/q1/o0/o1` stay in registers. Rows past `t_dec` are computed against a zero query and never written.
//
// What it replaces is SAM's materialized chain per window and query chunk: scores, `attn_relpos_add`, softmax,
// apply - one-thread-per-element kernels whose cost is the score slab (the unfused chain was 70% of the
// tower's device time). NOT bit-identical to it: an online softmax reassociates both sums, so a gate holds
// the two to a stated tolerance (`crates/sam1/tests`).
//
// @workgroup_size(256) = (BR/2 row groups) x LANES(4), checked against
// `DeviceCaps::max_workgroup_size` (queried, never assumed). Shared use is
// 48 KiB (ksh 8 + vsh 8 + part0 16 + part1 16), which is the Vulkan/NVIDIA
// 49152-byte compute limit exactly and far above the 16 KiB a Vulkan
// implementation is only REQUIRED to offer, so selection must also check
// `DeviceCaps::workgroup_mem_bytes` and fall back to the materialized trio.
// Two barriers per tile is more than the ONE the Cranelift CPU backend can
// split a kernel body at, so this kernel is GPU-only exactly like every other
// member of the family. Pascal-friendly: no subgroups, no atomics, no f16, one
// bind group, 4 storage buffers.

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
    t_dec: u32,        // query rows
    t_enc: u32,        // key/value rows
    head_dim: u32,     // <= 128
    q_stride: u32,     // row width of `q`
    q_off: u32,        // region offset within a `q` row
    k_stride: u32,     // row width of `k`
    k_off: u32,        // region offset within a `k` row
    v_stride: u32,     // row width of `v`
    v_off: u32,        // region offset within a `v` row
    d_model: u32,      // row width of `out` (heads*head_dim)
    kh_ext: u32,       // key grid height: row width of `rel_h`
    kw_ext: u32,       // key grid width: row width of `rel_w`, and key j is (j / kw_ext, j % kw_ext)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       k:   array<f32>;
@group(0) @binding(3) var<storage, read>       v:   array<f32>;
@group(0) @binding(4) var<storage, read>       rel_h: array<f32>;   // [bsz, heads, t_dec, kh_ext]
@group(0) @binding(5) var<storage, read>       rel_w: array<f32>;   // [bsz, heads, t_dec, kw_ext]
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> ksh:   array<vec4<f32>, 512>;  // BC*HDV  ->  8 KiB
var<workgroup> vsh:   array<vec4<f32>, 512>;  // BC*HDV  ->  8 KiB
var<workgroup> part0: array<vec4<f32>, 1024>; // BC*RG   -> 16 KiB
var<workgroup> part1: array<vec4<f32>, 1024>; // BC*RG   -> 16 KiB

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (b, h, query-tile).
    let wg = wgid.y * nwg.x + wgid.x;
    let ntiles_q = (Tq + BR - 1u) / BR;
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
    let live0 = i0 < Tq;
    let live1 = i1 < Tq;

    // Two query rows' q, and their output accumulators. Slot c holds channels
    // 4*(c*LANES + lane) .. +3.
    var q0: array<vec4<f32>, 8>;
    var q1: array<vec4<f32>, 8>;
    var o0: array<vec4<f32>, 8>;
    var o1: array<vec4<f32>, 8>;
    let qb0 = (b * Tq + i0) * p.q_stride + p.q_off + h * hd;
    let qb1 = (b * Tq + i1) * p.q_stride + p.q_off + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        var a0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var a1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (live0) {
            if (d0 + 0u < hd) { a0.x = q[qb0 + d0 + 0u]; }
            if (d0 + 1u < hd) { a0.y = q[qb0 + d0 + 1u]; }
            if (d0 + 2u < hd) { a0.z = q[qb0 + d0 + 2u]; }
            if (d0 + 3u < hd) { a0.w = q[qb0 + d0 + 3u]; }
        }
        if (live1) {
            if (d0 + 0u < hd) { a1.x = q[qb1 + d0 + 0u]; }
            if (d0 + 1u < hd) { a1.y = q[qb1 + d0 + 1u]; }
            if (d0 + 2u < hd) { a1.z = q[qb1 + d0 + 2u]; }
            if (d0 + 3u < hd) { a1.w = q[qb1 + d0 + 3u]; }
        }
        q0[c] = a0;
        q1[c] = a1;
        o0[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        o1[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Where in `k`/`v` this thread's STG staged vec4s live, as an offset from a
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

    let ntiles_k = (Tk + BC - 1u) / BC;

    // Prime the pipeline: tile 0 straight into shared.
    for (var s = 0u; s < STG; s = s + 1u) {
        let j = srow[s];
        let d0 = sd0[s];
        var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (j < Tk) {
            let bk = (b * Tk + j) * p.k_stride + p.k_off + h * hd + d0;
            let bv = (b * Tk + j) * p.v_stride + p.v_off + h * hd + d0;
            if (d0 + 3u < hd) {
                kv = vec4<f32>(k[bk], k[bk + 1u], k[bk + 2u], k[bk + 3u]);
                vv = vec4<f32>(v[bv], v[bv + 1u], v[bv + 2u], v[bv + 3u]);
            } else {
                if (d0 + 0u < hd) { kv.x = k[bk + 0u]; vv.x = v[bv + 0u]; }
                if (d0 + 1u < hd) { kv.y = k[bk + 1u]; vv.y = v[bv + 1u]; }
                if (d0 + 2u < hd) { kv.z = k[bk + 2u]; vv.z = v[bv + 2u]; }
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
        //    overlap this tile's multiply-adds instead of a barrier.
        let has_next = kt + 1u < ntiles_k;
        if (has_next) {
            let j0 = (kt + 1u) * BC;
            for (var s = 0u; s < STG; s = s + 1u) {
                let j = j0 + srow[s];
                let d0 = sd0[s];
                var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                if (j < Tk) {
                    let bk = (b * Tk + j) * p.k_stride + p.k_off + h * hd + d0;
                    let bv = (b * Tk + j) * p.v_stride + p.v_off + h * hd + d0;
                    if (d0 + 3u < hd) {
                        kv = vec4<f32>(k[bk], k[bk + 1u], k[bk + 2u], k[bk + 3u]);
                        vv = vec4<f32>(v[bv], v[bv + 1u], v[bv + 2u], v[bv + 3u]);
                    } else {
                        if (d0 + 0u < hd) { kv.x = k[bk + 0u]; vv.x = v[bv + 0u]; }
                        if (d0 + 1u < hd) { kv.y = k[bk + 1u]; vv.y = v[bv + 1u]; }
                        if (d0 + 2u < hd) { kv.z = k[bk + 2u]; vv.z = v[bv + 2u]; }
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
        let rem = Tk - kt * BC;
        if (rem < BC) { krows = rem; }
        var tmax0 = -3.4e38;
        var tmax1 = -3.4e38;
        // This (span, head)'s bias rows for this thread's two queries.
        let bh = b * p.n_heads + h;
        let rh0 = (bh * Tq + i0) * p.kh_ext;
        let rh1 = (bh * Tq + i1) * p.kh_ext;
        let rw0 = (bh * Tq + i0) * p.kw_ext;
        let rw1 = (bh * Tq + i1) * p.kw_ext;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s0 = -3.4e38;
            var s1 = -3.4e38;
            if (j < krows) {
                let po = j * RG + rg;
                let w0 = part0[po];
                let w1 = part1[po];
                s0 = (w0.x + w0.y + w0.z + w0.w) * scale;
                s1 = (w1.x + w1.y + w1.z + w1.w) * scale;
                let jk = kt * BC + j;
                let jh = jk / p.kw_ext;
                let jw = jk % p.kw_ext;
                // Rows past `t_dec` have no bias row; their (never written) result stays finite.
                if (live0) { s0 = s0 + rel_h[rh0 + jh] + rel_w[rw0 + jw]; }
                if (live1) { s1 = s1 + rel_h[rh1 + jh] + rel_w[rw1 + jw]; }
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
    let ob0 = (b * Tq + i0) * p.d_model + h * hd;
    let ob1 = (b * Tq + i1) * p.d_model + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        if (live0) {
            let w = o0[c] * inv0;
            if (d0 + 0u < hd) { out[ob0 + d0 + 0u] = w.x; }
            if (d0 + 1u < hd) { out[ob0 + d0 + 1u] = w.y; }
            if (d0 + 2u < hd) { out[ob0 + d0 + 2u] = w.z; }
            if (d0 + 3u < hd) { out[ob0 + d0 + 3u] = w.w; }
        }
        if (live1) {
            let w = o1[c] * inv1;
            if (d0 + 0u < hd) { out[ob1 + d0 + 0u] = w.x; }
            if (d0 + 1u < hd) { out[ob1 + d0 + 1u] = w.y; }
            if (d0 + 2u < hd) { out[ob1 + d0 + 2u] = w.z; }
            if (d0 + 3u < hd) { out[ob1 + d0 + 3u] = w.w; }
        }
    }
}
