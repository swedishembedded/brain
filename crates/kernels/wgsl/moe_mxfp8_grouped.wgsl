// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast prefill for MXFP8 mixture-of-experts models for its clients. If your team needs
// expertise in turning thousands of routed tokens into a few weight passes per expert then you can procure our
// services by sending an email to info@swedishembedded.com.

// @what  Sparse-MoE MXFP8 GEMM over a fused expert bank with each expert's slots grouped: an E4M3 weight chunk is decoded once for up to 8 slots - the prefill-regime expert GEMM
// @how   16 lanes per weight row, 8 slots per weight pass, register decode, one top-level barrier, ordered fold
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// The MXFP8 sibling of `moe_nvfp4_grouped.wgsl`: the slots of an expert (ordered by `moe_route_*.wgsl`) share each
// decoded weight chunk, up to `MR` of them per pass. THE ARITHMETIC IS `moe_mxfp8_gemv_gather.wgsl`'S, to the bit: per
// slot the same per-lane accumulation over 16-weight chunks and the same ascending lane fold, so a slot's output does
// not depend on which kernel or batch computed it.
//
//   params : u32 [tiles, k, n, xdiv, ne]   tiles bounds the router's real tile count (`ne + ceil(slots / 8)`)
//   x      : [rows, k] f32
//   tab    : [2 * (ne + 1)] u32            `moe_route_scan.wgsl`: slot starts per expert, then tile starts per expert
//   perm   : [slots] u32                   `moe_route_emit.wgsl`
//   wq     : [ne * n, k/4] u32             the bank (see the gather kernel)
//   sc     : [ne * n, k/128] u32
//   out    : [slots, n] f32
//
// Dispatch: `ceil(n / 4)` column tiles x `tiles` MR-slot tiles; a block past the table's real tile count does nothing.

struct Params {
    tiles: u32,
    k: u32,
    n: u32,
    xdiv: u32,
    ne: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       tab:  array<u32>;
@group(0) @binding(3) var<storage, read>       perm: array<u32>;
@group(0) @binding(4) var<storage, read>       wq:   array<vec4<u32>>;
@group(0) @binding(5) var<storage, read>       sc:   array<u32>;
@group(0) @binding(6) var<storage, read_write> out:  array<f32>;

const LANES: u32 = 16u;
const COLS: u32 = 4u;
const MR: u32 = 8u;

var<workgroup> partial: array<f32, 512>;   // [thread][slot of the tile]

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    var v = f32(m) * (1.0 / 512.0);
    if (e != 0u) { v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    if ((b & 128u) != 0u) { v = -v; }
    return v;
}

fn e8m0(b: u32) -> f32 {
    if (b == 0u) { return bitcast<f32>(0x00400000u); }
    if (b == 255u) { return bitcast<f32>(0x7fc00000u); }
    return bitcast<f32>(b << 23u);
}

fn dot4(w4: u32, xv: vec4<f32>) -> f32 {
    return e4m3(w4 & 255u) * xv.x + e4m3((w4 >> 8u) & 255u) * xv.y + e4m3((w4 >> 16u) & 255u) * xv.z + e4m3(w4 >> 24u) * xv.w;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let blk = wg.y * nwg.x + wg.x;
    let col_tiles = (p.n + COLS - 1u) / COLS;
    let tile = blk / col_tiles;
    let col = (blk % col_tiles) * COLS + li.x / LANES;
    let lane = li.x % LANES;
    // The expert whose tile range holds `tile` - the last e with tile_start[e] <= tile - or `ne` when `tile` is past
    // the last tile the router produced. (Inline, and again after the barrier: the CPU JIT splits the body there.)
    var e = p.ne;
    if (tile < tab[2u * p.ne + 1u]) {
        var lo = 0u;
        var hi = p.ne;
        while (hi - lo > 1u) {
            let mid = (lo + hi) / 2u;
            if (tab[p.ne + 1u + mid] <= tile) { lo = mid; } else { hi = mid; }
        }
        e = lo;
    }
    var first = 0u;
    var cnt = 0u;
    if (e < p.ne) {
        first = tab[e] + (tile - tab[p.ne + 1u + e]) * MR;
        cnt = min(MR, tab[e + 1u] - first);
    }

    var acc: array<f32, 8>;
    var xbase: array<u32, 8>;
    for (var r = 0u; r < MR; r = r + 1u) {
        acc[r] = 0.0;
        xbase[r] = 0u;
        if (r < cnt) { xbase[r] = (perm[first + r] / p.xdiv) * (p.k / 4u); }
    }
    if (e < p.ne && col < p.n) {
        let kv = p.k / 16u;
        let row_w = e * p.n + col;
        let wbase = row_w * kv;
        let sbase = row_w * (p.k / 128u);
        for (var j = lane; j < kv; j = j + LANES) {
            let w4 = wq[wbase + j];
            let b = j / 2u;
            let s = e8m0((sc[sbase + b / 4u] >> ((b % 4u) * 8u)) & 255u);
            for (var r = 0u; r < MR; r = r + 1u) {
                if (r < cnt) {
                    let x0 = xbase[r] + 4u * j;
                    let chunk = dot4(w4.x, x[x0]) + dot4(w4.y, x[x0 + 1u]) + dot4(w4.z, x[x0 + 2u]) + dot4(w4.w, x[x0 + 3u]);
                    acc[r] = acc[r] + chunk * s;
                }
            }
        }
    }
    for (var r = 0u; r < MR; r = r + 1u) { partial[li.x * MR + r] = acc[r]; }
    workgroupBarrier();
    // After the barrier the tile is located again, not carried across it (the CPU JIT re-initialises locals at a barrier).
    let blk2 = wg.y * nwg.x + wg.x;
    let tile2 = blk2 / col_tiles;
    let col2 = (blk2 % col_tiles) * COLS + li.x / LANES;
    var e2 = p.ne;
    if (tile2 < tab[2u * p.ne + 1u]) {
        var lo2 = 0u;
        var hi2 = p.ne;
        while (hi2 - lo2 > 1u) {
            let mid2 = (lo2 + hi2) / 2u;
            if (tab[p.ne + 1u + mid2] <= tile2) { lo2 = mid2; } else { hi2 = mid2; }
        }
        e2 = lo2;
    }
    if (li.x % LANES == 0u && e2 < p.ne && col2 < p.n) {
        let first2 = tab[e2] + (tile2 - tab[p.ne + 1u + e2]) * MR;
        let cnt2 = min(MR, tab[e2 + 1u] - first2);
        let c = (li.x / LANES) * LANES;
        for (var r = 0u; r < cnt2; r = r + 1u) {
            var t = 0.0;
            for (var i = 0u; i < LANES; i = i + 1u) { t = t + partial[(c + i) * MR + r]; }
            out[perm[first2 + r] * p.n + col2] = t;
        }
    }
}
