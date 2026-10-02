// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE int8 GEMM over a fused expert bank with each expert's slots grouped: a weight row is read once for up to 8 slots - the prefill-regime expert GEMM
// @how   DP4A packed int8, 16 lanes per weight row, 8 slots per weight pass, integer sum per 32-group, ordered fold
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// `moe_i8_gemv_gather.wgsl` computes one slot per weight pass: right at decode,
// where a token's eight experts are eight different weight matrices, and ruinous
// at prefill, where a few thousand tokens give every expert hundreds of slots and
// the same weights are streamed once per slot. This kernel takes the slots of an
// expert together (`moe_route_*.wgsl` order them) and applies each weight row to
// up to `MR` of them per pass, as `matmul_i8_gemv` does for rows of a dense
// activation.
//
//   out[s, col] = sx[s / xdiv] * sum_g  f32( dot_g ) * sw[row_w, g],   row_w = expert(s) * n + col
//
// The ARITHMETIC IS THE GATHER KERNEL'S, to the bit: per slot the same 16-lane
// accumulation over 32-groups in ascending group order and the same ascending
// lane fold, so a slot's output does not depend on which kernel or which batch
// computed it - a prefill and a decode step agree on a token's expert outputs.
//
//   params : u32 [ne, kg, n, xdiv]    ne = experts (n_experts + shared)
//   xq     : [rows, kg] u32           4 int8 per word
//   sx     : [rows] f32               per-row activation scale
//   tab    : [2 * (ne + 1)] u32       `moe_route_scan.wgsl`
//   perm   : [slots] u32              `moe_route_emit.wgsl`
//   wq     : [ne * n, kg] u32         the bank
//   sw     : [ne * n, kg/8] f32
//   out    : [slots, n] f32           slot-major, like the gather kernel's
//
// Dispatch: `ceil(n / 4)` column tiles x `max_tiles` MR-slot tiles, where
// `max_tiles` bounds the tile count the router can produce
// (`ne + ceil(slots / MR)`); a block past the table's real tile count does
// nothing. Tiles are numbered expert-major, column tile fastest, so blocks that
// run together read the same activation rows and consecutive tiles of an expert
// find its weights in L2.

struct Params {
    ne: u32,
    kg: u32,
    n: u32,
    xdiv: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:   array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       sx:   array<f32>;
@group(0) @binding(3) var<storage, read>       tab:  array<u32>;
@group(0) @binding(4) var<storage, read>       perm: array<u32>;
@group(0) @binding(5) var<storage, read>       wq:   array<vec4<u32>>;
@group(0) @binding(6) var<storage, read>       sw:   array<f32>;
@group(0) @binding(7) var<storage, read_write> out:  array<f32>;

const LANES: u32 = 16u;
const COLS: u32 = 4u;
const MR: u32 = 8u;

var<workgroup> partial: array<f32, 512>;   // [thread][slot of the tile]

fn dot32(x0: vec4<u32>, x1: vec4<u32>, w0: vec4<u32>, w1: vec4<u32>) -> i32 {
    return dot4I8Packed(x0.x, w0.x) + dot4I8Packed(x0.y, w0.y) + dot4I8Packed(x0.z, w0.z) + dot4I8Packed(x0.w, w0.w)
        + dot4I8Packed(x1.x, w1.x) + dot4I8Packed(x1.y, w1.y) + dot4I8Packed(x1.z, w1.z) + dot4I8Packed(x1.w, w1.w);
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

    // The expert whose tile range holds `tile` - the last e with tile_start[e] <=
    // tile - or `ne` when `tile` is past the last tile the router produced.
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
    let live = e < p.ne && col < p.n;
    // Position in `perm` of the tile's first slot, and how many slots it holds.
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
        if (r < cnt) { xbase[r] = (perm[first + r] / p.xdiv) * (p.kg / 4u); }
    }
    if (live) {
        let kg4 = p.kg / 4u;
        let ng = p.kg / 8u;
        let row_w = e * p.n + col;
        let wbase = row_w * kg4;
        let sbase = row_w * ng;
        for (var g = lane; g < ng; g = g + LANES) {
            let w0 = wq[wbase + 2u * g];
            let w1 = wq[wbase + 2u * g + 1u];
            let s = sw[sbase + g];
            for (var r = 0u; r < MR; r = r + 1u) {
                if (r < cnt) {
                    let d = dot32(xq[xbase[r] + 2u * g], xq[xbase[r] + 2u * g + 1u], w0, w1);
                    acc[r] = acc[r] + f32(d) * s;
                }
            }
        }
    }
    for (var r = 0u; r < MR; r = r + 1u) { partial[li.x * MR + r] = acc[r]; }
    workgroupBarrier();
    // After the barrier the tile is located again, not carried across it (the
    // CPU JIT re-initialises locals at a barrier).
    var e2 = p.ne;
    if (lane == 0u && tile < tab[2u * p.ne + 1u]) {
        var lo2 = 0u;
        var hi2 = p.ne;
        while (hi2 - lo2 > 1u) {
            let mid2 = (lo2 + hi2) / 2u;
            if (tab[p.ne + 1u + mid2] <= tile) { lo2 = mid2; } else { hi2 = mid2; }
        }
        e2 = lo2;
    }
    let col2 = (blk % col_tiles) * COLS + li.x / LANES;
    if (e2 < p.ne && col2 < p.n) {
        let first2 = tab[e2] + (tile - tab[p.ne + 1u + e2]) * MR;
        let cnt2 = min(MR, tab[e2 + 1u] - first2);
        let c = (li.x / LANES) * LANES;
        for (var r = 0u; r < cnt2; r = r + 1u) {
            var s = 0.0;
            for (var i = 0u; i < LANES; i = i + 1u) { s = s + partial[(c + i) * MR + r]; }
            let slot = perm[first2 + r];
            out[slot * p.n + col2] = s * sx[slot / p.xdiv];
        }
    }
}
