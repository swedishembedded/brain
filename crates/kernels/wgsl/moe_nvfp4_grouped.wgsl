// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast prefill for NVFP4 mixture-of-experts models for its clients. If your team needs
// expertise in turning thousands of routed tokens into a few 4-bit weight passes per expert then you can procure our
// services by sending an email to info@swedishembedded.com.

// @what  Sparse-MoE NVFP4 GEMM over a fused expert bank with each expert's slots grouped: a 4-bit weight row is decoded once for up to 8 slots - the prefill-regime expert GEMM
// @how   16 lanes per weight row, 8 slots per weight pass, E2M1 x E4M3 x f32 decoded once per chunk, ordered fold
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant nvfp4
// @dtype f32
//
// `moe_nvfp4_gemv_gather.wgsl` computes one slot per weight pass: right at decode, ruinous at prefill where every
// expert has hundreds of slots. This kernel takes the slots of an expert together (`moe_route_*.wgsl` order them)
// and applies each decoded weight chunk to up to `MR` of them per pass. The ARITHMETIC IS THE GATHER KERNEL'S, to
// the bit: per slot the same per-lane accumulation over 32-weight chunks and the same ascending lane fold, so a
// slot's output does not depend on which kernel or batch computed it.
//
//   params : u32 [tiles, k, n, xdiv, ne]   tiles bounds the router's real tile count (`ne + ceil(slots / 8)`)
//   x      : [rows, k] f32
//   tab    : [2 * (ne + 1)] u32            `moe_route_scan.wgsl`
//   perm   : [slots] u32                   `moe_route_emit.wgsl`
//   wq     : [ne * n, k/8] u32             the bank (see the gather kernel)
//   sc     : [ne * n, k/64] u32
//   gl     : [ne] f32
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
@group(0) @binding(6) var<storage, read>       gl:   array<f32>;
@group(0) @binding(7) var<storage, read_write> out:  array<f32>;

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

// The value of an E2M1 code: sign, 2-bit exponent (bias 1), 1-bit mantissa.
fn fp4_value(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let m = c & 1u;
    var v = f32(m) * 0.5;
    if (e != 0u) { v = bitcast<f32>(((126u + e) << 23u) | (m << 22u)); }
    if ((c & 8u) != 0u) { v = -v; }
    return v;
}

// byte -> the two E2M1 values it holds (low nibble first): decoding a weight word is four table reads, not 8 decodes.
var<workgroup> pair_lut: array<vec2<f32>, 256>;

fn init_pair_lut(t: u32) {
    for (var i = t; i < 256u; i = i + 64u) {
        pair_lut[i] = vec2<f32>(fp4_value(i & 15u), fp4_value(i >> 4u));
    }
}

// The dot product of the 8 codes of word `w` with the 8 activations `xa`, `xb` (low code first).
fn dot8(w: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let p0 = pair_lut[w & 255u];
    let p1 = pair_lut[(w >> 8u) & 255u];
    let p2 = pair_lut[(w >> 16u) & 255u];
    let p3 = pair_lut[w >> 24u];
    return p0.x * xa.x + p0.y * xa.y + p1.x * xa.z + p1.y * xa.w + p2.x * xb.x + p2.y * xb.y + p3.x * xb.z + p3.y * xb.w;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    init_pair_lut(li.x);
    workgroupBarrier();
    let blk = wg.y * nwg.x + wg.x;
    let col_tiles = (p.n + COLS - 1u) / COLS;
    let tile = blk / col_tiles;
    let col = (blk % col_tiles) * COLS + li.x / LANES;
    let lane = li.x % LANES;

    // The expert whose tile range holds `tile` - the last e with tile_start[e] <= tile - or `ne` when `tile` is
    // past the last tile the router produced.
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
    if (live) {
        let kv = p.k / 32u;
        let row_w = e * p.n + col;
        let wbase = row_w * kv;
        let sbase = row_w * (p.k / 64u);
        for (var j = lane; j < kv; j = j + LANES) {
            let w = wq[wbase + j];
            let sw = sc[sbase + j / 2u] >> ((j % 2u) * 16u);
            let s0 = e4m3(sw & 255u);
            let s1 = e4m3((sw >> 8u) & 255u);
            for (var r = 0u; r < MR; r = r + 1u) {
                if (r < cnt) {
                    let x0 = xbase[r] + 8u * j;
                    let b0 = dot8(w.x, x[x0], x[x0 + 1u]) + dot8(w.y, x[x0 + 2u], x[x0 + 3u]);
                    let b1 = dot8(w.z, x[x0 + 4u], x[x0 + 5u]) + dot8(w.w, x[x0 + 6u], x[x0 + 7u]);
                    acc[r] = acc[r] + b0 * s0;
                    acc[r] = acc[r] + b1 * s1;
                }
            }
        }
    }
    for (var r = 0u; r < MR; r = r + 1u) { partial[li.x * MR + r] = acc[r]; }
    workgroupBarrier();
    // After the barrier the tile is located again, not carried across it (the CPU JIT re-initialises locals at a barrier).
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
            out[slot * p.n + col2] = s * gl[e2];
        }
    }
}
