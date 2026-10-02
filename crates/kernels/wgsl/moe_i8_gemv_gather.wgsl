// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE int8 GEMV over a fused expert bank: each slot reads ITS expert's weight rows out of one buffer - the decode-regime expert GEMM
// @how   DP4A packed int8, 16 lanes per weight row, 16-byte vector loads, integer sum per 32-group, ordered fold
// @opt   5
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The int8 expert GEMM at decode row counts. A MoE layer's `n_experts`
// matrices of one projection live back to back in ONE bank (`[n_experts * n,
// k/4]` packed words, llama.cpp's own stacked layout), and a routed row ("slot")
// names the expert it wants by id, so no per-expert buffer, binding or dispatch
// exists: one launch computes every selected expert's projection for every
// row. The 256 experts a dense per-expert loop would visit - 5 dispatches
// each - become the 8 (+1 shared) a token really uses, in one.
//
//   out[s, col] = sx[s / xdiv] * sum_g  f32( dot_g ) * sw[row_w, g],   row_w = ids[s] * n + col
//
// `xdiv` picks the activation row a slot reads: `top_k + shared` for the
// gate/up projections (every slot of a token reads that token's one quantised
// `xn2`), `1` for the down projection (each slot reads its own quantised
// `h`). No gate weight is applied here: the combine step scales and sums the
// slots (`moe_slot_combine.wgsl`), so this kernel is a plain projection.
//
//   params : u32 [slots, kg, n, xdiv]   kg = k/4 words per row, k a multiple of 32
//   xq     : [rows, kg] u32             4 int8 per word
//   sx     : [rows]     f32             per-row activation scale
//   ids    : [slots]    u32             expert (bank block) of each slot
//   wq     : [n_blocks * n, kg] u32     the bank
//   sw     : [n_blocks * n, kg/8] f32   one scale per 32 weights
//   out    : [slots, n] f32
//
// ## The order of the arithmetic, which a native kernel must keep to the bit
//
// A weight row is `kg/8` groups of 32 int8. Sixteen lanes serve a row: lane `l`
// owns groups `l, l+16, l+32, ...`. For a group the 8 `dot4` products are summed
// in i32 (exact, so their order is free), converted once to f32 and multiplied
// by the group's scale; a lane adds its groups' terms in ascending group order
// into one f32 accumulator. The 16 lane accumulators are then folded in
// ascending lane order, `((a0 + a1) + a2) + ...`, and multiplied by `sx`. f32
// addition is not associative, so that order IS the contract:
// `kernels_cuda`'s `moe_i8_gemv_gather` reproduces it with warp shuffles and is
// gated on the raw bits against this kernel.
//
// One workgroup (64 invocations) covers 4 weight rows of one slot; the
// dispatch is `slots * ceil(n / 4)` workgroups, slot-major. A column past `n`
// computes nothing and writes nothing.

struct Params {
    slots: u32,
    kg: u32,
    n: u32,
    xdiv: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       sx:  array<f32>;
@group(0) @binding(3) var<storage, read>       ids: array<u32>;
@group(0) @binding(4) var<storage, read>       wq:  array<vec4<u32>>;
@group(0) @binding(5) var<storage, read>       sw:  array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

const LANES: u32 = 16u;
const COLS: u32 = 4u;

var<workgroup> partial: array<f32, 64>;

fn dot32(x0: vec4<u32>, x1: vec4<u32>, w0: vec4<u32>, w1: vec4<u32>) -> i32 {
    return dot4I8Packed(x0.x, w0.x) + dot4I8Packed(x0.y, w0.y) + dot4I8Packed(x0.z, w0.z) + dot4I8Packed(x0.w, w0.w)
        + dot4I8Packed(x1.x, w1.x) + dot4I8Packed(x1.y, w1.y) + dot4I8Packed(x1.z, w1.z) + dot4I8Packed(x1.w, w1.w);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let blk = wg.y * nwg.x + wg.x;
    let tiles = (p.n + COLS - 1u) / COLS;
    let slot = blk / tiles;
    let col = (blk % tiles) * COLS + li.x / LANES;
    let lane = li.x % LANES;
    let live = slot < p.slots && col < p.n;

    var acc = 0.0;
    if (live) {
        let kg4 = p.kg / 4u;                       // vec4 words per row
        let ng = p.kg / 8u;                        // 32-groups per row
        let row_w = ids[slot] * p.n + col;
        let wbase = row_w * kg4;
        let xbase = (slot / p.xdiv) * kg4;
        let sbase = row_w * ng;
        for (var g = lane; g < ng; g = g + LANES) {
            let d = dot32(xq[xbase + 2u * g], xq[xbase + 2u * g + 1u], wq[wbase + 2u * g], wq[wbase + 2u * g + 1u]);
            acc = acc + f32(d) * sw[sbase + g];
        }
    }
    partial[li.x] = acc;
    workgroupBarrier();
    if (lane == 0u && live) {
        var s = 0.0;
        let c = (li.x / LANES) * LANES;
        for (var i = 0u; i < LANES; i = i + 1u) { s = s + partial[c + i]; }
        out[slot * p.n + col] = s * sx[slot / p.xdiv];
    }
}
