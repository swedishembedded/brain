// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast serving of NVFP4 mixture-of-experts models for its clients. If your team needs
// expertise in 4-bit expert GEMMs then you can procure our services by sending an email to info@swedishembedded.com.

// @what  Sparse-MoE NVFP4 GEMV over a fused expert bank: each slot reads ITS expert's 4-bit weight rows out of one buffer, activations stay f32 (W4A16) - the decode-regime expert GEMM
// @how   16 lanes per weight row, 16-byte vector loads of 32 E2M1 codes, E4M3 scale per 16, f32 per-expert global scale, ordered fold
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant nvfp4
// @dtype f32
//
// The NVFP4 sibling of `moe_i8_gemv_gather.wgsl`, for checkpoints that store their experts as the NVIDIA
// ModelOpt "W4A16_NVFP4" format: two E2M1 codes per byte (element 2i in the low nibble), one E4M3 scale byte per 16
// weights along `k`, and one f32 `weight_scale_2` per expert matrix. A value is
// `e2m1(code) * e4m3(block scale) * global[expert]`, in that order, as `checkpoint::packed`'s codec decodes it.
// Activations are NOT quantised: the product is a float-read GEMV, so no activation scale exists.
//
//   out[s, col] = global[e] * sum_b  e4m3(sc[row_w, b]) * sum_{i<16} e2m1(w[row_w, 16b + i]) * x[s / xdiv, 16b + i]
//   with e = ids[s], row_w = e * n + col
//
//   params : u32 [slots, k, n, xdiv]   k a multiple of 32
//   x      : [rows, k] f32
//   ids    : [slots] u32               expert (bank block) of each slot
//   wq     : [blocks * n, k/8] u32     eight codes per word; the bank
//   sc     : [blocks * n, k/64] u32    four scale bytes per word (E4M3, one per 16 weights)
//   gl     : [blocks] f32              per-expert global scale
//   out    : [slots, n] f32
//
// ## Order of the arithmetic (shared with `moe_nvfp4_grouped.wgsl`, which must match it to the bit)
//
// Sixteen lanes serve a row; lane `l` owns 32-weight chunks `l, l+16, ...`. A chunk is two 16-weight blocks: each
// block's dot product is summed left to right in f32, scaled by its block scale, and the two terms are added into
// the lane's accumulator in block order. The 16 lane accumulators are folded in ascending lane order and scaled by
// the expert's global scale.

struct Params {
    slots: u32,
    k: u32,
    n: u32,
    xdiv: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       ids: array<u32>;
@group(0) @binding(3) var<storage, read>       wq:  array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       sc:  array<u32>;
@group(0) @binding(5) var<storage, read>       gl:  array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

const LANES: u32 = 16u;
const COLS: u32 = 4u;

var<workgroup> partial: array<f32, 64>;

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
    let tiles = (p.n + COLS - 1u) / COLS;
    let slot = blk / tiles;
    let col = (blk % tiles) * COLS + li.x / LANES;
    let lane = li.x % LANES;
    let live = slot < p.slots && col < p.n;

    var acc = 0.0;
    if (live) {
        let expert = ids[slot];
        let kv = p.k / 32u;                        // 32-weight chunks (one vec4<u32>) per row
        let row_w = expert * p.n + col;
        let wbase = row_w * kv;
        let sbase = row_w * (p.k / 64u);
        let xbase = (slot / p.xdiv) * (p.k / 4u);
        for (var j = lane; j < kv; j = j + LANES) {
            let w = wq[wbase + j];
            let sw = sc[sbase + j / 2u] >> ((j % 2u) * 16u);
            let x0 = xbase + 8u * j;
            let b0 = dot8(w.x, x[x0], x[x0 + 1u]) + dot8(w.y, x[x0 + 2u], x[x0 + 3u]);
            let b1 = dot8(w.z, x[x0 + 4u], x[x0 + 5u]) + dot8(w.w, x[x0 + 6u], x[x0 + 7u]);
            acc = acc + b0 * e4m3(sw & 255u);
            acc = acc + b1 * e4m3((sw >> 8u) & 255u);
        }
    }
    partial[li.x] = acc;
    workgroupBarrier();
    if (lane == 0u && live) {
        var s = 0.0;
        let c = (li.x / LANES) * LANES;
        for (var i = 0u; i < LANES; i = i + 1u) { s = s + partial[c + i]; }
        // The expert is read again: a local set before the barrier is not carried across it (the CPU JIT).
        out[slot * p.n + col] = s * gl[ids[slot]];
    }
}
