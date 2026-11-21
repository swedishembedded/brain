// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast serving of MXFP8 mixture-of-experts models for its clients. If your team needs
// expertise in microscaled expert GEMMs then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Sparse-MoE MXFP8 GEMV over a fused expert bank: each slot reads ITS expert's E4M3 weight rows (E8M0 scale per 32) out of one buffer, activations stay f32 - the decode-regime expert GEMM
// @how   16 lanes per weight row, 16-byte vector loads of 16 E4M3 bytes, register decode, one top-level barrier, ordered fold
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// The MXFP8 sibling of `moe_nvfp4_gemv_gather.wgsl`, for checkpoints that store their experts as OCP MXFP8: one E4M3
// byte per weight and one E8M0 power-of-two scale byte per 32 weights along `k`, no per-expert scale. A value is
// `e4m3(code) * 2^(e8m0(scale) - 127)`, as `checkpoint::packed`'s codec decodes it. Activations are not quantised.
// The bank is layout-agnostic: a fused gate/up bank (concatenated or row-interleaved) is read with `n = 2 * ff`, and
// the declared layout is applied where gate and up meet (`glu_act.wgsl`).
//
//   out[s, col] = sum_b  2^(e8m0(sc[row_w, b]) - 127) * sum_{i<32} e4m3(w[row_w, 32b + i]) * x[s / xdiv, 32b + i]
//   with e = ids[s], row_w = e * n + col
//
//   params : u32 [slots, k, n, xdiv]   k a multiple of 128
//   x      : [rows, k] f32
//   ids    : [slots] u32               expert (bank block) of each slot
//   wq     : [blocks * n, k/4] u32     four E4M3 bytes per word; the bank
//   sc     : [blocks * n, k/128] u32   four E8M0 scale bytes per word
//   out    : [slots, n] f32
//
// Dispatch: `slots * ceil(n / 4)` workgroups of 64.
//
// The arithmetic is `matmul_mxfp8_gemv.wgsl`'s (see its header), so an expert row and a dense row agree to the bit,
// and `moe_mxfp8_grouped.wgsl` matches it to the bit.

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
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

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
    let tiles = (p.n + COLS - 1u) / COLS;
    let slot = blk / tiles;
    let col = (blk % tiles) * COLS + li.x / LANES;
    let lane = li.x % LANES;

    var acc = 0.0;
    if (slot < p.slots && col < p.n) {
        let kv = p.k / 16u;
        let row_w = ids[slot] * p.n + col;
        let wbase = row_w * kv;
        let sbase = row_w * (p.k / 128u);
        let xbase = (slot / p.xdiv) * (p.k / 4u);
        for (var j = lane; j < kv; j = j + LANES) {
            let w4 = wq[wbase + j];
            let b = j / 2u;
            let s = e8m0((sc[sbase + b / 4u] >> ((b % 4u) * 8u)) & 255u);
            let x0 = xbase + 4u * j;
            let chunk = dot4(w4.x, x[x0]) + dot4(w4.y, x[x0 + 1u]) + dot4(w4.z, x[x0 + 2u]) + dot4(w4.w, x[x0 + 3u]);
            acc = acc + chunk * s;
        }
    }
    partial[li.x] = acc;
    workgroupBarrier();
    // After the barrier the slot is located again, not carried across it (the CPU JIT re-initialises locals there).
    let blk2 = wg.y * nwg.x + wg.x;
    let slot2 = blk2 / tiles;
    let col2 = (blk2 % tiles) * COLS + li.x / LANES;
    if (li.x % LANES == 0u && slot2 < p.slots && col2 < p.n) {
        var t = 0.0;
        let c = (li.x / LANES) * LANES;
        for (var i = 0u; i < LANES; i = i + 1u) { t = t + partial[c + i]; }
        out[slot2 * p.n + col2] = t;
    }
}
