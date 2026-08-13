// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  NLC -> NCHW with a per-channel bias - the epilogue of a conv lowered to a row-major GEMM
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// NLC -> NCHW with a per-channel bias — the epilogue of a conv lowered to a
// row-major GEMM.
//
//   x    : [L = H*W, C]   the GEMM output `y[HW, Cout] = col . Wᵀ`
//   bias : [C]
//   y    : [C, H*W]       y[c*L + l] = x[l*C + c] + bias[c]
//
// `nlc_nchw` already does the permutation; this fuses the conv bias into it so
// the lowered conv costs ONE extra pass over its output rather than two. Same
// role `conv_epilogue.wgsl` plays for the [Cout, HW]-oriented lowering, but for
// the transposed one — which is the orientation that lets the GEMM be chunked
// over spatial positions (see `im2col_at.wgsl`).
//
// A transpose has no coalesced element-wise indexing: whichever side the thread
// index follows, the other is strided, and on a P40 that costs the classic 8x
// sector amplification (measured: 158 ms for the FLUX.2 VAE decode's 35
// lowered-conv outputs, ~90 GB/s of 346). So stage a **64 x 64 tile in
// workgroup memory**: the load walks x along C (coalesced) and the store walks
// y along L (coalesced). The tile row stride is 65, not 64, so the store's
// column read `tile[t*65 + r]` lands on 32 distinct banks instead of one —
// without the pad this is a 32-way conflict and the transpose gains nothing.
//
// Dispatch: ceil(C/64) * ceil(L/64) workgroups of 64 invocations.

struct Params {
    total: u32,   // L*C (unused by the tiled form; kept so callers are stable)
    c: u32,
    l: u32,       // H*W
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:    array<f32>;

var<workgroup> tile: array<f32, 4160>;   // 64 rows x 65 (padded) columns

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let blk = wg.y * nwg.x + wg.x;
    let t = li.x;
    let nc = (p.c + 63u) / 64u;
    let nl = (p.l + 63u) / 64u;
    if (blk >= nc * nl) { return; }
    let c0 = (blk % nc) * 64u;
    let l0 = (blk / nc) * 64u;

    // Load: row r of the tile is x[l0+r, c0 .. c0+64) — thread t takes column t.
    let ct = c0 + t;
    for (var r = 0u; r < 64u; r = r + 1u) {
        var v = 0.0;
        if (ct < p.c && l0 + r < p.l) {
            v = x[(l0 + r) * p.c + ct];
        }
        tile[r * 65u + t] = v;
    }
    workgroupBarrier();

    // Store: row r of the OUTPUT is y[c0+r, l0 .. l0+64) — thread t takes l0+t.
    let lt = l0 + t;
    for (var r = 0u; r < 64u; r = r + 1u) {
        let cr = c0 + r;
        if (cr < p.c && lt < p.l) {
            y[cr * p.l + lt] = tile[t * 65u + r] + bias[cr];
        }
    }
}
