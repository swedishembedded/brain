// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal GQA flash attention backward, dQ: rebuilds each softmax weight from the forward's LSE instead of reading [H,T,T] probs
// @how   256-thread workgroup tile, 5 barriers (3 per key tile), causal early-exit over K tiles
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The query-side half of the backward of `flash_attn_causal_gqa.wgsl`, in the
// FlashAttention-2 formulation. With scaled scores `s_ij = scale·q_i·k_j`,
// weights `P_ij = exp(s_ij - lse_i)` (the forward wrote `lse`), `dP_ij =
// dO_i·v_j` and `D_i = Σ_d dO_i[d]·O_i[d]`:
//
//   dS_ij = P_ij·(dP_ij - D_i)          dQ_i = scale · Σ_{j<=i} dS_ij·k_j
//
// `D_i` is computed here once per row and ALSO written to `dsum`, where the
// key-side kernel (`flash_attn_causal_gqa_bwd_dkv.wgsl`, dispatched after this
// one) reads it - so the pair costs one extra `[B,H,T]` vector, not a pass.
//
// Nothing `[T,T]`-sized exists at any point: memory is O(T·head_dim), which is
// what makes a training sequence of tens of thousands of tokens fit where the
// materialised `gqa_scores`/`attn_softmax`/`gqa_bwd_*` chain needs
// `n_heads·T²` floats per layer.
//
// Layout and work split are the forward's own (see its header for why the
// head_dim is LANE-SPLIT: a one-thread-per-row kernel spills `q[128]` to
// local memory on Pascal): `lanes` threads per query row (4 up to head_dim 128
// with 64 rows per workgroup, 8 up to head_dim 256 with 32 rows), CH = 32
// channels per lane. Each thread keeps its q, dO and
// dQ slices in registers; K and V tiles of BC = 8 rows are staged in shared
// memory and read by every row of the tile. Per key, the two partial dot
// products (`q·k` and `dO·v`) are re-summed across the lanes through shared
// memory, as in the forward. Shared use is 32 KiB (ksh 8 + vsh 8 + two
// partial buffers 8 each); callers gate on the device's workgroup memory.

const WG: u32 = 256u;    // threads per workgroup == BR * lanes in both modes
const BC: u32 = 8u;      // key/value rows per shared tile
const CH: u32 = 32u;     // channels per lane (lanes*CH >= head_dim)

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,        // T
    head_dim: u32,     // <= 256
    group: u32,        // n_heads / n_kv_heads
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:    array<f32>;
@group(0) @binding(2) var<storage, read>       k:    array<f32>;
@group(0) @binding(3) var<storage, read>       v:    array<f32>;
@group(0) @binding(4) var<storage, read>       o:    array<f32>;  // forward output (ctx)
@group(0) @binding(5) var<storage, read>       d_o:  array<f32>;  // grad wrt ctx
@group(0) @binding(6) var<storage, read>       lse:  array<f32>;  // [B, n_heads, T]
@group(0) @binding(7) var<storage, read_write> dq:   array<f32>;
@group(0) @binding(8) var<storage, read_write> dsum: array<f32>;  // D, [B, n_heads, T]

var<workgroup> ksh:    array<f32, 2048>;  // BC*256
var<workgroup> vsh:    array<f32, 2048>;  // BC*256
var<workgroup> part_s: array<f32, 2048>;  // BC*WG
var<workgroup> part_p: array<f32, 2048>;  // BC*WG

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let T = p.tcols;
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));
    let lanes = select(4u, 8u, hd > 128u);   // threads per query row
    let BR = WG / lanes;                     // query rows per workgroup
    let hdt = lanes * CH;                    // K/V tile row width

    // Flat workgroup id -> (b, h, query-tile); uniform per workgroup, so the
    // early return below never splits a barrier.
    let wg = wgid.y * nwg.x + wgid.x;
    let ntiles_q = (T + BR - 1u) / BR;
    let qt = wg % ntiles_q;
    let r = wg / ntiles_q;
    let h = r % p.n_heads;
    let b = r / p.n_heads;
    if (b >= p.bsz) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;
    let row = lt / lanes;
    let lane = lt % lanes;
    let i = qt * BR + row;
    let live = i < T;

    var qv: array<f32, 32>;
    var dov: array<f32, 32>;
    var acc: array<f32, 32>;
    let q_base = (b * T + i) * q_row + h * hd;
    var dpart = 0.0;
    for (var c = 0u; c < CH; c = c + 1u) {
        let d = c * lanes + lane;
        if (live && d < hd) {
            qv[c] = q[q_base + d];
            let g = d_o[q_base + d];
            dov[c] = g;
            dpart = dpart + g * o[q_base + d];
        } else {
            qv[c] = 0.0;
            dov[c] = 0.0;
        }
        acc[c] = 0.0;
    }
    // D_i: re-sum the lane partials through shared memory.
    part_s[row * lanes + lane] = dpart;
    workgroupBarrier();
    let po = row * lanes;
    var dsum_i = 0.0;
    for (var l = 0u; l < lanes; l = l + 1u) { dsum_i = dsum_i + part_s[po + l]; }
    var lse_i = 0.0;
    let stat = (b * p.n_heads + h) * T + i;
    if (live) {
        lse_i = lse[stat];
        if (lane == 0u) { dsum[stat] = dsum_i; }
    }
    workgroupBarrier(); // part_s is reused by the key loop

    let max_i_in_wg = min(qt * BR + BR - 1u, T - 1u);
    let ntiles_k = (max_i_in_wg / BC) + 1u;

    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        for (var e = lt; e < BC * hdt; e = e + WG) {
            let jr = e / hdt;
            let d = e % hdt;
            let j = kt * BC + jr;
            if (j < T && d < hd) {
                let kv_base = (b * T + j) * kv_row + hkv * hd + d;
                ksh[e] = k[kv_base];
                vsh[e] = v[kv_base];
            } else {
                ksh[e] = 0.0;
                vsh[e] = 0.0;
            }
        }
        workgroupBarrier();

        for (var j = 0u; j < BC; j = j + 1u) {
            var s = 0.0;
            var dp = 0.0;
            let ko = j * hdt + lane;
            for (var c = 0u; c < CH; c = c + 1u) {
                s = s + qv[c] * ksh[ko + c * lanes];
                dp = dp + dov[c] * vsh[ko + c * lanes];
            }
            let slot = j * WG + row * lanes + lane;
            part_s[slot] = s;
            part_p[slot] = dp;
        }
        workgroupBarrier();

        for (var j = 0u; j < BC; j = j + 1u) {
            let key = kt * BC + j;
            if (live && key < T && key <= i) {
                let so = j * WG + row * lanes;
                var s_sum = 0.0;
                var dp = 0.0;
                for (var l = 0u; l < lanes; l = l + 1u) {
                    s_sum = s_sum + part_s[so + l];
                    dp = dp + part_p[so + l];
                }
                let ds = exp(s_sum * scale - lse_i) * (dp - dsum_i);
                let ko = j * hdt + lane;
                for (var c = 0u; c < CH; c = c + 1u) {
                    acc[c] = acc[c] + ds * ksh[ko + c * lanes];
                }
            }
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    if (live) {
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * lanes + lane;
            if (d < hd) { dq[q_base + d] = acc[c] * scale; }
        }
    }
}
