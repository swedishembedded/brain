// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Flash attention (bidirectional self-attention), TILED with shared-memory K/V reuse + online softmax — Pascal-friendly (sm_61
// @how   64-thread workgroup tile, 2 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
//
// Flash attention (bidirectional self-attention), TILED with shared-memory K/V
// reuse + online softmax — Pascal-friendly (sm_61: no subgroups, no f16, single
// bind group, only shared memory + workgroupBarrier). scores -> softmax -> apply
// are FUSED, so the [B,H,T,T] scores/probs matrices are NEVER materialised: peak
// attention memory is O(T*head_dim), which lets high-resolution latents run past
// the per-buffer binding limit and cuts training activations.
//
// One workgroup owns BR=64 query rows of one (b,h). It streams K/V in BC-row tiles
// THROUGH SHARED MEMORY, so all 64 queries reuse each loaded tile instead of each
// query re-reading all of K/V from global.
//
// Occupancy is the bottleneck on a P40, so this keeps shared memory SMALL: q and
// the output accumulator o[] live in REGISTERS (q loaded once, reused from
// registers across every tile — minimal q traffic), and shared holds ONLY the
// K/V tiles (BC*HD*2*4 = 16 KiB). At 16 KiB a P40 SM runs ~3 workgroups (≈6 warps)
// instead of 1 — 3× the warps to hide global-load latency. head_dim must be <=128
// (Z-Image = 128; q[]+o[] = 256 regs, one small spill on Pascal). Output layout
// matches attn_apply_bidir, so this drops in for the scores/softmax/apply trio.

const BR: u32 = 64u;   // query rows per workgroup (== workgroup size)
const BC: u32 = 16u;   // key/value rows per shared tile
const HD: u32 = 128u;  // max head_dim

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

var<workgroup> ksh: array<f32, 2048>;  // BC*HD = 16*128 -> 8 KiB
var<workgroup> vsh: array<f32, 2048>;  //                    8 KiB  (16 KiB total)

@compute @workgroup_size(64)
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

    let lt = lid.x;              // 0..63
    let i = qt * BR + lt;        // this thread's query row
    let live = i < T;

    // q_i and the output accumulator o[] in registers.
    var q: array<f32, 128>;
    var o: array<f32, 128>;
    if (live) {
        let q_base = (b * T + i) * p.qkv_stride + p.q_off + h * hd;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            q[d] = qkv[q_base + d];
            o[d] = 0.0;
        }
    }

    var m = -3.4e38;           // running max
    var l = 0.0;               // running sum of exp

    let ntiles_k = (T + BC - 1u) / BC;
    let tile_elems = BC * hd;
    for (var kt: u32 = 0u; kt < ntiles_k; kt = kt + 1u) {
        // Cooperatively stage the K,V tile [BC rows x hd] into shared. 64 threads
        // load BC*hd (contiguous) elements, strided by 64.
        for (var e: u32 = lt; e < tile_elems; e = e + 64u) {
            let row = e / hd;
            let d = e % hd;
            let j = kt * BC + row;
            if (j < T) {
                let base = (b * T + j) * p.qkv_stride + h * hd + d;
                ksh[e] = qkv[base + p.k_off];
                vsh[e] = qkv[base + p.v_off];
            } else {
                ksh[e] = 0.0;
                vsh[e] = 0.0;
            }
        }
        workgroupBarrier();

        // Each query streams this tile's keys through the online softmax.
        if (live) {
            var krows = BC;
            let rem = T - kt * BC;
            if (rem < BC) { krows = rem; }
            for (var jj: u32 = 0u; jj < krows; jj = jj + 1u) {
                let ko = jj * hd;
                var s = 0.0;
                for (var d: u32 = 0u; d < hd; d = d + 1u) {
                    s = s + q[d] * ksh[ko + d];
                }
                s = s * scale;
                let m_new = max(m, s);
                let corr = exp(m - m_new);
                let pj = exp(s - m_new);
                l = l * corr + pj;
                for (var d: u32 = 0u; d < hd; d = d + 1u) {
                    o[d] = o[d] * corr + pj * vsh[ko + d];
                }
                m = m_new;
            }
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    if (live) {
        let inv = 1.0 / l;
        let o_base = (b * T + i) * p.d_model + h * hd;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            out[o_base + d] = o[d] * inv;
        }
    }
}
