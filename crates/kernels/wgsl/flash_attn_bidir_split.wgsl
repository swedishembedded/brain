// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Flash attention (bidirectional self-attention), LANE-SPLIT across head_dim
// @how   256-thread workgroup tile, 3 barriers
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Flash attention (bidirectional self-attention), LANE-SPLIT across head_dim.
// Same math as flash_attn_bidir.wgsl (online softmax, K/V streamed through
// shared memory, scores never materialised), same Params, same output layout:
// a drop-in replacement wherever `flash_attn_bidir` runs, and MEASURABLY
// faster at every head_dim swept (128 / 96 / 64 / 32 at T=1536,
// d_model=3072). The margin is widest at head_dim 128 and narrows as head_dim
// falls, which is exactly the register-pressure story below: the wider the
// head, the more of it spills in the kernel this one replaces. A/B it with
// `crates/lfm2/src/bin/lfm_attn_ab.rs` rather than trusting a figure here.
//
// (agreement with flash_attn_bidir: cosine 1.00000000, max_abs 1.3e-6.)
//
// WHY IT EXISTS. flash_attn_bidir gives EVERY thread one query row and keeps
// that row's `q[128]` and output accumulator `o[128]` in `var<function>`
// arrays. 256 f32 per thread cannot live in registers on Pascal (255 max) and
// the loops index them by a RUNTIME `head_dim`, so they cannot be unrolled
// either — both arrays land in LOCAL memory, which is global-memory backed.
// The inner loop then does ~3 local-memory accesses (q[d], o[d] read, o[d]
// write) per 2 FLOP = 6 bytes/FLOP, pinning the kernel to memory bandwidth:
// measured at well under one percent of a Pascal card's fp32 peak at head_dim
// 128, T 1536, 24 heads - which on FLUX.2 klein-4B made this one kernel the
// bulk of the model's whole DiT forward.
//
// THE FIX. Split the head_dim across LANES=4 threads, so each thread owns only
// CH=32 channels: `q[32]` + `o[32]` = 64 values, indexed by a COMPILE-TIME
// constant trip count, hence real registers. The per-key dot product is then a
// partial per lane, summed through a small shared buffer once per key tile.
// Everything else stays in shared memory or registers, so the inner loops read
// only shared:
//
//   stage K/V tile [BC x HD] -> shared            (coalesced global reads)
//   per key j: partial dot over own 32 channels -> part[j][row][lane]
//   barrier; each lane re-sums LANES partials    (redundant, register-cheap)
//   tile-wide online-softmax update (m, l, corr)
//   per own channel c: o[c] = o[c]*corr + SUM_j p_j * vsh[j][c]
//
// The o rescale moved OUT of the per-key loop to once per TILE (the p_j are
// held in registers), which also cuts its FLOP by BC.
//
// LAYOUTS ARE BANK-CONFLICT FREE by construction. A lane owns the INTERLEAVED
// channels {lane, lane+4, lane+8, …} rather than a contiguous block, so the 4
// lanes of a query row touch 4 CONSECUTIVE shared words (4 consecutive banks,
// broadcast across the 8 rows in a warp). `part` is indexed
// [j][row][lane] = j*BR*LANES + row*LANES + lane, so a warp (8 rows x 4 lanes)
// spans banks 0,1,2,…,31 exactly once.
//
// TILES ARE ALWAYS HD=128 WIDE, zero-filled past `head_dim`. That keeps the
// channel loop's trip count a compile-time constant — the whole point, since a
// runtime trip count is exactly what forces the arrays out of registers. The
// cost is doing 128/head_dim of the work for narrower heads, which is why the
// speedup above shrinks with head_dim; it never inverts, because the baseline
// is paying local-memory latency regardless. flash_attn_bidir remains the
// fallback for devices whose `max_workgroup_size` is below 256.
//
// @workgroup_size(256) = BR(64 query rows) x LANES(4). 256 is the workgroup
// size the register-tiled matmuls already use and must be checked against
// `DeviceCaps::max_workgroup_size` (queried, never assumed). Shared use is
// 16 KiB (ksh 4 + vsh 4 + part 8), which leaves 3 workgroups resident per SM.
// Pascal-friendly: no subgroups, no atomics, no f16, one bind group, 2 storage
// buffers.

const BR: u32 = 64u;     // query rows per workgroup
const BC: u32 = 8u;      // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const CH: u32 = 32u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

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

var<workgroup> ksh:  array<f32, 1024>;  // BC*HD  -> 4 KiB
var<workgroup> vsh:  array<f32, 1024>;  // BC*HD  -> 4 KiB
var<workgroup> part: array<f32, 2048>;  // BC*BR*LANES -> 8 KiB

@compute @workgroup_size(256)
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

    let lt = lid.x;                 // 0..255
    let row = lt / LANES;           // 0..63  -> query row within the tile
    let lane = lt % LANES;          // 0..3   -> channel phase
    let i = qt * BR + row;          // this thread's query row
    let live = i < T;

    // This thread's slice of q, and its slice of the output accumulator.
    // Channels are interleaved: slot c holds channel c*LANES + lane.
    var q: array<f32, 32>;
    var o: array<f32, 32>;
    let q_base = (b * T + i) * p.qkv_stride + p.q_off + h * hd;
    for (var c = 0u; c < CH; c = c + 1u) {
        let d = c * LANES + lane;
        if (live && d < hd) { q[c] = qkv[q_base + d]; } else { q[c] = 0.0; }
        o[c] = 0.0;
    }

    var m = -3.4e38;   // running max
    var l = 0.0;       // running sum of exp

    var pj: array<f32, 8>;   // BC softmax weights for the current tile

    let ntiles_k = (T + BC - 1u) / BC;
    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        // Stage the K,V tile [BC x HD] into shared. 256 threads, HD-contiguous
        // so consecutive threads read consecutive addresses. Channels past
        // head_dim are zeroed, which is what makes the fixed CH loop correct.
        for (var e = lt; e < BC * HD; e = e + 256u) {
            let jr = e / HD;
            let d = e % HD;
            let j = kt * BC + jr;
            if (j < T && d < hd) {
                let base = (b * T + j) * p.qkv_stride + h * hd + d;
                ksh[e] = qkv[base + p.k_off];
                vsh[e] = qkv[base + p.v_off];
            } else {
                ksh[e] = 0.0;
                vsh[e] = 0.0;
            }
        }
        workgroupBarrier();

        // Partial dot products: this lane's 32 channels of every key in the tile.
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = 0.0;
            let ko = j * HD + lane;
            for (var c = 0u; c < CH; c = c + 1u) {
                s = s + q[c] * ksh[ko + c * LANES];
            }
            part[j * (BR * LANES) + row * LANES + lane] = s;
        }
        workgroupBarrier();

        // Every lane re-sums the LANES partials (redundant but register-cheap,
        // and it leaves each lane with the p_j it needs for its own channels).
        var krows = BC;
        let rem = T - kt * BC;
        if (rem < BC) { krows = rem; }
        var tmax = -3.4e38;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = -3.4e38;
            if (j < krows) {
                let po = j * (BR * LANES) + row * LANES;
                s = (part[po] + part[po + 1u] + part[po + 2u] + part[po + 3u]) * scale;
            }
            pj[j] = s;
            tmax = max(tmax, s);
        }
        let m_new = max(m, tmax);
        let corr = exp(m - m_new);
        var lsum = 0.0;
        for (var j = 0u; j < BC; j = j + 1u) {
            let e = exp(pj[j] - m_new);
            pj[j] = e;
            lsum = lsum + e;
        }
        l = l * corr + lsum;
        m = m_new;

        // One rescale of o per TILE, not per key.
        for (var c = 0u; c < CH; c = c + 1u) {
            let vo = c * LANES + lane;
            var acc = 0.0;
            for (var j = 0u; j < BC; j = j + 1u) {
                acc = acc + pj[j] * vsh[j * HD + vo];
            }
            o[c] = o[c] * corr + acc;
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    if (live) {
        let inv = 1.0 / l;
        let o_base = (b * T + i) * p.d_model + h * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { out[o_base + d] = o[c] * inv; }
        }
    }
}
