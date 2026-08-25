// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Flash attention (bidirectional self-attention), LANE-SPLIT across head_dim with VECTOR-TILED shared reads
// @how   256-thread workgroup tile, vec4 shared tiles, 3 barriers
// @opt   5
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Flash attention (bidirectional self-attention), LANE-SPLIT across head_dim
// with VECTOR-TILED shared reads. Same math as `flash_attn_bidir_split`, same
// Params, same output layout, same BR=64 query rows per workgroup and same
// @workgroup_size(256): a drop-in wherever that kernel runs, with the SAME
// dispatch geometry.
//
// WHY IT EXISTS. `flash_attn_bidir_split` already fixed the register-spill bug
// that pinned `flash_attn_bidir` to local-memory bandwidth, and its inner loops
// read only shared memory. But both of those loops issue exactly ONE shared
// load per fused multiply-add:
//
//     s   = s   + q[c] * ksh[ko + c*LANES]        // 1 LDS : 1 FFMA
//     acc = acc + pj[j] * vsh[j*HD + vo]          // 1 LDS : 1 FFMA
//
// A Pascal SM issues an FFMA warp-instruction every clock but retires a shared
// load-store warp-instruction only every fourth (32 LD/ST units against 128
// fp32 lanes), so a 1:1 mix caps the kernel at a QUARTER of the card's fp32
// rate no matter how well the loads are laid out. That is a structural
// instruction-mix ceiling, not a bandwidth or bank-conflict problem: the tile
// reads are already broadcast across the 8 query rows of a warp and hit 16
// distinct banks, so the shared memory itself is nowhere near saturated.
//
// The measured evidence for the ceiling: at Wan's shape (T=14040, 12 heads,
// head_dim 128) `flash_attn_bidir_split` runs at a fifth of the card's own
// MEASURED fp32 roof and does not move when the load is made sustained, while
// `matmul_reg3` - the same 256-thread workgroup, but a 1:4 mix from its 8x8
// register block - reaches twice that share on the same card. This kernel
// closes part of that gap at the same shape, and `flash_attn_bidir_reg2`,
// which adds a second query row per thread on top, closes most of the rest.
//
// THE FIX. Make each shared load feed FOUR multiply-adds instead of one, by
// holding the K/V tiles as `vec4<f32>` and giving each lane whole 4-channel
// groups. Registers are UNCHANGED (`q4[8]` + `o4[8]` vec4 = the same 64 f32
// the scalar kernel already held), so occupancy is unchanged and nothing new
// can spill; only the instruction mix moves, from 1:1 to 1:4:
//
//     acc = acc + q4[c]  * ksh[ko + c*LANES]      // 1 LDS : 4 FFMA
//     acc = acc + pj[j]  * vsh[j*HDV + vo]        // 1 LDS : 4 FFMA
//
// LANE OWNERSHIP AND BANKS. A lane owns the INTERLEAVED vec4 slots
// {lane, lane+4, lane+8, …}, i.e. channels {4·lane…4·lane+3, 16+4·lane…}, so
// for a fixed (key, slot) the 4 lanes of a query row read 4 CONSECUTIVE vec4s
// = 16 consecutive words = 16 distinct banks, broadcast across the 8 query
// rows that share a warp. Conflict-free by construction, exactly as the scalar
// kernel's scalar interleave was. `part` keeps that kernel's
// [j][row][lane] layout, which spans all 32 banks once per warp.
//
// The horizontal sum that closes each dot product (`acc.x+acc.y+acc.z+acc.w`)
// reassociates the 128-term sum relative to the scalar kernel, so results
// agree to f32 rounding rather than bit-exactly - the same relationship
// `flash_attn_bidir_split` already has with `flash_attn_bidir`.
//
// TILES ARE ALWAYS HD=128 WIDE, zero-filled past `head_dim`, so the channel
// loop's trip count stays a compile-time constant - the property that keeps
// `q4`/`o4` in registers, and the reason a runtime `head_dim` must never reach
// these loops. `head_dim` need not be a multiple of 4: the staging and the
// epilogue guard per element, the loops do not.
//
// @workgroup_size(256) = BR(64 query rows) x LANES(4), checked against
// `DeviceCaps::max_workgroup_size` (queried, never assumed). Shared use is
// 16 KiB (ksh 4 + vsh 4 + part 8), identical to `flash_attn_bidir_split`, so
// the same 3 workgroups stay resident per SM. Pascal-friendly: no subgroups,
// no atomics, no f16, one bind group, 2 storage buffers.

const BR: u32 = 64u;     // query rows per workgroup
const BC: u32 = 8u;      // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const V4: u32 = 8u;      // vec4 slots per lane (V4*4*LANES == HD)
const HDV: u32 = 32u;    // vec4 slots per key row (HD/4)
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

var<workgroup> ksh:  array<vec4<f32>, 256>;  // BC*HDV -> 4 KiB
var<workgroup> vsh:  array<vec4<f32>, 256>;  // BC*HDV -> 4 KiB
var<workgroup> part: array<f32, 2048>;       // BC*BR*LANES -> 8 KiB

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
    let lane = lt % LANES;          // 0..3   -> vec4-slot phase
    let i = qt * BR + row;          // this thread's query row
    let live = i < T;

    // This thread's slice of q, and its slice of the output accumulator.
    // Slot c holds channels 4*(c*LANES + lane) .. +3.
    var q4: array<vec4<f32>, 8>;
    var o4: array<vec4<f32>, 8>;
    let q_base = (b * T + i) * p.qkv_stride + p.q_off + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (live) {
            if (d0 + 0u < hd) { v.x = qkv[q_base + d0 + 0u]; }
            if (d0 + 1u < hd) { v.y = qkv[q_base + d0 + 1u]; }
            if (d0 + 2u < hd) { v.z = qkv[q_base + d0 + 2u]; }
            if (d0 + 3u < hd) { v.w = qkv[q_base + d0 + 3u]; }
        }
        q4[c] = v;
        o4[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var m = -3.4e38;   // running max
    var l = 0.0;       // running sum of exp

    var pj: array<f32, 8>;   // BC softmax weights for the current tile

    let ntiles_k = (T + BC - 1u) / BC;
    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        // Stage the K,V tile [BC x HDV vec4] into shared. 256 threads, one
        // vec4 each at BC=8, HDV-contiguous so a warp reads 512 consecutive
        // bytes. Channels past head_dim are zeroed, which is what makes the
        // fixed V4 loop correct.
        for (var e = lt; e < BC * HDV; e = e + 256u) {
            let jr = e / HDV;
            let d0 = (e % HDV) * 4u;
            let j = kt * BC + jr;
            var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (j < T) {
                let base = (b * T + j) * p.qkv_stride + h * hd + d0;
                let bk = base + p.k_off;
                let bv = base + p.v_off;
                if (d0 + 3u < hd) {
                    kv = vec4<f32>(qkv[bk], qkv[bk + 1u], qkv[bk + 2u], qkv[bk + 3u]);
                    vv = vec4<f32>(qkv[bv], qkv[bv + 1u], qkv[bv + 2u], qkv[bv + 3u]);
                } else {
                    if (d0 + 0u < hd) { kv.x = qkv[bk + 0u]; vv.x = qkv[bv + 0u]; }
                    if (d0 + 1u < hd) { kv.y = qkv[bk + 1u]; vv.y = qkv[bv + 1u]; }
                    if (d0 + 2u < hd) { kv.z = qkv[bk + 2u]; vv.z = qkv[bv + 2u]; }
                }
            }
            ksh[e] = kv;
            vsh[e] = vv;
        }
        workgroupBarrier();

        // Partial dot products: this lane's 32 channels of every key in the
        // tile, four channels per shared load.
        for (var j = 0u; j < BC; j = j + 1u) {
            var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            let ko = j * HDV + lane;
            for (var c = 0u; c < V4; c = c + 1u) {
                acc = acc + q4[c] * ksh[ko + c * LANES];
            }
            part[j * (BR * LANES) + row * LANES + lane] = acc.x + acc.y + acc.z + acc.w;
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

        // One rescale of o per TILE, not per key; four channels per load.
        for (var c = 0u; c < V4; c = c + 1u) {
            let vo = c * LANES + lane;
            var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            for (var j = 0u; j < BC; j = j + 1u) {
                acc = acc + pj[j] * vsh[j * HDV + vo];
            }
            o4[c] = o4[c] * corr + acc;
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    if (live) {
        let inv = 1.0 / l;
        let o_base = (b * T + i) * p.d_model + h * hd;
        for (var c = 0u; c < V4; c = c + 1u) {
            let d0 = (c * LANES + lane) * 4u;
            let v = o4[c] * inv;
            if (d0 + 0u < hd) { out[o_base + d0 + 0u] = v.x; }
            if (d0 + 1u < hd) { out[o_base + d0 + 1u] = v.y; }
            if (d0 + 2u < hd) { out[o_base + d0 + 2u] = v.z; }
            if (d0 + 3u < hd) { out[o_base + d0 + 3u] = v.w; }
        }
    }
}
