// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal GQA flash attention (decoder self-attention), LANE-SPLIT across head_dim
// @how   256-thread workgroup tile, 3 barriers, causal early-exit over K tiles
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Causal grouped-query attention, LANE-SPLIT across head_dim -- the causal/
// GQA/separate-q-k-v-buffer sibling of `flash_attn_bidir_split.wgsl`. Same
// online-softmax fusion (`gqa_scores.wgsl`/`attn_softmax.wgsl`/
// `gqa_apply.wgsl`'s scores -> softmax -> apply chain, but never materializing
// the dense `[H,T,T]` slab), same lane-split fix for the SAME real bug
// `flash_attn_bidir_split.wgsl`'s header documents and MEASURES (worth more
// than an order of magnitude at head_dim=128): a naive one-thread-per-query-row kernel with
// `q[128]`/`o[128]` in `var<function>` arrays cannot keep those in Pascal's
// 255-register budget, so they spill to local (global-memory-backed) memory
// and the whole kernel runs at ~6 bytes/FLOP local-memory bandwidth instead of
// compute. A first cut of THIS kernel repeated that exact mistake (measured,
// for one Thinker decoder layer's attention at t=7484, head_dim=128, at
// hundreds of times the compute-bound estimate; moving the same oversized
// arrays into workgroup shared memory instead of splitting the work made it
// WORSE still, by a further factor of two
// -- shared memory is faster than local memory but still far slower than a
// register for an access this hot, and halving the tile size to make room
// only added barrier overhead). `flash_attn_bidir_split` had already solved
// this for the non-causal case; this kernel is that same fix.
//
// THE FIX (identical strategy to `flash_attn_bidir_split`, see that file for
// the full derivation): split head_dim across LANES=4 threads so each thread
// owns CH=32 channels -- `q[32]` + `o[32]` = 64 values, a compile-time trip
// count, hence real registers. The per-key dot product is a partial per lane,
// summed through a small shared buffer once per key tile; the causal mask is
// applied at that same re-sum step (see below), not duplicated per lane.
//
// UNLIKE flash_attn_bidir_split, this kernel:
//  - takes SEPARATE q/k/v buffers (`gqa_scores.wgsl`/`gqa_apply.wgsl`'s own
//    layout: q is `[B*T, n_heads*head_dim]`, k/v are
//    `[B*T, n_kv_heads*head_dim]`), so it drops into `gqa_attn_sublayer_fwd`
//    without upstream layout changes;
//  - is grouped-query aware (`hkv = h / group` maps each query head to its
//    shared key/value head, staged into `ksh`/`vsh` once per tile and reused
//    by every query head in that group -- the same K/V tile a plain-MHA
//    kernel would stage anyway, just addressed through `hkv` instead of `h`);
//  - is CAUSAL: K tiles entirely beyond the workgroup's largest query row are
//    skipped ENTIRELY (`ntiles_k` depends on `qt`, unlike bidir's fixed
//    `ceil(T/BC)`), and a tile that straddles the causal boundary masks
//    individual keys to `-inf` in the re-sum step below (before the tile-wide
//    max/exp), so lower-`i` threads in the same workgroup as a higher-`i` one
//    still see the correct (smaller) causal window.
//
// This is deliberately the Pascal/no-subgroup TIER of attention, not the only
// one brain will ever need: newer hardware (subgroup shuffles, tensor cores)
// can do the cross-lane reduction with a shuffle instead of a shared-memory
// round trip, and NPU/CPU backends want their own native paths entirely. The
// selection point is the same one `flash_bidir_variant` already uses for
// `flash_attn_bidir`/`_split` -- `GqaAttnIds::flash_causal_gqa` in
// `crates/model/src/block.rs` picks a kernel index by `DeviceCaps`, so a
// faster variant for capable hardware is an additional registered kernel + a
// widened selector, not a rewrite of this one.
//
// @workgroup_size(256) = BR(64 query rows) x LANES(4), matching
// `flash_attn_bidir_split`'s own choice (already checked against
// `DeviceCaps::max_workgroup_size` by callers, never assumed). Shared use is
// 16 KiB (ksh 4 + vsh 4 + part 8) -- 3 workgroups resident per SM on Pascal.

const BR: u32 = 64u;     // query rows per workgroup
const BC: u32 = 8u;      // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const CH: u32 = 32u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,        // T
    head_dim: u32,     // <= 128
    group: u32,        // n_heads / n_kv_heads
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       k:   array<f32>;
@group(0) @binding(3) var<storage, read>       v:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

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

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;                 // 0..255
    let row = lt / LANES;           // 0..63  -> query row within the tile
    let lane = lt % LANES;          // 0..3   -> channel phase
    let i = qt * BR + row;          // this thread's query row
    let live = i < T;

    // This thread's slice of q, and its slice of the output accumulator.
    // Channels are interleaved: slot c holds channel c*LANES + lane.
    var qv: array<f32, 32>;
    var o: array<f32, 32>;
    let q_base = (b * T + i) * q_row + h * hd;
    for (var c = 0u; c < CH; c = c + 1u) {
        let d = c * LANES + lane;
        if (live && d < hd) { qv[c] = q[q_base + d]; } else { qv[c] = 0.0; }
        o[c] = 0.0;
    }

    var m = -3.4e38;   // running max
    var l = 0.0;       // running sum of exp

    var pj: array<f32, 8>;   // BC softmax weights for the current tile

    // Causal early-exit: no query in this workgroup ever attends past its OWN
    // row, and every live row is <= the workgroup's largest one
    // (`qt*BR + BR - 1`, clipped to T-1) -- so K tiles beyond that row need
    // never be loaded, let alone scored.
    let max_i_in_wg = min(qt * BR + BR - 1u, T - 1u);
    let ntiles_k = (max_i_in_wg / BC) + 1u;

    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        // Stage the K,V tile [BC x HD] into shared, through the SHARED kv
        // head `hkv` -- every query head in this group reads the same tile.
        // 256 threads, HD-contiguous so consecutive threads read consecutive
        // addresses. Channels past head_dim are zeroed, which is what makes
        // the fixed CH loop correct.
        for (var e = lt; e < BC * HD; e = e + 256u) {
            let jr = e / HD;
            let d = e % HD;
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

        // Partial dot products: this lane's 32 channels of every key in the tile.
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = 0.0;
            let ko = j * HD + lane;
            for (var c = 0u; c < CH; c = c + 1u) {
                s = s + qv[c] * ksh[ko + c * LANES];
            }
            part[j * (BR * LANES) + row * LANES + lane] = s;
        }
        workgroupBarrier();

        // Every lane re-sums the LANES partials (redundant but register-cheap,
        // and it leaves each lane with the p_j it needs for its own
        // channels). The causal mask lands here too: a key strictly past this
        // thread's own row `i` is excluded before the tile-wide max/exp, same
        // rule `gqa_scores.wgsl` uses (`j > i -> -inf`).
        var krows = BC;
        let rem = T - kt * BC;
        if (rem < BC) { krows = rem; }
        var tmax = -3.4e38;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = -3.4e38;
            if (j < krows && (kt * BC + j) <= i) {
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
        let o_base = (b * T + i) * q_row + h * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { out[o_base + d] = o[c] * inv; }
        }
    }
}
