// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused causal chunked paged-attention prefill: BR query rows/workgroup, online softmax, no [nh,N,N] slab
// @how   256-thread workgroup per (head, query-tile), LANE-SPLIT across head_dim, causal early-exit over K tiles, 3 barriers/tile
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Fused PREFILL-shaped paged attention: `flash_attn_causal_gqa.wgsl`'s
// BR-query-rows-per-workgroup causal tiling (many independent query rows,
// tiled over keys, lane-split head_dim to keep `qv`/`o` in registers - see
// that kernel's header for the full register-spill derivation this one
// inherits unchanged), ported from a DENSE `[B*T, ...]` K/V buffer onto
// `paged_flash_decode.wgsl`'s block-table-indexed paged pool. Same reason
// `paged_flash_decode` gives decode: no materialised `scores`/`probs` buffer
// at all, so nothing here scales with context capacity OR chunk length -
// unlike the three-stage triad this replaces (`paged_decode_scores{,_wg}` ->
// `decode_softmax_batched` -> `paged_decode_apply_batched`), which, called
// once per PREFILL CHUNK ROW the way `qwen3::serve::run_batched_steps`
// already does (`bsz` = chunk length, not sequence count), sizes its
// `scores`/`probs` scratch `[chunk_len, n_heads, cap]` - a genuine `[nh,N,N]`
// slab once `cap` grows to cover the same chunk (`Engine::from_map_with_gpu`'s
// own scratch-sizing comment names this shape explicitly).
//
//   q            : [bsz, n_heads*head_dim]           (bsz = chunk length, ONE sequence)
//   pool_k/pool_v: [num_blocks*block_size, n_kv_heads*head_dim]  (paged pool)
//   block_tables : [bsz, max_bt]  (physical block index per logical block)
//   seq_lens     : [bsz]          (live key count per query row)
//   ctx          : [bsz, n_heads*head_dim]
//
// SHARES THE TAPE: identical `Params` layout and buffer shapes to
// `paged_flash_decode`/the triad it replaces (`sc.q`, `sc.bt_buf`,
// `sc.seqlen_buf`, `sc.ctx`) - `qwen3::serve::prefill` already builds exactly
// this tape today (`seqlens[i] = start+i+1`, `bt` duplicated identically
// across every row of the chunk since a chunk is always ONE sequence, see
// that function's own comment), so wiring this kernel in behind the M1.1
// selector (M2.4) needs no host-side buffer changes - only the dispatch
// picks a different kernel.
//
// NOT THE SAME SHAPE AS DECODE: decode has exactly one query per (sequence,
// head) workgroup (1 of 256 threads would score if this kernel's BR-tiling
// were reused there - `paged_flash_decode`'s own header names this exact
// reason for its different, `LANES=8`/one-query tiling). Prefill has UP TO
// `bsz` live query rows sharing one paged sequence, so - like
// `flash_attn_causal_gqa` - a workgroup here owns `BR=64` query rows at once,
// re-using ONE staged K/V tile across all of them.
//
// TWO CONTRACTS THIS KERNEL RELIES ON, both already true of every row
// `qwen3::serve::prefill` builds for a chunk, and both DERIVE FROM "one
// prefill dispatch is one sequence's chunk" (checked against that function's
// source, not assumed - `run_batched(cc, ...)` is called once per prompt,
// never interleaving two sequences' rows in one call):
//   1. every query row in a workgroup's BR-tile shares the SAME physical
//      block table, so the K/V tile staged for that (head, query-tile) is
//      valid for every row in it. The kernel reads block_tables through the
//      tile's FIRST row (`qt*BR`), not `i`.
//   2. `seq_lens` is non-decreasing across a workgroup's BR-tile (causal:
//      row i's own boundary only ever grows with i), so the workgroup's
//      largest live-key count - and therefore how many K tiles it must visit
//      at all - is exactly the LAST row's own `seq_lens` value. The same
//      assumption `flash_attn_causal_gqa` already makes implicitly (row i's
//      causal boundary IS i there); this kernel makes it explicit because
//      `seq_lens` is data, not the row index itself.
//
// Per-row masking (unlike `flash_attn_causal_gqa`'s `j <= i`) compares
// against that row's OWN `seq_lens[i]`, exactly `paged_flash_decode`'s
// `j < t` convention generalised from "one t per workgroup" to "one t per
// row" - the mechanism that makes a chunk's OWN tokens (already scattered
// into the pool by `KV_APPEND` earlier in the SAME layer tape, before this
// dispatch) visible to later rows in the same chunk without any separate
// intra-chunk path.
//
// NOT bit-identical to the triad it replaces - same precedent
// `paged_flash_decode`'s own header already cites (`rmsnorm_rows`: "64
// partial sums fold in a different order, agreeing to ~3e-6"): the triad
// computes one exact max over the whole row before a single un-rescaled
// exp/sum pass; this kernel rescales its running sum/accumulator once per
// `BC=8`-key tile (textbook online softmax). Gated at `1e-3` absolute error,
// matching `paged_flash_decode_matches_batched_triad`'s own bound.
//
// Shared memory: ksh/vsh `BC*HD` = 4 KiB each, part `BC*BR*LANES` = 8 KiB ->
// 16 KiB total, IDENTICAL to `flash_attn_causal_gqa`'s own budget (sits
// exactly at WebGPU's guaranteed `maxComputeWorkgroupStorageSize` floor,
// already proven safe there).

const BR: u32 = 64u;     // query rows per workgroup
const BC: u32 = 8u;      // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const CH: u32 = 32u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

struct Params {
    bsz: u32,          // chunk length (query rows), ONE sequence
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 128
    group: u32,        // n_heads / n_kv_heads
    block_size: u32,
    max_bt: u32,       // block_tables row stride (blocks per sequence)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<f32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<f32>;
@group(0) @binding(4) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(5) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(6) var<storage, read_write> ctx:          array<f32>;

var<workgroup> ksh:  array<f32, 1024>;  // BC*HD  -> 4 KiB
var<workgroup> vsh:  array<f32, 1024>;  // BC*HD  -> 4 KiB
var<workgroup> part: array<f32, 2048>;  // BC*BR*LANES -> 8 KiB

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (h, query-tile) - no separate sequence axis: a
    // prefill dispatch's `bsz` rows all belong to the ONE sequence being
    // chunked (see the header's contract 1).
    let wg = wgid.y * nwg.x + wgid.x;
    let ntiles_q = (p.bsz + BR - 1u) / BR;
    let qt = wg % ntiles_q;
    let h = wg / ntiles_q;
    if (h >= p.n_heads) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;                 // 0..255
    let row = lt / LANES;           // 0..63  -> query row within the tile
    let lane = lt % LANES;          // 0..3   -> channel phase
    let i = qt * BR + row;          // this thread's absolute query row
    let live = i < p.bsz;

    // This thread's slice of q, and its slice of the output accumulator.
    // Channels are interleaved: slot c holds channel c*LANES + lane.
    var qv: array<f32, 32>;
    var o: array<f32, 32>;
    let q_base = i * q_row + h * hd;
    for (var c = 0u; c < CH; c = c + 1u) {
        let d = c * LANES + lane;
        if (live && d < hd) { qv[c] = q[q_base + d]; } else { qv[c] = 0.0; }
        o[c] = 0.0;
    }

    var m = -3.4e38;   // running max
    var l = 0.0;       // running sum of exp

    var pj: array<f32, 8>;   // BC softmax weights for the current tile

    // This row's own causal boundary (live key count) - `seq_lens[i]`, the
    // SAME per-row field `paged_flash_decode` reads per WORKGROUP; here it
    // varies per query row within the tile.
    let t_i = select(0u, seq_lens[i], live);

    // Causal early-exit (header contract 2): the workgroup's largest live
    // boundary is its LAST row's own `seq_lens`, since a chunk's boundaries
    // only grow with row index - so K tiles beyond it need never be staged.
    let max_i_in_wg = min(qt * BR + BR - 1u, p.bsz - 1u);
    let max_t = seq_lens[max_i_in_wg];
    let ntiles_k = (max_t + BC - 1u) / BC;

    // Every row in this tile shares one physical block table (header
    // contract 1) - read through the tile's first row.
    let bt_row = qt * BR;

    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        // Stage the K,V tile [BC x HD] into shared, through the SHARED kv
        // head `hkv` and the tile's shared block table. Coalesced: for a
        // fixed key-in-tile `jr`, consecutive threads read consecutive `d`
        // regardless of which physical block `j` resolves to (a block's own
        // `block_size` rows are contiguous in the pool, same layout
        // `paged_flash_decode`'s own staging loop addresses).
        for (var e = lt; e < BC * HD; e = e + 256u) {
            let jr = e / HD;
            let d = e % HD;
            let j = kt * BC + jr;
            if (j < max_t && d < hd) {
                let physical = block_tables[bt_row * p.max_bt + j / p.block_size];
                let slot = (physical * p.block_size + (j % p.block_size)) * kv_row + hkv * hd + d;
                ksh[e] = pool_k[slot];
                vsh[e] = pool_v[slot];
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
        // channels). The causal mask lands here, against THIS ROW's own
        // `t_i` (unlike `flash_attn_causal_gqa`'s shared `i` boundary): a key
        // at or past `t_i` is excluded before the tile-wide max/exp.
        var krows = BC;
        let rem = max_t - kt * BC;
        if (rem < BC) { krows = rem; }
        var tmax = -3.4e38;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = -3.4e38;
            if (j < krows && (kt * BC + j) < t_i) {
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
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = i * q_row + h * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { ctx[o_base + d] = o[c] * inv; }
        }
    }
}
