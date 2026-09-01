// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused paged-attention decode: block-table traversal, online softmax, context - one dispatch
// @how   64-thread workgroup per (sequence, head), tiled key staging, 5 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Fused decode-shaped paged attention: for each sequence `b`'s single query,
// walk its cached keys/values through its own block table and produce the
// attention context in ONE dispatch - no materialised `scores`/`probs`
// buffer, unlike the `paged_decode_scores{,_wg}` -> `decode_softmax_batched`
// -> `paged_decode_apply_batched` triad this replaces. The triad writes and
// re-reads a `[batch, n_heads, cap]` f32 slab twice per layer per token,
// sized by context CAPACITY rather than the live sequence length; this
// kernel keeps only the running online-softmax state (`m`, `l`) and the
// per-lane output accumulator in registers, so nothing scales with `cap` at
// all.
//
//   q            : [batch, n_heads*head_dim]
//   pool_k/pool_v: [num_blocks*block_size, n_kv_heads*head_dim]  (paged pool)
//   block_tables : [batch, max_bt]  (physical block index per logical block)
//   seq_lens     : [batch]          (live key count per sequence)
//   ctx          : [batch, n_heads*head_dim]
//
// THE ALGORITHM (`flash_attn_causal_gqa.wgsl`'s online-softmax fusion, ported
// from "many independent query rows, tiled over keys" to "one query per
// workgroup, keys parallelised WITHIN a tile" - decode has exactly one query
// per sequence, so tiling queries the way the prefill kernel does would use
// 1 of 256 threads):
//
// A workgroup owns one (b, h) pair. `BC = 8` keys are staged into shared
// memory per tile (same coalesced sweep as `flash_attn_causal_gqa`'s K/V
// staging - contiguous `d` across consecutive threads regardless of which
// physical block a key's `j` maps to, since a block's own `block_size` rows
// are contiguous in the pool). `LANES = 8` threads split each key's
// `head_dim` dot product - `paged_decode_scores_wg`'s own sweep at
// head_dim=128 measured the achieved bandwidth "unimodal, flat-topped across
// 8 and 4" (see that kernel's header), so 8 sits on the SAME measured
// optimum while halving `BC * HD` against the `LANES = 4` choice would
// otherwise need - so a tile uses exactly `BC * LANES = 64` threads: thread
// `lt` owns key-in-tile `row = lt / LANES` and channel group `lane = lt %
// LANES`.
//
// Per tile: stage K/V -> each thread's partial dot for its row/lane -> one
// thread per row (`lane == 0`) folds the `LANES` partials into that key's
// scaled score, masking a key past `seq_lens[b]` to `-inf` -> every thread
// redundantly re-derives the SAME tile max and exp-weights from the `BC`
// shared scores (a handful of shared-memory reads, far cheaper than another
// barrier round trip) -> the standard online-softmax rescale (`corr = exp(m
// - m_new)`) folds the previous running `(m, l, o)` state against this
// tile's contribution -> each thread accumulates its lane's channels of `o`
// by walking the (already-staged, already-shared) `BC` values of V, weighted
// by that tile's softmax numerators. Because every one of the `LANES`
// threads sharing a `lane` performs this identical, deterministic sequence
// from the same shared inputs, their private `(m, l, o)` registers stay
// bit-identical across the whole tile loop with no extra synchronisation -
// only thread `row == 0` needs to write the final normalised result.
//
// GPU-ONLY BY CONSTRUCTION (`@cpu no`), unlike the triad it replaces: FIVE
// top-level `workgroupBarrier()`s (one to stage the query, four per key
// tile: stage -> partial dot -> fold -> done-reading), not the CPU JIT's
// one-barrier-per-body limit (`softmax_rows.wgsl`'s own header names the
// same constraint for the same reason: a genuine multi-stage cooperative
// reduction cannot fit in one split). The existing three-stage path stays
// registered as the portable CPU/reference implementation; this is an
// additional GPU sibling, not a replacement of it.
//
// NOT bit-identical to the triad it replaces - checked against the same
// precedent `rmsnorm_rows`'s own header records ("64 partial sums fold in a
// different order, agreeing to ~3e-6"): the triad computes an exact row max
// over ALL `cap` keys before a single un-rescaled exp/sum pass, while this
// kernel rescales the running sum/accumulator once per 8-key tile (the
// textbook online-softmax reassociation). Same reason `flash_attn_causal_gqa`
// is checked against its materialized reference with a `1e-3` absolute-error
// gate, not `assert_eq`, in `crates/model/src/block.rs`'s own test.
//
// Shared memory: qsh 512 B + ksh/vsh 4 KiB each + part 256 B + sc 32 B ~=
// 8.8 KiB - comfortably under WebGPU's guaranteed 16 KiB
// `maxComputeWorkgroupStorageSize` floor (`flash_attn_causal_gqa`'s own
// ~16 KiB tile sits AT that floor; `BC = 16` here would have exceeded it).

const BC: u32 = 8u;      // keys staged and scored per tile
const LANES: u32 = 8u;   // threads cooperating on one key's head_dim reduction
const CH: u32 = 16u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

struct Params {
    batch: u32,
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

var<workgroup> qsh:  array<f32, 128>;  // HD
var<workgroup> ksh:  array<f32, 1024>; // BC*HD -> 4 KiB
var<workgroup> vsh:  array<f32, 1024>; // BC*HD -> 4 KiB
var<workgroup> part: array<f32, 64>;   // BC*LANES
var<workgroup> sc:   array<f32, 8>;    // BC

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (b, h) - one workgroup per sequence-head pair,
    // 2D-grid safe the same way every other batched kernel in this tree
    // reconstructs its index (`wg = wgid.y * nwg.x + wgid.x`).
    let wg = wgid.y * nwg.x + wgid.x;
    let h = wg % p.n_heads;
    let b = wg / p.n_heads;
    if (b >= p.batch) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;                  // 0..63
    let row = lt / LANES;            // 0..7 -> key-in-tile
    let lane = lt % LANES;           // 0..7 -> channel phase

    // Stage this workgroup's single query row into shared memory once -
    // every one of the 64 threads reads it every tile otherwise.
    let q_base = (b * q_row) + h * hd;
    for (var e = lt; e < HD; e = e + 64u) {
        qsh[e] = select(0.0, q[q_base + e], e < hd);
    }
    workgroupBarrier();

    var o: array<f32, 16>;
    for (var c = 0u; c < CH; c = c + 1u) { o[c] = 0.0; }
    var m = -3.4e38;
    var l = 0.0;

    let t = seq_lens[b];
    let ntiles = (t + BC - 1u) / BC;

    for (var kt = 0u; kt < ntiles; kt = kt + 1u) {
        // Stage the K,V tile [BC x HD] into shared, through the SHARED kv
        // head `hkv`. Coalesced: for a fixed key-in-tile `jr`, consecutive
        // threads read consecutive `d` regardless of which physical block
        // `j` resolves to (a block's `block_size` rows are contiguous in the
        // pool, same layout `paged_decode_scores_wg` addresses).
        for (var e = lt; e < BC * HD; e = e + 64u) {
            let jr = e / HD;
            let d = e % HD;
            let j = kt * BC + jr;
            if (j < t && d < hd) {
                let physical = block_tables[b * p.max_bt + j / p.block_size];
                let slot = (physical * p.block_size + (j % p.block_size)) * kv_row + hkv * hd + d;
                ksh[e] = pool_k[slot];
                vsh[e] = pool_v[slot];
            } else {
                ksh[e] = 0.0;
                vsh[e] = 0.0;
            }
        }
        workgroupBarrier();

        // Partial dot product: this lane's CH channels of this tile's key `row`.
        var sdot = 0.0;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            sdot = sdot + qsh[d] * ksh[row * HD + d];
        }
        part[row * LANES + lane] = sdot;
        workgroupBarrier();

        // One thread per row folds its LANES partials into the scaled score,
        // masking a key past the live sequence length to -inf.
        if (lane == 0u) {
            let j = kt * BC + row;
            if (j < t) {
                let po = row * LANES;
                var ssum = 0.0;
                for (var li = 0u; li < LANES; li = li + 1u) { ssum = ssum + part[po + li]; }
                sc[row] = ssum * scale;
            } else {
                sc[row] = -3.4e38;
            }
        }
        workgroupBarrier();

        // Every thread redundantly re-derives the SAME tile max and
        // exp-weights from the `BC` shared scores - cheap shared-memory
        // reads, and it keeps every thread's private (m, l, o) in lockstep
        // without another barrier round trip (`flash_attn_causal_gqa`'s own
        // "redundant but register-cheap" pattern, applied to the whole tile
        // instead of just the LANES-wide fold).
        var tile_max = -3.4e38;
        for (var r = 0u; r < BC; r = r + 1u) { tile_max = max(tile_max, sc[r]); }
        let m_new = max(m, tile_max);
        let corr = exp(m - m_new);

        var pj: array<f32, 8>;
        var tile_l = 0.0;
        for (var r = 0u; r < BC; r = r + 1u) {
            let e = exp(sc[r] - m_new);
            pj[r] = e;
            tile_l = tile_l + e;
        }
        l = l * corr + tile_l;
        m = m_new;

        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            var acc = 0.0;
            for (var r = 0u; r < BC; r = r + 1u) {
                acc = acc + pj[r] * vsh[r * HD + d];
            }
            o[c] = o[c] * corr + acc;
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    if (row == 0u) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = (b * q_row) + h * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { ctx[o_base + d] = o[c] * inv; }
        }
    }
}
