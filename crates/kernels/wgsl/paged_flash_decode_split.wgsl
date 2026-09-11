// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Split-key phase of a two-pass FlashDecode: each workgroup owns one (sequence, head, key-split) triple and writes a PARTIAL online-softmax state, not a final context
// @how   64-thread workgroup per (sequence, head, split), tiled key staging, 5 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// M2.7's occupancy fix `paged_flash_decode`'s own header (M2.1) named as a
// follow-up ("a split-key-then-combine two-pass shape") and this repo's own
// decision-4 convention required a fresh profile of, not a skip-by-
// precedent, before landing: M2.1 measured `paged_flash_decode` losing to
// the three-stage triad at every batch size because ONE workgroup per
// (sequence, head) serialises `ntiles = cap / BC` barrier-synced tile
// iterations one after another, dispatching far fewer workgroups than the
// triad's per-SCORE parallelism can ever match. This kernel splits that
// serial tile walk itself across MORE workgroups: each one covers only a
// contiguous `tiles_per_split`-tile RANGE of the sequence's key/value
// history (`p.tiles_per_split` is a runtime `Params` field, not a compile
// constant - it changes neither this kernel's control flow shape nor its
// shared-memory footprint, only how many tiles one dispatch instance walks,
// so sweeping it needs no recompile) and writes its own partial `(m, l, o)`
// online-softmax state instead of a normalised context. A caller dispatches
// `batch * n_heads * n_splits` workgroups (`n_splits` chosen host-side from
// the batch's own max live sequence length), giving this design the SAME
// per-workgroup body `paged_flash_decode` already has but MORE independent
// workgroups at the same batch size - directly targeting the serialised-
// tile-walk root cause M2.1's own entry named, not just batch-size
// occupancy.
//
// SAME algorithm as `paged_flash_decode.wgsl` (see that file's header for
// the full derivation: BC=8 keys staged per tile, LANES=8 threads splitting
// each key's head_dim dot product, online-softmax rescale once per tile) -
// this kernel is NOT a different attention formulation, only a bounded tile
// RANGE and a partial-state output. The two-pass split-then-combine shape
// itself is the same one `flash-decoding` (Dao et al.) and vLLM's
// `paged_attention_v2` partition/reduce design use for the identical
// occupancy reason at small batch - no code from either is transcribed here
// (both are independent implementations of the same well-known online-
// softmax merge identity `paged_flash_decode_combine.wgsl` also documents),
// so this is a design correspondence, not a literal port needing a NOTICE.md
// entry.
//
// A workgroup whose split lies entirely past the sequence's own live length
// (`kt0 >= ntiles`) needs NO special case: its tile loop `for kt in
// kt0..kt_end` simply runs zero times, leaving `m = -3.4e38`, `l = 0.0`,
// `o = [0.0; CH]` - the exact degenerate values `paged_flash_decode_combine`
// treats as "no contribution" (its own online-softmax merge already folds a
// `-inf`/`0` partial in as a no-op, the same identity this kernel's own
// per-tile rescale already relies on).
//
//   q             : [batch, n_heads*head_dim]
//   pool_k/pool_v : [num_blocks*block_size, n_kv_heads*head_dim]  (paged pool)
//   block_tables  : [batch, max_bt]  (physical block index per logical block)
//   seq_lens      : [batch]          (live key count per sequence)
//   part_m/part_l : [batch, n_heads, n_splits]        (running max / sum)
//   part_o        : [batch, n_heads, n_splits, head_dim]  (UNNORMALISED accumulator)
//
// Shared memory: identical to `paged_flash_decode.wgsl` - qsh 512 B + ksh/vsh
// 4 KiB each + part 256 B + sc 32 B ~= 8.8 KiB, comfortably under WebGPU's
// guaranteed 16 KiB `maxComputeWorkgroupStorageSize` floor. `tiles_per_split`
// bounds ONLY the loop trip count, never a shared-memory array size, so this
// budget is exactly `paged_flash_decode`'s own regardless of how it is
// tuned.
//
// 8 storage buffers total (5 in + 3 out) - exactly the WebGPU guaranteed
// floor, the same worst case M2.2's `paged_flash_decode_i8` already landed
// at.

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
    max_bt: u32,        // block_tables row stride (blocks per sequence)
    n_splits: u32,      // dispatched split count per (sequence, head)
    tiles_per_split: u32, // contiguous BC-tiles each split's workgroup walks
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<f32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<f32>;
@group(0) @binding(4) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(5) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(6) var<storage, read_write> part_m:       array<f32>;
@group(0) @binding(7) var<storage, read_write> part_l:       array<f32>;
@group(0) @binding(8) var<storage, read_write> part_o:       array<f32>;

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

    // Flat workgroup id -> (b, h, s) - split is the fastest-varying index,
    // matching the host's own flat dispatch order below.
    let wg = wgid.y * nwg.x + wgid.x;
    let s = wg % p.n_splits;
    let wg2 = wg / p.n_splits;
    let h = wg2 % p.n_heads;
    let b = wg2 / p.n_heads;
    if (b >= p.batch) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;                  // 0..63
    let row = lt / LANES;            // 0..7 -> key-in-tile
    let lane = lt % LANES;           // 0..7 -> channel phase

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
    let kt0 = s * p.tiles_per_split;
    let kt_end = min(kt0 + p.tiles_per_split, ntiles);

    for (var kt = kt0; kt < kt_end; kt = kt + 1u) {
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

        var sdot = 0.0;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            sdot = sdot + qsh[d] * ksh[row * HD + d];
        }
        part[row * LANES + lane] = sdot;
        workgroupBarrier();

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
        workgroupBarrier();
    }

    if (row == 0u) {
        let part_idx = (b * p.n_heads + h) * p.n_splits + s;
        if (lane == 0u) {
            part_m[part_idx] = m;
            part_l[part_idx] = l;
        }
        let o_base = part_idx * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { part_o[o_base + d] = o[c]; }
        }
    }
}
