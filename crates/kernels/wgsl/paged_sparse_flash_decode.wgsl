// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused paged attention over a per-query list of selected key positions (sparse attention read)
// @how   64-thread workgroup per (query row, head), tiled key staging through the block table, online softmax
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32|bf16
// @tpl   pool_k,pool_v -> bf16 storage variant, the same chained
//        `kernels::template::dtype_variant` pair `paged_flash_decode` takes
//        (both pools index with the bare identifier `slot`).
//
// Swedish Embedded AB implements attention that reads only the keys a query
// needs out of a paged KV cache, for clients whose contexts outgrow dense
// attention. If your team needs expertise in sparse GPU attention kernels,
// you can procure our services by sending an email to
// info@swedishembedded.com.
//
// `paged_flash_decode` with the dense key walk `j = 0..seq_len` replaced by a
// walk over an explicit list: query row `r` reads exactly the logical
// positions `sel[r * sel_width + i]`, `i < row_meta[r].len`, resolved
// through its sequence's block table. This is the attention half of any
// selection-based sparse attention (latent sparse attention's bucket read,
// a top-k indexer, block-sparse or window-plus-global patterns): the index
// decides which keys, this kernel reads them, and no selected K/V is ever
// copied into a contiguous buffer.
//
//   q            : [rows, n_heads*head_dim]
//   pool_k/pool_v: [num_blocks*block_size, n_kv_heads*head_dim]  (paged pool)
//   block_tables : [n_tables, max_bt]  physical block per logical block
//   sel          : [rows, sel_width]   selected logical positions (u32)
//   row_meta     : [rows, 3]           (selected count, block-table row,
//                                       live keys of that sequence)
//   ctx          : [rows, n_heads*head_dim]
//
// Rows are independent queries: decode passes one row per sequence, prefill
// or a chunked step passes many rows sharing one block-table row, each with
// its own causal selection.
//
// A selected position at or past its sequence's live length is masked out
// (`-inf` score), never read: a stale or corrupt selection must not reach
// another sequence's KV through a block-table slot that is not yet its own.
// The selection's own contract (unique, causal positions) is the indexer's;
// a duplicate is attended twice, exactly as listed.
//
// Selecting `0..seq_len` in ascending order reproduces `paged_flash_decode`
// bit for bit: same tiles, same keys per tile, same reduction order. That is
// the gate that says the indirection adds nothing of its own.
//
// GPU-only (`@cpu no`) for the same reason as `paged_flash_decode`: five
// top-level barriers. Shared memory is that kernel's ~8.8 KiB.

const BC: u32 = 8u;      // keys staged and scored per tile
const LANES: u32 = 8u;   // threads cooperating on one key's head_dim reduction
const CH: u32 = 16u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

struct Params {
    rows: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 128
    group: u32,        // n_heads / n_kv_heads
    block_size: u32,
    max_bt: u32,       // block_tables row stride (blocks per sequence)
    sel_width: u32,    // sel row stride
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<f32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<f32>;
@group(0) @binding(4) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(5) var<storage, read>       sel:          array<u32>;
@group(0) @binding(6) var<storage, read>       row_meta:     array<u32>;
@group(0) @binding(7) var<storage, read_write> ctx:          array<f32>;

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

    let wg = wgid.y * nwg.x + wgid.x;
    let h = wg % p.n_heads;
    let r = wg / p.n_heads;
    if (r >= p.rows) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;
    let count = row_meta[r * 3u];
    let table = row_meta[r * 3u + 1u] * p.max_bt;
    let live = row_meta[r * 3u + 2u];
    let sel_base = r * p.sel_width;

    let lt = lid.x;
    let row = lt / LANES;
    let lane = lt % LANES;

    let q_base = (r * q_row) + h * hd;
    for (var e = lt; e < HD; e = e + 64u) {
        qsh[e] = select(0.0, q[q_base + e], e < hd);
    }
    workgroupBarrier();

    var o: array<f32, 16>;
    for (var c = 0u; c < CH; c = c + 1u) { o[c] = 0.0; }
    var m = -3.4e38;
    var l = 0.0;

    let ntiles = (count + BC - 1u) / BC;

    for (var kt = 0u; kt < ntiles; kt = kt + 1u) {
        // Stage the tile's selected K,V rows. For a fixed selected key,
        // consecutive threads read consecutive channels, so each key's row
        // is one coalesced sweep wherever its block lives.
        for (var e = lt; e < BC * HD; e = e + 64u) {
            let jr = e / HD;
            let d = e % HD;
            let i = kt * BC + jr;
            var kval = 0.0;
            var vval = 0.0;
            if (i < count && d < hd) {
                let j = sel[sel_base + i];
                if (j < live) {
                    let physical = block_tables[table + j / p.block_size];
                    let slot = (physical * p.block_size + (j % p.block_size)) * kv_row + hkv * hd + d;
                    kval = pool_k[slot];
                    vval = pool_v[slot];
                }
            }
            ksh[e] = kval;
            vsh[e] = vval;
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
            let i = kt * BC + row;
            var keep = false;
            if (i < count) { keep = sel[sel_base + i] < live; }
            if (keep) {
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
        for (var t = 0u; t < BC; t = t + 1u) { tile_max = max(tile_max, sc[t]); }
        let m_new = max(m, tile_max);
        let corr = exp(m - m_new);

        var pj: array<f32, 8>;
        var tile_l = 0.0;
        for (var t = 0u; t < BC; t = t + 1u) {
            let e = exp(sc[t] - m_new);
            pj[t] = e;
            tile_l = tile_l + e;
        }
        l = l * corr + tile_l;
        m = m_new;

        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            var acc = 0.0;
            for (var t = 0u; t < BC; t = t + 1u) {
                acc = acc + pj[t] * vsh[t * HD + d];
            }
            o[c] = o[c] * corr + acc;
        }
        workgroupBarrier();
    }

    if (row == 0u) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = (r * q_row) + h * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { ctx[o_base + d] = o[c] * inv; }
        }
    }
}
