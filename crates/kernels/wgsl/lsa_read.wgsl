// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Latent sparse attention read: score a query's visible paged bucket centroids, pick the top-p, expand them to key positions
// @how   64-thread workgroup per query row: centroid scores to a row scratch, radix select of the p-th score, index-ordered compaction, bitonic sort, scanned expansion
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements long-context attention whose key selection
// never leaves the GPU, for clients that cannot afford a host round trip per
// decoded token. If your team needs expertise in GPU selection and indexing
// kernels, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The device half of `model::lsa::select` (`read_scores`). Query row `r` sits
// at position `pos = seqlen[r] - 1` (its own key is cached, as in every paged
// attention row). Its page-table row is `[built, page_0, page_1, ...]`
// (`1 + max_pages` words): `built` closed cluster blocks have index pages.
// It may read the buckets of the blocks closed before its own whose pages
// are built: `visible = min(pos / cluster_block, built) * buckets_per_block`.
// Bucket `b` lives in row `row0 + page_(b / buckets_per_block) *
// buckets_per_block + b % buckets_per_block` of the pool (`row0` is the
// layer's slot in a pool shared by every sparse layer). A closed block whose page is not built yet is read
// through the window, which then reaches back to `built * cluster_block`: a
// lagging index widens the read, it never drops a key. The selection row has
// room for one such block; the engine keeps the lag within it, and the window
// is clamped there so a larger lag cannot write past the row. This kernel:
//
//   0. scores every visible bucket, `iq[r] . centroid`, into the row's
//      scratch `scores[r * score_stride ..]`;
//   1. finds the p-th largest score by a 32-step radix bisection over the
//      order-preserving u32 image of the f32 scores (`p = min(top_p,
//      visible)`), counting with a workgroup reduction each step;
//   2. compacts the picked buckets in bucket order: every score above the
//      threshold, and of those equal to it the lowest-indexed ones until `p`
//      are held - the reference's "a tie to the lower bucket";
//   3. bitonic-sorts the picks best first (ties to the lower bucket);
//   4. expands them in that order: each bucket's members that are neither
//      sinks nor in the window, prefix-summed across the workgroup, the
//      first `top_k` kept (the last bucket cut short);
//   5. appends the window `[min(pos + 1 - window, built * cluster_block),
//      pos]` and the sinks below it,
//      writing the row's header `[count, r, seqlen[r]]` first - the exact
//      selection rows the sparse attention kernels read, block-table row `r`.
//
// Given the same f32 scores the result is identical to `read_scores`,
// position for position. Members are absolute positions, `0xFFFFFFFF` past a
// bucket's size.
//
// `top_p` (and an early budget's) must not exceed `P_MAX`, and the index
// dimension 64; the host checks both. Workgroup memory: 2 x 1024 u32 picks +
// 64 f32 query + 2 x 64 u32 scan slots ~= 8.8 KiB.
//
//   iq        : [rows, index_dim] unit index queries
//   centroids : [pages * buckets_per_block, index_dim]
//   members   : [pages * buckets_per_block, bucket_cap] u32
//   page_tab  : [rows, 1 + max_pages] built pages, then each block's page
//   seqlen    : [rows] live keys of the row's sequence
//   scores    : [rows, score_stride] scratch
//   sel       : [rows, sel_width] out: header (count, table row, live keys),
//               then positions

const P_MAX: u32 = 1024u;
const WG: u32 = 64u;
const NO_KEY: u32 = 0xFFFFFFFFu;

struct Params {
    rows: u32,
    score_stride: u32,
    max_pages: u32,
    index_dim: u32,
    bucket_cap: u32,
    cluster_block: u32,
    buckets_per_block: u32,
    top_p: u32,
    top_k: u32,
    sink: u32,
    window: u32,
    sel_width: u32,
    early_position: u32,   // 0 = no early budget
    early_top_p: u32,
    early_top_k: u32,
    row0: u32,             // the layer's first bucket row in the shared pool
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       iq:        array<f32>;
@group(0) @binding(2) var<storage, read>       centroids: array<f32>;
@group(0) @binding(3) var<storage, read>       members:   array<u32>;
@group(0) @binding(4) var<storage, read>       page_tab:  array<u32>;
@group(0) @binding(5) var<storage, read>       seqlen:    array<u32>;
@group(0) @binding(6) var<storage, read_write> scores:    array<f32>;
@group(0) @binding(7) var<storage, read_write> sel:       array<u32>;

var<workgroup> pick_key: array<u32, 1024>; // P_MAX
var<workgroup> pick_idx: array<u32, 1024>; // P_MAX
var<workgroup> part_a:   array<u32, 64>;   // WG
var<workgroup> part_b:   array<u32, 64>;   // WG
var<workgroup> qsh:      array<f32, 64>;   // index_dim <= 64
var<workgroup> total_a:  u32;
var<workgroup> total_b:  u32;

// The centroid/member row of visible bucket `b` of the row whose page table starts at `ptab`.
fn bucket_row(ptab: u32, b: u32) -> u32 {
    return p.row0 + page_tab[ptab + 1u + b / p.buckets_per_block] * p.buckets_per_block + b % p.buckets_per_block;
}

// Order-preserving image of an f32 in u32: larger float, larger key.
fn order_key(s: f32) -> u32 {
    let bits = bitcast<u32>(s);
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

// Whether pick (ka, ia) sorts before (kb, ib): higher score first, a tie to the lower bucket.
fn before(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    return ka > kb || (ka == kb && ia < ib);
}

// Exclusive scan of part_a (and part_b) by thread 0; totals in total_a/b.
fn scan_parts(lt: u32) {
    workgroupBarrier();
    if (lt == 0u) {
        var a = 0u;
        var b = 0u;
        for (var t = 0u; t < WG; t = t + 1u) {
            let va = part_a[t];
            let vb = part_b[t];
            part_a[t] = a;
            part_b[t] = b;
            a = a + va;
            b = b + vb;
        }
        total_a = a;
        total_b = b;
    }
    workgroupBarrier();
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wgid.y * nwg.x + wgid.x;
    if (r >= p.rows) { return; }
    let lt = lid.x;

    let live = seqlen[r];
    let pos = live - 1u;
    var top_p = p.top_p;
    var top_k = p.top_k;
    if (pos < p.early_position) {
        top_p = p.early_top_p;
        top_k = p.early_top_k;
    }
    let ptab = r * (p.max_pages + 1u);
    let built = min(page_tab[ptab], p.max_pages);
    let visible = min(pos / p.cluster_block, built) * p.buckets_per_block;
    let picked = min(min(top_p, visible), P_MAX);
    let srow = r * p.score_stride;

    // 0. Score every visible bucket into the row's scratch.
    for (var c = lt; c < p.index_dim; c = c + WG) {
        qsh[c] = iq[r * p.index_dim + c];
    }
    workgroupBarrier();
    for (var b = lt; b < visible; b = b + WG) {
        let crow = bucket_row(ptab, b);
        var acc = 0.0;
        for (var c = 0u; c < p.index_dim; c = c + 1u) {
            acc = acc + qsh[c] * centroids[crow * p.index_dim + c];
        }
        scores[srow + b] = acc;
    }
    storageBarrier();
    workgroupBarrier();

    // 1. Radix bisection for the threshold key: the largest t with at least
    //    `picked` scores >= t, i.e. the picked-th largest score.
    var threshold = 0u;
    if (picked > 0u) {
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let candidate = threshold | (0x80000000u >> bit);
            var count = 0u;
            for (var b = lt; b < visible; b = b + WG) {
                count = count + select(0u, 1u, order_key(scores[srow + b]) >= candidate);
            }
            part_a[lt] = count;
            part_b[lt] = 0u;
            scan_parts(lt);
            if (workgroupUniformLoad(&total_a) >= picked) {
                threshold = candidate;
            }
        }
    }

    // 2. Compaction in bucket order. Each thread owns a contiguous range so
    //    an exclusive scan of its counts places its picks in global order.
    let chunk = (visible + WG - 1u) / WG;
    let lo = min(lt * chunk, visible);
    let hi = min(lo + chunk, visible);
    var above = 0u;
    var ties = 0u;
    for (var b = lo; b < hi; b = b + 1u) {
        let k = order_key(scores[srow + b]);
        above = above + select(0u, 1u, k > threshold);
        ties = ties + select(0u, 1u, k == threshold);
    }
    part_a[lt] = above;
    part_b[lt] = ties;
    scan_parts(lt);
    let need = picked - min(workgroupUniformLoad(&total_a), picked);
    if (picked > 0u) {
        var tie_rank = part_b[lt];
        var slot = part_a[lt] + min(tie_rank, need);
        for (var b = lo; b < hi; b = b + 1u) {
            let k = order_key(scores[srow + b]);
            var take = k > threshold;
            if (k == threshold) {
                take = tie_rank < need;
                tie_rank = tie_rank + 1u;
            }
            if (take) {
                pick_key[slot] = k;
                pick_idx[slot] = b;
                slot = slot + 1u;
            }
        }
    }

    // 3. Bitonic sort of the picks, padded to a power of two with entries
    //    that sort last.
    var n2 = 1u;
    while (n2 < picked) { n2 = n2 << 1u; }
    for (var i = picked + lt; i < n2; i = i + WG) {
        pick_key[i] = 0u;
        pick_idx[i] = 0xFFFFFFFFu;
    }
    workgroupBarrier();
    for (var k = 2u; k <= n2; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            for (var i = lt; i < n2; i = i + WG) {
                let l = i ^ j;
                if (l > i) {
                    let ki = pick_key[i];
                    let ii = pick_idx[i];
                    let kl = pick_key[l];
                    let il = pick_idx[l];
                    var swap = before(kl, il, ki, ii);
                    if ((i & k) != 0u) { swap = before(ki, ii, kl, il); }
                    if (swap) {
                        pick_key[i] = kl;
                        pick_idx[i] = il;
                        pick_key[l] = ki;
                        pick_idx[l] = ii;
                    }
                }
            }
            workgroupBarrier();
        }
    }

    // 4. Expansion, best bucket first: eligible members are neither sinks nor
    //    in the window; a scan of per-thread counts orders them.
    var window_start = 0u;
    if (pos + 1u > p.window) { window_start = pos + 1u - p.window; }
    window_start = min(window_start, built * p.cluster_block);
    if (pos + 1u > p.window + p.cluster_block) {
        window_start = max(window_start, pos + 1u - p.window - p.cluster_block);
    }
    let bchunk = (picked + WG - 1u) / WG;
    let blo = min(lt * bchunk, picked);
    let bhi = min(blo + bchunk, picked);
    var eligible = 0u;
    for (var s = blo; s < bhi; s = s + 1u) {
        let mrow = bucket_row(ptab, pick_idx[s]) * p.bucket_cap;
        for (var m = 0u; m < p.bucket_cap; m = m + 1u) {
            let key = members[mrow + m];
            if (key == NO_KEY) { break; }
            eligible = eligible + select(0u, 1u, key >= p.sink && key < window_start);
        }
    }
    part_a[lt] = eligible;
    part_b[lt] = 0u;
    scan_parts(lt);
    let long_range = min(workgroupUniformLoad(&total_a), top_k);
    let out = r * p.sel_width + 3u;
    var at = part_a[lt];
    for (var s = blo; s < bhi; s = s + 1u) {
        let mrow = bucket_row(ptab, pick_idx[s]) * p.bucket_cap;
        for (var m = 0u; m < p.bucket_cap; m = m + 1u) {
            let key = members[mrow + m];
            if (key == NO_KEY) { break; }
            if (key >= p.sink && key < window_start) {
                if (at < top_k) { sel[out + at] = key; }
                at = at + 1u;
            }
        }
    }

    // 5. The window, then the sinks below it, then the row's metadata.
    let window_len = pos + 1u - window_start;
    let sinks = min(p.sink, window_start);
    for (var i = lt; i < window_len; i = i + WG) {
        sel[out + long_range + i] = window_start + i;
    }
    for (var i = lt; i < sinks; i = i + WG) {
        sel[out + long_range + window_len + i] = i;
    }
    if (lt == 0u) {
        sel[r * p.sel_width] = long_range + window_len + sinks;
        sel[r * p.sel_width + 1u] = r;
        sel[r * p.sel_width + 2u] = live;
    }
}
