// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Latent sparse attention tree read: a beam down a query's node bucket levels, then its top-p buckets expanded to key positions
// @how   64-thread workgroup per query row: each level's candidates scored into a row scratch, a radix select of the best (score, then lower id), children carried down, the bucket picks bitonic-sorted and expanded as the flat read does
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements attention whose read stays bounded as the
// context grows into the millions of tokens, for clients whose documents and
// sessions outgrow a flat index. If your team needs expertise in GPU beam
// search and hierarchical indexing, you can procure our services by sending
// an email to info@swedishembedded.com.
//
// The device half of `model::lsa::tree::select_tree`. Query row `r`'s table
// row (`table_stride` words from `r * table_stride`) holds its live keys, its
// built pages and their bucket pages (as `lsa_read`'s page table), then from
// `growth_off` the levels as they stood at its page's start - covered
// buckets, then made and closed node buckets a level - and from each level's
// `tab_off` that level's node pages. Logical node `n` of level `l` lives in
// node row `row0_l + page_(n / npp_l) * npp_l + n % npp_l` of `nodes`, a row
// being the centroid (`index_dim` floats) then the node's children (logical
// rows of the level below, as u32 bits).
//
//   1. The beam: from the start level (the highest made, or the lowest of at
//      most `start_nodes` node buckets, scored whole) down to level 1, the
//      carried children beside the level's node buckets no level above
//      covers are scored and the best `beam_l` kept - higher score first, a
//      tie to the lower node - and their children carried down.
//   2. The buckets: the last beam's children beside the visible buckets no
//      node bucket covers are scored and the best `top_p` picked (the same
//      order), sorted best first, and expanded with the window and sinks
//      exactly as `lsa_read` does.
//
// The best `m` of a candidate list is found without sorting it: a radix
// bisection over the order-preserving image of the scores, then over the
// inverted ids among the scores equal to the threshold. Scratch per row
// (`cand_stride` words each): candidate ids and their score keys. Candidate
// `e` is only ever written and read by thread `e % 64` (a stage's stale
// values must never reach another thread's later read); what crosses threads
// - a level's kept node buckets, the picked buckets - goes through workgroup
// memory. Limits: beam widths <= 2048, top_p <= 1024.

const WG: u32 = 64u;
const P_MAX: u32 = 1024u;
const MAX_LEVELS: u32 = 8u;
const NO_KEY: u32 = 0xFFFFFFFFu;

struct Params {
    rows: u32,
    index_dim: u32,
    bucket_cap: u32,
    cluster_block: u32,
    buckets_per_block: u32,
    top_p: u32,
    top_k: u32,
    sink: u32,
    window: u32,
    sel_width: u32,
    max_pages: u32,
    table_stride: u32,
    growth_off: u32,
    top_level: u32,
    node_bucket_size: u32,
    start_nodes: u32,     // 0: start at the highest level made
    cand_stride: u32,
    row0: u32,            // the layer's first bucket row in the shared pool
    pad0: u32,
    pad1: u32,
    levels: array<vec4<u32>, MAX_LEVELS>,   // (tab_off, npp, beam, row0) a level, level 1 first
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       iq:        array<f32>;
@group(0) @binding(2) var<storage, read>       centroids: array<f32>;
@group(0) @binding(3) var<storage, read>       members:   array<u32>;
@group(0) @binding(4) var<storage, read>       nodes:     array<f32>;
@group(0) @binding(5) var<storage, read>       tables:    array<u32>;
@group(0) @binding(6) var<storage, read_write> cand:      array<u32>;
@group(0) @binding(7) var<storage, read_write> sel:       array<u32>;
@group(0) @binding(8) var<storage, read_write> stats:     array<atomic<u32>>;

var<workgroup> pick_key: array<u32, 1024>;
var<workgroup> pick_idx: array<u32, 1024>;
var<workgroup> part_a:   array<u32, 64>;
var<workgroup> part_b:   array<u32, 64>;
var<workgroup> qsh:      array<f32, 1024>;
var<workgroup> total_a:  u32;
var<workgroup> total_b:  u32;

// A level's kept node buckets, in the two pick arrays (2048 entries).
fn set_kept(i: u32, id: u32) {
    if (i < P_MAX) { pick_idx[i] = id; } else { pick_key[i - P_MAX] = id; }
}

fn get_kept(i: u32) -> u32 {
    if (i < P_MAX) { return pick_idx[i]; }
    return pick_key[i - P_MAX];
}

fn count(at: u32, v: u32) {
    let old = atomicAdd(&stats[at], v);
    if (old > 0xFFFFFFFFu - v) { atomicAdd(&stats[at + 1u], 1u); }
}

fn order_key(s: f32) -> u32 {
    let bits = bitcast<u32>(s);
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn before(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    return ka > kb || (ka == kb && ia < ib);
}

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

fn bucket_row(tab: u32, b: u32) -> u32 {
    return p.row0 + tables[tab + 2u + b / p.buckets_per_block] * p.buckets_per_block + b % p.buckets_per_block;
}

fn node_row(tab: u32, level: u32, n: u32) -> u32 {
    let lv = p.levels[level - 1u];
    return lv.w + tables[tab + lv.x + n / lv.y] * lv.y + n % lv.y;
}

fn dot_query(row: u32, stride: u32, from_nodes: bool) -> f32 {
    var acc = 0.0;
    for (var c = 0u; c < p.index_dim; c = c + 1u) {
        var x = 0.0;
        if (from_nodes) { x = nodes[row * stride + c]; } else { x = centroids[row * p.index_dim + c]; }
        acc = acc + qsh[c] * x;
    }
    return acc;
}

// How many of the `n` candidates in the row's scratch from `base` have key
// >= t (by_id false), or key == t_key and inverted id >= t (by_id true).
fn count_at_least(lt: u32, base: u32, n: u32, t: u32, by_id: bool, t_key: u32) -> u32 {
    var c = 0u;
    for (var e = lt; e < n; e = e + WG) {
        let k = cand[base + p.cand_stride + e];
        if (by_id) {
            c = c + select(0u, 1u, k == t_key && ~cand[base + e] >= t);
        } else {
            c = c + select(0u, 1u, k >= t);
        }
    }
    part_a[lt] = c;
    part_b[lt] = 0u;
    scan_parts(lt);
    return workgroupUniformLoad(&total_a);
}

// The best `m` of `n` scored candidates: returns (key threshold, inverted-id
// threshold); candidate e is kept iff key > tk, or key == tk and ~id >= ti.
fn best_threshold(lt: u32, base: u32, n: u32, m: u32) -> vec2<u32> {
    if (m >= n) { return vec2<u32>(0u, 0u); }
    var tk = 0u;
    for (var bit = 0u; bit < 32u; bit = bit + 1u) {
        let c = tk | (0x80000000u >> bit);
        if (count_at_least(lt, base, n, c, false, 0u) >= m) { tk = c; }
    }
    var above = 0u;
    if (tk < 0xFFFFFFFFu) { above = count_at_least(lt, base, n, tk + 1u, false, 0u); }
    let need = m - above;
    var ti = 0u;
    for (var bit = 0u; bit < 32u; bit = bit + 1u) {
        let c = ti | (0x80000000u >> bit);
        if (count_at_least(lt, base, n, c, true, tk) >= need) { ti = c; }
    }
    return vec2<u32>(tk, ti);
}

fn kept(base: u32, e: u32, th: vec2<u32>, all: bool) -> bool {
    let k = cand[base + p.cand_stride + e];
    return all || k > th.x || (k == th.x && ~cand[base + e] >= th.y);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wgid.y * nwg.x + wgid.x;
    if (r >= p.rows) { return; }
    let lt = lid.x;
    let tab = r * p.table_stride;
    let live = tables[tab];
    let pos = live - 1u;
    let built = min(tables[tab + 1u], p.max_pages);
    let visible = min(pos / p.cluster_block, built) * p.buckets_per_block;
    let base = r * p.cand_stride * 2u;     // ids, keys
    let nbs = p.node_bucket_size;
    let node_stride = p.index_dim + nbs;
    for (var c = lt; c < p.index_dim; c = c + WG) {
        qsh[c] = iq[r * p.index_dim + c];
    }
    workgroupBarrier();

    // The levels the row's page saw, and where the beam starts.
    let g = tab + p.growth_off;
    var levels = 0u;
    for (var l = 1u; l <= p.top_level; l = l + 1u) {
        if (tables[g + 1u + 2u * (l - 1u)] > 0u && levels == l - 1u) { levels = l; }
    }
    var start = levels;
    if (p.start_nodes > 0u) {
        for (var l = levels; l >= 1u; l = l - 1u) {
            if (tables[g + 1u + 2u * (l - 1u)] <= p.start_nodes) { start = l; }
        }
    }

    // 1. The beam. `carried` candidates sit at the front of the id scratch.
    var carried = 0u;
    for (var level = start; level >= 1u; level = level - 1u) {
        let made = tables[g + 1u + 2u * (level - 1u)];
        var first_open = tables[g + 2u + 2u * (level - 1u)];
        if (level == start && p.start_nodes > 0u) { first_open = 0u; }
        let n = min(carried + made - first_open, p.cand_stride);
        // The carried children were written by this thread; the open ones
        // follow them.
        for (var e = lt; e < n; e = e + WG) {
            var id = first_open + e - carried;
            if (e < carried) { id = cand[base + e]; }
            cand[base + e] = id;
            cand[base + p.cand_stride + e] = order_key(dot_query(node_row(tab, level, id), node_stride, true));
        }
        let beam = p.levels[level - 1u].z;
        let m = min(beam, n);
        let th = best_threshold(lt, base, n, m);
        // Kept, in any order, into workgroup memory.
        var mine = 0u;
        for (var e = lt; e < n; e = e + WG) { mine = mine + select(0u, 1u, kept(base, e, th, m >= n)); }
        part_a[lt] = mine;
        part_b[lt] = 0u;
        scan_parts(lt);
        var slot = part_a[lt];
        for (var e = lt; e < n; e = e + WG) {
            if (kept(base, e, th, m >= n)) {
                set_kept(slot, cand[base + e]);
                slot = slot + 1u;
            }
        }
        workgroupBarrier();
        // Their children, carried down.
        carried = min(m * nbs, p.cand_stride);
        for (var e = lt; e < carried; e = e + WG) {
            let row = node_row(tab, level, get_kept(e / nbs));
            cand[base + e] = bitcast<u32>(nodes[row * node_stride + p.index_dim + e % nbs]);
        }
        workgroupBarrier();
    }

    // 2. The buckets: the carried children beside the uncovered visible ones.
    let covered = select(0u, tables[g], levels > 0u);
    var uncovered = 0u;
    if (visible > covered) { uncovered = visible - covered; }
    let n = min(carried + uncovered, p.cand_stride);
    for (var e = lt; e < n; e = e + WG) {
        var id = covered + e - carried;
        if (e < carried) { id = cand[base + e]; }
        cand[base + e] = id;
        cand[base + p.cand_stride + e] = order_key(dot_query(bucket_row(tab, id), 0u, false));
    }
    let picked = min(min(p.top_p, n), P_MAX);
    let th = best_threshold(lt, base, n, picked);
    var mine = 0u;
    for (var e = lt; e < n; e = e + WG) { mine = mine + select(0u, 1u, kept(base, e, th, picked >= n)); }
    part_a[lt] = mine;
    part_b[lt] = 0u;
    scan_parts(lt);
    var slot = part_a[lt];
    for (var e = lt; e < n; e = e + WG) {
        if (kept(base, e, th, picked >= n)) {
            pick_key[slot] = cand[base + p.cand_stride + e];
            pick_idx[slot] = cand[base + e];
            slot = slot + 1u;
        }
    }

    // 3. Best first: bitonic sort padded with entries that sort last.
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

    // 4. Expansion, best bucket first, as `lsa_read`.
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
        let mrow = bucket_row(tab, pick_idx[s]) * p.bucket_cap;
        for (var m = 0u; m < p.bucket_cap; m = m + 1u) {
            let key = members[mrow + m];
            if (key == NO_KEY) { break; }
            eligible = eligible + select(0u, 1u, key >= p.sink && key < window_start);
        }
    }
    part_a[lt] = eligible;
    part_b[lt] = 0u;
    scan_parts(lt);
    let long_range = min(workgroupUniformLoad(&total_a), p.top_k);
    let out = r * p.sel_width + 3u;
    var at = part_a[lt];
    for (var s = blo; s < bhi; s = s + 1u) {
        let mrow = bucket_row(tab, pick_idx[s]) * p.bucket_cap;
        for (var m = 0u; m < p.bucket_cap; m = m + 1u) {
            let key = members[mrow + m];
            if (key == NO_KEY) { break; }
            if (key >= p.sink && key < window_start) {
                if (at < p.top_k) { sel[out + at] = key; }
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
        count(0u, 1u);
        count(2u, long_range + window_len + sinks);
        count(4u, live);
    }
}
